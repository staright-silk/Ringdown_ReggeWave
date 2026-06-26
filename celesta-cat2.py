 #!/usr/bin/env python3

"""
FT-Transformer (Feature Tokenizer + Transformer) for KOI Binary Classification
================================================================================

Implementation follows Gorishniy et al. 2021, "Revisiting Deep Learning Models
for Tabular Data" (NeurIPS 2021) -- the paper that introduced FT-Transformer.

Architecture:
  1. Feature Tokenizer: each numerical feature is projected into a d-dimensional
     embedding via a learned per-feature linear map. Categorical features (if any)
     use a learned embedding table per category. A [CLS] token is prepended.
  2. Transformer encoder: standard multi-head self-attention blocks operate over
     the sequence of per-feature tokens, letting the model learn feature
     interactions directly via attention (rather than via tree splits or MLP
     layers mixing all features together).
  3. Prediction head: the [CLS] token's final representation is passed through
     a small head to produce the classification logit.

This is a binary classification setup: PLANET_SIGNAL (Confirmed + Candidate)
vs FALSE POSITIVE, matching the leakage-free feature pipeline used elsewhere
in this project (celesta_2.py / celesta-binary.py).
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = "KOI_Cumulative_clean.csv"
N_SPLITS = 5
RANDOM_STATE = 42

D_TOKEN = 64           # embedding dimension per feature token
N_LAYERS = 4           # number of transformer encoder blocks
N_HEADS = 8            # attention heads per block
FFN_MULT = 2           # feed-forward hidden size multiplier (FFN dim = D_TOKEN * FFN_MULT)
ATTN_DROPOUT = 0.15
FFN_DROPOUT = 0.15
RESIDUAL_DROPOUT = 0.0

MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 256
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10
LABEL_SMOOTHING = 0.05

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# DATA LOADING + LEAKAGE-FREE FEATURE PIPELINE
# (identical to celesta_2.py / celesta-binary.py for fair comparison)
# ============================================================================

df = pd.read_csv(DATA_PATH)

target_col = "koi_disposition"

DROP_COLS = [
    "rowid", "kepid", "kepoi_name", "kepler_name", "koi_score",
    "koi_pdisposition", "koi_vet_stat", "koi_vet_date", "koi_disp_prov",
    "koi_comment", "koi_datalink_dvr", "koi_datalink_dvs", "koi_trans_mod",
    "koi_limbdark_mod", "koi_quarters", "koi_parm_prov", "koi_sparprov",
    "koi_tce_delivname", "koi_fittype",
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

y_raw = df[target_col].copy()
y_raw = y_raw.replace({"CANDIDATE": "PLANET_SIGNAL", "CONFIRMED": "PLANET_SIGNAL"})
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

for col in ["koi_period", "koi_depth", "koi_prad", "koi_insol"]:
    if col in X.columns:
        X[f"{col}_log"] = np.log1p(np.maximum(X[col].fillna(0), 0))

eps = 1e-6

if {"koi_depth", "koi_duration"}.issubset(X.columns):
    X["eng_depth_per_duration"] = X["koi_depth"] / (X["koi_duration"] + eps)

if {"koi_prad", "koi_srad"}.issubset(X.columns):
    X["eng_prad_to_srad"] = X["koi_prad"] / (X["koi_srad"] + eps)

if {"koi_duration", "koi_period"}.issubset(X.columns):
    X["eng_duration_frac_period"] = X["koi_duration"] / (X["koi_period"] + eps)

if {"koi_duration", "koi_period", "koi_dor"}.issubset(X.columns):
    expected_duration = X["koi_period"] / (np.pi * (X["koi_dor"] + eps))
    X["eng_duration_vs_expected"] = X["koi_duration"] / (expected_duration + eps)

if {"koi_dicco_msky", "koi_dikco_msky"}.issubset(X.columns):
    X["eng_centroid_offset_diff"] = X["koi_dicco_msky"] - X["koi_dikco_msky"]

if {"koi_dicco_msky", "koi_model_snr"}.issubset(X.columns):
    X["eng_centroid_offset_per_snr"] = X["koi_dicco_msky"] * X["koi_model_snr"]

if {"koi_model_snr", "koi_num_transits"}.issubset(X.columns):
    X["eng_snr_x_transits"] = X["koi_model_snr"] * X["koi_num_transits"]

if {"koi_insol", "koi_teq"}.issubset(X.columns):
    X["eng_insol_per_teq4"] = X["koi_insol"] / (X["koi_teq"] ** 4 + eps)

if {"koi_prad", "koi_period"}.issubset(X.columns):
    X["eng_prad_x_period_log"] = (
        np.log1p(np.maximum(X["koi_prad"].fillna(0), 0))
        * np.log1p(np.maximum(X["koi_period"].fillna(0), 0))
    )

core_measurement_cols = [
    c for c in [
        "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_srad",
        "koi_smass", "koi_steff", "koi_slogg", "koi_smet", "koi_dor",
        "koi_insol", "koi_teq", "koi_model_snr", "koi_num_transits",
        "koi_dicco_msky", "koi_dikco_msky",
    ] if c in X.columns
]
X["eng_n_missing_core"] = X[core_measurement_cols].isna().sum(axis=1)

X = X.replace([np.inf, -np.inf], np.nan)

all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    X = X.drop(columns=all_nan_cols)

X = X.astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

print(f"Classes: {le.classes_}")
print(f"Feature count (tokens): {X.shape[1]}")
assert num_classes == 2, "This script is configured for binary classification."


# ============================================================================
# FT-TRANSFORMER ARCHITECTURE
# ============================================================================

class NumericalFeatureTokenizer(nn.Module):
    """
    Tokenizes continuous numerical features into d_token-dimensional embeddings.

    For each feature x_i, the token is: x_i * W_i + b_i
    where W_i, b_i are learned per-feature (i.e. NOT shared across features --
    this is what lets the transformer treat each column as a distinct,
    learnable "word" in its attention vocabulary).
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # weight: (n_features, d_token), one learned projection vector per feature
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(n_features), 1 / math.sqrt(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> tokens: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CLSToken(nn.Module):
    """Learned [CLS] token prepended to the feature-token sequence, whose final
    representation after the transformer encodes the whole-sample prediction."""

    def __init__(self, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        nn.init.uniform_(self.weight, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_token, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out_proj = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_token: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_token * mult
        self.net = nn.Sequential(
            nn.Linear(d_token, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual, LN -> FFN -> residual."""

    def __init__(self, d_token: int, n_heads: int, ffn_mult: int,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = MultiHeadSelfAttention(d_token, n_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = FeedForward(d_token, ffn_mult, ffn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attn(self.norm1(x)))
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """
    Full FT-Transformer for tabular binary classification.

    Pipeline:
      numerical features -> per-feature tokenizer -> prepend [CLS] ->
      N transformer encoder blocks -> take [CLS] representation ->
      LayerNorm -> GELU -> Linear -> single logit (binary classification)
    """

    def __init__(self, n_features: int, d_token: int, n_layers: int, n_heads: int,
                 ffn_mult: int, attn_dropout: float, ffn_dropout: float,
                 residual_dropout: float, n_classes: int = 2):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = CLSToken(d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_mult, attn_dropout,
                              ffn_dropout, residual_dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token, 1 if n_classes == 2 else n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.tokenizer(x)                          # (B, n_features, d_token)
        cls = self.cls_token(B)                              # (B, 1, d_token)
        seq = torch.cat([cls, tokens], dim=1)                # (B, 1+n_features, d_token)

        for block in self.blocks:
            seq = block(seq)

        cls_out = self.final_norm(seq[:, 0])                 # (B, d_token) -- [CLS] representation
        logit = self.head(cls_out)                           # (B, 1) for binary
        return logit.squeeze(-1)


# ============================================================================
# TRAINING LOOP (per fold)
# ============================================================================

def cosine_warmup_lr(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one_fold(X_train, y_train, X_valid, y_valid, device):
    pos_weight_value = (
        (y_train == 0).sum() / max(1, (y_train == 1).sum())
    )
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    valid_X_t = torch.tensor(X_valid, dtype=torch.float32, device=device)
    valid_y_t = torch.tensor(y_valid, dtype=torch.float32, device=device)

    model = FTTransformer(
        n_features=X_train.shape[1],
        d_token=D_TOKEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        attn_dropout=ATTN_DROPOUT,
        ffn_dropout=FFN_DROPOUT,
        residual_dropout=RESIDUAL_DROPOUT,
        n_classes=2,
    ).to(device)

    # No weight decay on biases / norm params -- standard transformer training practice
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "cls_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=BASE_LR,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS),
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Label smoothing for binary targets: pull hard 0/1 labels slightly
            # toward 0.5, which discourages overconfident logits and improves
            # calibration -- standard practice in modern transformer training.
            yb_smoothed = yb * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_smoothed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(valid_X_t)
            val_loss = criterion(val_logits, valid_y_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(valid_X_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        preds = (val_probs >= 0.5).astype(int)

    return preds, val_probs, epoch + 1


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

acc_scores, prec_scores, rec_scores, f1_scores, auc_scores = [], [], [], [], []
conf_matrix_sum = None

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\nFold {fold}/{N_SPLITS}")

    X_train_raw = X.iloc[train_idx].copy()
    X_valid_raw = X.iloc[valid_idx].copy()
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_valid_imp = imputer.transform(X_valid_raw)

    # FT-Transformer, like the MLP, requires standardized inputs since the
    # per-feature tokenizer weights are shared across all samples and expect
    # roughly comparable input scales across features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)

    preds, probs, n_epochs = train_one_fold(
        X_train_scaled, y_train, X_valid_scaled, y_valid, device=device
    )

    acc = accuracy_score(y_valid, preds)
    prec = precision_score(y_valid, preds, average="weighted", zero_division=0)
    rec = recall_score(y_valid, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_valid, preds, average="weighted", zero_division=0)
    auc = roc_auc_score(y_valid, probs)

    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)
    auc_scores.append(auc)

    cm = confusion_matrix(y_valid, preds)
    conf_matrix_sum = cm if conf_matrix_sum is None else conf_matrix_sum + cm

    print(f"Accuracy={acc:.5f}  F1={f1:.5f}  AUC={auc:.5f}  (stopped at epoch {n_epochs})")

print("\n" + "=" * 80)
print("FT-TRANSFORMER CROSS VALIDATION RESULTS")
print("=" * 80)
print(f"Accuracy : {np.mean(acc_scores):.5f} +/- {np.std(acc_scores):.5f}")
print(f"Precision: {np.mean(prec_scores):.5f}")
print(f"Recall   : {np.mean(rec_scores):.5f}")
print(f"F1 Score : {np.mean(f1_scores):.5f}")
print(f"ROC AUC  : {np.mean(auc_scores):.5f}")

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print("Class order:", le.classes_)
print(conf_matrix_sum)

n_params = sum(p.numel() for p in FTTransformer(
    n_features=X.shape[1], d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
    ffn_mult=FFN_MULT, attn_dropout=ATTN_DROPOUT, ffn_dropout=FFN_DROPOUT,
    residual_dropout=RESIDUAL_DROPOUT, n_classes=2,
).parameters())
print(f"\nModel parameter count: {n_params:,}")

print("\nFinished Successfully.")
