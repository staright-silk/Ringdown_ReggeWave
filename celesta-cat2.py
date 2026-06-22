#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from catboost import (
    CatBoostClassifier
)


# ==========================
# CONFIG
# ==========================

DATA_PATH = (
    "KOI_Cumulative_clean.csv"
)

N_SPLITS = 5
RANDOM_STATE = 42


# ==========================
# LOAD
# ==========================

df = pd.read_csv(
    DATA_PATH
)

TARGET = (
    "koi_disposition"
)


DROP_COLS = [

    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",

    "koi_score",

    "koi_pdisposition",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_disp_prov",

    "koi_comment",

    "koi_datalink_dvr",
    "koi_datalink_dvs",

    "koi_trans_mod",
    "koi_limbdark_mod",

    "koi_quarters",

    "koi_parm_prov",

    "koi_sparprov",

    "koi_tce_delivname",

    "koi_fittype"
]


# ==========================
# FEATURE ENGINEERING
# ==========================

df["planet_star_ratio"] = (
    df["koi_prad"] /
    (df["koi_srad"] + 1e-6)
)

df["temp_flux_ratio"] = (
    df["koi_teq"] /
    (df["koi_insol"] + 1)
)

df["transit_strength"] = (
    df["koi_depth"]
    *
    df["koi_model_snr"]
)

df["duration_period_ratio"] = (
    df["koi_duration"]
    /
    (df["koi_period"] + 1e-6)
)


# ==========================
# REMOVE UNUSED
# ==========================

df = df.drop(
    columns=[
        c
        for c
        in DROP_COLS
        if c in df.columns
    ]
)


# only candidate vs confirmed

df = df[
    df[TARGET]
    !=
    "FALSE POSITIVE"
]


# ==========================
# TARGET
# ==========================

y_raw = (
    df[TARGET]
)

X = (
    df.drop(
        columns=[
            TARGET
        ]
    )
)


# ==========================
# CATEGORICAL
# ==========================

cat_cols = (
    X.select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)

cat_idx = [
    X.columns
    .get_loc(c)
    for c
    in cat_cols
]


# ==========================
# REMOVE EMPTY COLS
# ==========================

X = X.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)

empty = (
    X.columns[
        X.isna()
        .all()
    ]
)

X = (
    X.drop(
        columns=empty
    )
)


# ==========================
# LABELS
# ==========================

le = (
    LabelEncoder()
)

y = (
    le.fit_transform(
        y_raw
    )
)


# ==========================
# CV
# ==========================

skf = (
    StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
)


acc_scores = []
f1_scores = []
macro_scores = []

conf_matrix = None


# ==========================
# TRAIN
# ==========================

for fold, (
    train_idx,
    valid_idx

) in enumerate(

    skf.split(
        X,
        y
    ),

    1
):

    print(
        f"\nFold {fold}"
    )

    X_train = (
        X.iloc[
            train_idx
        ]
        .copy()
    )

    X_valid = (
        X.iloc[
            valid_idx
        ]
        .copy()
    )

    y_train = (
        y[
            train_idx
        ]
    )

    y_valid = (
        y[
            valid_idx
        ]
    )


    # impute

    imp = (
        SimpleImputer(
            strategy="median"
        )
    )

    num_cols = (
        X_train
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    X_train[
        num_cols
    ] = (
        imp.fit_transform(
            X_train[
                num_cols
            ]
        )
    )

    X_valid[
        num_cols
    ] = (
        imp.transform(
            X_valid[
                num_cols
            ]
        )
    )


    # weights

    cw = (
        compute_class_weight(
            "balanced",
            classes=np.unique(
                y_train
            ),
            y=y_train
        )
    )

    class_weights = (
        cw.tolist()
    )


    # model

    model = (

        CatBoostClassifier(

            task_type="GPU",

            iterations=5000,

            depth=16,

            learning_rate=0.0005,

            loss_function="Logloss",

            eval_metric="F1",

            class_weights=class_weights,

            l2_leaf_reg=5,

            bootstrap_type="Bernoulli",

            subsample=0.85,

            od_type="Iter",

            od_wait=150,

            random_seed=RANDOM_STATE,

            verbose=False
        )

    )


    model.fit(

        X_train,

        y_train,

        cat_features=cat_idx,

        eval_set=(

            X_valid,

            y_valid

        ),

        use_best_model=True
    )


    preds = (
        model.predict(
            X_valid
        )
        .reshape(-1)
        .astype(int)
    )


    acc = (
        accuracy_score(
            y_valid,
            preds
        )
    )

    f1 = (
        f1_score(

            y_valid,

            preds,

            average="weighted"
        )
    )

    macro = (
        f1_score(

            y_valid,

            preds,

            average="macro"
        )
    )

    acc_scores.append(
        acc
    )

    f1_scores.append(
        f1
    )

    macro_scores.append(
        macro
    )


    cm = confusion_matrix(
        y_valid,
        preds
    )

    if conf_matrix is None:
        conf_matrix = cm
    else:
        conf_matrix += cm


    print(
        f"Accuracy={acc:.5f} "
        f"F1={f1:.5f} "
        f"MacroF1={macro:.5f}"
    )

    
    print("\n" + "=" * 80)
    print("CROSS VALIDATION RESULTS")
    print("=" * 80)
    
    print(
        f"Accuracy : {np.mean(acc_scores):.5f} Â± {np.std(acc_scores):.5f}"
    )
    
    print(
        f"Weighted F1 : {np.mean(f1_scores):.5f}"
    )
    
    print(
        f"Macro F1    : {np.mean(macro_scores):.5f}"
    )
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    
    print(
        conf_matrix
    )
    
    print("\nFinished Successfully.")
    
    print("\nClass mapping:")
    
    print(
        le.classes_
    )
