
import torch
import torch.nn as nn

class LightCurveTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, num_classes=10):
        super().__init__()
        self.time_projector = nn.Linear(1, d_model)
        self.feature_projector = nn.Linear(2, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, times, features):
        time_emb = self.time_projector(times)
        feat_emb = self.feature_projector(features)
        x = time_emb + feat_emb + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LightCurveTransformer().to(device)
dummy_times = torch.randn(4, 200, 1).to(device)
dummy_features = torch.randn(4, 200, 2).to(device)

with torch.no_grad():
    predictions = model(dummy_times, dummy_features)

print(f"Final output shape: {predictions.shape}")
print(f"Device: {predictions.device}")
