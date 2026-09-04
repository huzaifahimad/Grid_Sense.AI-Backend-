"""
GridSense AI — LSTM+CNN Hybrid Load Forecaster

Implements the architecture from ML Model Design doc Section 1.1:
Conv1D layers -> LSTM layers -> quantile output heads (P10/P50/P90).

Only run this AFTER models/train_baselines.py — you need the LightGBM
number to know whether this added complexity is earning its keep.

Usage:
    python -m models.train_lstm_cnn
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import DATA_FEATURES_DIR, MODELS_DIR, LOOKBACK_HOURS

FEATURE_COLS = [
    "load_mw", "temperature_c", "humidity_pct", "wind_speed_ms",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
]
QUANTILES = [0.1, 0.5, 0.9]


class SequenceDataset(Dataset):
    """Builds sliding windows of LOOKBACK_HOURS -> single-step-ahead target."""

    def __init__(self, df: pd.DataFrame, lookback: int, feature_cols: list, target_col: str = "load_mw"):
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.samples = []

        for zone, g in df.groupby("zone"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            values = g[feature_cols].values
            targets = g[target_col].values
            for i in range(lookback, len(g) - 1):
                x = values[i - lookback:i]
                y = targets[i + 1]  # 1-hour-ahead target; extend for multi-horizon
                if not (np.isnan(x).any() or np.isnan(y)):
                    self.samples.append((x.astype(np.float32), np.float32(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y)


class LSTMCNNForecaster(nn.Module):
    """Matches ML Model Design Section 1.1: Conv1D -> LSTM -> Dense quantile heads."""

    def __init__(self, n_features: int, n_quantiles: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        self.lstm1 = nn.LSTM(32, 128, batch_first=True, dropout=0.0)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, n_quantiles)  # one output per quantile

    def forward(self, x):
        # x: [batch, seq_len, n_features] -> conv expects [batch, n_features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # back to [batch, seq_len, channels]

        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, (h_n, _) = self.lstm2(x)
        x = self.dropout2(h_n[-1])  # last hidden state

        x = self.relu(self.fc1(x))
        return self.fc_out(x)  # [batch, n_quantiles]


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """Quantile/pinball loss — required for the P10/P50/P90 uncertainty bands (TRD FR-5)."""
    target = target.unsqueeze(1)
    errors = target - preds
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(torch.max((q - 1) * errors[:, i], q * errors[:, i]))
    return torch.stack(losses, dim=1).mean()


def train(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run features/build_features.py first.")

    df = pd.read_parquet(feat_path)

    # Same walk-forward split logic as baselines — MUST stay consistent for fair comparison
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * 0.8)
    cutoff_time = df.iloc[cutoff_idx]["timestamp"]
    train_df = df[df["timestamp"] < cutoff_time]
    test_df = df[df["timestamp"] >= cutoff_time]

    # Normalize using train statistics only — leaking test stats into normalization
    # is a subtle way to inflate your reported accuracy, avoid it
    means = train_df[FEATURE_COLS].mean()
    stds = train_df[FEATURE_COLS].std().replace(0, 1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[FEATURE_COLS] = (train_df[FEATURE_COLS] - means) / stds
    test_df[FEATURE_COLS] = (test_df[FEATURE_COLS] - means) / stds

    train_ds = SequenceDataset(train_df, LOOKBACK_HOURS, FEATURE_COLS)
    test_ds = SequenceDataset(test_df, LOOKBACK_HOURS, FEATURE_COLS)
    print(f"[lstm_cnn] train sequences: {len(train_ds)}, test sequences: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMCNNForecaster(n_features=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = pinball_loss(preds, y, QUANTILES)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += pinball_loss(preds, y, QUANTILES).item() * x.size(0)

        print(f"[lstm_cnn] epoch {epoch+1:2d}/{epochs}  "
              f"train_loss={total_loss/len(train_ds):.4f}  val_loss={val_loss/len(test_ds):.4f}")

    model_path = MODELS_DIR / "lstm_cnn_forecast.pt"
    torch.save({
        "model_state": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "quantiles": QUANTILES,
        "lookback_hours": LOOKBACK_HOURS,
        "norm_means": means.to_dict(),
        "norm_stds": stds.to_dict(),
    }, model_path)
    print(f"[lstm_cnn] saved model -> {model_path}")
    print("[lstm_cnn] IMPORTANT: compute MAPE/MAE/RMSE on test set and compare against "
          "data/models/baseline_results.csv before deciding this model ships.")


if __name__ == "__main__":
    train()
