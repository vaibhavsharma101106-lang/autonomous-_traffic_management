"""
Module 4: Congestion Prediction
IT 207 Human Intelligence Project

Implements three prediction models with time-series features:
    1. Random Forest (baseline)
    2. XGBoost (primary ML model)
    3. LSTM (deep learning, PyTorch)

Human Intelligence Mapping — Learning:
    Humans learn traffic patterns from accumulated daily experience — they know
    rush hour is busy on weekdays at 8 AM, and adjust expectations accordingly.
    This module trains on historical observations to learn such temporal patterns,
    then generalises to unseen traffic conditions (transfer of learning).
    Working Memory is also engaged: lag features and rolling statistics encode
    the agent's short-term 'memory' of recent conditions into the feature vector.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils import CONFIG, PROJECT_ROOT, engineer_features, classify_traffic_state


# ── PyTorch LSTM architecture ─────────────────────────────────────────────────
class TrafficLSTM(nn.Module):
    """
    Bidirectional LSTM for multi-step traffic count forecasting.

    Analogous to human 'sequential memory' — the LSTM hidden state acts as
    a compressed representation of recent temporal context, similar to how
    humans encode experience into retrievable patterns.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_steps: int = 6):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_steps = output_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last    = self.dropout(out[:, -1, :])  # last time step
        return self.head(last)         # (batch, output_steps)


class SequenceDataset(Dataset):
    """Sliding-window sequence dataset for LSTM training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int, horizon: int):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        # Build valid index windows
        self.indices = []
        for i in range(seq_len, len(features) - horizon + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i  = self.indices[idx]
        xw = self.X[i - self.seq_len: i]
        yw = self.y[i: i + self.horizon, 0]   # target = vehicle_count
        return xw, yw


# ── Random Forest model ───────────────────────────────────────────────────────
class RandomForestPredictor:
    """Random Forest baseline — tabular features, single-step output."""

    def __init__(self, **kwargs):
        rf_cfg = CONFIG["prediction"]["random_forest"]
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", rf_cfg["n_estimators"]),
            max_depth=kwargs.get("max_depth", rf_cfg["max_depth"]),
            min_samples_split=kwargs.get("min_samples_split", rf_cfg["min_samples_split"]),
            random_state=CONFIG["prediction"].get("random_seed", 42),
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self.scaler = MinMaxScaler()
        self.trained = False

    def fit(self, df: pd.DataFrame, target_col: str = "vehicle_count",
            horizon: int = 1):
        df = engineer_features(df, target_col=target_col)
        self.feature_cols = [c for c in df.columns
                             if c not in [target_col, "timestamp", "source",
                                          "lane_id", "traffic_state",
                                          "density", "queue_length", "state_color"]]
        X = df[self.feature_cols].values
        y = df[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.trained = True
        train_pred = self.model.predict(X)
        mae  = mean_absolute_error(y, train_pred)
        rmse = np.sqrt(mean_squared_error(y, train_pred))
        logger.info(f"[RF] Train MAE={mae:.3f}, RMSE={rmse:.3f}")
        return {"mae": mae, "rmse": rmse}

    def predict(self, df: pd.DataFrame,
                target_col: str = "vehicle_count") -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        df = engineer_features(df, target_col=target_col)
        missing = [c for c in self.feature_cols if c not in df.columns]
        for c in missing:
            df[c] = 0.0
        X = df[self.feature_cols].values
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_cols": self.feature_cols}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model        = data["model"]
        self.scaler       = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.trained      = True


# ── XGBoost model ─────────────────────────────────────────────────────────────
class XGBoostPredictor:
    """XGBoost gradient-boosted tree predictor — primary ML model."""

    def __init__(self, **kwargs):
        cfg = CONFIG["prediction"]["xgboost"]
        device = "cuda" if _cuda_available() else "cpu"
        self.model = xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", cfg["n_estimators"]),
            max_depth=kwargs.get("max_depth", cfg["max_depth"]),
            learning_rate=kwargs.get("learning_rate", cfg["learning_rate"]),
            subsample=kwargs.get("subsample", cfg["subsample"]),
            colsample_bytree=kwargs.get("colsample_bytree", cfg["colsample_bytree"]),
            tree_method="hist",
            device=device,
            random_state=CONFIG["prediction"].get("random_seed", 42),
        )
        self.feature_cols: List[str] = []
        self.scaler  = MinMaxScaler()
        self.trained = False
        self._importances: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target_col: str = "vehicle_count",
            horizon: int = 6,
            val_df: Optional[pd.DataFrame] = None):
        df = engineer_features(df, target_col=target_col)
        self.feature_cols = [c for c in df.columns
                             if c not in [target_col, "timestamp", "source",
                                          "lane_id", "traffic_state",
                                          "density", "queue_length", "state_color"]]
        X = df[self.feature_cols].values
        y = df[target_col].shift(-horizon).dropna().values
        X = X[:len(y)]
        X = self.scaler.fit_transform(X)

        if val_df is not None and not val_df.empty:
            val_df = engineer_features(val_df, target_col=target_col)
            Xv = self.scaler.transform(val_df[self.feature_cols].fillna(0).values)
            yv = val_df[target_col].shift(-horizon).dropna().values
            Xv = Xv[:len(yv)]
            eval_set = [(Xv, yv)]
        else:
            eval_set = None

        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.trained = True
        self._importances = self.model.feature_importances_

        train_pred = self.model.predict(X)
        mae  = mean_absolute_error(y, train_pred)
        rmse = np.sqrt(mean_squared_error(y, train_pred))
        logger.info(f"[XGB] Train MAE={mae:.3f}, RMSE={rmse:.3f}")
        return {"mae": mae, "rmse": rmse}

    def predict(self, df: pd.DataFrame,
                target_col: str = "vehicle_count") -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        df = engineer_features(df, target_col=target_col)
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = self.scaler.transform(df[self.feature_cols].values)
        return self.model.predict(X)

    def feature_importance(self) -> Dict[str, float]:
        if self._importances is None or not self.feature_cols:
            return {}
        return dict(zip(self.feature_cols, self._importances))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path + ".xgb")
        with open(path + ".meta", "wb") as f:
            pickle.dump({"scaler": self.scaler,
                         "feature_cols": self.feature_cols}, f)

    def load(self, path: str):
        self.model = xgb.XGBRegressor()
        self.model.load_model(path + ".xgb")
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        self.scaler       = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        self.trained      = True


# ── LSTM Trainer ──────────────────────────────────────────────────────────────
class LSTMPredictor:
    """PyTorch LSTM multi-step traffic forecaster."""

    def __init__(self, seq_len: int = 12, horizon: int = 6, **kwargs):
        cfg = CONFIG["prediction"]["lstm"]
        self.seq_len     = seq_len
        self.horizon     = horizon
        self.hidden_size = kwargs.get("hidden_size", cfg["hidden_size"])
        self.num_layers  = kwargs.get("num_layers", cfg["num_layers"])
        self.dropout     = kwargs.get("dropout",  cfg["dropout"])
        self.epochs      = kwargs.get("epochs",   cfg["epochs"])
        self.lr          = kwargs.get("lr",       cfg["lr"])
        self.batch_size  = kwargs.get("batch_size", cfg["batch_size"])
        self.patience    = kwargs.get("patience", cfg.get("patience", 15))
        self.device      = "cuda" if _cuda_available() else "cpu"
        self.scaler_X    = MinMaxScaler()
        self.scaler_y    = MinMaxScaler()
        self.model: Optional[TrafficLSTM] = None
        self.feature_cols: List[str] = []
        self.trained = False
        self._train_losses: List[float] = []
        self._val_losses:   List[float] = []

    def fit(self, df: pd.DataFrame, target_col: str = "vehicle_count",
            val_df: Optional[pd.DataFrame] = None):
        df = engineer_features(df, target_col=target_col)
        self.feature_cols = [c for c in df.columns
                             if c not in ["timestamp", "source", "lane_id",
                                          "traffic_state", "state_color"]]

        X_raw = df[self.feature_cols].values.astype(np.float32)
        y_raw = df[[target_col]].values.astype(np.float32)

        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)

        train_ds = SequenceDataset(X_scaled, y_scaled, self.seq_len, self.horizon)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if val_df is not None and not val_df.empty:
            val_df   = engineer_features(val_df, target_col=target_col)
            Xv = self.scaler_X.transform(val_df[self.feature_cols].fillna(0).values.astype(np.float32))
            yv = self.scaler_y.transform(val_df[[target_col]].values.astype(np.float32))
            val_ds = SequenceDataset(Xv, yv, self.seq_len, self.horizon)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_dl = None

        input_size = X_scaled.shape[1]
        self.model = TrafficLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_steps=self.horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        no_improve    = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred    = self.model(xb)
                loss    = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_dl), 1)
            self._train_losses.append(train_loss)

            if val_dl:
                val_loss = self._eval_loss(val_dl, criterion)
                self._val_losses.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    no_improve    = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"[LSTM] Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0:
                v = f", val_loss={val_loss:.4f}" if val_dl else ""
                logger.info(f"[LSTM] Epoch {epoch+1}/{self.epochs} "
                            f"train_loss={train_loss:.4f}{v}")

        self.trained = True
        logger.success("[LSTM] Training complete.")
        return {"train_losses": self._train_losses, "val_losses": self._val_losses}

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += criterion(self.model(xb), yb).item()
        return total / max(len(loader), 1)

    def predict(self, df: pd.DataFrame,
                target_col: str = "vehicle_count") -> np.ndarray:
        if not self.trained or self.model is None:
            raise RuntimeError("LSTM not trained.")
        df = engineer_features(df, target_col=target_col)
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        X  = self.scaler_X.transform(df[self.feature_cols].values.astype(np.float32))
        # Take the last seq_len rows as input
        seq = X[-self.seq_len:]
        xb  = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(xb).cpu().numpy()   # (1, horizon)
        # Inverse transform: reshape to (horizon, 1)
        dummy = np.zeros((self.horizon, len(self.feature_cols)))
        # We only need to inverse the target column — use scaler_y
        pred_raw = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return np.clip(pred_raw, 0, None)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":   self.model.state_dict(),
            "input_size":   self.model.lstm.input_size,
            "hidden_size":  self.hidden_size,
            "num_layers":   self.num_layers,
            "dropout":      self.dropout,
            "output_steps": self.horizon,
            "seq_len":      self.seq_len,
        }, path + ".pt")
        with open(path + ".scalers", "wb") as f:
            pickle.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
                         "feature_cols": self.feature_cols}, f)

    def load(self, path: str):
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.model = TrafficLSTM(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            output_steps=ckpt["output_steps"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.seq_len = ckpt["seq_len"]
        with open(path + ".scalers", "rb") as f:
            meta = pickle.load(f)
        self.scaler_X     = meta["scaler_X"]
        self.scaler_y     = meta["scaler_y"]
        self.feature_cols = meta["feature_cols"]
        self.trained      = True


# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "model") -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for a regression prediction."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    logger.info(f"[{model_name}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compare_models(test_df: pd.DataFrame,
                   models: Dict[str, Any],
                   target_col: str = "vehicle_count") -> pd.DataFrame:
    """Run all models on test set and return a comparison DataFrame."""
    y_true = test_df[target_col].values
    rows   = []
    for name, mdl in models.items():
        try:
            y_pred = mdl.predict(test_df.copy(), target_col=target_col)
            n      = min(len(y_true), len(y_pred))
            metrics = evaluate_model(y_true[:n], y_pred[:n], name)
            metrics["model"] = name
            rows.append(metrics)
        except Exception as e:
            logger.warning(f"[{name}] prediction failed: {e}")
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


# ── Utility ───────────────────────────────────────────────────────────────────
def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def train_all_models(train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     target_col: str = "vehicle_count",
                     horizon: int = 6,
                     save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Train RF, XGBoost, and LSTM on the provided splits.
    Returns dict of trained models and evaluation results.
    """
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "models" / "prediction")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    # Random Forest
    rf = RandomForestPredictor()
    rf_metrics = rf.fit(train_df, target_col=target_col, horizon=horizon)
    rf.save(str(Path(save_dir) / "random_forest"))
    results["RandomForest"] = {"model": rf, "train": rf_metrics}

    # XGBoost
    xgb_pred = XGBoostPredictor()
    xgb_metrics = xgb_pred.fit(train_df, target_col=target_col,
                                horizon=horizon, val_df=val_df)
    xgb_pred.save(str(Path(save_dir) / "xgboost"))
    results["XGBoost"] = {"model": xgb_pred, "train": xgb_metrics}

    # LSTM
    lstm = LSTMPredictor(seq_len=12, horizon=horizon)
    lstm_metrics = lstm.fit(train_df, target_col=target_col, val_df=val_df)
    lstm.save(str(Path(save_dir) / "lstm"))
    results["LSTM"] = {"model": lstm, "train": lstm_metrics}

    # Evaluate on test set
    test_comparisons = compare_models(
        test_df,
        {name: v["model"] for name, v in results.items()},
        target_col=target_col,
    )
    results["test_comparison"] = test_comparisons
    logger.success("All models trained and evaluated.")
    return results
