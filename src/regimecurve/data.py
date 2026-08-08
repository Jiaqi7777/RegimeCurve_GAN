from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

MATURITY_YEARS = {
    "1 Mo": 1 / 12, "2 Mo": 2 / 12, "3 Mo": 3 / 12, "4 Mo": 4 / 12,
    "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}


def nelson_siegel_basis(maturities: np.ndarray, decay: float) -> np.ndarray:
    x = decay * maturities
    loading = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), loading, loading - np.exp(-x)])


def load_curves(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Input must be CSV or Excel. Export Apple Numbers as .xlsx first.")
    if "Date" not in frame:
        raise ValueError("Dataset must contain a Date column")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise", format="mixed")
    available = [column for column in MATURITY_YEARS if column in frame.columns]
    if len(available) < 5:
        raise ValueError("At least five recognised Treasury maturity columns are required")
    frame = frame[["Date", *available]].sort_values("Date").drop_duplicates("Date")
    # The Treasury archive contains an occasional dated row with no reported curve at all.
    frame = frame.dropna(subset=available, how="all")
    # Interpolate across log-maturity within each date. Edge gaps use nearest maturity.
    values = frame[available].astype(float)
    values = values.interpolate(axis=1, limit_direction="both")
    frame[available] = values
    return frame.set_index("Date")


@dataclass
class CurveTransform:
    maturities: np.ndarray
    decay: float
    residual_components: int
    factor_scaler: StandardScaler
    residual_scaler: StandardScaler
    residual_pca: PCA

    @classmethod
    def fit(cls, curves: np.ndarray, maturities: np.ndarray, decay: float,
            residual_components: int) -> CurveTransform:
        basis = nelson_siegel_basis(maturities, decay)
        beta = curves @ np.linalg.pinv(basis).T
        residuals = curves - beta @ basis.T
        pca = PCA(n_components=min(residual_components, curves.shape[1] - 3)).fit(residuals)
        scores = pca.transform(residuals)
        return cls(maturities, decay, pca.n_components_, StandardScaler().fit(beta),
                   StandardScaler().fit(scores), pca)

    @property
    def state_dim(self) -> int:
        return 3 + self.residual_components

    def encode(self, curves: np.ndarray) -> np.ndarray:
        shape = curves.shape
        flat = curves.reshape(-1, shape[-1])
        basis = nelson_siegel_basis(self.maturities, self.decay)
        beta = flat @ np.linalg.pinv(basis).T
        residual = flat - beta @ basis.T
        states = np.concatenate([
            self.factor_scaler.transform(beta),
            self.residual_scaler.transform(self.residual_pca.transform(residual)),
        ], axis=1)
        return states.reshape(*shape[:-1], self.state_dim).astype(np.float32)

    def decode(self, states: np.ndarray) -> np.ndarray:
        shape = states.shape
        flat = states.reshape(-1, shape[-1])
        beta = self.factor_scaler.inverse_transform(flat[:, :3])
        scores = self.residual_scaler.inverse_transform(flat[:, 3:])
        curves = beta @ nelson_siegel_basis(self.maturities, self.decay).T
        curves += self.residual_pca.inverse_transform(scores)
        return curves.reshape(*shape[:-1], len(self.maturities)).astype(np.float32)

    def torch_decoder_parameters(self) -> dict[str, np.ndarray]:
        basis = nelson_siegel_basis(self.maturities, self.decay)
        return {
            "basis": basis.astype(np.float32),
            "beta_scale": self.factor_scaler.scale_.astype(np.float32),
            "beta_mean": self.factor_scaler.mean_.astype(np.float32),
            "res_scale": self.residual_scaler.scale_.astype(np.float32),
            "res_mean": self.residual_scaler.mean_.astype(np.float32),
            "res_components": self.residual_pca.components_.astype(np.float32),
            "residual_mean": self.residual_pca.mean_.astype(np.float32),
        }


class WindowDataset(Dataset):
    def __init__(self, states: np.ndarray, indices: np.ndarray, context_days: int,
                 horizon_days: int):
        self.states = states
        self.indices = indices
        self.context_days = context_days
        self.horizon_days = horizon_days

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        import torch
        end_context = int(self.indices[item])
        context = self.states[end_context - self.context_days:end_context]
        future = self.states[end_context:end_context + self.horizon_days]
        return torch.from_numpy(context), torch.from_numpy(future)


def valid_window_indices(dates: pd.DatetimeIndex, context_days: int, horizon_days: int,
                         start: str | None, end: str | None) -> np.ndarray:
    candidates = np.arange(context_days, len(dates) - horizon_days + 1)
    target_end_dates = dates[candidates + horizon_days - 1]
    mask = np.ones(len(candidates), dtype=bool)
    if start:
        mask &= target_end_dates >= pd.Timestamp(start)
    if end:
        mask &= target_end_dates <= pd.Timestamp(end)
    return candidates[mask]


@dataclass
class DataBundle:
    frame: pd.DataFrame
    transform: CurveTransform
    states: np.ndarray
    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset


def prepare_data(config: dict) -> DataBundle:
    cfg = config["data"]
    frame = load_curves(cfg["path"])
    columns = list(frame.columns)
    maturities = np.array([MATURITY_YEARS[c] for c in columns], dtype=np.float64)
    values = frame.to_numpy(dtype=np.float64)
    train_rows = frame.index <= pd.Timestamp(cfg["train_end"])
    transform = CurveTransform.fit(values[train_rows], maturities,
                                   cfg["nelson_siegel_lambda"], cfg["residual_components"])
    states = transform.encode(values)
    c, h = cfg["context_days"], cfg["horizon_days"]
    val_start = str(pd.Timestamp(cfg["train_end"]) + pd.Timedelta(days=1))
    test_start = str(pd.Timestamp(cfg["validation_end"]) + pd.Timedelta(days=1))
    return DataBundle(
        frame, transform, states,
        WindowDataset(states, valid_window_indices(frame.index, c, h, None, cfg["train_end"]), c, h),
        WindowDataset(states, valid_window_indices(frame.index, c, h, val_start,
                                                   cfg["validation_end"]), c, h),
        WindowDataset(states, valid_window_indices(frame.index, c, h, test_start, None), c, h),
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=shuffle, persistent_workers=workers > 0)
