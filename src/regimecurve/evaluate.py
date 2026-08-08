from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_curves
from .utils import save_json


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    # Monte Carlo energy distance on flattened paths.
    cross = np.linalg.norm(x[:, None] - y[None, :], axis=-1).mean()
    within_x = np.linalg.norm(x[:, None] - x[None, :], axis=-1).mean()
    within_y = np.linalg.norm(y[:, None] - y[None, :], axis=-1).mean()
    return float(2 * cross - within_x - within_y)


def evaluate(generated_path: str, historical_path: str, output_dir: str) -> dict:
    generated = pd.read_csv(generated_path)
    maturity_columns = [c for c in generated.columns if c not in {"scenario", "day"}]
    historical = load_curves(historical_path)
    hist = historical[maturity_columns].to_numpy(float)
    gen = generated[maturity_columns].to_numpy(float)
    hist_changes, gen_changes = np.diff(hist, axis=0), generated.groupby("scenario")[maturity_columns].diff().dropna().to_numpy()
    metrics = {
        "mean_absolute_yield_difference": float(np.abs(hist.mean(0) - gen.mean(0)).mean()),
        "mean_absolute_volatility_difference": float(np.abs(hist_changes.std(0) - gen_changes.std(0)).mean()),
        "correlation_matrix_error": float(np.linalg.norm(np.corrcoef(hist_changes.T) - np.corrcoef(gen_changes.T), ord="fro")),
        "path_diversity": float(generated.groupby("scenario")[maturity_columns].mean().std().mean()),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output / "metrics.json")

    latest = hist[-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(maturity_columns, latest, color="black", linewidth=2, label="last observed")
    for scenario, frame in generated.groupby("scenario"):
        ax.plot(maturity_columns, frame.iloc[-1][maturity_columns], alpha=0.45,
                label="generated" if scenario == 0 else None)
    ax.set(title="Generated 10-day terminal yield curves", ylabel="Yield (%)", xlabel="Maturity")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "terminal_curves.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", default="outputs/generated_curves.csv")
    parser.add_argument("--historical", default="data/data.xlsx")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    print(evaluate(args.generated, args.historical, args.output_dir))


if __name__ == "__main__":
    main()
