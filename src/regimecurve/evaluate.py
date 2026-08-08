from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_curves
from .utils import save_json

FACTOR_NAMES = ["Level (10Y)", "Slope (10Y-2Y)", "Curvature (2x5Y-2Y-10Y)"]


def curve_factors(curves: np.ndarray, columns: list[str]) -> np.ndarray:
    """Return level, slope, and curvature in percentage points."""
    index = {column: position for position, column in enumerate(columns)}
    required = {"2 Yr", "5 Yr", "10 Yr"}
    if not required.issubset(index):
        raise ValueError(f"Factor diagnostics require columns {sorted(required)}")
    two, five, ten = (curves[..., index[name]] for name in ("2 Yr", "5 Yr", "10 Yr"))
    return np.stack([ten, ten - two, 2.0 * five - two - ten], axis=-1)


def conditional_neighbours(hist: np.ndarray, columns: list[str], horizon: int,
                           context_days: int = 3, neighbours: int = 250
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find historical starts resembling the current factor state and recent volatility."""
    factors = curve_factors(hist, columns)
    starts = np.arange(context_days - 1, len(hist) - horizon)
    recent_volatility = np.array([
        np.diff(hist[i - context_days + 1:i + 1], axis=0).std() for i in starts
    ])
    candidate_features = np.column_stack([factors[starts], recent_volatility])
    current_volatility = np.diff(hist[-context_days:], axis=0).std()
    current_features = np.r_[factors[-1], current_volatility]
    scale = candidate_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    distances = np.linalg.norm((candidate_features - current_features) / scale, axis=1)
    chosen = np.argsort(distances)[:min(neighbours, len(starts))]
    selected_starts = starts[chosen]
    paths = np.stack([
        factors[start + 1:start + horizon + 1] - factors[start] for start in selected_starts
    ])
    return selected_starts, distances[chosen], paths


def factor_path_plot(generated_changes: np.ndarray, historical_changes: np.ndarray,
                     output: Path) -> None:
    horizon = generated_changes.shape[1]
    days = np.arange(horizon + 1)
    generated = np.concatenate([
        np.zeros((generated_changes.shape[0], 1, 3)), generated_changes
    ], axis=1) * 100.0
    historical = np.concatenate([
        np.zeros((historical_changes.shape[0], 1, 3)), historical_changes
    ], axis=1) * 100.0
    quantiles = np.quantile(historical, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for factor, axis in enumerate(axes):
        axis.fill_between(days, quantiles[0, :, factor], quantiles[4, :, factor],
                          color="steelblue", alpha=0.15, label="conditional 5–95%")
        axis.fill_between(days, quantiles[1, :, factor], quantiles[3, :, factor],
                          color="steelblue", alpha=0.25, label="conditional 25–75%")
        axis.plot(days, quantiles[2, :, factor], color="steelblue", linestyle="--",
                  linewidth=1.5, label="conditional median")
        for scenario in range(generated.shape[0]):
            axis.plot(days, generated[scenario, :, factor], alpha=0.65, linewidth=1.2,
                      label="generated" if scenario == 0 else None)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=FACTOR_NAMES[factor], xlabel="Forecast business day", ylabel="Change (bp)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Generated factor paths against conditionally similar historical episodes", y=1.04)
    fig.tight_layout()
    fig.savefig(output / "factor_paths_conditional.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def terminal_comparison_plot(generated_terminal: np.ndarray, historical_terminal: np.ndarray,
                             output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    rng = np.random.default_rng(42)
    for factor, axis in enumerate(axes):
        historical_bp = historical_terminal[:, factor] * 100.0
        generated_bp = generated_terminal[:, factor] * 100.0
        axis.boxplot(historical_bp, widths=0.45, showfliers=False)
        jitter = rng.normal(1.0, 0.025, len(generated_bp))
        axis.scatter(jitter, generated_bp, color="darkorange", edgecolor="black", s=35,
                     zorder=3, label="generated scenarios")
        axis.set_xticks([])
        axis.set(title=FACTOR_NAMES[factor], ylabel="10-day change (bp)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Generated terminal changes versus conditional historical distribution")
    fig.tight_layout()
    fig.savefig(output / "conditional_terminal_changes.png", dpi=180)
    plt.close(fig)


def terminal_curve_plot(generated: pd.DataFrame, hist: np.ndarray, columns: list[str],
                        output: Path) -> None:
    terminal_day = generated["day"].max()
    terminal = generated[generated["day"] == terminal_day]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(columns, hist[-1], color="black", linewidth=2.5, label="last observed")
    for scenario, row in terminal.groupby("scenario"):
        axis.plot(columns, row.iloc[0][columns].to_numpy(float), alpha=0.55,
                  label="generated" if scenario == 0 else None)
    axis.set(title="Generated 10-day terminal yield curves", ylabel="Yield (%)", xlabel="Maturity")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "terminal_curves.png", dpi=180)
    plt.close(fig)


def evaluate(generated_path: str, historical_path: str, output_dir: str,
             neighbours: int = 250) -> dict:
    generated = pd.read_csv(generated_path).sort_values(["scenario", "day"])
    columns = [column for column in generated.columns if column not in {"scenario", "day"}]
    historical_frame = load_curves(historical_path)[columns]
    hist = historical_frame.to_numpy(float)
    horizon = int(generated["day"].max())
    scenario_ids = sorted(generated["scenario"].unique())
    generated_paths = np.stack([
        generated[generated["scenario"] == scenario][columns].to_numpy(float)
        for scenario in scenario_ids
    ])
    generated_factors = curve_factors(generated_paths, columns)
    starting_factors = curve_factors(hist[-1][None], columns)[0]
    generated_factor_changes = generated_factors - starting_factors
    starts, distances, historical_factor_changes = conditional_neighbours(
        hist, columns, horizon, neighbours=neighbours
    )
    generated_terminal = generated_factor_changes[:, -1]
    historical_terminal = historical_factor_changes[:, -1]
    lower, median, upper = np.quantile(historical_terminal, [0.05, 0.50, 0.95], axis=0)
    outside_by_factor = ((generated_terminal < lower) | (generated_terminal > upper)).mean(axis=0)
    outside_any = ((generated_terminal < lower) | (generated_terminal > upper)).any(axis=1).mean()

    full_paths = np.concatenate([hist[-1][None, None].repeat(len(scenario_ids), axis=0),
                                 generated_paths], axis=1)
    generated_daily_changes = np.diff(full_paths, axis=1)
    historical_daily_changes = np.diff(hist, axis=0)
    generated_cov = np.corrcoef(generated_daily_changes.reshape(-1, len(columns)).T)
    historical_cov = np.corrcoef(historical_daily_changes.T)
    maximum_jump = np.abs(generated_daily_changes).max(axis=(1, 2)) * 100.0
    terminal_curve_std = generated_paths[:, -1].std(axis=0) * 100.0

    warnings = []
    if outside_any > 0.20:
        warnings.append(
            f"{outside_any:.0%} of scenarios fall outside at least one conditional 5–95% factor range."
        )
    if maximum_jump.max() > 25.0:
        warnings.append(
            f"The largest one-day maturity move is {maximum_jump.max():.1f} bp; inspect path realism."
        )
    if terminal_curve_std.mean() < 5.0:
        warnings.append("Mean terminal dispersion is below 5 bp; possible conditional mode collapse.")

    metrics = {
        "conditional_neighbours": len(starts),
        "conditional_neighbour_distance_mean": float(distances.mean()),
        "terminal_factor_change_bp": {
            name: {
                "generated_mean": float(generated_terminal[:, i].mean() * 100.0),
                "generated_std": float(generated_terminal[:, i].std() * 100.0),
                "historical_p05": float(lower[i] * 100.0),
                "historical_median": float(median[i] * 100.0),
                "historical_p95": float(upper[i] * 100.0),
                "generated_outside_5_95_fraction": float(outside_by_factor[i]),
            } for i, name in enumerate(FACTOR_NAMES)
        },
        "generated_outside_any_factor_5_95_fraction": float(outside_any),
        "mean_terminal_curve_dispersion_bp": float(terminal_curve_std.mean()),
        "mean_absolute_daily_move_bp": float(np.abs(generated_daily_changes).mean() * 100.0),
        "maximum_daily_move_bp": float(maximum_jump.max()),
        "maximum_daily_move_bp_by_scenario": [float(value) for value in maximum_jump],
        "correlation_matrix_error": float(np.linalg.norm(generated_cov - historical_cov, ord="fro")),
        "warnings": warnings,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output / "metrics.json")
    pd.DataFrame({
        "start_date": historical_frame.index[starts].astype(str), "distance": distances,
    }).to_csv(output / "conditional_neighbours.csv", index=False)
    factor_path_plot(generated_factor_changes, historical_factor_changes, output)
    terminal_comparison_plot(generated_terminal, historical_terminal, output)
    terminal_curve_plot(generated, hist, columns, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", default="outputs/generated_curves.csv")
    parser.add_argument("--historical", default="data/data.xlsx")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--neighbours", type=int, default=250)
    args = parser.parse_args()
    metrics = evaluate(args.generated, args.historical, args.output_dir, args.neighbours)
    print(metrics)


if __name__ == "__main__":
    main()
