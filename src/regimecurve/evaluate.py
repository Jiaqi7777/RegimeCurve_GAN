from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from .data import load_curves
from .utils import save_json

FACTOR_NAMES = ["Level (10Y)", "Slope (10Y-2Y)", "Curvature (2x5Y-2Y-10Y)"]


def curve_factors(curves: np.ndarray, columns: list[str]) -> np.ndarray:
    index = {column: position for position, column in enumerate(columns)}
    required = {"2 Yr", "5 Yr", "10 Yr"}
    if not required.issubset(index):
        raise ValueError(f"Factor diagnostics require columns {sorted(required)}")
    two, five, ten = (curves[..., index[name]] for name in ("2 Yr", "5 Yr", "10 Yr"))
    return np.stack([ten, ten - two, 2.0 * five - two - ten], axis=-1)


def conditional_neighbours(hist: np.ndarray, columns: list[str], horizon: int,
                           context_days: int = 3, neighbours: int = 250,
                           minimum_separation: int | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = curve_factors(hist, columns)
    starts = np.arange(context_days - 1, len(hist) - horizon)
    recent_volatility = np.array([
        np.diff(hist[i - context_days + 1:i + 1], axis=0).std() for i in starts
    ])
    candidate_features = np.column_stack([factors[starts], recent_volatility])
    current_features = np.r_[factors[-1], np.diff(hist[-context_days:], axis=0).std()]
    scale = candidate_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    distances = np.linalg.norm((candidate_features - current_features) / scale, axis=1)
    separation = horizon if minimum_separation is None else minimum_separation
    selected_positions = []
    for position in np.argsort(distances):
        candidate = starts[position]
        if all(abs(candidate - starts[chosen]) >= separation for chosen in selected_positions):
            selected_positions.append(position)
            if len(selected_positions) == neighbours:
                break
    chosen = np.asarray(selected_positions, dtype=int)
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
                          color="steelblue", alpha=0.15, label="historical 5–95%")
        axis.fill_between(days, quantiles[1, :, factor], quantiles[3, :, factor],
                          color="steelblue", alpha=0.28, label="historical 25–75%")
        axis.plot(days, quantiles[2, :, factor], color="steelblue", linestyle="--",
                  linewidth=1.5, label="historical median")
        for scenario in range(generated.shape[0]):
            axis.plot(days, generated[scenario, :, factor], alpha=0.65, linewidth=1.2,
                      label="generated" if scenario == 0 else None)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=FACTOR_NAMES[factor], xlabel="Forecast business day", ylabel="Change (bp)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False)
    fig.suptitle("Generated factor paths versus conditionally similar history", y=1.10)
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
        axis.scatter(rng.normal(1.0, 0.025, len(generated_bp)), generated_bp,
                     color="darkorange", edgecolor="black", s=35, zorder=3,
                     label="generated scenarios")
        axis.set_xticks([])
        axis.set(title=FACTOR_NAMES[factor], ylabel="10-day change (bp)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Generated terminal changes versus conditional historical distribution")
    fig.tight_layout()
    fig.savefig(output / "conditional_terminal_changes.png", dpi=180)
    plt.close(fig)


def maturity_envelope_plot(generated_changes: np.ndarray, historical_changes: np.ndarray,
                           columns: list[str], output: Path) -> None:
    quantiles = np.quantile(historical_changes * 100.0, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    x = np.arange(len(columns))
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.fill_between(x, quantiles[0], quantiles[4], color="steelblue", alpha=0.16,
                      label="historical 5–95%")
    axis.fill_between(x, quantiles[1], quantiles[3], color="steelblue", alpha=0.30,
                      label="historical 25–75%")
    axis.plot(x, quantiles[2], color="steelblue", linestyle="--", label="historical median")
    for scenario, curve in enumerate(generated_changes * 100.0):
        axis.plot(x, curve, alpha=0.65, linewidth=1.3,
                  label="generated" if scenario == 0 else None)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(xticks=x, xticklabels=columns, ylabel="10-day yield change (bp)",
             xlabel="Maturity", title="Generated changes within conditional historical envelopes")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=2, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "maturity_change_envelope.png", dpi=180)
    plt.close(fig)


def nearest_neighbour_percentiles(historical: np.ndarray, generated: np.ndarray) -> np.ndarray:
    centre = historical.mean(axis=0)
    scale = historical.std(axis=0)
    scale[scale < 1e-8] = 1.0
    hist_standard = (historical - centre) / scale
    gen_standard = (generated - centre) / scale
    hist_distances = np.linalg.norm(hist_standard[:, None] - hist_standard[None, :], axis=-1)
    np.fill_diagonal(hist_distances, np.inf)
    reference = hist_distances.min(axis=1)
    generated_nearest = np.linalg.norm(
        gen_standard[:, None] - hist_standard[None, :], axis=-1
    ).min(axis=1)
    return np.array([(reference <= distance).mean() for distance in generated_nearest])


def representativeness_pca_plot(historical: np.ndarray, generated: np.ndarray,
                                output: Path) -> np.ndarray:
    centre, scale = historical.mean(0), historical.std(0)
    scale[scale < 1e-8] = 1.0
    historical_standard = (historical - centre) / scale
    generated_standard = (generated - centre) / scale
    pca = PCA(n_components=2).fit(historical_standard)
    historical_scores = pca.transform(historical_standard)
    generated_scores = pca.transform(generated_standard)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(historical_scores[:, 0], historical_scores[:, 1], s=22, alpha=0.35,
                 color="steelblue", label="conditional historical episodes")
    axis.scatter(generated_scores[:, 0], generated_scores[:, 1], s=70, color="darkorange",
                 edgecolor="black", label="generated scenarios", zorder=3)
    covariance = np.cov(historical_scores.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    radius = np.sqrt(4.605)  # 90% chi-square radius in two dimensions
    ellipse = Ellipse(historical_scores.mean(0), 2 * radius * np.sqrt(values[0]),
                      2 * radius * np.sqrt(values[1]), angle=angle, fill=False,
                      color="navy", linewidth=2, linestyle="--", label="historical 90% ellipse")
    axis.add_patch(ellipse)
    axis.axhline(0, color="grey", linewidth=0.6)
    axis.axvline(0, color="grey", linewidth=0.6)
    axis.set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)",
        title="Representativeness of generated 10-day curve changes",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "representativeness_pca.png", dpi=180)
    plt.close(fig)
    return nearest_neighbour_percentiles(historical, generated)


def terminal_curve_plot(generated: pd.DataFrame, hist: np.ndarray, columns: list[str],
                        output: Path) -> None:
    terminal = generated[generated["day"] == generated["day"].max()]
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
    historical_curve_changes = np.stack([hist[start + horizon] - hist[start] for start in starts])
    generated_curve_changes = generated_paths[:, -1] - hist[-1]
    lower, median, upper = np.quantile(historical_terminal, [0.05, 0.50, 0.95], axis=0)
    outside_by_factor = ((generated_terminal < lower) | (generated_terminal > upper)).mean(axis=0)
    outside_any = ((generated_terminal < lower) | (generated_terminal > upper)).any(axis=1).mean()
    full_paths = np.concatenate([hist[-1][None, None].repeat(len(scenario_ids), axis=0),
                                 generated_paths], axis=1)
    generated_daily_changes = np.diff(full_paths, axis=1)
    historical_daily_changes = np.diff(hist, axis=0)
    maximum_jump = np.abs(generated_daily_changes).max(axis=(1, 2)) * 100.0
    terminal_curve_std = generated_paths[:, -1].std(axis=0) * 100.0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    representativeness_percentiles = representativeness_pca_plot(
        historical_curve_changes, generated_curve_changes, output
    )
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
    if (representativeness_percentiles > 0.95).mean() > 0.20:
        warnings.append("More than 20% of scenarios are less representative than 95% of historical episodes.")
    metrics = {
        "conditional_neighbours": len(starts),
        "minimum_historical_start_separation_days": horizon,
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
        "representativeness_nearest_neighbour_percentiles": [
            float(value) for value in representativeness_percentiles
        ],
        "generated_less_representative_than_95pct_fraction": float(
            (representativeness_percentiles > 0.95).mean()
        ),
        "mean_terminal_curve_dispersion_bp": float(terminal_curve_std.mean()),
        "mean_absolute_daily_move_bp": float(np.abs(generated_daily_changes).mean() * 100.0),
        "maximum_daily_move_bp": float(maximum_jump.max()),
        "correlation_matrix_error": float(np.linalg.norm(
            np.corrcoef(generated_daily_changes.reshape(-1, len(columns)).T)
            - np.corrcoef(historical_daily_changes.T), ord="fro"
        )),
        "warnings": warnings,
    }
    save_json(metrics, output / "metrics.json")
    pd.DataFrame({
        "start_date": historical_frame.index[starts].astype(str), "distance": distances,
    }).to_csv(output / "conditional_neighbours.csv", index=False)
    factor_path_plot(generated_factor_changes, historical_factor_changes, output)
    terminal_comparison_plot(generated_terminal, historical_terminal, output)
    maturity_envelope_plot(generated_curve_changes, historical_curve_changes, columns, output)
    terminal_curve_plot(generated, hist, columns, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", default="outputs/generated_curves.csv")
    parser.add_argument("--historical", default="data/data.xlsx")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--neighbours", type=int, default=250)
    args = parser.parse_args()
    print(evaluate(args.generated, args.historical, args.output_dir, args.neighbours))


if __name__ == "__main__":
    main()
