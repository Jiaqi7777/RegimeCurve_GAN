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
    index = {name: i for i, name in enumerate(columns)}
    two, five, ten = (curves[..., index[name]] for name in ("2 Yr", "5 Yr", "10 Yr"))
    return np.stack([ten, ten - two, 2 * five - two - ten], axis=-1)


def conditional_neighbours(hist: np.ndarray, columns: list[str], horizon: int,
                           neighbours: int = 250, context_days: int = 3
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = curve_factors(hist, columns)
    starts = np.arange(context_days - 1, len(hist) - horizon)
    volatility = np.array([
        np.diff(hist[i - context_days + 1:i + 1], axis=0).std() for i in starts
    ])
    features = np.column_stack([factors[starts], volatility])
    current = np.r_[factors[-1], np.diff(hist[-context_days:], axis=0).std()]
    scale = features.std(0)
    scale[scale < 1e-8] = 1.0
    distances = np.linalg.norm((features - current) / scale, axis=1)
    selected = []
    for position in np.argsort(distances):
        if all(abs(starts[position] - starts[other]) >= horizon for other in selected):
            selected.append(position)
            if len(selected) == neighbours:
                break
    selected = np.asarray(selected)
    selected_starts = starts[selected]
    factor_paths = np.stack([
        factors[start + 1:start + horizon + 1] - factors[start] for start in selected_starts
    ])
    return selected_starts, distances[selected], factor_paths


def _display_indices(count: int, maximum: int = 20) -> np.ndarray:
    return np.linspace(0, count - 1, min(count, maximum), dtype=int)


def plot_factor_paths(generated: np.ndarray, historical: np.ndarray, output: Path) -> None:
    days = np.arange(generated.shape[1] + 1)
    generated = np.concatenate([np.zeros((len(generated), 1, 3)), generated], axis=1) * 100
    historical = np.concatenate([np.zeros((len(historical), 1, 3)), historical], axis=1) * 100
    q = np.quantile(historical, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for factor, axis in enumerate(axes):
        axis.fill_between(days, q[0, :, factor], q[4, :, factor], color="steelblue", alpha=.15,
                          label="historical 5–95%")
        axis.fill_between(days, q[1, :, factor], q[3, :, factor], color="steelblue", alpha=.28,
                          label="historical 25–75%")
        axis.plot(days, q[2, :, factor], "--", color="steelblue", label="historical median")
        for j, scenario in enumerate(_display_indices(len(generated))):
            axis.plot(days, generated[scenario, :, factor], alpha=.65,
                      label="generated (20 shown)" if j == 0 else None)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(title=FACTOR_NAMES[factor], xlabel="Forecast business day", ylabel="Change (bp)")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=4, frameon=False)
    fig.suptitle("Generated factor paths versus conditionally similar history", y=1.10)
    fig.tight_layout()
    fig.savefig(output / "factor_paths_conditional.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_factors(generated: np.ndarray, historical: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for factor, axis in enumerate(axes):
        axis.boxplot(historical[:, factor] * 100, showfliers=False)
        axis.scatter(rng.normal(1, .025, len(generated)), generated[:, factor] * 100,
                     color="darkorange", edgecolor="black", s=32, label="generated scenarios")
        axis.set(xticks=[], title=FACTOR_NAMES[factor], ylabel="10-day change (bp)")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Generated terminal changes versus conditional historical distribution")
    fig.tight_layout()
    fig.savefig(output / "conditional_terminal_changes.png", dpi=180)
    plt.close(fig)


def plot_maturity_envelope(generated: np.ndarray, historical: np.ndarray,
                           columns: list[str], output: Path) -> None:
    x = np.arange(len(columns))
    q = np.quantile(historical * 100, [.05, .25, .5, .75, .95], axis=0)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.fill_between(x, q[0], q[4], color="steelblue", alpha=.15, label="historical 5–95%")
    axis.fill_between(x, q[1], q[3], color="steelblue", alpha=.28, label="historical 25–75%")
    axis.plot(x, q[2], "--", color="steelblue", label="historical median")
    for j, scenario in enumerate(_display_indices(len(generated))):
        axis.plot(x, generated[scenario] * 100, alpha=.65,
                  label="generated (20 shown)" if j == 0 else None)
    axis.axhline(0, color="black", linewidth=.7)
    axis.set(xticks=x, xticklabels=columns, xlabel="Maturity", ylabel="10-day change (bp)",
             title="Generated changes within conditional historical envelopes")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "maturity_change_envelope.png", dpi=180)
    plt.close(fig)


def plot_level_neutral_changes(generated: np.ndarray, historical: np.ndarray,
                               columns: list[str], output: Path) -> None:
    """Expose twists and butterflies hidden by the dominant parallel shift."""
    generated = generated - generated.mean(axis=-1, keepdims=True)
    historical = historical - historical.mean(axis=-1, keepdims=True)
    x = np.arange(len(columns))
    q = np.quantile(historical * 100, [.05, .25, .5, .75, .95], axis=0)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.fill_between(x, q[0], q[4], color="mediumpurple", alpha=.14,
                      label="historical 5–95%")
    axis.fill_between(x, q[1], q[3], color="mediumpurple", alpha=.28,
                      label="historical 25–75%")
    axis.plot(x, q[2], "--", color="indigo", label="historical median")
    for j, scenario in enumerate(_display_indices(len(generated))):
        axis.plot(x, generated[scenario] * 100, alpha=.68,
                  label="generated (20 shown)" if j == 0 else None)
    axis.axhline(0, color="black", linewidth=.7)
    axis.set(xticks=x, xticklabels=columns, xlabel="Maturity",
             ylabel="Level-neutral 10-day change (bp)",
             title="Shape diversity after removing each scenario's parallel shift")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "level_neutral_shape_changes.png", dpi=180)
    plt.close(fig)


def nearest_neighbour_percentiles(historical: np.ndarray, generated: np.ndarray) -> np.ndarray:
    centre, scale = historical.mean(0), historical.std(0)
    scale[scale < 1e-8] = 1
    historical = (historical - centre) / scale
    generated = (generated - centre) / scale
    historical_distances = np.linalg.norm(historical[:, None] - historical[None], axis=-1)
    np.fill_diagonal(historical_distances, np.inf)
    reference = historical_distances.min(1)
    generated_nearest = np.linalg.norm(generated[:, None] - historical[None], axis=-1).min(1)
    return np.array([(reference <= distance).mean() for distance in generated_nearest])


def plot_pca(historical: np.ndarray, generated: np.ndarray, output: Path) -> tuple[np.ndarray, list[float]]:
    centre, scale = historical.mean(0), historical.std(0)
    scale[scale < 1e-8] = 1
    historical_z, generated_z = (historical - centre) / scale, (generated - centre) / scale
    component_count = min(4, historical.shape[1])
    pca = PCA(n_components=component_count).fit(historical_z)
    hist_score, gen_score = pca.transform(historical_z), pca.transform(generated_z)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axis = axes[0]
    axis.scatter(hist_score[:, 0], hist_score[:, 1], alpha=.3, s=22, label="conditional history")
    axis.scatter(gen_score[:, 0], gen_score[:, 1], color="darkorange", edgecolor="black",
                 s=55, label="generated", zorder=3)
    covariance = np.cov(hist_score.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    radius = np.sqrt(4.605)
    axis.add_patch(Ellipse(hist_score.mean(0), 2 * radius * np.sqrt(values[0]),
                           2 * radius * np.sqrt(values[1]), angle=angle, fill=False,
                           color="navy", linestyle="--", linewidth=2, label="historical 90% ellipse"))
    axis.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
             ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
             title="Representativeness and shape diversity")
    axis.axhline(0, color="grey", linewidth=.5)
    axis.axvline(0, color="grey", linewidth=.5)
    axis.legend(frameon=False)
    axis = axes[1]
    axis.scatter(hist_score[:, 2], hist_score[:, 3], alpha=.3, s=22,
                 label="conditional history")
    axis.scatter(gen_score[:, 2], gen_score[:, 3], color="darkorange", edgecolor="black",
                 s=55, label="generated", zorder=3)
    axis.set(xlabel=f"PC3 ({pca.explained_variance_ratio_[2]:.0%})",
             ylabel=f"PC4 ({pca.explained_variance_ratio_[3]:.0%})",
             title="Local twists and butterflies")
    axis.axhline(0, color="grey", linewidth=.5)
    axis.axvline(0, color="grey", linewidth=.5)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "representativeness_pca.png", dpi=180)
    plt.close(fig)
    spread_ratio = (gen_score.std(0) / hist_score.std(0).clip(1e-8)).tolist()
    return nearest_neighbour_percentiles(historical, generated), spread_ratio


def plot_shape_covariance_heatmaps(historical: np.ndarray, generated: np.ndarray,
                                   columns: list[str], output: Path
                                   ) -> tuple[float, float, float, list[float]]:
    historical = historical - historical.mean(axis=-1, keepdims=True)
    generated = generated - generated.mean(axis=-1, keepdims=True)
    historical_covariance = np.cov((historical * 100).T)
    generated_covariance = np.cov((generated * 100).T)
    difference = np.abs(historical_covariance - generated_covariance)
    limit = max(np.abs(historical_covariance).max(), np.abs(generated_covariance).max())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, matrix, title, limits in zip(
        axes, [historical_covariance, generated_covariance, difference],
        ["Historical shape covariance", "Generated shape covariance", "Absolute difference"],
        [(-limit, limit), (-limit, limit), (0, difference.max())], strict=True,
    ):
        image = axis.imshow(matrix, cmap="coolwarm" if limits[0] < 0 else "magma",
                            vmin=limits[0], vmax=limits[1])
        axis.set(xticks=range(len(columns)), yticks=range(len(columns)),
                 xticklabels=columns, yticklabels=columns, title=title)
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=axis, fraction=.046)
    fig.suptitle("Level-neutral 10-day shape covariance (bp²)")
    fig.tight_layout()
    fig.savefig(output / "shape_covariance_heatmaps.png", dpi=180)
    plt.close(fig)
    absolute_error = np.linalg.norm(historical_covariance - generated_covariance, ord="fro")
    relative_error = absolute_error / np.linalg.norm(historical_covariance, ord="fro").clip(1e-8)
    trace_ratio = np.trace(generated_covariance) / np.trace(historical_covariance).clip(1e-8)
    historical_eigenvalues = np.linalg.eigvalsh(historical_covariance)[::-1]
    generated_eigenvalues = np.linalg.eigvalsh(generated_covariance)[::-1]
    mode_ratios = (generated_eigenvalues[:6] / historical_eigenvalues[:6].clip(1e-8)).tolist()
    return float(absolute_error), float(relative_error), float(trace_ratio), mode_ratios


def plot_correlation_heatmaps(historical_changes: np.ndarray, generated_changes: np.ndarray,
                              columns: list[str], output: Path) -> float:
    historical = np.corrcoef(historical_changes.reshape(-1, len(columns)).T)
    generated = np.corrcoef(generated_changes.reshape(-1, len(columns)).T)
    difference = np.abs(historical - generated)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, matrix, title, limits in zip(
        axes, [historical, generated, difference],
        ["Historical correlation", "Generated correlation", "Absolute difference"],
        [(-1, 1), (-1, 1), (0, 1)], strict=True,
    ):
        image = axis.imshow(matrix, cmap="coolwarm" if limits[0] < 0 else "magma",
                            vmin=limits[0], vmax=limits[1])
        axis.set(xticks=range(len(columns)), yticks=range(len(columns)),
                 xticklabels=columns, yticklabels=columns, title=title)
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=axis, fraction=.046)
    fig.suptitle("Daily-change cross-maturity correlation")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmaps.png", dpi=180)
    plt.close(fig)
    return float(np.linalg.norm(historical - generated, ord="fro"))


def plot_terminal_curves(paths: np.ndarray, latest: np.ndarray, columns: list[str], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(columns, latest, color="black", linewidth=2.5, label="last observed")
    for j, scenario in enumerate(_display_indices(len(paths), maximum=30)):
        axis.plot(columns, paths[scenario, -1], alpha=.55,
                  label="generated (30 shown)" if j == 0 else None)
    axis.set(title="Generated 10-day terminal yield curves", xlabel="Maturity", ylabel="Yield (%)")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "terminal_curves.png", dpi=180)
    plt.close(fig)


def evaluate(generated_path: str, historical_path: str, output_dir: str,
             neighbours: int = 250) -> dict:
    generated = pd.read_csv(generated_path).sort_values(["scenario", "day"])
    columns = [name for name in generated.columns if name not in {"scenario", "day"}]
    historical_frame = load_curves(historical_path)[columns]
    hist = historical_frame.to_numpy(float)
    scenarios = sorted(generated.scenario.unique())
    paths = np.stack([generated[generated.scenario == s][columns].to_numpy(float) for s in scenarios])
    horizon = paths.shape[1]
    starts, distances, historical_factor_paths = conditional_neighbours(
        hist, columns, horizon, neighbours
    )
    generated_factor_paths = curve_factors(paths, columns) - curve_factors(hist[-1:], columns)[0]
    historical_curve_changes = np.stack([hist[start + horizon] - hist[start] for start in starts])
    generated_curve_changes = paths[:, -1] - hist[-1]
    generated_terminal, historical_terminal = generated_factor_paths[:, -1], historical_factor_paths[:, -1]
    lower, median, upper = np.quantile(historical_terminal, [.05, .5, .95], axis=0)
    outside = (generated_terminal < lower) | (generated_terminal > upper)
    historical_daily = np.stack([np.diff(hist[start:start + horizon + 1], axis=0) for start in starts])
    generated_daily = np.diff(np.concatenate([np.repeat(hist[-1][None, None], len(paths), axis=0), paths], axis=1), axis=1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    representative, pca_spread_ratio = plot_pca(historical_curve_changes, generated_curve_changes, output)
    correlation_error = plot_correlation_heatmaps(historical_daily, generated_daily, columns, output)
    (shape_covariance_error, relative_shape_covariance_error,
     shape_variance_ratio, shape_mode_variance_ratios) = plot_shape_covariance_heatmaps(
        historical_curve_changes, generated_curve_changes, columns, output
    )
    maturity_lower, maturity_upper = np.quantile(historical_curve_changes, [.05, .95], axis=0)
    maturity_outside = ((generated_curve_changes < maturity_lower)
                        | (generated_curve_changes > maturity_upper)).mean(axis=0)
    sensitivity = {}
    for count in (100, min(250, len(starts))):
        lo, hi = np.quantile(historical_factor_paths[:count, -1], [.05, .95], axis=0)
        sensitivity[str(count)] = float(((generated_terminal < lo) | (generated_terminal > hi)).any(1).mean())
    metrics = {
        "conditional_neighbours": len(starts),
        "terminal_factor_change_bp": {
            name: {"generated_mean": float(generated_terminal[:, i].mean() * 100),
                   "generated_std": float(generated_terminal[:, i].std() * 100),
                   "historical_p05": float(lower[i] * 100), "historical_median": float(median[i] * 100),
                   "historical_p95": float(upper[i] * 100),
                   "outside_fraction": float(outside[:, i].mean())}
            for i, name in enumerate(FACTOR_NAMES)
        },
        "outside_any_factor_fraction": float(outside.any(1).mean()),
        "outside_any_factor_sensitivity_by_neighbours": sensitivity,
        "maturity_outside_5_95_fraction": {
            name: float(value) for name, value in zip(columns, maturity_outside, strict=True)
        },
        "pca_generated_to_historical_spread_ratio": {
            f"PC{i + 1}": float(value) for i, value in enumerate(pca_spread_ratio)
        },
        "representativeness_percentiles": [float(x) for x in representative],
        "less_representative_than_95pct_fraction": float((representative > .95).mean()),
        "mean_terminal_curve_dispersion_bp": float(paths[:, -1].std(0).mean() * 100),
        "mean_absolute_daily_move_bp": float(np.abs(generated_daily).mean() * 100),
        "maximum_daily_move_bp": float(np.abs(generated_daily).max() * 100),
        "correlation_matrix_error": correlation_error,
        "level_neutral_shape_covariance_error": shape_covariance_error,
        "relative_shape_covariance_error": relative_shape_covariance_error,
        "generated_to_historical_shape_variance_ratio": shape_variance_ratio,
        "generated_to_historical_shape_mode_variance_ratio": {
            f"mode_{i + 1}": float(value)
            for i, value in enumerate(shape_mode_variance_ratios)
        },
    }
    warnings = []
    if metrics["outside_any_factor_fraction"] > .2:
        warnings.append("More than 20% of scenarios fall outside conditional factor ranges.")
    if pca_spread_ratio[1] < .5:
        warnings.append("Generated PC2 spread is below 50% of history; possible shape under-dispersion.")
    if correlation_error > 2.0:
        warnings.append("Cross-maturity correlation error remains high.")
    if shape_variance_ratio < .6:
        warnings.append("Generated level-neutral shape variance is below 60% of history.")
    if shape_variance_ratio > 1.4:
        warnings.append("Generated level-neutral shape variance exceeds 140% of history.")
    if metrics["maximum_daily_move_bp"] > 25.0:
        warnings.append("At least one generated daily maturity move exceeds 25 bp.")
    if max(maturity_outside) > .25:
        warnings.append("At least one maturity has more than 25% of scenarios outside its 5–95% range.")
    metrics["warnings"] = warnings
    save_json(metrics, output / "metrics.json")
    pd.DataFrame({"start_date": historical_frame.index[starts].astype(str),
                  "distance": distances}).to_csv(output / "conditional_neighbours.csv", index=False)
    plot_factor_paths(generated_factor_paths, historical_factor_paths, output)
    plot_terminal_factors(generated_terminal, historical_terminal, output)
    plot_maturity_envelope(generated_curve_changes, historical_curve_changes, columns, output)
    plot_level_neutral_changes(generated_curve_changes, historical_curve_changes, columns, output)
    plot_terminal_curves(paths, hist[-1], columns, output)
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
