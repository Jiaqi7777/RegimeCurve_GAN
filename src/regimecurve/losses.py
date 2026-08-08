from __future__ import annotations

import torch


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor,
                     context: torch.Tensor | None = None) -> torch.Tensor:
    alpha_shape = [real.shape[0]] + [1] * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(context, mixed) if context is not None else critic(mixed)
    gradients = torch.autograd.grad(score.sum(), mixed, create_graph=True)[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def smoothness_loss(curves: torch.Tensor) -> torch.Tensor:
    second = curves[..., 2:] - 2 * curves[..., 1:-1] + curves[..., :-2]
    return second.square().mean()


def daily_changes(curves: torch.Tensor) -> torch.Tensor:
    return 100.0 * (curves[:, 1:] - curves[:, :-1])


def moment_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_changes, fake_changes = daily_changes(real), daily_changes(fake)
    scale = real_changes.std((0, 1)).detach().clamp_min(0.5)
    mean = ((real_changes.mean((0, 1)) - fake_changes.mean((0, 1))) / scale).square().mean()
    std = ((real_changes.std((0, 1)) - fake_changes.std((0, 1))) / scale).square().mean()
    return mean + std


def covariance_matrix(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1, values.shape[-1])
    centred = flattened - flattened.mean(dim=0, keepdim=True)
    return centred.T @ centred / max(flattened.shape[0] - 1, 1)


def covariance_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_covariance = covariance_matrix(daily_changes(real))
    fake_covariance = covariance_matrix(daily_changes(fake))
    normaliser = real_covariance.detach().square().mean().clamp_min(1e-6)
    return (real_covariance - fake_covariance).square().mean() / normaliser


def level_neutral_shape_covariance_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Match terminal maturity geometry after removing parallel curve movement.

    The ordinary covariance objective is dominated by the common level factor.
    This objective uses horizon changes, removes each path's cross-maturity mean,
    and therefore only rewards slope, butterfly, and local spline variation.
    """
    real_change = 100.0 * (real[:, -1] - real[:, 0])
    fake_change = 100.0 * (fake[:, -1] - fake[:, 0])
    real_shape = real_change - real_change.mean(dim=-1, keepdim=True)
    fake_shape = fake_change - fake_change.mean(dim=-1, keepdim=True)
    real_covariance = covariance_matrix(real_shape)
    fake_covariance = covariance_matrix(fake_shape)
    normaliser = real_covariance.detach().square().mean().clamp_min(1e-4)
    return (real_covariance - fake_covariance).square().mean() / normaliser


def correlation_matrix(values: torch.Tensor) -> torch.Tensor:
    flattened = values.reshape(-1, values.shape[-1])
    centred = flattened - flattened.mean(dim=0, keepdim=True)
    standardised = centred / centred.std(dim=0, unbiased=False).clamp_min(1e-4)
    return standardised.T @ standardised / max(standardised.shape[0], 1)


def correlation_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return (
        correlation_matrix(daily_changes(real))
        - correlation_matrix(daily_changes(fake))
    ).square().mean()


def lag_one_autocorrelation(curves: torch.Tensor) -> torch.Tensor:
    changes = daily_changes(curves)
    previous = changes[:, :-1].reshape(-1, changes.shape[-1])
    following = changes[:, 1:].reshape(-1, changes.shape[-1])
    previous = previous - previous.mean(dim=0, keepdim=True)
    following = following - following.mean(dim=0, keepdim=True)
    covariance = (previous * following).mean(dim=0)
    denominator = previous.square().mean(dim=0).sqrt() * following.square().mean(dim=0).sqrt()
    return covariance / denominator.clamp_min(1e-6)


def autocorrelation_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return (lag_one_autocorrelation(real) - lag_one_autocorrelation(fake)).square().mean()


def daily_tail_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_absolute = daily_changes(real).abs().reshape(-1, real.shape[-1])
    fake_absolute = daily_changes(fake).abs().reshape(-1, fake.shape[-1])
    quantiles = real_absolute.new_tensor([0.95, 0.99])
    real_tail = torch.quantile(real_absolute, quantiles, dim=0)
    fake_tail = torch.quantile(fake_absolute, quantiles, dim=0)
    return ((real_tail - fake_tail) / real_tail.detach().clamp_min(1.0)).square().mean()


def diversity_loss(first: torch.Tensor, second: torch.Tensor,
                   noise_first: torch.Tensor, noise_second: torch.Tensor) -> torch.Tensor:
    output_distance = (first - second).abs().flatten(1).mean(1)
    noise_distance = (noise_first - noise_second).abs().mean(1).clamp_min(1e-4)
    return -(output_distance / noise_distance).mean()


def raw_economic_curve_features(curves: torch.Tensor) -> torch.Tensor:
    """Level, 2s10s slope, and 2y/5y/10y curvature for standard column order."""
    level = curves[..., 10]
    slope = curves[..., 10] - curves[..., 6]
    curvature = 2.0 * curves[..., 8] - curves[..., 6] - curves[..., 10]
    return torch.stack([level, slope, curvature], dim=-1)


def economic_curve_features(curves: torch.Tensor) -> torch.Tensor:
    weights = curves.new_tensor([1.0, 1.5, 2.0])
    return raw_economic_curve_features(curves) * weights


def terminal_factor_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_terminal = 100.0 * raw_economic_curve_features(real[:, -1])
    fake_terminal = 100.0 * raw_economic_curve_features(fake[:, -1])
    scale = real_terminal.std(dim=0).detach().clamp_min(1.0)
    mean_error = ((real_terminal.mean(0) - fake_terminal.mean(0)) / scale).square().mean()
    std_error = ((real_terminal.std(0) - fake_terminal.std(0)) / scale).square().mean()
    quantiles = real_terminal.new_tensor([0.05, 0.50, 0.95])
    quantile_error = (
        (torch.quantile(real_terminal, quantiles, dim=0)
         - torch.quantile(fake_terminal, quantiles, dim=0)) / scale
    ).square().mean()
    return mean_error + std_error + quantile_error


def conditional_factor_spread_loss(grouped_curves: torch.Tensor) -> torch.Tensor:
    """Keep same-context scenarios diverse in level, slope, and curvature."""
    terminal_features = 100.0 * raw_economic_curve_features(grouped_curves[:, :, -1])
    conditional_spread = terminal_features.std(dim=1, unbiased=False).mean(dim=0)
    targets = terminal_features.new_tensor([10.0, 6.0, 4.0])
    return (torch.relu(targets - conditional_spread) / targets).square().mean()


def shape_repulsion_loss(grouped_curves: torch.Tensor, bandwidth: float = 0.5) -> torch.Tensor:
    """Repel maturity shapes after removing each curve's parallel level component."""
    terminal = grouped_curves[:, :, -1]
    shapes = terminal - terminal.mean(dim=-1, keepdim=True)
    shapes = shapes / shapes.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    distances = torch.cdist(shapes, shapes)
    paths = shapes.shape[1]
    off_diagonal = 1.0 - torch.eye(paths, device=shapes.device)[None]
    return (torch.exp(-distances / bandwidth) * off_diagonal).sum() / (
        shapes.shape[0] * max(paths * (paths - 1), 1)
    )


def repulsion_loss(grouped_paths: torch.Tensor, bandwidth: float = 0.5) -> torch.Tensor:
    """Penalise similar paths generated from the same context."""
    batch, paths = grouped_paths.shape[:2]
    flattened = grouped_paths.reshape(batch, paths, -1)
    distances = torch.cdist(flattened, flattened)
    off_diagonal = 1.0 - torch.eye(paths, device=grouped_paths.device)[None]
    return (torch.exp(-distances / bandwidth) * off_diagonal).sum() / (
        batch * max(paths * (paths - 1), 1)
    )


def regime_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_probability = probabilities.mean(dim=0).clamp_min(1e-8)
    return (mean_probability * (mean_probability * probabilities.shape[-1]).log()).sum()
