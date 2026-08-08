import pytest

torch = pytest.importorskip("torch")

from regimecurve.losses import (
    autocorrelation_loss,
    conditional_factor_spread_loss,
    correlation_loss,
    covariance_loss,
    daily_tail_loss,
    economic_curve_features,
    level_neutral_shape_covariance_loss,
    repulsion_loss,
    shape_repulsion_loss,
    terminal_factor_loss,
    whitened_shape_covariance_loss,
    within_context_shape_covariance_loss,
    within_context_shape_trace_loss,
    within_context_whitened_shape_loss,
)
from regimecurve.model import RegimeGenerator, ShapeCritic, TemporalCritic


def build_generator():
    return RegimeGenerator(
        state_dim=6, hidden_dim=16, latent_dim=8, daily_noise_dim=8,
        horizon=10, num_regimes=4, gumbel_temperature=0.5,
        noise_df=5.0, noise_scale=1.5, regime_scales=[0.7, 1.0, 1.5, 2.5],
    )


def test_generator_shapes_and_discrete_regimes():
    generator = build_generator()
    future, probabilities, selections, latent = generator(torch.randn(8, 3, 6))
    assert future.shape == (8, 10, 6)
    assert probabilities.shape == selections.shape == (8, 4)
    assert latent.shape == (8, 8)
    assert torch.allclose(probabilities.sum(1), torch.ones(8), atol=1e-6)
    assert torch.allclose(selections.sum(1), torch.ones(8), atol=1e-6)
    assert set(selections.unique().tolist()).issubset({0.0, 1.0})


def test_generator_rejects_insufficient_daily_noise_dimension():
    with pytest.raises(ValueError):
        RegimeGenerator(6, 16, 8, 4, 10, 4, 0.5, 5.0, 1.5, [1, 1, 1, 1])


def test_critics_return_one_score_per_path():
    context, future = torch.randn(8, 3, 6), torch.randn(8, 10, 6)
    assert TemporalCritic(6, 16)(context, future).shape == (8,)
    assert ShapeCritic(13, 16)(torch.randn(8, 10, 13)).shape == (8,)


def test_covariance_loss_is_zero_for_identical_curves():
    curves = torch.randn(8, 10, 13)
    assert covariance_loss(curves, curves).item() == pytest.approx(0.0)
    assert level_neutral_shape_covariance_loss(curves, curves).item() == pytest.approx(0.0)
    assert whitened_shape_covariance_loss(curves, curves).item() == pytest.approx(0.0, abs=1e-5)


def test_economic_features_and_repulsion_are_finite():
    curves = torch.randn(3, 4, 10, 13)
    features = economic_curve_features(curves)
    assert features.shape == (3, 4, 10, 3)
    assert torch.isfinite(repulsion_loss(features))


def test_calibration_losses_are_zero_for_identical_paths():
    curves = torch.randn(32, 10, 13)
    assert correlation_loss(curves, curves).item() == pytest.approx(0.0)
    assert autocorrelation_loss(curves, curves).item() == pytest.approx(0.0)
    assert terminal_factor_loss(curves, curves).item() == pytest.approx(0.0)
    assert daily_tail_loss(curves, curves).item() == pytest.approx(0.0)


def test_conditional_shape_losses_are_finite():
    grouped = torch.randn(8, 4, 10, 13)
    real = torch.randn(8, 10, 13)
    assert torch.isfinite(conditional_factor_spread_loss(grouped))
    assert torch.isfinite(shape_repulsion_loss(grouped))
    assert torch.isfinite(within_context_shape_covariance_loss(real, grouped))
    assert torch.isfinite(within_context_whitened_shape_loss(real, grouped))
    assert torch.isfinite(within_context_shape_trace_loss(real, grouped))
