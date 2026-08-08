from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def append_minibatch_std(features: torch.Tensor) -> torch.Tensor:
    """Append one batch-diversity feature to every observation."""
    if features.shape[0] == 1:
        batch_std = features.new_zeros(())
    else:
        centred = features - features.mean(dim=0, keepdim=True)
        batch_std = (centred.square().mean(dim=0) + 1e-8).sqrt().mean()
    return torch.cat([features, batch_std.expand(features.shape[0], 1)], dim=-1)


class ContextEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_regimes: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.regime_head = nn.Linear(hidden_dim, num_regimes)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(context)
        encoded = hidden[-1]
        return encoded, self.regime_head(encoded)


class StochasticRegimeExpert(nn.Module):
    """Autoregressive expert with explicit drift and conditional volatility."""

    def __init__(self, state_dim: int, context_dim: int, latent_dim: int,
                 noise_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.condition = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim), nn.SiLU(),
        )
        self.initial_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.cell = nn.GRUCell(state_dim + noise_dim + hidden_dim, hidden_dim)
        self.drift_head = nn.Linear(hidden_dim, state_dim)
        self.volatility_head = nn.Linear(hidden_dim, state_dim)
        # Stochastic shocks remain factor-specific; drift is deliberately much smaller.
        initial = torch.full((state_dim,), inverse_softplus(0.05))
        if state_dim >= 3:
            initial[2] = inverse_softplus(0.07)
        self.raw_shock_scale = nn.Parameter(initial)
        self.raw_drift_scale = nn.Parameter(torch.tensor(-0.6931))
        self.raw_mean_reversion = nn.Parameter(torch.tensor(-0.6931))

    def forward(self, encoded: torch.Tensor, path_noise: torch.Tensor,
                daily_noise: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        condition = self.condition(torch.cat([encoded, path_noise], dim=-1))
        hidden = torch.tanh(self.initial_hidden(condition))
        state = initial_state
        generated = []
        shock_scale = F.softplus(self.raw_shock_scale)
        drift_scale = 0.03 * torch.sigmoid(self.raw_drift_scale)
        mean_reversion = 0.20 * torch.sigmoid(self.raw_mean_reversion)
        for day in range(self.horizon):
            hidden = self.cell(
                torch.cat([state, daily_noise[:, day], condition], dim=-1), hidden
            )
            drift = torch.tanh(self.drift_head(hidden))
            volatility = F.softplus(self.volatility_head(hidden)) + 1e-4
            innovation = daily_noise[:, day, :self.state_dim]
            increment = (
                drift_scale * drift
                + shock_scale * volatility * innovation
                + mean_reversion * (initial_state - state)
            )
            state = state + increment
            generated.append(state)
        return torch.stack(generated, dim=1)


class RegimeGenerator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int,
                 daily_noise_dim: int, horizon: int, num_regimes: int,
                 gumbel_temperature: float, noise_df: float, noise_scale: float,
                 regime_scales: list[float]):
        super().__init__()
        if daily_noise_dim < state_dim:
            raise ValueError("daily_noise_dim must be at least state_dim")
        if len(regime_scales) != num_regimes:
            raise ValueError("One noise scale is required per regime")
        self.latent_dim = latent_dim
        self.daily_noise_dim = daily_noise_dim
        self.horizon = horizon
        self.num_regimes = num_regimes
        self.gumbel_temperature = gumbel_temperature
        self.noise_df = noise_df
        self.noise_scale = noise_scale
        self.encoder = ContextEncoder(state_dim, hidden_dim, num_regimes)
        self.experts = nn.ModuleList([
            StochasticRegimeExpert(state_dim, hidden_dim, latent_dim, daily_noise_dim,
                                   hidden_dim, horizon)
            for _ in range(num_regimes)
        ])
        self.register_buffer("regime_scales", torch.tensor(regime_scales, dtype=torch.float32))

    def sample_daily_noise(self, batch: int, device: torch.device) -> torch.Tensor:
        distribution = torch.distributions.StudentT(
            df=torch.tensor(self.noise_df, device=device)
        )
        noise = self.noise_scale * distribution.sample(
            (batch, self.horizon, self.daily_noise_dim)
        )
        # Preserve heavy tails while preventing a single extreme draw from destroying a batch.
        return noise.clamp(-8.0, 8.0)

    def select_regime(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.gumbel_softmax(
                logits, tau=self.gumbel_temperature, hard=True, dim=-1
            )
        probabilities = torch.softmax(logits, dim=-1)
        indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        return F.one_hot(indices, num_classes=self.num_regimes).float()

    def forward(self, context: torch.Tensor, path_noise: torch.Tensor | None = None,
                daily_noise: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        encoded, regime_logits = self.encoder(context)
        probabilities = torch.softmax(regime_logits, dim=-1)
        regime_weights = self.select_regime(regime_logits)
        if path_noise is None:
            path_noise = torch.randn(batch, self.latent_dim, device=context.device)
        if daily_noise is None:
            daily_noise = self.sample_daily_noise(batch, context.device)
        selected_scale = (regime_weights * self.regime_scales).sum(dim=-1)
        scaled_daily_noise = daily_noise * selected_scale[:, None, None]
        paths = torch.stack([
            expert(encoded, path_noise, scaled_daily_noise, context[:, -1])
            for expert in self.experts
        ], dim=1)
        selected_path = (paths * regime_weights[:, :, None, None]).sum(dim=1)
        return selected_path, probabilities, regime_weights, path_noise


class TemporalCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(torch.cat([context, future], dim=1))
        features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.head(append_minibatch_std(features)).squeeze(-1)


class ShapeCritic(nn.Module):
    def __init__(self, maturity_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(maturity_dim * 3 + 1, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1),
        )

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        first = curves[:, :, 1:] - curves[:, :, :-1]
        second = first[:, :, 1:] - first[:, :, :-1]
        features = torch.cat([
            curves.mean(1), F.pad(first.mean(1), (0, 1)), F.pad(second.mean(1), (0, 2)),
        ], dim=-1)
        return self.net(append_minibatch_std(features)).squeeze(-1)


class LatentRecoveryNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, generated_path: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(generated_path)
        return self.head(hidden[-1])


class TorchCurveDecoder(nn.Module):
    def __init__(self, parameters: dict):
        super().__init__()
        for name, value in parameters.items():
            self.register_buffer(name, torch.as_tensor(value))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        beta = states[..., :3] * self.beta_scale + self.beta_mean
        scores = states[..., 3:] * self.res_scale + self.res_mean
        return beta @ self.basis.T + scores @ self.res_components + self.residual_mean


class RegimeCurveGAN(nn.Module):
    def __init__(self, state_dim: int, maturity_dim: int, hidden_dim: int,
                 latent_dim: int, daily_noise_dim: int, horizon: int,
                 num_regimes: int, gumbel_temperature: float, noise_df: float,
                 noise_scale: float, regime_scales: list[float], decoder_parameters: dict):
        super().__init__()
        self.generator = RegimeGenerator(
            state_dim, hidden_dim, latent_dim, daily_noise_dim, horizon, num_regimes,
            gumbel_temperature, noise_df, noise_scale, regime_scales,
        )
        self.temporal_critic = TemporalCritic(state_dim, hidden_dim)
        self.shape_critic = ShapeCritic(maturity_dim, hidden_dim)
        self.latent_recovery = LatentRecoveryNetwork(state_dim, hidden_dim, latent_dim)
        self.decoder = TorchCurveDecoder(decoder_parameters)
