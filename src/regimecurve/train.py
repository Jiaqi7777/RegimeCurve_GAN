from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import Adam

from .data import make_loader, prepare_data
from .losses import (
    autocorrelation_loss,
    conditional_factor_spread_loss,
    correlation_loss,
    covariance_loss,
    daily_tail_loss,
    diversity_loss,
    economic_curve_features,
    gradient_penalty,
    moment_loss,
    regime_balance_loss,
    repulsion_loss,
    shape_repulsion_loss,
    smoothness_loss,
    terminal_factor_loss,
    within_context_shape_covariance_loss,
    within_context_shape_trace_loss,
    within_context_whitened_shape_loss,
)
from .model import RegimeCurveGAN
from .utils import ensure_output, load_config, set_seed


def build_model(config: dict, bundle) -> RegimeCurveGAN:
    model, noise = config["model"], config["noise"]
    return RegimeCurveGAN(
        state_dim=bundle.transform.state_dim,
        maturity_dim=len(bundle.frame.columns),
        hidden_dim=model["hidden_dim"],
        latent_dim=model["latent_dim"],
        daily_noise_dim=model["daily_noise_dim"],
        horizon=config["data"]["horizon_days"],
        num_regimes=model["num_regimes"],
        gumbel_temperature=model["gumbel_temperature"],
        noise_df=noise["degrees_of_freedom"],
        noise_scale=noise["base_scale"],
        regime_scales=noise["regime_scales"],
        decoder_parameters=bundle.transform.torch_decoder_parameters(),
    )


@torch.no_grad()
def validation_score(model: RegimeCurveGAN, loader, device: torch.device,
                     training_config: dict) -> float:
    model.eval()
    scores = []
    paths = training_config["paths_per_context"]
    for context, future in loader:
        context, future = context.to(device), future.to(device)
        expanded_context = context.repeat_interleave(paths, dim=0)
        generated, _, _, _ = model.generator(expanded_context)
        real_curves = model.decoder(future)
        fake_curves = model.decoder(generated)
        real_expanded = real_curves.repeat_interleave(paths, dim=0)
        grouped_fake = fake_curves.view(context.shape[0], paths, *fake_curves.shape[1:])
        scores.append((
            moment_loss(real_expanded, fake_curves)
            + training_config["covariance_weight"] * covariance_loss(real_expanded, fake_curves)
            + training_config["shape_covariance_weight"]
            * within_context_shape_covariance_loss(real_curves, grouped_fake)
            + training_config["whitened_shape_weight"]
            * within_context_whitened_shape_loss(real_curves, grouped_fake)
            + training_config["shape_trace_weight"]
            * within_context_shape_trace_loss(real_curves, grouped_fake)
            + training_config["correlation_weight"] * correlation_loss(real_expanded, fake_curves)
            + training_config["autocorrelation_weight"]
            * autocorrelation_loss(real_expanded, fake_curves)
            + training_config["daily_tail_weight"] * daily_tail_loss(real_expanded, fake_curves)
            + training_config["terminal_weight"] * terminal_factor_loss(real_expanded, fake_curves)
        ).item())
    model.train()
    return float(sum(scores) / max(len(scores), 1))


def train(config: dict) -> Path:
    set_seed(config["seed"])
    bundle = prepare_data(config)
    cfg = config["training"]
    train_loader = make_loader(bundle.train, cfg["batch_size"], True, cfg["num_workers"])
    validation_loader = make_loader(bundle.validation, cfg["batch_size"], False,
                                    cfg["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, bundle).to(device)
    generator_parameters = list(model.generator.parameters()) + list(model.latent_recovery.parameters())
    generator_optimizer = Adam(generator_parameters, lr=cfg["learning_rate"], betas=(0.0, 0.9))
    critic_parameters = list(model.temporal_critic.parameters()) + list(model.shape_critic.parameters())
    critic_optimizer = Adam(critic_parameters, lr=cfg["learning_rate"], betas=(0.0, 0.9))
    output = ensure_output(config["output_dir"])
    best_score = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        generator_total = critic_total = 0.0
        for step, (context, real_future) in enumerate(train_loader):
            context, real_future = context.to(device), real_future.to(device)
            with torch.no_grad():
                fake_future, _, _, _ = model.generator(context)
            real_curves, fake_curves = model.decoder(real_future), model.decoder(fake_future)
            temporal_loss = (model.temporal_critic(context, fake_future).mean()
                             - model.temporal_critic(context, real_future).mean())
            shape_loss = model.shape_critic(fake_curves).mean() - model.shape_critic(real_curves).mean()
            penalty = gradient_penalty(model.temporal_critic, real_future, fake_future, context)
            penalty += gradient_penalty(model.shape_critic, real_curves, fake_curves)
            critic_loss = temporal_loss + shape_loss + cfg["gradient_penalty"] * penalty
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters, max_norm=10.0)
            critic_optimizer.step()
            critic_total += critic_loss.item()

            if (step + 1) % cfg["critic_steps"] == 0:
                paths = cfg["paths_per_context"]
                expanded_context = context.repeat_interleave(paths, dim=0)
                expanded_real = real_future.repeat_interleave(paths, dim=0)
                generated, probabilities, _, latent = model.generator(expanded_context)
                curves = model.decoder(generated)
                real_expanded_curves = model.decoder(expanded_real)
                adversarial = -model.temporal_critic(expanded_context, generated).mean()
                adversarial -= model.shape_critic(curves).mean()
                recovered_latent = model.latent_recovery(generated)
                grouped_curves = curves.view(context.shape[0], paths, *curves.shape[1:])
                grouped_latent = latent.view(context.shape[0], paths, -1)
                economic = economic_curve_features(grouped_curves)
                if paths >= 2:
                    economic_diversity = diversity_loss(
                        economic[:, 0], economic[:, 1], grouped_latent[:, 0], grouped_latent[:, 1]
                    )
                else:
                    economic_diversity = curves.new_zeros(())
                generator_loss = adversarial
                generator_loss += cfg["smoothness_weight"] * smoothness_loss(curves)
                generator_loss += cfg["moment_weight"] * moment_loss(real_expanded_curves, curves)
                generator_loss += cfg["covariance_weight"] * covariance_loss(real_expanded_curves, curves)
                real_curves = model.decoder(real_future)
                generator_loss += cfg["shape_covariance_weight"] * (
                    within_context_shape_covariance_loss(real_curves, grouped_curves)
                )
                generator_loss += cfg["whitened_shape_weight"] * (
                    within_context_whitened_shape_loss(real_curves, grouped_curves)
                )
                generator_loss += cfg["shape_trace_weight"] * (
                    within_context_shape_trace_loss(real_curves, grouped_curves)
                )
                generator_loss += cfg["correlation_weight"] * correlation_loss(
                    real_expanded_curves, curves
                )
                generator_loss += cfg["autocorrelation_weight"] * autocorrelation_loss(
                    real_expanded_curves, curves
                )
                generator_loss += cfg["daily_tail_weight"] * daily_tail_loss(
                    real_expanded_curves, curves
                )
                generator_loss += cfg["terminal_weight"] * terminal_factor_loss(
                    real_expanded_curves, curves
                )
                generator_loss += cfg["diversity_weight"] * economic_diversity
                generator_loss += cfg["conditional_spread_weight"] * conditional_factor_spread_loss(
                    grouped_curves
                )
                generator_loss += cfg["shape_repulsion_weight"] * shape_repulsion_loss(grouped_curves)
                generator_loss += cfg["information_weight"] * F.mse_loss(recovered_latent, latent)
                generator_loss += cfg["repulsion_weight"] * repulsion_loss(economic)
                generator_loss += cfg["regime_balance_weight"] * regime_balance_loss(probabilities)
                generator_optimizer.zero_grad(set_to_none=True)
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator_parameters, max_norm=10.0)
                generator_optimizer.step()
                generator_total += generator_loss.item()

        score = validation_score(model, validation_loader, device, cfg)
        if score < best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(), "config": config, "transform": bundle.transform,
                "columns": list(bundle.frame.columns), "validation_score": score,
            }, output / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:04d} critic={critic_total/len(train_loader):.4f} "
                  f"generator={generator_total/max(len(train_loader)//cfg['critic_steps'], 1):.4f} "
                  f"validation={score:.6f}")
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    checkpoint = train(load_config(args.config))
    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
