from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_curves
from .model import RegimeCurveGAN


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config, transform = checkpoint["config"], checkpoint["transform"]
    model_cfg, noise = config["model"], config["noise"]
    model = RegimeCurveGAN(
        transform.state_dim, len(checkpoint["columns"]), model_cfg["hidden_dim"],
        model_cfg["latent_dim"], model_cfg["daily_noise_dim"],
        config["data"]["horizon_days"], model_cfg["num_regimes"],
        model_cfg["gumbel_temperature"], noise["degrees_of_freedom"],
        noise["base_scale"], noise["regime_scales"],
        transform.torch_decoder_parameters(),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate(checkpoint_path: str, data_path: str, scenarios: int, output_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    transform, config = checkpoint["transform"], checkpoint["config"]
    frame = load_curves(data_path)[checkpoint["columns"]]
    context_days = config["data"]["context_days"]
    context = transform.encode(frame.to_numpy()[-context_days:])[None]
    context = torch.from_numpy(np.repeat(context, scenarios, axis=0)).to(device)
    future, probabilities, selections, _ = model.generator(context)
    curves = model.decoder(future).cpu().numpy()
    records = []
    for scenario in range(scenarios):
        for day in range(curves.shape[1]):
            records.append([scenario, day + 1, *curves[scenario, day]])
    output = pd.DataFrame(records, columns=["scenario", "day", *checkpoint["columns"]])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    regime_frame = pd.DataFrame(
        probabilities.cpu().numpy(), columns=[f"regime_{i}_probability" for i in range(probabilities.shape[1])]
    )
    regime_frame.insert(0, "sampled_regime", selections.argmax(dim=-1).cpu().numpy())
    regime_frame.to_csv(Path(output_path).with_name("regime_probabilities.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--data", default="data/data.xlsx")
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--output", default="outputs/generated_curves.csv")
    args = parser.parse_args()
    generate(args.checkpoint, args.data, args.scenarios, args.output)
    print(f"Saved {args.scenarios} scenarios to {args.output}")


if __name__ == "__main__":
    main()
