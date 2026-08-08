# RegimeCurve-GAN

RegimeCurve-GAN is a conditional generative model for US Treasury yield curves. Given the preceding three business days of the curve, it generates ten alternative scenarios, each covering the next ten business days.

The project was designed for scenario generation rather than point forecasting. A useful model should therefore satisfy two objectives at the same time:

1. each simulated path should be financially and temporally plausible;
2. repeated simulations from the same context should produce diverse but credible futures.

The implementation combines a regime-aware conditional Wasserstein GAN, an interpretable Nelson–Siegel factor representation, low-rank residual factors, heavy-tailed hierarchical noise, autoregressive stochastic experts, two specialised critics, latent-information recovery, and explicit moment and covariance matching.

## Contents

- [Architecture](#architecture)
- [Design choices](#design-choices)
- [Data and preprocessing](#data-and-preprocessing)
- [Training objective](#training-objective)
- [Running the project](#running-the-project)
- [Reproducing results](#reproducing-results)
- [Evaluation](#evaluation)
- [Assumptions and constraints](#assumptions-and-constraints)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Use of AI tools](#use-of-ai-tools)

## Architecture

### Overview

```text
Previous three curves
        │
        ▼
Nelson–Siegel and residual-factor encoder
        │
        ▼
GRU context encoder ───────────────► regime probabilities
        │                                      │
        ├──────── path-level noise              │
        └──────── daily shock noise             │
                        │                       │
                        ▼                       ▼
       sampled stochastic expert ◄──────── discrete regime
                        │
                        ▼
       autoregressive drift and volatility
                        │
                        ▼
          learnable factor-specific step scales
                        │
                        ▼
        differentiable yield-curve reconstruction
                 │                         │
                 ▼                         ▼
          temporal critic             shape critic
```

### 1. Financial factor representation

Generating all maturities independently makes it easy for a neural network to create irregular curve shapes and difficult to learn the strong dependence between maturities. Each observed curve is instead represented as

\[
y_t(\tau)
=
\beta_{0,t}
+\beta_{1,t}\frac{1-e^{-\lambda\tau}}{\lambda\tau}
+\beta_{2,t}\left(
\frac{1-e^{-\lambda\tau}}{\lambda\tau}-e^{-\lambda\tau}
\right)
+\varepsilon_t(\tau).
\]

The first three coefficients approximately describe level, slope, and curvature. A PCA representation of the Nelson–Siegel residuals retains local maturity effects that a strict three-factor model cannot capture. The default model uses three residual components, giving a six-dimensional latent curve state.

Both the factor transform and its standardisation statistics are fitted using training data only.

### 2. Context encoder and latent regimes

A GRU encodes the previous three latent curve states. Its output is used both as a conditioning vector and to infer probabilities over four latent regimes. Straight-through Gumbel–Softmax selects one expert per scenario during training, while categorical sampling is used during inference. This prevents averaging incompatible regime behaviours.

The regimes are not given economic labels during training. They are intended to let different experts specialise in behaviours such as parallel shifts, steepening, flattening, or high-volatility movement. Because the regimes are latent, their numerical labels are arbitrary and must be interpreted after training.

### 3. Heavy-tailed hierarchical noise

Two types of randomness are supplied to the generator:

- path-level noise controls the overall ten-day scenario;
- daily noise introduces local shocks along the simulated path.

This division encourages coherence across the horizon while avoiding deterministic trajectories. Daily innovations follow a Student-\(t_5\) prior rather than a Gaussian prior. Regime-dependent scales allow calm and stressed scenarios, while factor-specific positive scales let curvature receive larger shocks without injecting independent noise directly into every maturity.

### 4. Autoregressive stochastic regime experts

Each regime expert is a GRUCell decoder. At every forecast day it consumes the previous latent state, encoded context, path code, and a fresh daily innovation. Separate heads predict drift and conditional volatility. One sampled expert produces the complete scenario, and increments accumulate from the last observed state.

A fixed global multiplier initially produced movements that were systematically too small. It was replaced with positive, learnable, factor-specific step scales. Level, slope, curvature, and residual factors can consequently learn different shock amplitudes while the positivity constraint keeps their interpretation stable.

### 5. Differentiable curve decoder

The generated latent states are converted back to all 13 maturities using the fitted Nelson–Siegel basis and residual PCA components. Because this reconstruction is implemented in PyTorch, gradients from the curve-level losses and critic pass directly into the generator.

### 6. Dual critics

One discriminator was not expected to assess temporal behaviour and maturity geometry equally well. The model therefore uses two Wasserstein critics.

The **temporal critic** is a bidirectional GRU applied to the complete sequence consisting of three conditioning days and ten future days. It evaluates continuity at the forecast boundary, path dynamics, and temporal dependence.

The **shape critic** processes yield levels together with first and second differences across maturity. It focuses on cross-maturity consistency, excessive kinks, slope, and curvature. Both critics receive a minibatch-standard-deviation feature so batches of nearly identical scenarios are detectable.

## Design choices

### Why generate factors rather than yields directly?

Treasury maturities are highly correlated, and most curve variation lies in a small number of economic directions. The factor representation reduces dimensionality, improves sample efficiency, and makes generated scenarios easier to diagnose. Residual PCA components prevent the model from being forced into an exact Nelson–Siegel family.

### Why WGAN-GP?

The dataset is modest relative to typical image-generation datasets, and the target distribution is continuous and strongly correlated. Wasserstein training supplies a smoother critic signal than a binary GAN objective. Gradient penalties constrain both critics and improve stability without clipping their weights.

### Why use learned regimes?

Yield-curve dynamics are non-stationary. A single generator can average incompatible behaviours and produce conservative scenarios. A learned mixture of experts provides regime-dependent capacity without requiring subjective historical labels.

### Why use explicit statistical losses?

Adversarial loss alone does not guarantee correct marginal volatility or cross-maturity covariance. The initial experiment produced visually plausible curves but underestimated daily movement and showed limited curvature diversity. Moment and covariance losses were therefore included as auditable, financially meaningful training targets.

## Data and preprocessing

The supplied workbook contains 7,758 dated observations from 1990 to 2024 and 13 Treasury maturities from one month to 30 years.

Preprocessing performs the following steps:

1. parse dates and sort observations from oldest to newest;
2. remove duplicate dates;
3. discard the single archive row containing no reported yields;
4. interpolate missing tenors across maturity within the same date;
5. fit the Nelson–Siegel and residual-PCA transform on training rows only;
6. construct rolling windows with three context days and ten target days;
7. assign a window to a split using the final date of its target horizon.

The default chronological split is:

| Partition | Target-horizon end date |
|---|---|
| Training | Up to 31 December 2017 |
| Validation | 1 January 2018 to 31 December 2020 |
| Test | 1 January 2021 onward |

A random split was deliberately avoided because neighbouring rolling windows overlap heavily and would leak near-identical market states between training and evaluation.

## Training objective

The critics minimise the conditional Wasserstein objectives with gradient penalties. The generator objective is

\[
\mathcal L_G
=
\mathcal L_{\mathrm{adv}}
+\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}
+\lambda_{\mathrm{moment}}\mathcal L_{\mathrm{moment}}
+\lambda_{\mathrm{cov}}\mathcal L_{\mathrm{cov}}
+\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}
+\lambda_{\mathrm{regime}}\mathcal L_{\mathrm{regime}}.
\]

- **Adversarial loss** encourages temporal and cross-maturity realism.
- **Smoothness loss** penalises excessive second differences across maturity.
- **Moment loss** matches the mean and volatility of daily yield changes.
- **Covariance loss** matches the covariance matrix of daily changes across maturities.
- **Economic diversity loss** rewards latent noise that changes level, slope, and curvature, with additional weight on curvature.
- **Latent-information loss** trains an auxiliary network to recover the path code from the generated scenario, discouraging the generator from ignoring noise.
- **Within-context repulsion** separates four paths sampled from the same conditioning history.
- **Regime balance regularisation** discourages premature collapse to a single expert.

Yield changes are converted to basis points before covariance matching. This makes the magnitude of the loss numerically interpretable. Loss weights should still be monitored because covariance loss can otherwise dominate adversarial training.

The recommended initial weights are:

```yaml
smoothness_weight: 0.02
moment_weight: 1.0
covariance_weight: 0.05
diversity_weight: 0.20
information_weight: 0.10
repulsion_weight: 0.05
regime_balance_weight: 0.01
paths_per_context: 4
```

## Running the project

### Requirements

- Python 3.11 or newer
- `uv`
- a CUDA-capable GPU is recommended but not required

From the repository root, install the project and development dependencies:

```bash
uv sync --extra dev
```

Run the tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check src tests
```

### Train

```bash
uv run regimecurve-train --config configs/default.yaml
```

The best validation checkpoint is saved as:

```text
outputs/best.pt
```

The checkpoint contains model parameters, configuration, maturity columns, fitted factor transform, and validation score.

### Generate the required scenarios

```bash
uv run regimecurve-generate \
  --checkpoint outputs/best.pt \
  --data data/data.xlsx \
  --scenarios 10 \
  --output outputs/generated_curves.csv
```

This writes 100 rows: ten scenarios multiplied by ten forecast days. Regime probabilities are written to `outputs/regime_probabilities.csv`.

### Evaluate and visualise

```bash
uv run regimecurve-evaluate \
  --generated outputs/generated_curves.csv \
  --historical data/data.xlsx \
  --output-dir outputs/evaluation
```

The evaluation directory contains summary metrics and a terminal-curve plot.

## Reproducing results

The default random seed is stored in `configs/default.yaml`. To reproduce a run:

1. use the provided workbook without changing row order or maturity names;
2. use the committed configuration file;
3. install dependencies from the project definition;
4. run the tests before training;
5. train from a fresh output directory;
6. generate scenarios using the saved best checkpoint;
7. retain the checkpoint, configuration, metrics, and plots together.

For a fast end-to-end smoke test, temporarily use:

```yaml
model:
  hidden_dim: 32
training:
  batch_size: 1024
  epochs: 1
  critic_steps: 2
```

The smoke test verifies execution only. Its generated curves must not be reported as final model performance.

GPU kernels and dependency versions can cause small numerical differences even with fixed seeds. Conclusions should therefore be based on several seeds rather than one visually attractive run.

## Evaluation

Realism and diversity are evaluated separately.

### Realism diagnostics

- distributions of daily changes by maturity;
- volatility term structure;
- level, slope, and curvature distributions;
- cross-maturity correlation and covariance matrices;
- lag-one autocorrelation;
- PCA eigenvalue spectrum;
- continuity between the last context day and first generated day;
- maximum daily movement and path roughness.

### Diversity diagnostics

- terminal yield dispersion by maturity;
- pairwise distance between scenarios from the same context;
- dispersion of level, slope, and curvature;
- sensitivity of generated paths to latent noise;
- expert usage and regime entropy.

Generating only ten scenarios gives a useful visualisation but an unstable estimate of tail behaviour. Quantitative evaluation should use at least 100 scenarios per test context, even though the submitted example contains the ten scenarios requested by the task.

Unconditional comparison with the complete 1990–2024 level distribution is not an appropriate measure for scenarios anchored to a 2024 context. Conditional evaluation should compare generated changes with historical ten-day changes starting from similar level, slope, curvature, and recent volatility.

## Assumptions and constraints

- Observations are treated as business-day sequences. The model does not attempt to simulate weekends or holidays.
- The supplied rates are modelled as a multivariate par-yield curve, not as bootstrapped zero-coupon discount factors.
- Missing maturities are interpolated across maturity within a date. No future observation is used to fill a past curve.
- The three-day context is the only conditioning information. Macroeconomic announcements, policy rates, inflation, and trading-volume variables are not included.
- Nelson–Siegel decay is fixed in the default configuration for stability and identifiability.
- Regime identities are latent and identifiable only up to permutation.
- The model is intended for research scenario generation, not trading or valuation without additional calibration and validation.

## Limitations

### Limited conditioning information

Three days may not distinguish persistent macroeconomic regimes. The regime encoder can only infer state from the recent curve itself.

### No strict no-arbitrage guarantee

Smooth par-yield curves are not equivalent to arbitrage-free discount curves. A genuine no-arbitrage constraint would require instrument conventions and a differentiable bootstrapping layer.

### Historical missingness

Interpolating tenors that were not published in early history creates usable rectangular data but can understate uncertainty at those maturities.

### GAN optimisation

WGAN-GP is more stable than a vanilla GAN but remains sensitive to critic strength, loss weights, learning rate, and random seed. Low critic loss does not imply good scenario calibration.

### Aggressive heavy-tailed priors

Large Student-\(t\) shocks improve exploration but can create implausible paths or destabilise the critics. Prior scale is therefore treated as a tunable stress parameter, not as a substitute for diversity objectives. Results must be checked against historical conditional volatility and tail movements.

### Evaluation sample size

Ten requested scenarios are insufficient for reliable coverage or tail-risk estimates. Larger Monte Carlo samples are needed for model selection.

## Future work

The current version uses straight-through Gumbel–Softmax during training and categorical regime sampling during inference. A useful extension would replace the fixed Student-\(t\) degrees of freedom and regime scales with calibrated or learned conditional distributions.

Other useful extensions include:

1. a Transformer or state-space decoder as a higher-capacity alternative to the autoregressive GRUCell;
2. rolling conditional coverage, energy-score, and variogram-score evaluation;
3. nearest-neighbour conditional historical bootstrap and dynamic Nelson–Siegel baselines;
4. diffusion or flow-matching baselines to test whether the remaining diversity problem is specific to adversarial training;
5. missingness-aware training rather than deterministic interpolation;
6. macroeconomic and policy-rate conditioning;
7. differentiable discount-curve bootstrapping and soft no-arbitrage penalties;
8. systematic ablation of regimes, residual factors, dual critics, covariance matching, and diversity regularisation.

## Use of AI tools

AI coding assistance was used for:

- brainstorming the initial non-standard factor and regime architecture;
- scaffolding the package structure and command-line entry points;
- suggesting unit tests and reproducibility checks;
- reviewing tensor shapes and data-leakage risks;
- drafting documentation;
- diagnosing the first generated scenarios and proposing targeted changes.

AI output was treated as a source of proposals, not as an authority. All generated code was inspected, executed, tested, and revised. The end-to-end smoke test exposed a fully empty row in the Treasury archive that the initial implementation had not handled, and preprocessing was corrected accordingly.

The most important disagreement concerned diversity. The initial AI-proposed architecture and metrics suggested that the generated scenarios had non-zero dispersion. Visual inspection showed that this was not sufficient: most variation resembled parallel shifts, curvature dispersion was small, and daily volatility was underestimated. The conclusion was therefore changed from “adequately diverse” to “under-dispersed in economically important directions.” The fixed increment multiplier was replaced by learnable factor-specific scales, and covariance matching was added to the generator objective.

Another rejected suggestion was to describe the model as arbitrage-aware merely because it generated smooth curves. Smoothness of Treasury par yields does not establish absence of arbitrage. The final description explicitly avoids that claim and treats differentiable curve bootstrapping as future work.

The candidate should be able to explain every architectural component and reproduce the diagnostic reasoning without relying on AI-generated text. This section should be updated if further tools or manual changes are used before submission.
