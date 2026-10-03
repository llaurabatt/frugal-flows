# Example fits: MorphoMNIST 8×8, n = 5000, experiment E2, dataset 1

Saved weights for every model of the n = 5000 paper grid on ONE dataset, so the models can be loaded and compared side by side.

## The data

- All ten digits, 8×8 (K = 64 pixels, logit scale), `seed_data` 101.
- A fixed subsample of 5000 images.
- E2: treatment confounded through stroke thickness, homogeneous true effect; assignment seed 1.
- Each model is the single fit with fit seed 1, at the paper settings.

## The models

`models.json` lists the source runs.

| Folder | Model | Y scaling | ATE MAE |
|---|---|---|---|
| `G-flex-std` | Gaussian-scale flexible frugal flow (`flexible_continuous_gaussian`), the recommended model | standardised | 0.0105 |
| `frengression` | frengression baseline (engression + frugal parametrisation, PyTorch) | per-pixel | 0.0144 |
| `U-flex-raw` | uniform-base flexible frugal flow (`flexible_continuous`) | raw logits | 0.0247 |
| `U-flex-std` | the same, standardised Y (control) | standardised | 0.0396 |
| `G-LT-head` | Gaussian-scale location translation, per-pixel naive start, 10× shift lr | standardised | 0.0035 |
| `G-LT-std` | Gaussian-scale location translation, scalar start 0.5 | standardised | 0.166 |
| `U-LT-raw` | uniform-base location translation | raw logits | 0.366 |

How to read these numbers:
- **Dataset choice.** This dataset was chosen because the Gaussian spline does well on it (its best of the 10 E2 datasets), so these errors are optimistic for that model.
- **Typical values.** Across all 10 datasets, the mean ATE MAE on E2 is:

  | Model | Single fit | Average of 5 fits |
  |---|---|---|
  | Gaussian spline | 0.0143 | 0.0097 |
  | frengression | 0.0135 | 0.0135 |
  | Uniform flexible | 0.0294 | — |

- **Location translation on E2.** It wins here only because E2's effect is a pure shift. It also recovers the truth with shuffled covariates, so its E2 result does not show adjustment.

## Files in each folder

- `model.eqx` (flows) or `model.pt` (frengression): the weights.
- `config.json`: every setting, plus the dataset id and hash.
- `metrics.json`: ATE MAE and the other scores.
- `arrays.npz`: `tau_hat` (the 64-pixel ATE map, logit scale), covariate ranks `u_z`, loss curves.
- `wandb.json`: the W&B run (proj-lb / Frugal Images).
- `G-flex-std/model_spec.json` additionally holds the build settings and the fitted Y standardiser.

## Loading

The Gaussian spline loads with no runner and no data:

```python
import jax.random as jr
import frugal_flows as ff

flow, ot = ff.load_gaussian_flow("examples/morphomnist_8x8_n5000/G-flex-std")
s = ff.interventional_samples(jr.key(0), flow, 1, 5000, outcome_transform=ot, dim_y=64)
s["ate"]          # (64,) ATE map on the logit scale; matches arrays.npz["tau_hat"] up to Monte Carlo error
```

Any model, through the MorphoMNIST runner, needs the `validation` extras and the MNIST data in `data/`. The runner rebuilds the dataset from `config.json` and checks its hash:

```python
# from validation/morphomnist/
import exp_ate_recovery as E, exp_frengression_recovery as F
flow = E.load_model("../../examples/morphomnist_8x8_n5000/U-flex-raw")
model, inputs = F.load_model("../../examples/morphomnist_8x8_n5000/frengression")
```

## The full set

All 300 Gaussian-spline fits are shared separately (Google Drive package with its own README and index):
- 6 experiments × 10 datasets × 5 fit seeds;
- including the averaging seeds 1001–1004.

Regenerate this folder with `validation/morphomnist/scripts/exp_ate_recovery/make_examples.py`.
