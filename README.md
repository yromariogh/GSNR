# GSNR: Graph-Smooth Null-Space Representation

This repository contains code for learning and using graph-structured null-space representations for linear imaging inverse problems. The current codebase is focused on two tasks:

- super-resolution (`sr`)
- demosaicing (`demosaicing`, with `demo` accepted as an alias in some scripts)

The core idea is to build a low-dimensional null-space basis `S` from graph-smooth modes of the sensing operator, train a predictor `G(y) ~= Sx`, and use that prediction inside downstream reconstruction solvers such as Plug-and-Play (PnP), Deep Image Prior (DIP), and DiffPIR.


## Repository layout

- [comput_gnsr_matrix.py](./comput_gnsr_matrix.py): builds graph-constrained null-space projection matrices `S` for SR or demosaicing.
- [Compute_coverage.py](./Compute_coverage.py): computes coverage curves for generated `S` matrices.
- [predict_sx.py](./predict_sx.py): trains the predictor network `G(y)` that estimates `Sx` from measurements.
- [pnp_gsnr.py](./pnp_gsnr.py): PnP reconstruction with or without GSNR regularization.
- [test_gsnr_sr.py](./test_gsnr_sr.py): batch evaluation script for the learned GSNR pipeline on SR or demosaicing.
- [deep_image_prior.py](./deep_image_prior.py): DIP-based reconstruction variant. Currently SR-oriented.
- [diffpir_dinv.py](./diffpir_dinv.py): DiffPIR-based reconstruction variant. Currently SR-oriented.
- [load_data.py](./load_data.py): local dataset loading and `test_data` routing.
- [other_models.py](./other_models.py): predictor backbones, including `UNetLeon`.
- [algos_dinv.py](./algos_dinv.py): custom optimization operators used by PnP-style solvers.
- [laplacians.py](./laplacians.py): graph Laplacian builders and related operators.
- [utils.py](./utils.py): path helpers for `RESULTS`, `S` matrices, demosaicing `H`, and model checkpoints.
- [utils_sr.py](./utils_sr.py), [utils_coverage.py](./utils_coverage.py): task utilities, metrics, coverage helpers.
- [deepinv](./deepinv): vendored local copy of `deepinv`.
- [RESULTS](./RESULTS): precomputed graph matrices, `H` matrices, trained models, and outputs.
- [test_data](./test_data): lightweight evaluation subsets used by the PnP, DIP, and DiffPIR scripts.

## Requirements

There is no pinned environment file in the repo at the moment. The main scripts assume a Python environment with at least:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`
- `tqdm`
- `torchmetrics`
- `einops`
- `scipy`

Optional:

- `wandb` for logging in [predict_sx.py](./predict_sx.py)

The project ships its own local [deepinv](./deepinv) package, so no separate `pip install deepinv` is required for this repo layout.

## Data layout

There are two different data paths in the current code:

1. Full datasets for training and large-scale experiments are hardcoded in [load_data.py](./load_data.py) under `DATASET_DIRS`.
2. Small evaluation subsets live under [test_data](./test_data) and are used by default in the PnP, DIP, and DiffPIR evaluation scripts through `--use_test_data`.

Supported datasets in [load_data.py](./load_data.py):

- `celeba`
- `div2k`
- `places`
- `sar`

Current lightweight test subsets available in [test_data](./test_data):

- `celeba`
- `places`
- `sar`

If you are running this repo on a new machine, you will likely need to update the hardcoded dataset roots in [load_data.py](./load_data.py).

## Results layout

The repo uses a stable output structure under [RESULTS](./RESULTS):

- `RESULTS/GRAPH_MATRICES`: saved `S` matrices and coverage curves.
- `RESULTS/H_matrices`: sensing matrices for demosaicing.
- `RESULTS/MODELS`: trained predictor checkpoints `G`.
- `RESULTS/PNP`: reconstruction metrics and figures from solver runs.

Important filenames are built through [utils.py](./utils.py):

- SR graph matrices: `S_full_<variant>_SR_<srf>_H_<n>.pt`
- Demosaicing graph matrices: `S_full_<variant>_demo_H_<n>.pt`
- SR models: `<dataset>_SR_<srf>_<variant>_p_<p>_lr_<lr>/model.pt`
- Demosaicing models: `<dataset>_demo_<variant>_p_<p>_lr_<lr>/model.pt`

## End-to-end workflow

### 1. Generate the graph-constrained null-space basis `S`

Super-resolution:

```bash
python comput_gnsr_matrix.py --task sr --n 128 --srf 4 --train_L_variant grid_4nn
```

Demosaicing:

```bash
python comput_gnsr_matrix.py --task demosaicing --n 64 --train_L_variant grid_4nn --channels 3
```

This saves matrices into `RESULTS/GRAPH_MATRICES`.

### 2. Compute coverage curves

Super-resolution:

```bash
python Compute_coverage.py --task sr --n 64 --srf 4
```

Demosaicing:

```bash
python Compute_coverage.py --task demosaicing --n 64 --channels 3
```

This produces `coverage_curves_*.npz` and summary plots in `RESULTS/GRAPH_MATRICES`.

Note: the coverage code currently loads its reference images through [utils_coverage.py](./utils_coverage.py), which uses CIFAR-10 splits rather than [load_data.py](./load_data.py).

### 3. Train the predictor `G(y) -> Sx`

Super-resolution example:

```bash
python predict_sx.py --task sr --dataset celeba --variant grid_4nn --p 0.1 --n 128 --srf 4
```

Demosaicing example:

```bash
python predict_sx.py --task demosaicing --dataset places --variant grid_4nn --p 0.7 --n 64
```

This trains the `UNetLeon` predictor and saves checkpoints under `RESULTS/MODELS`.

### 4. Run reconstruction

PnP with GSNR, super-resolution:

```bash
python pnp_gsnr.py --task sr --dataset celeba --variant grid_4nn --p 0.1 --denoiser dncnn --n 128 --srf 4
```

PnP with GSNR, demosaicing:

```bash
python pnp_gsnr.py --task demosaicing --dataset places --variant grid_4nn --p 0.5 --denoiser dncnn --n 64
```

Batch evaluation:

```bash
python test_gsnr_sr.py --task sr --dataset celeba --variant grid_4nn --p 0.1 --denoiser dncnn
python test_gsnr_sr.py --task demosaicing --dataset celeba --variant grid_4nn --p 0.5 --denoiser dncnn
```

Additional SR-oriented solvers:

```bash
python deep_image_prior.py --dataset celeba --variant grid_8nn --p 0.1 --use_test_data
python diffpir_dinv.py --dataset celeba --variant identity --p 0.1 --use_test_data
```

## Mathematical overview

### Inverse problem

The starting point is the linear inverse model

$$
y = Hx^\ast + \omega, \qquad \omega \sim \mathcal{N}(0, \sigma^2 I),
$$

where `x*` is the unknown image, `H` is the sensing operator, and `y` is the observation.

A standard variational formulation is

$$
\hat{x} = \arg\min_{\tilde{x}} \frac{1}{2}\|H\tilde{x} - y\|_2^2 + \eta f(\tilde{x}),
$$

with `f` representing an image prior.

### Null-space decomposition

Because `H` is typically ill-posed or rank-deficient, the reconstruction is ambiguous along `Null(H)`. The image can be decomposed into

$$
x = x_r + x_n,
$$

with

$$
x_r = P_r x, \qquad x_n = P_n x, \qquad P_n = I - H^\dagger H.
$$

The null component `x_n` is invisible to the measurements because `Hx_n = 0`.

### Graph-smooth null-space operator

GSNR introduces structure only on the invisible component. Given a graph Laplacian `L`, the method defines the null-restricted Laplacian

$$
T = P_n L P_n.
$$

Where `L` is an image-grid Laplacian, typically `4nn` or `8nn`, so smooth graph modes correspond to spatially coherent null-space directions.

If

$$
T = V \operatorname{diag}(\mu_1,\dots,\mu_n)V^\top
$$

is the eigendecomposition of `T`, the GSNR projection matrix is formed from the `p` smoothest null modes:

$$
S = V_p^\top \in \mathbb{R}^{p \times n}.
$$

This means `Sx` extracts the coefficients of the null component along the smoothest graph-aligned directions.

### Learning the predictor

The predictor `G` is trained to estimate those coefficients directly from the measurements:

$$
G^\ast = \arg\min_G \mathbb{E}\left[\|G(y) - Sx^\ast\|_2^2\right].
$$

In the current code, [predict_sx.py](./predict_sx.py) uses `UNetLeon` from [other_models.py](./other_models.py) as the predictor backbone.

### Reconstruction with GSNR

At inference time, GSNR augments the base inverse-problem objective with two null-space-aware terms:

$$
\min_{\tilde{x}} \; g(\tilde{x}) + \eta f(\tilde{x})
+ \gamma \|G^\ast(y) - S\tilde{x}\|_2^2
+ \frac{\gamma_g}{2}\tilde{x}^\top T \tilde{x}.
$$

Interpretation:

- `g(tilde{x})` is the measurement fidelity.
- `f(tilde{x})` is the image prior, often implemented implicitly through a denoiser.
- `||G*(y) - S tilde{x}||^2` forces the reconstruction to match the predicted null-space coefficients.
- `tilde{x}^T T tilde{x}` is an optional null-only graph regularizer.

This is the main mechanism used by the PnP code in [pnp_gsnr.py](./pnp_gsnr.py), and conceptually reused by the DIP and DiffPIR variants.

### Coverage and predictability

Two ideas drive the method:

- Coverage: how much null-space variability is captured by the first `p` rows of `S`.
- Predictability: how accurately those projected null components can be inferred from `y`.

The repo includes explicit coverage computation in [Compute_coverage.py](./Compute_coverage.py), which is useful for selecting `p` and comparing graph designs such as `identity`, `grid_4nn`, `grid_8nn`, `sym`, and `rw`.

## Practical notes

- The unified scripts for both tasks are [comput_gnsr_matrix.py](./comput_gnsr_matrix.py), [Compute_coverage.py](./Compute_coverage.py), [predict_sx.py](./predict_sx.py), [pnp_gsnr.py](./pnp_gsnr.py), and [test_gsnr_sr.py](./test_gsnr_sr.py).
- [deep_image_prior.py](./deep_image_prior.py) and [diffpir_dinv.py](./diffpir_dinv.py) are still written around the SR setup.
- Some training data paths are machine-specific Windows paths in [load_data.py](./load_data.py). Those should be updated before running on another system.
- The repo already contains precomputed matrices and checkpoints under [RESULTS](./RESULTS), so you do not need to regenerate everything from scratch if those artifacts match your experiment.



## Citation

If you use this repository, cite:

```bibtex
@article{gualdron2026gsnr,
  title={GSNR: Graph Smooth Null-Space Representation for Inverse Problems},
  author={Gualdr{\'o}n-Hurtado, Romario and Jacome, Roman and Suarez, Rafael S and Arguello, Henry},
  journal={arXiv preprint arXiv:2602.20328},
  year={2026}
}
```
