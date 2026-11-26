# The model, as implemented

Everything below describes `submission/diffusion.py` and `submission/unet.py`, which is what the
marker runs. Numbers are from `results/ab/mixed60k/config.json` and
`results/ab/eval_mixed60k_n1000.json`.

## 1. Diffusion

A denoising diffusion probabilistic model with a cosine noise schedule. Images live in `[-1, 1]`
after the standard rescale. For a training image `x0` and noise `eps ~ N(0, I)`:

```
q(x_t | x0) = N( sqrt(abar_t) x0 , (1 - abar_t) I )

abar_t = cos^2( ((t/T + s) / (1 + s)) * pi/2 ) / cos^2( (s / (1 + s)) * pi/2 )
```

with `T = 1000` and `s = 0.008` (`cosine_abar`, `diffusion.py`). The cosine schedule keeps more
signal at small `t` than the linear one and is the reason this stack trains at 32×32 without a
learned variance.

The network predicts the noise under the plain unweighted objective:

```
L = E_{t, x0, eps, y} || eps - eps_theta( sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, t, y ) ||^2
```

A Min-SNR weighted variant of the *same* objective is implemented behind `snr_gamma`
(`w(t) = min(SNR(t), gamma) / SNR(t)`); `snr_gamma = 0` reproduces the plain loss bit-for-bit.
The shipped checkpoint uses the plain loss, because weighting it was measured and lost (see
`docs/EXPERIMENTS.md`, row 4).

## 2. Network

A UNet with base width 64 and a 128-dimensional conditioning path: two downsamplings take 32×32 to
an 8×8 bottleneck, with one self-attention layer at 16×16 and skip connections back to full
resolution. Every normalisation layer is GroupNorm (8 groups). BatchNorm would make a sample depend
on the other samples in its batch, which the interface forbids, and GroupNorm behaves identically in
train and eval mode.

The step index enters through a sinusoidal embedding and a two-layer MLP; the class through a
two-row embedding table. Both are summed into one vector `e`:

```
e = time_mlp( sinusoidal(t) ) + class_emb(y)
h <- conv(h) + proj_block(e)         # a per-channel bias, added after the block's first convolution
```

`proj_block` is a linear map from the 128-dimensional `e` to that block's channel count — the
per-block bias of Dhariwal & Nichol. Total: **5,145,731 parameters**, 19.74 MB on disk.

## 3. Sampling

Deterministic DDIM with `eta = 0`, 50 steps. Each step predicts `x0` and re-noises it to the next
level:

```
x0_hat = ( x_t - sqrt(1 - abar_t) eps_theta ) / sqrt(abar_t)
x_{t-1} = sqrt(abar_{t-1}) x0_hat + sqrt(1 - abar_{t-1}) eps_theta
```

The 50-step grid is `linspace(999, 0, 51).round()`, so the same weights always take the same
trajectory given the same starting noise.

Guidance uses **the other class as the negative**, because the shipped checkpoint was trained
without label dropout and therefore has no null token:

```
eps_tilde = eps_theta(x_t, t, 1-y) + w * [ eps_theta(x_t, t, y) - eps_theta(x_t, t, 1-y) ],  w = 2.5
```

Both terms are computed per sample, so guidance does not couple samples and does not break batch
independence. The weight was selected with the official evaluator at 250 images per class and
confirmed at 1,000; above 2.5 FID and recall turn back down. A second reason to stop there:
guidance multiplies the per-step numerical drift by `(2w - 1)`, so at `w = 4.5` the evaluator's
repeat check fails outright and the run returns 0/15 rather than a lower score.

## 4. Determinism and the batch contract

Sample `i` gets its initial noise from a CPU generator seeded with

```
(seed * 1,000,003 + i) mod 2^63 - 1
```

and nothing else in the loop is random. An image therefore depends on its own label and the seed
only, and a CPU run matches a GPU run. The evaluator's own checks agree (`checks` in
`eval_mixed60k_n1000.json`):

| Check | Result |
|---|---|
| `deterministic` | true — same seed, identical tensors |
| `seed_sensitive` | true — different seed, different images |
| batch composition | 0.9600 accuracy on single-class batches vs 0.8700 on mixed, a 0.09 drop against the 0.20 limit |
| `violations` | `[]` |

The batch probe is the important one: a sampler that drew its noise from the batch (or normalised
across it) would collapse when the batch changed shape, and this one does not.

## 5. Training

| Setting | Value |
|---|---|
| Steps × batch | 60,000 × 64 |
| Passes over 3,000 rows | 1,280 by sample count (the loader's own counter reads 1,305, because `drop_last` drops 56 rows a pass) |
| Optimiser | Adam, constant 2×10⁻⁴, no warmup |
| EMA | 0.999, weights shipped are the EMA parameters at step 60,000 |
| Seed | 0 |
| Loss | 1.1493 at the first step, 0.0583 at the last |
| Wall clock | 5,507 s on one RTX 2060, 0.0918 s/step |

Step count is tied to **passes**, not to steps alone: the 1,000-row recipe was tuned at 1,280 passes,
so 3,000 rows needs 60,000 steps to see the same amount of training. The same rows at 20,000 steps
is 427 passes and scores 11/15 — that arm is in `docs/EXPERIMENTS.md` because the arithmetic is the
reusable part.

## 6. Interface

`submission/generate.py` implements the two required functions:

- `load_model(device)` returns one `nn.Module` in `.eval()` holding every generation parameter, and
  refuses anything it cannot load rather than silently falling back.
- `generate(model, labels, seed)` returns `[N, 3, 32, 32]` float in `[0, 1]`, no NaN, dispatching
  on the checkpoint kind (`kind="ddpm"` for the shipped one).

`submission/` imports only `torch` and its own siblings — no network access, no downloads, no
`classifier.pt`, no Inception, and no pretrained weights anywhere in the generator.

## References

1. Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
2. Nichol & Dhariwal. Improved Denoising Diffusion Probabilistic Models. ICML 2021.
3. Song, Meng & Ermon. Denoising Diffusion Implicit Models. ICLR 2021.
4. Ho & Salimans. Classifier-Free Diffusion Guidance. arXiv:2207.12598, 2022.
5. Hang et al. Efficient Diffusion Training via Min-SNR Weighting Strategy. ICCV 2023.
6. Dhariwal & Nichol. Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021 (per-block
   conditioning bias, GroupNorm at 32×32).
