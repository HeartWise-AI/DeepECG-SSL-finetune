# Using the beat models

Three checkpoints, three jobs: **generate** an occluded beat for a named territory,
**reconstruct** or embed a real beat, and **score** any beat. Every snippet below was executed
against the shipped checkpoints before being written down; the printed values are what it
actually returned.

Set `CKPT_DIR` to wherever the checkpoints live — `data/acs_final/checkpoints` in the training
container, or the NAS directory you copied them to.

```python
import sys, numpy as np, torch
sys.path.insert(0, "scripts/beatmodel")
CKPT_DIR = "data/acs_final/checkpoints"
```

Beats are `(N, 12, 250)` float32 in **millivolts**, 250 Hz, lead order I, II, III, aVR, aVL, aVF,
V1–V6, windowed −0.28 s to +0.72 s around the R peak with the PR segment as baseline. That is
what every model here consumes and produces. To cut beats from a raw 10-second record, use
`build_beats_nk.py`.

---

## 1. Generate an occluded beat for a territory

```python
from beat_diffusion import BeatUNet, Schedule, ddim_sample

CLASSES = ["normal", "LEFT", "RCA", "LCX"]        # LEFT = LAD or Left Main

ck = torch.load(f"{CKPT_DIR}/beat_diffusion/beat_diffusion.pt",
                map_location="cpu", weights_only=False)
gen = BeatUNet(base=ck["base_width"]).cuda().eval()
gen.load_state_dict(ck["ema"])                    # EMA weights, NOT ck["model"]
mean = torch.from_numpy(ck["mean"]).cuda()
std  = torch.from_numpy(ck["std"]).cuda()
sched = Schedule(device="cuda")

y = torch.full((8,), CLASSES.index("RCA"), device="cuda")
x = ddim_sample(gen, sched, len(y), y, device="cuda", steps=100, guidance=0.0)
beats = (x * std + mean).cpu().numpy()            # (8, 12, 250) millivolts
```

Returns `(8, 12, 250)`, amplitude 99th percentile 0.61 mV.

- **`ck["ema"]`, not `ck["model"]`.** The raw weights are noticeably worse; the EMA copy is the
  one every reported result used.
- **Denormalise with the checkpoint's own `mean` and `std`.** They are per-lead statistics from
  the training beats and are stored alongside the weights. Skipping this gives you standardised
  units, not millivolts.
- **`guidance=0.0` is the default for a reason.** Classifier-free guidance sharpens morphology
  but pushes apparent severity past real occlusions. At 0 the samples sit at the same occlusion
  probability as real occluded beats; at 5 they read roughly twice as severe. Use guidance only
  if you want a caricature and say so.
- `steps=100` DDIM steps is the reported setting. Fewer is faster and slightly blurrier.
- Mixing classes in one batch is fine: pass a `y` with different values per row.

## 2. Reconstruct or embed a real beat

```python
from beat_vae import BeatVAE       # from DeepECG-PNG/scripts/explainability/beatmodel

vk = torch.load(f"{CKPT_DIR}/beat_drvae/beat_drvae_best.pt",
                map_location="cpu", weights_only=False)
vae = BeatVAE(z_dim=vk["z_dim"], width=vk["width"]).cuda().eval()
vae.load_state_dict(vk["model"])

real = np.load("data/acs_final/beat_cache_nk/test_beats.npy")[:8].astype("float32")
with torch.no_grad():
    xb = torch.from_numpy(real).cuda()            # millivolts, no scaling needed
    mu, logvar = vae.encode(xb)                   # mu: (N, 512) latent
    recon = vae.decode(mu).cpu().numpy()
```

Returns a 512-dim latent and reconstructions at **correlation 0.997** against the input.

- **Feed millivolts directly.** `BeatVAE` applies its own internal scaling; do not pre-normalise.
- Use `mu` for a deterministic embedding. Sampling `mu + eps * exp(0.5*logvar)` gives you the
  stochastic version, which is what training used.
- The decoder is the one to reach for when you want to move a beat in latent space; see
  `morph_territory.py` for the counterfactual loop, and read the warning in the project README
  before trusting what a gradient morph draws.

## 3. Score any beat

```python
from sample_conditional import load_classifier, score

clf = load_classifier()
p_acco, p_terr = score(clf, beats)                # beats: (N, 12, 250) millivolts
```

On the generated RCA beats above: `P(occlusion)` 0.43, and the territory head calls RCA in 88%
of them. `p_terr` columns are `[LEFT, RCA, LCX]`.

- `load_classifier()` needs a fairseq-signals backbone checkpoint as an architecture donor; the
  path is set at the top of `sample_conditional.py`.
- The territory head is **conditional on an occlusion being present**. On a beat with no
  occlusion its output is meaningless, which is what the ACCO gate exists to enforce. Do not read
  it as "which artery has a lesion".
- **If you need gradients with respect to the input** — saliency, morphing, anything — set
  `clf.base.encoder.feature_grad_mult = 1.0` first. The backbone runs its conv front end under
  `torch.no_grad()` when that value is 0, and returns silently zero gradients otherwise.

---

## Sanity checks worth running after a move

```python
# generated beats should read as occlusions of the intended territory
y = torch.arange(4, device="cuda").repeat_interleave(64)
x = ddim_sample(gen, sched, len(y), y, device="cuda", steps=100, guidance=0.0)
p_acco, p_terr = score(clf, ((x * std + mean)).cpu().numpy())
```

Expected, and what the shipped checkpoints give: `P(occlusion)` around 0.02 for the normal class
and 0.42 / 0.42 / 0.37 for LEFT / RCA / LCx, against 0.41 / 0.47 / 0.36 for real held-out
occluded beats. Territory called correctly in about 83% of samples. If the normal class comes out
high, or the territories land far from those values, the most likely cause is a denormalisation
mistake or loading `ck["model"]` instead of `ck["ema"]`.

## What these models are not for

The generator produces single beats, not 10-second strips, and it is trained on one cohort from
one centre. It is an instrument for explaining and stress-testing the classifier, for building
reader-study material, and for augmenting a rare class. No cardiologist has yet reviewed its
output, so a generated beat is not evidence about a patient and should not be presented as a
real recording.

---

# Retraining on a new cohort

All three models retrain from the same beat cache, so build that once and the rest follow. Wall
times below are what this cohort took on one H200 shared with other jobs: about 95 minutes end to
end for all three.

## Step 0 — point the scripts at your data

Two module-level constants near the top of `build_beats_nk.py`:

| Constant | What it needs |
|---|---|
| `SP` | A labels CSV, one row per record, with columns `Split` (`train`/`val`/`test`), `ACS`, `ACCO`, the territory flags, and `new_PatientID` |
| `BASE` | A directory holding `arrays/X_{split}.npy`, each `(n_records, 2500, 12)` in raw ADC counts, row-aligned with the CSV rows for that split |

Row alignment is the one thing that fails silently. The array for a split must be in exactly the
order the CSV yields after filtering and `reset_index`, or every label attaches to the wrong beat.

Different territories or a different number of classes means editing `CLASSES` in each script,
and `N_CLASSES` in `beat_diffusion.py`. Keep the convention that class 0 is the negative class:
the generator's null token for classifier-free guidance is `N_CLASSES`, so an off-by-one there
silently trains guidance against a real class.

## Step 1 — beat cache (about 30 seconds)

```bash
PYTHONPATH=. python scripts/beatmodel/build_beats_nk.py
```

`MAX_PER_REC` (default 4) sets beats kept per record, `WORKERS` (24) the process pool. Expect
about 96% of records to yield beats; a much lower rate means the R-peak detector is unhappy with
your signal scaling, so check that ADC-to-millivolt factor first.

Do not filter on the `quality` column it writes. It records the NeuroKit verdict, which rejects
acutely occluded ECGs far more often than normals — 39% of single-territory occlusion beats were
lost when it was used as a filter on this cohort.

## Step 2 — beat classifier (about 10 minutes)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beatmodel/train_beat_classifier.py
```

`BEAT_CACHE`, `BEAT_OUT`, `BEAT_BATCH` (128), `BEAT_EPOCHS` (25). Early stops on
`0.5·occlusion AUROC + 0.5·territory macro` with patience 6; this cohort stopped at epoch 7 and
selected epoch 1, so do not be alarmed by an early best.

It also writes the risk percentiles the morph's stopping rule needs. Check them: if the territory
head's 90th percentile comes back at 0.999 as it did here, the head is saturated and any
percentile-based stopping rule on it is meaningless. Label smoothing is the fix if you care.

## Step 3 — DR-VAE (about 35 minutes)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 VAE_EPOCHS=30 python scripts/beatmodel/train_beat_drvae.py
```

`VAE_EPOCHS` (60), `VAE_BATCH` (128), `VAE_LR` (2e-4), `VAE_BETA_KL` (5e-4), `VAE_WIDTH` (96),
`VAE_DR_TARGET` (0.2). The discriminative term needs the step-2 classifier and is auto-balanced
on the first batch to `VAE_DR_TARGET` of the reconstruction loss, so the weight transfers across
cohorts and unit conventions without retuning.

Watch per-beat reconstruction correlation, printed each epoch. It should pass 0.98. If it
plateaus near 0.89, your beats are misaligned rather than your architecture being too small —
that was the false conclusion on this project, and NeuroKit alignment took the same architecture
from 0.89 to 0.99.

## Step 4 — conditional generator (about 48 minutes)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 DIFF_EPOCHS=80 python scripts/beatmodel/train_beat_diffusion.py
```

`DIFF_EPOCHS` (120), `DIFF_BATCH` (256), `DIFF_LR` (2e-4), `DIFF_WIDTH` (96), `DIFF_SPE` (400
steps per epoch). Classes are drawn in equal proportion regardless of prevalence, so the model's
implicit class prior is uniform and must never be read as an epidemiological rate.

Every fifth epoch it samples and scores with the classifier, which is the metric to watch: on
this cohort the normal class fell below 0.05 and territory top-1 passed 0.9 by epoch 25, then
plateaued. Loss alone will keep drifting down long after the samples stop improving.

## Step 5 — verify before trusting it

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/beatmodel/sample_conditional.py
```

`N_PER` (256 per class), `GUIDANCES` (`0,1,2,3,5`), `DDIM_STEPS` (100). It reports, per guidance
setting, the occlusion probability the classifier assigns, territory top-1, the cosine of the
sampled morphology against real ischemia, amplitude percentiles, and a memorisation check. It
picks the guidance whose severity is closest to real occluded beats.

Three numbers decide whether the run is usable:

| Check | Healthy | Failure means |
|---|---|---|
| Cosine vs real ischemia | above 0.9 | samples do not carry the real morphology; suspect labels |
| Sampled vs real occlusion probability | within about 0.05 | severity is miscalibrated; drop guidance |
| Nearest-neighbour distance vs real-to-real | at or above | the generator is memorising training beats |

## Adapting to a different outcome

The pipeline is not specific to coronary occlusion. Any per-record label with a negative class
and a small number of mutually exclusive positive classes fits, as long as the morphology lives
within a single beat. Swap the label columns in step 0, rename `CLASSES`, and the rest runs
unchanged.

Where it will not transfer: outcomes carried by rhythm, by beat-to-beat variation, or by anything
spanning more than one beat. A single beat gave up nearly all the occlusion signal here, which is
why this works, but that is a property of ischemia rather than of the method. Check it on your
outcome before committing to a beat-level pipeline, by comparing a beat-level classifier against
your existing full-strip model as step 2 does.
