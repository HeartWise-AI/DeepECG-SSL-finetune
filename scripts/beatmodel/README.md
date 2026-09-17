# Beat-level explainability for DeepECG-ACS

Implements the beat-model protocol of Obermeyer/Schubert et al., *Nature* 2026
(`s41586-026-10674-6`), and extends it with a class-conditional generator.

Territories throughout are `LEFT` (LAD or Left Main), `RCA`, `LCX`, matching the softmax
culprit head in `scripts/train_acs_3head_softmax.py`.

## Pipeline

Run in order. Each step writes under `data/acs_final/`.

| # | Script | What it does |
|---|--------|--------------|
| 1 | `build_beats_nk.py` | Segments Database A records into single beats with **NeuroKit2** (the "standard methods" the paper cites). Detection on cleaned lead II, fixed −0.28/+0.72 s window sliced from the raw 12 leads. → `beat_cache_nk/` |
| 2 | `train_beat_classifier.py` | Retrains the 3-head classifier on single beats, same architecture and recipe as the 10-second model, as the paper requires. → `checkpoints/beat_3head_nk/` |
| 3 | `train_beat_drvae.py` | 512-dim **discriminatively regularised** VAE; the DR term is the authors' released loss, forcing reconstructions to carry the same predicted risk as the real beat. → `checkpoints/beat_drvae/` |
| 4 | `morph_territory.py` | The counterfactual morph: latent gradient ascent on occlusion risk plus the one-vs-rest territory margin, stopping at the 90th percentile of real occluded beats. → `explainability/morphs.npz` |
| 5 | `plot_morphs.py` | 12-lead ACO-negative vs ACO-positive panels, and the lead-by-lead validation against real ischemia. |
| 6 | `beat_diffusion.py` / `train_beat_diffusion.py` | Class-conditional diffusion model over beats: `P(beat \| territory)`. → `checkpoints/beat_diffusion/` |
| 7 | `sample_conditional.py` | Samples per territory, audits them with the frozen classifier, checks memorisation and amplitude, and sweeps classifier-free guidance. |
| 8 | `plot_conditional.py` | Sampled ACO-positive panels per territory, and sampling vs morphing vs real ischemia. |

## Start here

`HANDOVER.md` — what exists, what is settled and should not be re-derived, the traps that fail
silently, and what is still open. Read it before running anything.

## How to use these models

`USAGE.md` in this directory: loading each checkpoint, generating an occluded beat for a
territory, reconstructing or embedding a real beat, scoring, the sanity checks to run after
moving the weights, and a step-by-step guide to retraining all three on a new cohort. Every
snippet there was executed against the shipped checkpoints before being written down.

## Result in one line

Conditional sampling reproduces real ischemic morphology (cosine +0.99 / +0.99 / +0.94 for
LEFT / RCA / LCx); gradient morphing of the same model does not (+0.17 / +0.32 / +0.34) and
overshoots to roughly twice the occlusion probability of real occluded beats. Use the morph to
audit where a classifier's boundary lies; use the generator to show what the disease looks like.

## Three traps recorded here so they are not rediscovered

1. **Input gradients are silently zero.** The WCR backbone runs its conv feature extractor under
   `torch.no_grad()` whenever `feature_grad_mult == 0`
   (`fairseq_signals/models/ecg_transformer.py`, `get_embeddings`). Any saliency or morphing
   method returns nothing. Set `model.base.encoder.feature_grad_mult = 1.0`; freeze weights
   separately with `requires_grad_(False)`.
2. **Do not filter on NeuroKit's quality verdict.** `ecg_quality(method="zhao2018")` rejects
   acutely occluded ECGs far more often than normals — 39% of single-territory occlusion beats
   were lost when it was used as a filter. It is stored as a `quality` column instead.
3. **The cosine noise schedule needs clipping.** `alpha_bar` reaches ~1e-15 at the terminal
   step and `predict_x0` divides by its square root, amplifying any epsilon error by ~1e7.
   `beat_diffusion.py` clips to `ALPHA_MIN = 1e-4` and starts sampling at `t_max`.

## Extra dependencies

See `requirements.txt`. `neurokit2` is required by step 1 only. Steps 4 and the analysis scripts
in `scripts/explain_acs_softmax.py` import beat-segmentation helpers from the sibling
**DeepECG-PNG** checkout at `/volume/DeepECG-PNG/scripts/explainability/beatmodel/`; adjust that
path or vendor the module if you run this elsewhere.
