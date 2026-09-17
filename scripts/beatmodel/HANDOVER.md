# Handover — DeepECG-ACS culprit territory and beat models

Written for whoever picks this up next. Read this before running anything: several questions here
are settled and re-deriving them wastes a day, and several traps are invisible until they have
already corrupted a result.

Work done 13–17 September 2026. Cohort throughout is Database A, the post-exclusion v6-final ACS
cohort (no LBBB, paced, prior CABG, graft-only PCI or non-coronary).

---

## 1. What exists

| Thing | Where | State |
|---|---|---|
| Softmax culprit-territory model (candidate 2.2) | `data/acs_final/checkpoints/acs_3head_softmax/` | Candidate, **not** promoted to the NAS deployment path |
| Beat classifier | `data/acs_final/checkpoints/beat_3head_nk/` | Guides the morph, judges the generator |
| DR-VAE, 512-dim | `data/acs_final/checkpoints/beat_drvae/` | Reconstruction r = 0.99 |
| Conditional beat generator | `data/acs_final/checkpoints/beat_diffusion/` | The most novel artefact here |
| Beat cache, NeuroKit-aligned | `data/acs_final/beat_cache_nk/` | Rebuilds in ~30 s, do not archive it |
| Results and figures | `data/acs_final/explainability/` | Source data behind every reported number |

Code is on `main` in this repo (PR #3, commit `04e944b`; docs in PR #4). The image-model suite is
on `main` in DeepECG-PNG (PR #189, commit `85345bf`). The model card is the DeepECG-ACS page in
Notion, with sections 2.2 candidate, Explainability, Beat-level generative explainability, and
Shipped.

**`/volume` is local NVMe, not the NAS.** Everything above is container-local and dies with the
container. Only the older full-cohort deployed model is on the NAS. The Notion card still names
container `4aeeec59fd40`; the container as of this writing is `4370b27764c3`. Verify with
`hostname` rather than trusting either.

---

## 2. Settled — do not re-derive

**The softmax culprit head does not beat four sigmoids on localisation.** Paired on the same 297
held-out single-territory complete occlusions, in the same 3-class label space: macro AUROC 0.91
both, McNemar p = 0.86, top-1 difference −0.7 points (95% CI −4.4 to +3.0). It was adopted
because it is explicitly conditional on an occlusion, not because it is more accurate. Re-running
this comparison will give the same answer.

**A single beat carries nearly all the signal.** Beat-level classifier: occlusion AUROC 0.93,
territory macro 0.91, top-1 82.4%. The 10-second model: 0.94, 0.91, 79.8%. Rhythm and the rest of
the strip add very little for this outcome.

**The 0.89 autoencoder reconstruction ceiling was not architectural.** It was R-peak jitter from a
Pan-Tompkins-style detector. NeuroKit alignment took the same architecture to 0.99. The earlier
handover in DeepECG-PNG calls it an architectural ceiling; that is wrong.

**Conditional sampling recovers real ischemic morphology; gradient morphing does not.** Cosine
against real ischemia 0.99 / 0.99 / 0.94 for LEFT / RCA / LCx, against 0.17 / 0.32 / 0.34 for the
morph, which also overshoots to roughly twice the occlusion probability of real occluded beats.
Use the morph to audit where a classifier's boundary lies; use the generator to show what the
disease looks like. They are not interchangeable.

**The "image model is the sharper localiser" claim is dead.** It came from a second sigmoid
applied to the signal model's vessel head, whose outputs are already probabilities. Corrected, the
waveform model's territory specificity is 0.98 on RCA and 0.91 on LAD. Fixed in DeepECG-PNG and
that handover is updated, but anything downstream that assumed the image model localises better
should be revisited.

---

## 3. Traps

Each of these cost real time and none of them announce themselves.

1. **Input gradients are silently zero.** The WCR backbone runs its conv feature extractor under
   `torch.no_grad()` whenever `feature_grad_mult == 0`, which is the default in these checkpoints
   (`fairseq_signals/models/ecg_transformer.py`, `get_embeddings`). Saliency, morphing and any
   input-attribution method return nothing and raise no error. Set
   `clf.base.encoder.feature_grad_mult = 1.0`; freeze weights separately with
   `requires_grad_(False)`. **Any earlier gradient-based explainability on this backbone should be
   re-checked against this.**
2. **Never filter on NeuroKit's `ecg_quality(method="zhao2018")`.** It rejects acutely occluded
   ECGs far more often than normals: 39% of single-territory occlusion beats were lost when it was
   used as a filter. It is stored as a `quality` column for that reason.
3. **Load `ck["ema"]` from the generator, not `ck["model"]`.** Both are in the checkpoint. Every
   reported number used the EMA copy.
4. **Denormalise generator output** with the checkpoint's own per-lead `mean` and `std`, or you
   are looking at standardised units and will misjudge amplitudes.
5. **The territory softmax is saturated.** Median probability for the called territory on real
   occluded beats is 1.000, so percentile-based thresholds on it are meaningless. This is why the
   morph's stopping rule uses an explicit 0.5 threshold for the territory half. Label smoothing or
   temperature scaling is the fix if anyone needs calibrated territory probabilities.
6. **The cosine noise schedule needs clipping.** `alpha_bar` reaches ~1e-15 at the terminal step
   and `predict_x0` divides by its square root. `beat_diffusion.py` clips at 1e-4 and starts
   sampling at `t_max`; this was a documented bug in the earlier diffusion work.
7. **Row alignment between the labels CSV and the arrays fails silently** and mislabels every
   beat. Check it first whenever results look strange after a data change.

---

## 4. Open, roughly in priority order

1. **No cardiologist has reviewed the generated beats.** The classifier agreeing with them is not
   evidence they are real ischemic morphology. This is the single highest-value next step and the
   one the computational work cannot substitute for.
2. **No seed sweep on anything.** Every model here is one training run. The cosine values, the
   detection AUROC gains (0.88→0.91 and 0.91→0.94) and the generator's fidelity all rest on single
   runs. The detection gains in particular are confounded with batch size and stopping epoch and
   should not be claimed without this.
3. **LCx is not explained.** The model localises it at AUROC 0.81 while the interpretable ST/T
   morphology localises it at 0.54, so whatever it uses for LCx is not the mean ischemic pattern.
   The generator shows why the territory is hard — nearly silent on standard leads, appearing as
   reciprocal anterior depression — but not what the model keys on.
4. **The 2.2 candidate is not promoted.** The operating-point tables on the model card were fitted
   on the full-cohort model and need refitting before any promotion.
5. **No inference rule for excluded rhythms.** The cohort excludes LBBB, paced and post-CABG ECGs
   and nothing suppresses the output when one arrives at inference.
6. **PR #189 merged with automated review only.** ~8,400 lines of research code went into
   DeepECG-PNG's main; the methodology pass the Cursor review asked for was never done by a human.

---

## 5. How to work here

Read `USAGE.md` in this directory for loading, generating, reconstructing, scoring and retraining,
with verified snippets and measured wall times. Read `README.md` for the pipeline order.

Practical notes for an agent:

- **GPUs are shared and usually near-full.** Check free memory before launching; three long jobs
  from other projects run on this host. Never kill a process without confirming it is yours — I
  killed a colleague's training run this week by matching on a pattern instead of a PID I had
  launched.
- **Do not check out branches in `/volume/DeepECG-PNG`.** A multi-day training job runs from that
  working tree. Use `git worktree add --detach` in a scratch directory instead.
- **Verify before documenting.** Every snippet in `USAGE.md` was executed first. Two of the bugs
  in section 3 were found by smoke-testing a pipeline on four samples before the full run.
- **The `.gitignore` on this cohort's repos carries a governance rule**
  (`/reproducibility/acs/inference_pack/`). A merge from an older branch deleted it once. Do not
  let that rule disappear.
- Cohort CSV: `/media/data1/datasets/DeepECG/acs_v6_final_labels.csv`. Territories are `LEFT`
  (LAD or Left Main), `RCA`, `LCX`; Left Main is merged because Database A has 2 single-vessel
  Left Main ECGs in train and 1 in test.
