#!/usr/bin/env python3
"""End-to-end smoke test for the WCR v2 multi-task stack.

Builds a tiny synthetic dataset in the same on-disk format as
``build_wcrv2_multitask_dataset.py`` (NaN labels, NaN aux), fine-tunes the real
WCR v2 encoder for a handful of updates with ``masked_bce`` and
``ecg_transformer_aux_classifier``, then runs ``fairseq-hydra-inference`` and
checks the saved logits. Runs in ~2 minutes on one GPU.

    PYTHONPATH=/volume/DeepECG-SSL-finetune /volume/venvs/fss39/bin/python \
        scripts/tests/smoke_wcrv2_multitask.py --device 0
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "examples/w2v_cmsc/config/finetuning/ecg_transformer"
DEFAULT_ENCODER = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt"
SKIP = 77  # exit code when prerequisites (encoder checkpoint, CUDA) are missing
K, AUXD, T, L = 16, 2, 2500, 12


def make_split(root, name, n, rng):
    x = (rng.standard_normal((n, T, L, 1)) * 0.3).astype(np.float32)  # ~mV-scale noise
    y = rng.integers(0, 2, size=(n, K)).astype(np.float32)
    y[rng.random((n, K)) < 0.4] = np.nan                               # 40 % missing labels
    y[: n // 3, 14:16] = np.nan                                        # "EchoNext-like" rows: no AF labels
    aux = rng.standard_normal((n, AUXD)).astype(np.float32)
    aux[:, 1] = (aux[:, 1] > 0).astype(np.float32)
    aux[rng.random(n) < 0.5, 0] = np.nan                               # half without age
    np.save(root / f"{name}_x.npy", x)
    np.save(root / f"{name}_y.npy", y)
    np.save(root / f"{name}_aux.npy", aux)
    with open(root / f"{name}.tsv", "w") as f:
        f.write(f"x_path:{root / f'{name}_x.npy'}\n")
        f.write(f"x_shape:({T}, {L}, 1)\n")
        f.write(f"y_path:{root / f'{name}_y.npy'}\n")
        f.write(f"aux_path:{root / f'{name}_aux.npy'}\n")
        f.write(f"label_indexes:{list(range(K))}\n")
    return y


def check_masked_loss():
    """NaN targets must contribute nothing: changing logits at missing positions must not change the loss."""
    import torch
    from types import SimpleNamespace
    from fairseq_signals.criterions.masked_binary_cross_entropy_with_logits import (
        MaskedBinaryCrossEntropyWithLogitsCriterion as Crit,
    )

    class Echo(torch.nn.Module):
        def forward(self, **k):
            return {"out": k["logits"]}

        def get_logits(self, o):
            return o["out"]

        def get_targets(self, smp, o):
            return smp["label"]

    cfg = SimpleNamespace(threshold=0.5, report_auc=False, pos_weight=None, label_weights=None,
                          label_names=None, log_per_label=True, sample_size_mode="positives")
    crit = Crit(cfg, task=None)
    nan = float("nan")
    t = torch.tensor([[1, 0, nan], [0, 1, 1], [nan, nan, nan], [1, 1, 0]])
    lg = torch.randn(4, 3)
    lg2 = lg.clone()
    lg2[torch.isnan(t)] = 50.0
    l1, _, _ = crit(Echo(), {"net_input": {"logits": lg}, "label": t, "id": torch.arange(4)})
    l2, _, _ = crit(Echo(), {"net_input": {"logits": lg2}, "label": t, "id": torch.arange(4)})
    assert torch.isfinite(l1) and abs(float(l1) - float(l2)) < 1e-6, (float(l1), float(l2))
    print("[ok] masked_bce ignores missing labels")


def check_aux_collated(work):
    """The NPY dataset must load aux and the collator must place it in net_input."""
    from fairseq_signals.data import NpECGDataset

    ds = NpECGDataset(manifest_path=str(work / "val.tsv"), sample_rate=None, label=True, pad=True)
    batch = ds.collator([ds[i] for i in range(4)])
    aux = batch["net_input"].get("aux")
    assert aux is not None and tuple(aux.shape) == (4, AUXD), "aux missing from net_input"
    print("[ok] dataset + collator deliver aux to net_input")


def check_aux_used(model_path, device):
    """A trained checkpoint's output must depend on the aux inputs (sex flipped, age shifted)."""
    import torch
    from fairseq_signals import tasks
    from fairseq_signals.utils import checkpoint_utils

    state = checkpoint_utils.load_checkpoint_to_cpu(str(model_path))
    task = tasks.setup_task(state["cfg"]["task"])
    model = task.build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"], strict=True)
    dev = torch.device("cuda", 0) if device != "cpu" else torch.device("cpu")
    model = model.to(dev).eval()
    x = torch.randn(4, L, T, device=dev) * 0.3
    a = torch.tensor([[0.0, 0.0], [0.5, 1.0], [-1.0, 0.0], [1.5, 1.0]], device=dev)
    b = a.clone()
    b[:, 0] += 2.0
    b[:, 1] = 1.0 - b[:, 1]
    with torch.no_grad():
        oa = model(source=x, padding_mask=None, aux=a)["out"]
        ob = model(source=x, padding_mask=None, aux=b)["out"]
    delta = float((oa - ob).abs().max())
    assert delta > 1e-5, f"output does not depend on age/sex (max delta {delta})"
    print(f"[ok] model output depends on age/sex (max |delta logit| {delta:.4f})")


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(p.stdout.splitlines()[-40:])
    if p.returncode != 0:
        print(tail)
        raise SystemExit(f"command failed with code {p.returncode}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--encoder", default=os.environ.get("WCRV2_ENCODER", DEFAULT_ENCODER),
                    help="WCR v2 SSL encoder checkpoint (env WCRV2_ENCODER)")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    check_masked_loss()

    import torch
    if not Path(args.encoder).exists():
        print(f"[skip] encoder checkpoint not found: {args.encoder} (pass --encoder or set WCRV2_ENCODER)")
        sys.exit(SKIP)
    if not torch.cuda.is_available():
        print("[skip] CUDA is not available; the end-to-end part of this test trains on a GPU")
        sys.exit(SKIP)

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="wcrv2_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    make_split(work, "train", args.n, rng)
    y_val = make_split(work, "val", args.n // 2, rng)
    check_aux_collated(work)
    ckpt = work / "ckpt"
    results = work / "results"
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=args.device, WANDB_MODE="disabled",
               HYDRA_FULL_ERROR="1")
    py = sys.executable

    results.mkdir(parents=True, exist_ok=True)  # hydra_inference expects results_path to exist
    train_log = run([py, "-m", "fairseq_cli.hydra_train",
         "--config-dir", str(CFG_DIR), "--config-name", "diagnosis_wcrv2_multitask",
         f"task.data={work}", f"checkpoint.save_dir={ckpt}", f"model.model_path={args.encoder}",
         "dataset.batch_size=16", "dataset.num_workers=0", f"optimization.max_update={args.updates}",
         "optimization.max_epoch=1", "common.log_interval=1", "checkpoint.save_interval_updates=0",
         "common.wandb_project=null"], env)
    best = ckpt / "checkpoint_best.pt"
    last = ckpt / "checkpoint_last.pt"
    assert last.exists(), f"no checkpoint written in {ckpt}"
    model_path = best if best.exists() else last
    print(f"[ok] training wrote {model_path}")
    valid_lines = [l for l in train_log.splitlines() if '"val_auroc"' in l]
    if valid_lines:
        print("[valid]", valid_lines[-1][:600])
    check_aux_used(model_path, args.device)

    run([py, "-m", "fairseq_cli.hydra_inference",
         "--config-dir", str(CFG_DIR), "--config-name", "eval",
         f"task.data={work}", f"common_eval.path={model_path}", f"common_eval.results_path={results}",
         "task.npy_dataset=true", f"model.num_labels={K}", "dataset.valid_subset=val",
         "dataset.batch_size=16", "dataset.num_workers=0", "common.wandb_project=null"], env)
    out = results / "outputs_val.npy"
    hdr = results / "outputs_val_header.pkl"
    assert out.exists() and hdr.exists(), f"inference did not write {out}"
    with open(hdr, "rb") as f:
        h = pickle.load(f)
    logits = np.memmap(out, dtype=h["dtype"], mode="r", shape=tuple(h["shape"]))
    assert logits.shape == (len(y_val), K), (logits.shape, (len(y_val), K))
    assert np.isfinite(np.asarray(logits, dtype=np.float32)).all(), "non-finite logits"
    print(f"[ok] inference logits {logits.shape}, finite, mean {float(np.asarray(logits, dtype=np.float32).mean()):.4f}")
    print(json.dumps({"workdir": str(work), "checkpoint": str(model_path), "outputs": str(out)}, indent=2))


if __name__ == "__main__":
    main()
