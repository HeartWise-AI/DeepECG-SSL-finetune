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
ENCODER = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt"
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
    args = ap.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="wcrv2_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    make_split(work, "train", args.n, rng)
    y_val = make_split(work, "val", args.n // 2, rng)
    ckpt = work / "ckpt"
    results = work / "results"
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=args.device, WANDB_MODE="disabled",
               HYDRA_FULL_ERROR="1")
    py = sys.executable

    results.mkdir(parents=True, exist_ok=True)  # hydra_inference expects results_path to exist
    train_log = run([py, "-m", "fairseq_cli.hydra_train",
         "--config-dir", str(CFG_DIR), "--config-name", "diagnosis_wcrv2_multitask",
         f"task.data={work}", f"checkpoint.save_dir={ckpt}", f"model.model_path={ENCODER}",
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