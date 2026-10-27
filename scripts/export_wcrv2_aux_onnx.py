#!/usr/bin/env python3
"""Export a WCR v2 auxiliary-input classifier to ONNX and verify parity.

The exported graph accepts a millivolt ECG plus raw age and sex inputs. Age
normalization is embedded in the graph using ``labels.json``. NaN demographic
values are preserved so the model's learned missingness indicators behave the
same way as in PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from scripts.infer_wcrv2_aux import load


EXPECTED_LABELS = ["incident_af_2y", "incident_af_5y"]


class WCRv2AuxOnnxWrapper(nn.Module):
    """Give the auxiliary classifier a deployment-friendly input contract."""

    def __init__(self, model: nn.Module, age_mean: float, age_std: float):
        super().__init__()
        self.model = model
        self.register_buffer("age_mean", torch.tensor(float(age_mean), dtype=torch.float32))
        self.register_buffer("age_std", torch.tensor(float(age_std), dtype=torch.float32))

    def forward(
        self,
        ecg_12lead_mv: torch.Tensor,
        age_years: torch.Tensor,
        sex_male: torch.Tensor,
    ) -> torch.Tensor:
        age_z = (age_years - self.age_mean) / self.age_std
        aux = torch.stack((age_z, sex_male), dim=1)
        return self.model(
            source=ecg_12lead_mv,
            padding_mask=None,
            aux=aux,
        )["out"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: MODEL_DIR/deepecg-ssl-v2-afib-2y-5y.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Export and validate the graph without PyTorch/ONNXRuntime comparisons.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_metadata(
    path: Path,
    *,
    labels: list[str],
    age_mean: float,
    age_std: float,
    checkpoint_name: str,
) -> None:
    model = onnx.load(str(path))
    metadata = {
        "model_name": "DeepECG-SSL v2 Incident AF 2y/5y",
        "source_checkpoint": checkpoint_name,
        "labels": json.dumps(labels),
        "output_semantics": "raw_logits",
        "waveform_units": "mV",
        "waveform_shape": "[batch,12,2500]",
        "age_mean": repr(float(age_mean)),
        "age_std": repr(float(age_std)),
        "sex_encoding": "1=male,0=female,NaN=unknown",
        "missing_demographics": "NaN; missingness indicators embedded in model",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def make_inputs(batch_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ecg = rng.normal(0.0, 0.25, size=(batch_size, 12, 2500)).astype(np.float32)
    age = rng.uniform(18.0, 95.0, size=batch_size).astype(np.float32)
    sex = rng.integers(0, 2, size=batch_size).astype(np.float32)
    if batch_size >= 2:
        age[-1] = np.nan
        sex[-1] = np.nan
    return ecg, age, sex


def verify_parity(
    wrapper: nn.Module,
    output_path: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    results: list[dict[str, Any]] = []
    for batch_size, seed in ((1, 1001), (3, 1003)):
        ecg, age, sex = make_inputs(batch_size, seed)
        with torch.no_grad():
            torch_logits = wrapper(
                torch.from_numpy(ecg),
                torch.from_numpy(age),
                torch.from_numpy(sex),
            ).cpu().numpy()
        ort_logits = session.run(
            ["logits"],
            {
                "ecg_12lead_mv": ecg,
                "age_years": age,
                "sex_male": sex,
            },
        )[0]
        np.testing.assert_allclose(torch_logits, ort_logits, atol=atol, rtol=rtol)
        delta = np.abs(torch_logits - ort_logits)
        results.append(
            {
                "batch_size": batch_size,
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
                "contains_missing_demographics": bool(np.isnan(age).any()),
            }
        )
    return {"atol": atol, "rtol": rtol, "cases": results}


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else model_dir / "deepecg-ssl-v2-afib-2y-5y.onnx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load(model_dir, device="cpu", ckpt_name=args.checkpoint)
    labels = list(bundle["labels"])
    if labels != EXPECTED_LABELS:
        raise ValueError(f"Unexpected label order: {labels}; expected {EXPECTED_LABELS}")
    if bundle["aux_dim"] != 2:
        raise ValueError(f"Expected aux_dim=2, got {bundle['aux_dim']}")

    age_mean = float(bundle["aux_norm"]["age_mean"])
    age_std = float(bundle["aux_norm"]["age_std"])
    wrapper = WCRv2AuxOnnxWrapper(bundle["model"].float().eval(), age_mean, age_std).eval()

    dummy_ecg, dummy_age, dummy_sex = make_inputs(batch_size=2, seed=77)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                torch.from_numpy(dummy_ecg),
                torch.from_numpy(dummy_age),
                torch.from_numpy(dummy_sex),
            ),
            str(output_path),
            input_names=["ecg_12lead_mv", "age_years", "sex_male"],
            output_names=["logits"],
            dynamic_axes={
                "ecg_12lead_mv": {0: "batch_size"},
                "age_years": {0: "batch_size"},
                "sex_male": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    add_metadata(
        output_path,
        labels=labels,
        age_mean=age_mean,
        age_std=age_std,
        checkpoint_name=args.checkpoint,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    parity = None if args.skip_parity else verify_parity(
        wrapper,
        output_path,
        atol=args.atol,
        rtol=args.rtol,
    )
    manifest = {
        "onnx_path": str(output_path),
        "sha256": sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "opset": args.opset,
        "inputs": {
            "ecg_12lead_mv": ["batch_size", 12, 2500],
            "age_years": ["batch_size"],
            "sex_male": ["batch_size"],
        },
        "outputs": {"logits": ["batch_size", 2]},
        "labels": labels,
        "aux_norm": {"age_mean": age_mean, "age_std": age_std},
        "parity": parity,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
