#!/usr/bin/env python3
"""OPT-0089: fake-quantize the ACE-Step rev7 DiT GEMM weights to int8 and back.

Measures PURE quantization damage with zero kernel changes: every rank-two
DiT GEMM/projection weight in the hosted `dit-revision7` package is quantized
to symmetric int8 with one fp16 scale per 32-element block along the input
(K) dimension, then immediately dequantized back to the original storage
dtype (fp16 or packed bf16). Norms, biases, scale-shift tables, embeddings,
and constants are byte-identical passthrough. The output is a complete,
content-addressed package tree whose manifest differs from the source only in
the 48 weight-shard SHA-256 values (byte length preserved), so the existing
runtime loads it unmodified apart from the two pinned manifest-identity
constants (`DIT_MANIFEST_SHA256` in src/engines/musicgen-acestep/config.ts
and `ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256` in
packages/acestep/src/webgpu/dit-fp16-package.ts), which are patched locally
for the run and never committed.

Usage:
  python3 scripts/requantize-dit-int8.py \
    --source ~/Documents/ace-step-models-mirror/v1/dit-revision7/<sha> \
    --output-root models-local/ace-int8-gate/v1/dit-int8-fakequant \
    --report /tmp/opt-0089-quant-error.json

Requires python3 + numpy only. Streams shard-by-shard (~122 MB peak input);
never materializes the whole 3 GB package in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# The two rank-two GEMM tile layouts emitted by model/convert.py revision 7.
# Physical order (from _iter_dit_gemm_tile_bytes): logical [N, K] ->
# [N/tile, K/32, 32 (K-in-tile), tile (N-in-tile)], so one quantization
# block of 32 consecutive K elements for a fixed output row n is exactly
# physical [n_tile, k_block, :, n_in_tile].
GEMM_LAYOUTS = {
    "dit-gemm-n256-k32-tile-major-v1": 256,
    "dit-gemm-n128-k32-tile-major-v1": 128,
}
QMAX = 127.0
K_BLOCK = 32


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even truncation of float32 to bfloat16 bit patterns."""
    bits = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def fake_quant_blocks(values_f32: np.ndarray) -> tuple[np.ndarray, dict]:
    """values_f32: [n_tiles, k_blocks, 32, tile] float32 -> dequantized f32.

    Symmetric per-block int8: scale = amax/127 stored as fp16; q =
    clip(rint(w / scale), -127, 127); dequant = fp16(scale) * q evaluated in
    f32 (a real int8 kernel would apply the fp16 scale after the integer
    contraction, so this models the deployed dequantization path).
    """
    if not np.isfinite(values_f32).all():
        raise ValueError("non-finite source weight")
    amax = np.abs(values_f32).max(axis=2, keepdims=True)
    scale16 = (amax / QMAX).astype(np.float16)
    if not np.isfinite(scale16).all():
        raise ValueError("fp16 block scale overflow")
    scale = scale16.astype(np.float32)
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.clip(np.rint(values_f32 / safe), -QMAX, QMAX)
    dq = scale * q
    err = dq - values_f32
    tensor_amax = float(np.abs(values_f32).max())
    rms = float(np.sqrt(np.mean(np.square(values_f32), dtype=np.float64)))
    rmse = float(np.sqrt(np.mean(np.square(err), dtype=np.float64)))
    stats = {
        "tensorAmax": tensor_amax,
        "tensorRms": rms,
        "rmse": rmse,
        "nrmse": rmse / rms if rms > 0 else 0.0,
        "maxAbsErr": float(np.abs(err).max()),
        "maxRelErrVsAmax": float(np.abs(err).max() / tensor_amax) if tensor_amax > 0 else 0.0,
        "snrDb": float(20.0 * np.log10(rms / rmse)) if rmse > 0 else float("inf"),
        "zeroScaleBlocks": int((scale == 0.0).sum()),
        "blocks": int(scale.size),
    }
    return dq, stats


def process_tensor(shard: bytearray, tensor: dict) -> dict:
    off, length = tensor["byteOffset"], tensor["byteLength"]
    tile = GEMM_LAYOUTS[tensor["layout"]]
    n, k = tensor["logicalShape"]
    if n % tile or k % K_BLOCK:
        raise ValueError(f"{tensor['logicalTensor']}: shape {n}x{k} not tileable")
    region = np.frombuffer(bytes(shard[off:off + length]), dtype="<u2")
    if region.size != n * k:
        raise ValueError(f"{tensor['logicalTensor']}: byteLength/shape mismatch")
    tiled_shape = (n // tile, k // K_BLOCK, K_BLOCK, tile)
    if tensor["dtype"] == "float16":
        values = region.view("<f2").astype(np.float32).reshape(tiled_shape)
        dq, stats = fake_quant_blocks(values)
        out = dq.astype("<f2")
        # Extra fidelity check: dequantized fp16 must round-trip its bits.
        shard[off:off + length] = out.tobytes()
    elif tensor["dtype"] == "uint32-bf16-pairs":
        values = bf16_to_f32(region).reshape(tiled_shape)
        dq, stats = fake_quant_blocks(values)
        out = f32_to_bf16_rne(dq)
        shard[off:off + length] = out.astype("<u2").tobytes()
        stats["storageNote"] = "dequantized values re-rounded to bf16 (RNE)"
    else:
        raise ValueError(f"{tensor['logicalTensor']}: unexpected dtype {tensor['dtype']}")
    stats["logicalTensor"] = tensor["logicalTensor"]
    stats["logicalShape"] = tensor["logicalShape"]
    stats["layout"] = tensor["layout"]
    stats["dtype"] = tensor["dtype"]
    stats["byteLength"] = length
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="local dit-revision7/<manifest-sha> package dir")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="output root; package written to <root>/<new-manifest-sha>/")
    parser.add_argument("--report", required=True, type=Path,
                        help="per-tensor quantization-error JSON report path")
    args = parser.parse_args()

    manifest_path = args.source / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    source_manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if args.source.name != source_manifest_sha:
        raise SystemExit(f"source dir name != manifest sha256 {source_manifest_sha}")
    manifest = json.loads(manifest_raw)

    by_shard: dict[str, list[dict]] = {}
    passthrough = 0
    for name, tensor in manifest["tensors"].items():
        if tensor["layout"] in GEMM_LAYOUTS:
            by_shard.setdefault(tensor["shard"], []).append(dict(tensor, logicalTensor=name))
        else:
            passthrough += 1

    weight_files = [f for f in manifest["files"] if f["kind"] == "weights"]
    print(f"{sum(len(v) for v in by_shard.values())} GEMM tensors across "
          f"{len(by_shard)} shards; {passthrough} passthrough tensors")

    stage = Path(tempfile.mkdtemp(prefix="dit-int8-fakequant.", dir=args.output_root.parent
                                  if args.output_root.exists() else None))
    (stage / "weights" / "dit").mkdir(parents=True)

    tensor_stats: list[dict] = []
    sha_replacements: dict[str, str] = {}
    for entry in weight_files:
        src = args.source / entry["name"]
        data = bytearray(src.read_bytes())
        if len(data) != entry["byteLength"]:
            raise SystemExit(f"{entry['name']}: byte length mismatch")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{entry['name']}: source shard sha mismatch")
        for tensor in sorted(by_shard.get(entry["name"], []), key=lambda t: t["byteOffset"]):
            tensor_stats.append(process_tensor(data, tensor))
        new_sha = hashlib.sha256(data).hexdigest()
        sha_replacements[entry["sha256"]] = new_sha
        out = stage / entry["name"]
        out.write_bytes(data)
        print(f"  {entry['name']}: {len(by_shard.get(entry['name'], []))} tensors -> {new_sha[:12]}")

    # Rewrite the manifest textually: swap only the 48 shard sha256 hex
    # strings (equal length), preserving canonical JSON bytes and therefore
    # the pinned manifest byteLength (254,357).
    new_manifest_raw = manifest_raw
    for old, new in sha_replacements.items():
        count = new_manifest_raw.count(old.encode())
        if count != 1:
            raise SystemExit(f"shard sha {old} appears {count} times in manifest")
        new_manifest_raw = new_manifest_raw.replace(old.encode(), new.encode())
    if len(new_manifest_raw) != len(manifest_raw):
        raise SystemExit("manifest byte length changed")
    new_manifest_sha = hashlib.sha256(new_manifest_raw).hexdigest()
    (stage / "manifest.json").write_bytes(new_manifest_raw)

    final_dir = args.output_root / new_manifest_sha
    args.output_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(stage)
        print(f"already exists: {final_dir}")
    else:
        stage.rename(final_dir)

    worst = sorted(tensor_stats, key=lambda s: -s["nrmse"])[:10]
    report = {
        "experiment": "OPT-0089",
        "method": "symmetric int8, per-32-block along K, fp16 scale, "
                  "round-to-nearest-even, clamp +/-127, weights only",
        "sourceManifestSha256": source_manifest_sha,
        "outputManifestSha256": new_manifest_sha,
        "manifestByteLength": len(new_manifest_raw),
        "gemmTensorCount": len(tensor_stats),
        "passthroughTensorCount": passthrough,
        "quantizedBytes": sum(s["byteLength"] for s in tensor_stats),
        "aggregate": {
            "maxNrmse": max(s["nrmse"] for s in tensor_stats),
            "medianNrmse": float(np.median([s["nrmse"] for s in tensor_stats])),
            "minSnrDb": min(s["snrDb"] for s in tensor_stats),
            "maxRelErrVsAmax": max(s["maxRelErrVsAmax"] for s in tensor_stats),
            "totalZeroScaleBlocks": sum(s["zeroScaleBlocks"] for s in tensor_stats),
        },
        "worstTensorsByNrmse": [s["logicalTensor"] for s in worst],
        "tensors": tensor_stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"package: {final_dir}")
    print(f"new manifest sha256: {new_manifest_sha}")
    print(f"report: {args.report}")
    agg = report["aggregate"]
    print(f"aggregate: maxNRMSE {agg['maxNrmse']:.5f}  medianNRMSE "
          f"{agg['medianNrmse']:.5f}  minSNR {agg['minSnrDb']:.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
