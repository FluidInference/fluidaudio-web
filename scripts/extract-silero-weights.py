# Extract Silero VAD v5 (16 kHz path) weights from silero_vad.onnx into a flat
# weights.bin + manifest.json for the ORT-free JS forward (raw-silero.js).
#
#   python3 scripts/extract-silero-weights.py /tmp/silero_vad.onnx /tmp/silero-raw
#
# CRITICAL: the ONNX inlines TWO sample-rate branches (8 kHz / 16 kHz) with
# SEPARATE encoder/decoder/LSTM weights — they are NOT shared. A global
# first-match grab silently mixes the 8 kHz enc1/2/3/decoder into the 16 kHz path
# and the forward diverges. So everything is pulled strictly from the 16 kHz
# subgraph (identified by its STFT basis dims [258,1,256]).
import onnx, numpy as np, json, os, sys
from onnx import numpy_helper

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/silero_vad.onnx"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/silero-raw"

m = onnx.load(src)
top = [n for n in m.graph.node if n.op_type == "If"][0]
g16 = None
for a in top.attribute:
    if a.type == onnx.AttributeProto.GRAPH and any(list(w.dims) == [258, 1, 256] for w in a.g.initializer):
        g16 = a.g
assert g16 is not None, "16 kHz subgraph (STFT basis [258,1,256]) not found"

inits = {w.name: numpy_helper.to_array(w).astype(np.float32) for w in g16.initializer}
def by(suffix):
    for k in inits:
        if k.endswith(suffix):
            return inits[k]
    raise SystemExit("missing " + suffix)

lstm = [n for n in g16.node if n.op_type == "LSTM"][0]  # inputs: X, W, R, B, seqlen, h0, c0
want = {
    "stft_basis": by("stft.forward_basis_buffer"),
    "enc0_w": by("encoder.0.reparam_conv.weight"), "enc0_b": by("encoder.0.reparam_conv.bias"),
    "enc1_w": by("encoder.1.reparam_conv.weight"), "enc1_b": by("encoder.1.reparam_conv.bias"),
    "enc2_w": by("encoder.2.reparam_conv.weight"), "enc2_b": by("encoder.2.reparam_conv.bias"),
    "enc3_w": by("encoder.3.reparam_conv.weight"), "enc3_b": by("encoder.3.reparam_conv.bias"),
    "dec_w": by("decoder.decoder.2.weight"), "dec_b": by("decoder.decoder.2.bias"),
    "lstm_W": inits[lstm.input[1]], "lstm_R": inits[lstm.input[2]], "lstm_B": inits[lstm.input[3]],
}
os.makedirs(out, exist_ok=True)
manifest, blob = {}, bytearray()
for k, a in want.items():
    manifest[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a.size)}
    blob += a.astype(np.float32).tobytes()
open(os.path.join(out, "weights.bin"), "wb").write(blob)
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))
print(f"wrote {len(blob)} bytes, {len(manifest)} tensors -> {out}")
