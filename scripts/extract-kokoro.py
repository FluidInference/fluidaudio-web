# Extract ALL Kokoro weights from model.onnx → flat fp32 blob + manifest + role map.
# The JS port (src/gpu/kokoro.js) resolves tensors by node-name SUFFIX exactly like
# the numpy reference (/tmp/kfinal.py's R()): roles[nodeSuffix] = {w,b(,r)} → keys
# into the manifest. Covers predictor (LSTM/Gemm/Conv AdaINResBlocks), decoder
# (AdaIN resblocks), generator (convs/ConvTranspose/alphas), iSTFT basis, ALBERT.
#   python3 scripts/extract-kokoro.py /tmp/kokoro/model.onnx /tmp/kokoro/kw
import onnx, numpy as np, json, sys, os
from onnx import numpy_helper

src, out = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: numpy_helper.to_array(w).astype(np.float32) for w in g.initializer}

os.makedirs(out, exist_ok=True)
man, blob = {}, bytearray()
def add(name, a):
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[name] = {"offset": len(blob) // 4, "len": int(a2.size), "dims": list(a.shape)}
    blob.extend(a2.tobytes())
for name, a in I.items():
    add(name, a)

# role map: node-name → the initializer name(s) it consumes (weights/bias/recurrent).
roles = {}
for n in g.node:
    if not n.name:
        continue
    ins = [i for i in n.input if i in I]
    if n.op_type in ("Conv", "ConvTranspose", "Gemm"):
        if ins:
            roles[n.name] = {"w": ins[0], "b": ins[1] if len(ins) > 1 else None}
    elif n.op_type == "MatMul":
        if ins:
            roles[n.name] = {"w": ins[0]}
    elif n.op_type == "LSTM":
        # ONNX LSTM inputs: X, W, R, B, ... → W/R/B are initializers
        if len(ins) >= 3:
            roles[n.name] = {"w": ins[0], "r": ins[1], "b": ins[2] if len(ins) > 2 else None}

open(f"{out}/weights.bin", "wb").write(blob)
json.dump(man, open(f"{out}/manifest.json", "w"))
json.dump(roles, open(f"{out}/roles.json", "w"))
print(f"kokoro: {len(man)} tensors, {len(blob)//1024//1024}MB, {len(roles)} roles")
