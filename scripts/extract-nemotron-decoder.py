# Extract the Nemotron RNNT decoder (2-layer LSTM prediction net) + joint (enc/pred
# proj + relu + out) weights to a flat fp32 blob for raw-decoder-nemotron.js.
#   python3 scripts/extract-nemotron-decoder.py /tmp/nemo-fp16 /tmp/nemo-raw/dec
import onnx, numpy as np, json, sys, os
from onnx import numpy_helper
src, out = sys.argv[1], sys.argv[2]
dg = onnx.load(f"{src}/decoder.onnx").graph
jg = onnx.load(f"{src}/joint.onnx").graph
DI = {w.name: w for w in dg.initializer}
JI = {w.name: w for w in jg.initializer}
arr = lambda I, k: numpy_helper.to_array(I[k]).astype(np.float32)
# decoder LSTM weight names appear in initializer order: layer0 W/R/B then layer1 W/R/B
lstm_names = [k for k in DI if "LSTM" in k]  # 6 names, order = graph order
# map by finding which LSTM node consumes them (node order = layer order)
lstm_nodes = [n for n in dg.node if n.op_type == "LSTM"]
l0 = lstm_nodes[0].input[1:4]  # W,R,B
l1 = lstm_nodes[1].input[1:4]
# joint MatMul weights by their consuming node names
enc_w = pred_w = out_w = None
for n in jg.node:
    if n.op_type == "MatMul":
        wn = [i for i in n.input if i in JI][0]
        if "/enc/" in n.name: enc_w = wn
        elif "/pred/" in n.name: pred_w = wn
        else: out_w = wn
tens = {
    "embed": arr(DI, "decoder.prediction.embed.weight"),
    "l0_W": arr(DI, l0[0]), "l0_R": arr(DI, l0[1]), "l0_B": arr(DI, l0[2]),
    "l1_W": arr(DI, l1[0]), "l1_R": arr(DI, l1[1]), "l1_B": arr(DI, l1[2]),
    "encW": arr(JI, enc_w), "encB": arr(JI, "joint.enc.bias"),
    "predW": arr(JI, pred_w), "predB": arr(JI, "joint.pred.bias"),
    "outW": arr(JI, out_w), "outB": arr(JI, "joint.joint_net.2.bias"),
}
os.makedirs(out, exist_ok=True); man = {}; blob = bytearray()
for k, a in tens.items():
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a2.size)}
    blob += a2.tobytes()
open(f"{out}/weights.bin", "wb").write(blob)
json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"decoder+joint: {len(man)} tensors, {len(blob)//1024//1024}MB; embed {list(tens['embed'].shape)} out {list(tens['outW'].shape)}")
