# Extract Parakeet TDT v3 encoder weights (fp32) from encoder-model.onnx(+.data)
# into a flat weights.bin + manifest.json for the raw-WebGPU forward (raw-encoder.js).
#   python3 scripts/extract-parakeet-encoder-weights.py /tmp/pkv3/encoder-model.onnx /tmp/pk-raw/enc
# Weight roles are read from the graph node names (/layers.L/<role>/...), so the
# anonymous onnx::MatMul_* / onnx::Conv_* initializers are mapped unambiguously.
import onnx, numpy as np, json, os, sys
from onnx import numpy_helper
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pkv3/encoder-model.onnx"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pk-raw/enc"
g = onnx.load(src).graph
I = {w.name: numpy_helper.to_array(w).astype(np.float32) for w in g.initializer}
role = {}
for n in g.node:
    if n.op_type in ("MatMul", "Conv") and n.name and "/layers." in n.name:
        L = int(n.name.split("/layers.")[1].split("/")[0])
        role[(L, n.name.split(f"/layers.{L}/")[1])] = [i for i in n.input if i in I]
R = lambda L, r: I[role[(L, r)][0]]
os.makedirs(out, exist_ok=True)
man, blob = {}, bytearray()
def add(k, a):
    a = np.ascontiguousarray(a, np.float32)
    man[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a.size)}
    blob.extend(a.tobytes())
convs = []
for n in g.node:
    if n.op_type == "Conv": convs.append((n.input[1], n.input[2] if len(n.input) > 2 else None))
    if len(convs) == 5: break
for i, (w, b) in enumerate(convs): add(f"c{i}w", I[w]); add(f"c{i}b", I[b])
addn = [n for n in g.node if n.output and n.output[0] == "/pre_encode/out/Add_output_0"][0]
mmn = [n for n in g.node if addn.input[1] in n.output][0]
add("linw", I[mmn.input[1]]); add("linb", I["pre_encode.out.bias"])
for L in range(24):
    p = f"layers.{L}"
    add(f"L{L}_lnff1_w", I[f"{p}.norm_feed_forward1.weight"]); add(f"L{L}_lnff1_b", I[f"{p}.norm_feed_forward1.bias"])
    add(f"L{L}_ff1w1", R(L, "feed_forward1/linear1/MatMul")); add(f"L{L}_ff1w2", R(L, "feed_forward1/linear2/MatMul"))
    add(f"L{L}_lnatt_w", I[f"{p}.norm_self_att.weight"]); add(f"L{L}_lnatt_b", I[f"{p}.norm_self_att.bias"])
    add(f"L{L}_q", R(L, "self_attn/linear_q/MatMul")); add(f"L{L}_k", R(L, "self_attn/linear_k/MatMul")); add(f"L{L}_v", R(L, "self_attn/linear_v/MatMul"))
    add(f"L{L}_pos", R(L, "self_attn/linear_pos/MatMul")); add(f"L{L}_out", R(L, "self_attn/linear_out/MatMul"))
    add(f"L{L}_pbu", I[f"{p}.self_attn.pos_bias_u"]); add(f"L{L}_pbv", I[f"{p}.self_attn.pos_bias_v"])
    add(f"L{L}_lnconv_w", I[f"{p}.norm_conv.weight"]); add(f"L{L}_lnconv_b", I[f"{p}.norm_conv.bias"])
    add(f"L{L}_pw1", R(L, "conv/pointwise_conv1/Conv"))
    dw = [I[x] for x in role[(L, "conv/depthwise_conv/Conv")]]; add(f"L{L}_dw", dw[0]); add(f"L{L}_dwb", dw[1])
    add(f"L{L}_pw2", R(L, "conv/pointwise_conv2/Conv"))
    add(f"L{L}_lnff2_w", I[f"{p}.norm_feed_forward2.weight"]); add(f"L{L}_lnff2_b", I[f"{p}.norm_feed_forward2.bias"])
    add(f"L{L}_ff2w1", R(L, "feed_forward2/linear1/MatMul")); add(f"L{L}_ff2w2", R(L, "feed_forward2/linear2/MatMul"))
    add(f"L{L}_lnout_w", I[f"{p}.norm_out.weight"]); add(f"L{L}_lnout_b", I[f"{p}.norm_out.bias"])
open(f"{out}/weights.bin", "wb").write(blob)
json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"{len(man)} tensors, {len(blob)//1024//1024} MB -> {out}")
