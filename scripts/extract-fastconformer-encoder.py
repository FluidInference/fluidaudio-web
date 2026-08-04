# Generic NeMo FastConformer encoder weight extractor → the L{n}_* + c0..4/lin scheme
# consumed by raw-encoder.js. Handles the /layers.N/ or /encoder/layers.N/ node-name
# prefix and named-or-anonymous conv/matmul weights (via node role map).
#   python3 scripts/extract-fastconformer-encoder.py <enc.onnx> <outdir>
import onnx, numpy as np, json, os, sys, re
from onnx import numpy_helper
src, out = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: w for w in g.initializer}
def arr(n): return numpy_helper.to_array(I[n]).astype(np.float32)
# prefix: does the layer path have /encoder/ ?
enc_pref = "/encoder/layers." if any(n.name and "/encoder/layers.0/" in n.name for n in g.node) else "/layers."
w_pref = "encoder." if any(k.startswith("encoder.layers.") for k in I) else ""
role = {}
for n in g.node:
    if n.op_type in ("MatMul", "Conv") and n.name and enc_pref in n.name:
        mm = re.search(re.escape(enc_pref) + r"(\d+)/", n.name)
        if not mm: continue
        L = int(mm.group(1)); r = n.name.split(f"{enc_pref}{L}/")[1]
        w = [i for i in n.input if i in I]
        if w: role[(L, r)] = w[0]
R = lambda L, r: arr(role[(L, r)])
nl = max(L for (L, _) in role) + 1
os.makedirs(out, exist_ok=True); man = {}; blob = bytearray()
def add(k, a, dims):
    a = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(dims), "offset": len(blob) // 4, "len": int(a.size)}; blob.extend(a.tobytes())
# subsampling: first 5 Conv nodes
convs = []
for n in g.node:
    if n.op_type == "Conv": convs.append((n.input[1], n.input[2] if len(n.input) > 2 else None))
    if len(convs) == 5: break
for i, (wn, bn) in enumerate(convs): add(f"c{i}w", arr(wn), I[wn].dims); add(f"c{i}b", arr(bn), I[bn].dims)
# pre_encode.out linear: the Add consuming the named pre_encode.out.bias (its output
# name varies — "pre_encode/out/Add" for some exports, a graph-output alias for others).
out_bias = f"{w_pref}pre_encode.out.bias"
addn = [n for n in g.node if n.op_type == "Add" and out_bias in n.input][0]
mm_in = [i for i in addn.input if i != out_bias][0]
mmn = [n for n in g.node if mm_in in n.output][0]
add("linw", arr(mmn.input[1]), I[mmn.input[1]].dims); add("linb", arr(out_bias), I[out_bias].dims)
W = w_pref
for L in range(nl):
    p = f"{W}layers.{L}"
    add(f"L{L}_lnff1_w", arr(f"{p}.norm_feed_forward1.weight"), I[f"{p}.norm_feed_forward1.weight"].dims); add(f"L{L}_lnff1_b", arr(f"{p}.norm_feed_forward1.bias"), I[f"{p}.norm_feed_forward1.bias"].dims)
    add(f"L{L}_ff1w1", R(L, "feed_forward1/linear1/MatMul"), I[role[(L, "feed_forward1/linear1/MatMul")]].dims); add(f"L{L}_ff1w2", R(L, "feed_forward1/linear2/MatMul"), I[role[(L, "feed_forward1/linear2/MatMul")]].dims)
    add(f"L{L}_lnatt_w", arr(f"{p}.norm_self_att.weight"), I[f"{p}.norm_self_att.weight"].dims); add(f"L{L}_lnatt_b", arr(f"{p}.norm_self_att.bias"), I[f"{p}.norm_self_att.bias"].dims)
    for nm, rl in [("q", "self_attn/linear_q/MatMul"), ("k", "self_attn/linear_k/MatMul"), ("v", "self_attn/linear_v/MatMul"), ("pos", "self_attn/linear_pos/MatMul"), ("out", "self_attn/linear_out/MatMul")]:
        add(f"L{L}_{nm}", R(L, rl), I[role[(L, rl)]].dims)
    add(f"L{L}_pbu", arr(f"{p}.self_attn.pos_bias_u"), I[f"{p}.self_attn.pos_bias_u"].dims); add(f"L{L}_pbv", arr(f"{p}.self_attn.pos_bias_v"), I[f"{p}.self_attn.pos_bias_v"].dims)
    add(f"L{L}_lnconv_w", arr(f"{p}.norm_conv.weight"), I[f"{p}.norm_conv.weight"].dims); add(f"L{L}_lnconv_b", arr(f"{p}.norm_conv.bias"), I[f"{p}.norm_conv.bias"].dims)
    add(f"L{L}_pw1", R(L, "conv/pointwise_conv1/Conv"), I[role[(L, "conv/pointwise_conv1/Conv")]].dims)
    dww = role[(L, "conv/depthwise_conv/Conv")]
    add(f"L{L}_dw", arr(dww), I[dww].dims)
    dwnode = [n for n in g.node if n.name and n.name.endswith(f"{enc_pref}{L}/conv/depthwise_conv/Conv")][0]
    dwins = [i for i in dwnode.input if i in I]
    if len(dwins) >= 2: add(f"L{L}_dwb", arr(dwins[1]), I[dwins[1]].dims)  # depthwise has bias
    else:
        cout = I[dww].dims[0]; add(f"L{L}_dwb", np.zeros(cout, np.float32), [cout])  # no bias → zeros
    # conv-module norm after depthwise (EOU: explicit batch_norm applied as LN; Parakeet folds it)
    bnw = f"{p}.conv.batch_norm.weight"
    if bnw in I: add(f"L{L}_bnw", arr(bnw), I[bnw].dims); add(f"L{L}_bnb", arr(f"{p}.conv.batch_norm.bias"), I[f"{p}.conv.batch_norm.bias"].dims)
    add(f"L{L}_pw2", R(L, "conv/pointwise_conv2/Conv"), I[role[(L, "conv/pointwise_conv2/Conv")]].dims)
    add(f"L{L}_lnff2_w", arr(f"{p}.norm_feed_forward2.weight"), I[f"{p}.norm_feed_forward2.weight"].dims); add(f"L{L}_lnff2_b", arr(f"{p}.norm_feed_forward2.bias"), I[f"{p}.norm_feed_forward2.bias"].dims)
    add(f"L{L}_ff2w1", R(L, "feed_forward2/linear1/MatMul"), I[role[(L, "feed_forward2/linear1/MatMul")]].dims); add(f"L{L}_ff2w2", R(L, "feed_forward2/linear2/MatMul"), I[role[(L, "feed_forward2/linear2/MatMul")]].dims)
    add(f"L{L}_lnout_w", arr(f"{p}.norm_out.weight"), I[f"{p}.norm_out.weight"].dims); add(f"L{L}_lnout_b", arr(f"{p}.norm_out.bias"), I[f"{p}.norm_out.bias"].dims)
open(f"{out}/weights.bin", "wb").write(blob); json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"{len(man)} tensors, {len(blob)//1024//1024}MB, {nl} layers (prefix {enc_pref})")
