# Extract Whisper encoder weights → flat fp32 blob for raw-whisper-encoder.js.
# conv1(80→512,k3,s1)+GELU, conv2(512→512,k3,s2)+GELU, +sinusoidal pos, 6 PRE-LN
# transformer layers (MHA no-rel-pos, erf-GELU FFN), final LN. Linear weights are
# anonymous ONNX MatMuls; traced by NODE NAME (nodes are cleanly named). LN weights
# are named initializers (LN itself is decomposed into ReduceMean/Sub/.../Mul/Add).
#   python3 scripts/extract-whisper-encoder.py /tmp/whisper/encoder_model.onnx /tmp/whisper/enc
import onnx, numpy as np, json, sys, os, re
from onnx import numpy_helper
src, out = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: w for w in g.initializer}
arr = lambda n: numpy_helper.to_array(I[n]).astype(np.float32)
# node-name → weight initializer (for MatMul/Conv)
role = {}
for n in g.node:
    if n.op_type in ("MatMul", "Conv") and n.name:
        w = [i for i in n.input if i in I]
        if w: role[n.name] = arr(w[0])
def R(suffix):  # find the single node whose name ends with suffix
    hits = [k for k in role if k.endswith(suffix)]
    assert len(hits) == 1, (suffix, hits)
    return role[hits[0]]
man = {}; blob = bytearray()
def add(k, a):
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a2.size)}; blob.extend(a2.tobytes())
add("conv1w", arr("conv1.weight")); add("conv1b", arr("conv1.bias"))
add("conv2w", arr("conv2.weight")); add("conv2b", arr("conv2.bias"))
add("posw", arr("embed_positions.weight"))
add("lnf_w", arr("layer_norm.weight")); add("lnf_b", arr("layer_norm.bias"))
nl = max(int(re.search(r"layers.(\d+)", k).group(1)) for k in I if k.startswith("layers.")) + 1
for L in range(nl):
    p = f"layers.{L}"
    add(f"L{L}_qw", R(f"/{p}/self_attn/q_proj/MatMul")); add(f"L{L}_qb", arr(f"{p}.self_attn.q_proj.bias"))
    add(f"L{L}_kw", R(f"/{p}/self_attn/k_proj/MatMul"))  # k_proj has no bias
    add(f"L{L}_vw", R(f"/{p}/self_attn/v_proj/MatMul")); add(f"L{L}_vb", arr(f"{p}.self_attn.v_proj.bias"))
    add(f"L{L}_ow", R(f"/{p}/self_attn/out_proj/MatMul")); add(f"L{L}_ob", arr(f"{p}.self_attn.out_proj.bias"))
    add(f"L{L}_ln1w", arr(f"{p}.self_attn_layer_norm.weight")); add(f"L{L}_ln1b", arr(f"{p}.self_attn_layer_norm.bias"))
    add(f"L{L}_f1w", R(f"/{p}/fc1/MatMul")); add(f"L{L}_f1b", arr(f"{p}.fc1.bias"))
    add(f"L{L}_f2w", R(f"/{p}/fc2/MatMul")); add(f"L{L}_f2b", arr(f"{p}.fc2.bias"))
    add(f"L{L}_ln2w", arr(f"{p}.final_layer_norm.weight")); add(f"L{L}_ln2b", arr(f"{p}.final_layer_norm.bias"))
os.makedirs(out, exist_ok=True)
open(f"{out}/weights.bin", "wb").write(blob); json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"whisper encoder: {len(man)} tensors, {len(blob)//1024//1024}MB, {nl} layers")
