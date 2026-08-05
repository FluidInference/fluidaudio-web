# Extract Whisper decoder weights → flat fp32 blob for raw-whisper-decoder.js.
# embed_tokens[51865,512] + embed_positions[448,512]; 6 layers each = self-attn
# (causal) + cross-attn (to encoder) + GELU FFN, PRE-LN; final LN; logits tied to
# embed_tokens. Linear weights are anonymous MatMuls traced by node name; k_proj
# (self & cross) has no bias. Use the NON-merged decoder_model.onnx (flat nodes).
#   python3 scripts/extract-whisper-decoder.py /tmp/whisper/decoder_model.onnx /tmp/whisper/dec
import onnx, numpy as np, json, sys, os, re
from onnx import numpy_helper
src, out = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: w for w in g.initializer}
arr = lambda n: numpy_helper.to_array(I[n]).astype(np.float32)
role = {}
for n in g.node:
    if n.op_type == "MatMul" and n.name:
        w = [i for i in n.input if i in I]
        if w: role[n.name] = arr(w[0])
def R(suffix):
    hits = [k for k in role if k.endswith(suffix)]
    assert len(hits) == 1, (suffix, len(hits))
    return role[hits[0]]
man = {}; blob = bytearray()
def add(k, a):
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a2.size)}; blob.extend(a2.tobytes())
add("embed", arr("model.decoder.embed_tokens.weight")); add("pos", arr("model.decoder.embed_positions.weight"))
add("lnf_w", arr("model.decoder.layer_norm.weight")); add("lnf_b", arr("model.decoder.layer_norm.bias"))
nl = max(int(re.search(r"decoder.layers.(\d+)", k).group(1)) for k in I if "decoder.layers." in k) + 1
for L in range(nl):
    p = f"model.decoder.layers.{L}"
    np_ = f"/model/decoder/layers.{L}"
    for a, an in [("s", "self_attn"), ("c", "encoder_attn")]:
        add(f"L{L}_{a}qw", R(f"{np_}/{an}/q_proj/MatMul")); add(f"L{L}_{a}qb", arr(f"{p}.{an}.q_proj.bias"))
        add(f"L{L}_{a}kw", R(f"{np_}/{an}/k_proj/MatMul"))  # no k bias
        add(f"L{L}_{a}vw", R(f"{np_}/{an}/v_proj/MatMul")); add(f"L{L}_{a}vb", arr(f"{p}.{an}.v_proj.bias"))
        add(f"L{L}_{a}ow", R(f"{np_}/{an}/out_proj/MatMul")); add(f"L{L}_{a}ob", arr(f"{p}.{an}.out_proj.bias"))
    add(f"L{L}_ln1w", arr(f"{p}.self_attn_layer_norm.weight")); add(f"L{L}_ln1b", arr(f"{p}.self_attn_layer_norm.bias"))
    add(f"L{L}_ln2w", arr(f"{p}.encoder_attn_layer_norm.weight")); add(f"L{L}_ln2b", arr(f"{p}.encoder_attn_layer_norm.bias"))
    add(f"L{L}_f1w", R(f"{np_}/fc1/MatMul")); add(f"L{L}_f1b", arr(f"{p}.fc1.bias"))
    add(f"L{L}_f2w", R(f"{np_}/fc2/MatMul")); add(f"L{L}_f2b", arr(f"{p}.fc2.bias"))
    add(f"L{L}_ln3w", arr(f"{p}.final_layer_norm.weight")); add(f"L{L}_ln3b", arr(f"{p}.final_layer_norm.bias"))
os.makedirs(out, exist_ok=True)
open(f"{out}/weights.bin", "wb").write(blob); json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"whisper decoder: {len(man)} tensors, {len(blob)//1024//1024}MB, {nl} layers")
