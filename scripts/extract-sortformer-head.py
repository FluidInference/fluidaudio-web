# Extract Sortformer's post-encoder head: encoder_proj (512→192), the 18-layer
# transformer_encoder (post-LN, MHA + ReLU-FFN), and first_hidden_to_hidden /
# single_hidden_to_spks. Anonymous MatMul weights traced via the named bias each feeds.
#   python3 scripts/extract-sortformer-head.py /tmp/sf/sf.onnx /tmp/sf-raw/head
import onnx, numpy as np, json, sys, os, re
from onnx import numpy_helper
src, out = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: w for w in g.initializer}
prod = {o: n for n in g.node for o in n.output}
arr = lambda n: numpy_helper.to_array(I[n]).astype(np.float32)
def wfor(biasname):  # weight of the MatMul whose Add consumes biasname
    add = [n for n in g.node if n.op_type == "Add" and biasname in n.input][0]
    mm = prod[[i for i in add.input if i != biasname][0]]
    return arr([i for i in mm.input if i in I][0])
man = {}; blob = bytearray()
def add(k, a):
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(a.shape), "offset": len(blob) // 4, "len": int(a2.size)}; blob.extend(a2.tobytes())
# encoder_proj + head MLPs
add("encoder_proj_w", wfor("sortformer_modules.encoder_proj.bias")); add("encoder_proj_b", arr("sortformer_modules.encoder_proj.bias"))
add("fhh_w", wfor("sortformer_modules.first_hidden_to_hidden.bias")); add("fhh_b", arr("sortformer_modules.first_hidden_to_hidden.bias"))
add("spks_w", wfor("sortformer_modules.single_hidden_to_spks.bias")); add("spks_b", arr("sortformer_modules.single_hidden_to_spks.bias"))
nl = max(int(re.search(r"transformer_encoder.layers.(\d+)", w.name).group(1)) for w in g.initializer if "transformer_encoder.layers." in w.name) + 1
for L in range(nl):
    p = f"transformer_encoder.layers.{L}"
    for nm, sub in [("q", "first_sub_layer.query_net"), ("k", "first_sub_layer.key_net"), ("v", "first_sub_layer.value_net"), ("o", "first_sub_layer.out_projection"), ("din", "second_sub_layer.dense_in"), ("dout", "second_sub_layer.dense_out")]:
        add(f"T{L}_{nm}w", wfor(f"{p}.{sub}.bias")); add(f"T{L}_{nm}b", arr(f"{p}.{sub}.bias"))
    add(f"T{L}_ln1w", arr(f"{p}.layer_norm_1.weight")); add(f"T{L}_ln1b", arr(f"{p}.layer_norm_1.bias"))
    add(f"T{L}_ln2w", arr(f"{p}.layer_norm_2.weight")); add(f"T{L}_ln2b", arr(f"{p}.layer_norm_2.bias"))
os.makedirs(out, exist_ok=True)
open(f"{out}/weights.bin", "wb").write(blob); json.dump(man, open(f"{out}/manifest.json", "w"))
print(f"head: {len(man)} tensors, {len(blob)//1024//1024}MB, {nl} transformer layers")
