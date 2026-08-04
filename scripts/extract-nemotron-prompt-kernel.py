# Append Nemotron's multilingual prompt_kernel weights to a FastConformer extraction
# (extract-fastconformer-encoder.py doesn't capture them — they're not layers.N).
# The prompt_kernel is a 2-layer MLP applied AFTER the conformer stack:
#   encoded_output = MLP(concat([conformer_out[1024], language_onehot[128]]))
#   MLP: Linear(1152->2048) -> Relu -> Linear(2048->1024)
# Keys added: pk0w[1152,2048] pk0b[2048] pk2w[2048,1024] pk2b[1024].
#   python3 scripts/extract-nemotron-prompt-kernel.py /tmp/nemo-fp16/encoder.onnx /tmp/nemo-raw/enc
import onnx, numpy as np, json, sys
from onnx import numpy_helper
src, dst = sys.argv[1], sys.argv[2]
g = onnx.load(src).graph
I = {w.name: w for w in g.initializer}
# find the two prompt_kernel MatMul weight names + biases
w0 = w2 = None
for n in g.node:
    if n.op_type == "MatMul" and "prompt_kernel.0" in n.name:
        w0 = [i for i in n.input if i in I][0]
    if n.op_type == "MatMul" and "prompt_kernel.2" in n.name:
        w2 = [i for i in n.input if i in I][0]
arr = lambda k: numpy_helper.to_array(I[k]).astype(np.float32)
pk = {"pk0w": arr(w0), "pk0b": arr("prompt_kernel.0.bias"), "pk2w": arr(w2), "pk2b": arr("prompt_kernel.2.bias")}
man = json.load(open(f"{dst}/manifest.json"))
blob = bytearray(open(f"{dst}/weights.bin", "rb").read())
for k, a in pk.items():
    a = np.ascontiguousarray(a.reshape(-1), np.float32)
    man[k] = {"dims": list(pk[k].shape), "offset": len(blob) // 4, "len": int(a.size)}
    blob += a.tobytes()
open(f"{dst}/weights.bin", "wb").write(blob)
json.dump(man, open(f"{dst}/manifest.json", "w"))
print(f"added prompt_kernel: pk0w{list(pk['pk0w'].shape)} pk2w{list(pk['pk2w'].shape)}; {len(man)} tensors total")
