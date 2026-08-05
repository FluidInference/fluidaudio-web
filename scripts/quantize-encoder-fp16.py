# Convert fp32 encoder weights (from extract-fastconformer-encoder.py) to fp16 for
# shipping — half the size, near-exact quality. Every tensor is stored as fp16;
# the loader (raw-encoder.js) expands to fp32 at load via a 65536-entry LUT.
# Use this for int8-sensitive models (EOU 120M RNNT) where int8 degrades the transcript.
#   python3 scripts/quantize-encoder-fp16.py /tmp/eou-raw/enc /tmp/eou-raw/enc-fp16
import numpy as np, json, os, sys
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eou-raw/enc"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/eou-raw/enc-fp16"
man = json.load(open(f"{src}/manifest.json"))
def readbig(p):
    sz = os.path.getsize(p); u8 = np.empty(sz, np.uint8)
    with open(p, "rb") as f:
        off = 0
        while off < sz:
            n = f.readinto(memoryview(u8)[off:]); off += n
            if not n: break
    return u8.view(np.float32)
bin = readbig(f"{src}/weights.bin")
newman = {}; blob = bytearray()
for k, m in man.items():
    a = np.array(bin[m["offset"]:m["offset"] + m["len"]], np.float32).astype(np.float16)
    newman[k] = {"dims": m["dims"], "dtype": "f16", "offset": len(blob) // 2, "count": int(a.size)}
    blob += np.ascontiguousarray(a).tobytes()
os.makedirs(out, exist_ok=True)
open(f"{out}/weights.bin", "wb").write(blob)
json.dump(newman, open(f"{out}/manifest.json", "w"))
print(f"fp16 artifact: {len(blob)//1024//1024} MB, {len(man)} tensors")
