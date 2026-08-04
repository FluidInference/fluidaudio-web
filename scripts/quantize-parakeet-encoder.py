# Quantize the fp32 Parakeet encoder weights (from extract-parakeet-encoder-weights.py)
# to per-channel symmetric int8 for shipping. Validated WER-neutral (identical
# transcript vs fp32). matmul weights [in,out] → per output col; conv weights (flat,
# first dim Cout) → per Cout; small tensors (LN, bias, pos_bias) stay fp32.
#   python3 scripts/quantize-parakeet-encoder.py /tmp/pk-raw/enc /tmp/pk-raw/enc-int8
# Output: weights.bin (fp32 section [scales + fp32 tensors] first, then int8 section)
# + manifest.json. Loader dequants per-tensor (raw-encoder.js).
import numpy as np, json, os, sys
from pathlib import Path
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pk-raw/enc"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pk-raw/enc-int8"
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
def is_mat(k): return len(man[k]["dims"]) == 2 and (k == "linw" or k.split("_")[-1] in ("q","k","v","pos","out","ff1w1","ff1w2","ff2w1","ff2w2"))
def is_conv(k): return __import__("re").match(r"^c[0-4]w$", k) or k.split("_")[-1] in ("pw1","pw2","dw")

f32_blocks = []  # (key-or-scalekey, np.float32 array)
i8_blocks = []
newman = {}
for k, m in man.items():
    a = np.array(bin[m["offset"]:m["offset"]+m["len"]], np.float32)
    dims = m["dims"]
    if is_mat(k):
        inn, o = dims; a2 = a.reshape(inn, o)
        scale = (np.abs(a2).max(0) / 127.0); scale[scale == 0] = 1
        q = np.clip(np.round(a2 / scale), -127, 127).astype(np.int8)
        newman[k] = {"dims": dims, "dtype": "i8", "quant": "col", "count": a.size, "scaleCount": o}
        i8_blocks.append((k, q.reshape(-1))); f32_blocks.append(("__scale__"+k, scale.astype(np.float32)))
    elif is_conv(k):
        cout = dims[0]; rest = a.size // cout; a2 = a.reshape(cout, rest)
        scale = (np.abs(a2).max(1) / 127.0); scale[scale == 0] = 1
        q = np.clip(np.round(a2 / scale[:, None]), -127, 127).astype(np.int8)
        newman[k] = {"dims": dims, "dtype": "i8", "quant": "grp", "count": a.size, "scaleCount": cout}
        i8_blocks.append((k, q.reshape(-1))); f32_blocks.append(("__scale__"+k, scale.astype(np.float32)))
    else:
        newman[k] = {"dims": dims, "dtype": "f32", "count": a.size}
        f32_blocks.append((k, a))
# layout: all fp32 first (4-aligned), then all int8
blob = bytearray()
for key, arr in f32_blocks:
    off = len(blob) // 4
    if key.startswith("__scale__"): newman[key[9:]]["scaleOffset"] = off
    else: newman[key]["offset"] = off
    blob += np.ascontiguousarray(arr, np.float32).tobytes()
i8_base = len(blob)  # bytes
for key, arr in i8_blocks:
    newman[key]["i8ByteOffset"] = len(blob)
    blob += np.ascontiguousarray(arr, np.int8).tobytes()
os.makedirs(out, exist_ok=True)
open(f"{out}/weights.bin", "wb").write(blob)
json.dump(newman, open(f"{out}/manifest.json", "w"))
print(f"int8 artifact: {len(blob)//1024//1024} MB (fp32 section {i8_base//1024//1024} MB + int8 {(len(blob)-i8_base)//1024//1024} MB)")
