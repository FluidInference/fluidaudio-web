# VoiceChat-11B STT chain extractor: safetensors → the L{n}_* + c0..4/lin manifest
# scheme consumed by raw-encoder.js (encoder, fp16) and the flat fp32 blob consumed
# by raw-decoder-voicechat.js (RNNT prediction net + joint), plus vocab.json.
#
# Source: NVIDIA VoiceChat-11B user-transcription chain — 609M causal FastConformer
# (24L, d1024, h8, ff4096, dwK 9, dw_striding 8x, conv_norm_type layer_norm,
# use_bias False, att_context [70,0] chunked_limited) + 2-layer LSTM RNNT
# (hidden 640, vocab 1024 + blank 1024). stt_model.perception.proj (1024→4480,
# the LLM adapter) is deliberately EXCLUDED — the RNNT taps asr_emb directly.
#
#   uv run --with safetensors,numpy,sentencepiece python3 scripts/extract-voicechat-stt.py \
#       ~/Documents/models/voicechat-11b models-local/voicechat-stt
import json, os, sys
import numpy as np
from safetensors import safe_open

src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Documents/models/voicechat-11b")
out = sys.argv[2] if len(sys.argv) > 2 else "models-local/voicechat-stt"
os.makedirs(out, exist_ok=True)

# ── encoder → fp16 manifest ──────────────────────────────────────────────────
E = "stt_model.perception.encoder."
ef = safe_open(f"{src}/components/encoder.safetensors", "np")
get = lambda k: ef.get_tensor(E + k).astype(np.float32)

man = {}
blob = bytearray()


def add(key, a):
    h = np.ascontiguousarray(a, np.float32).astype(np.float16)
    man[key] = {"dims": list(a.shape), "dtype": "f16", "offset": len(blob) // 2, "count": int(h.size)}
    blob.extend(h.reshape(-1).tobytes())


# dw_striding subsampling: conv.0 (regular 3x3 s2) / conv.2 (depthwise) / conv.3
# (pointwise) / conv.5 (depthwise) / conv.6 (pointwise); ReLU after 0, 3, 6.
for i, n in enumerate([0, 2, 3, 5, 6]):
    add(f"c{i}w", get(f"pre_encode.conv.{n}.weight"))
    add(f"c{i}b", get(f"pre_encode.conv.{n}.bias"))
# pre_encode.out: PyTorch Linear [out, in] → matmul layout [in, out]
add("linw", get("pre_encode.out.weight").T)
add("linb", get("pre_encode.out.bias"))

NL = 24
for L in range(NL):
    p = f"layers.{L}."
    g = lambda s: get(p + s)
    add(f"L{L}_lnff1_w", g("norm_feed_forward1.weight"))
    add(f"L{L}_lnff1_b", g("norm_feed_forward1.bias"))
    add(f"L{L}_ff1w1", g("feed_forward1.linear1.weight").T)
    add(f"L{L}_ff1w2", g("feed_forward1.linear2.weight").T)
    add(f"L{L}_lnatt_w", g("norm_self_att.weight"))
    add(f"L{L}_lnatt_b", g("norm_self_att.bias"))
    for nm, an in [("q", "linear_q"), ("k", "linear_k"), ("v", "linear_v"), ("pos", "linear_pos"), ("out", "linear_out")]:
        add(f"L{L}_{nm}", g(f"self_attn.{an}.weight").T)  # use_bias False → no biases
    add(f"L{L}_pbu", g("self_attn.pos_bias_u"))
    add(f"L{L}_pbv", g("self_attn.pos_bias_v"))
    add(f"L{L}_lnconv_w", g("norm_conv.weight"))
    add(f"L{L}_lnconv_b", g("norm_conv.bias"))
    add(f"L{L}_pw1", g("conv.pointwise_conv1.weight"))  # [2D, D, 1] as-is (pwT reads [cout, cin])
    add(f"L{L}_dw", g("conv.depthwise_conv.weight"))  # [D, 1, 9], no bias
    add(f"L{L}_dwb", np.zeros(1024, np.float32))
    # conv_norm_type=layer_norm: NeMo names it batch_norm but it IS a LayerNorm →
    # the raw-encoder bnw/bnb path (depthwise → LN over channels → SiLU).
    add(f"L{L}_bnw", g("conv.batch_norm.weight"))
    add(f"L{L}_bnb", g("conv.batch_norm.bias"))
    add(f"L{L}_pw2", g("conv.pointwise_conv2.weight"))
    add(f"L{L}_lnff2_w", g("norm_feed_forward2.weight"))
    add(f"L{L}_lnff2_b", g("norm_feed_forward2.bias"))
    add(f"L{L}_ff2w1", g("feed_forward2.linear1.weight").T)
    add(f"L{L}_ff2w2", g("feed_forward2.linear2.weight").T)
    add(f"L{L}_lnout_w", g("norm_out.weight"))
    add(f"L{L}_lnout_b", g("norm_out.bias"))

open(f"{out}/encoder-f16.bin", "wb").write(blob)
json.dump(man, open(f"{out}/encoder-f16.manifest.json", "w"))
print(f"encoder: {len(man)} tensors, {len(blob) // 1024 // 1024} MB fp16, {NL} layers")

# ── RNNT decoder + joint → flat fp32 blob ────────────────────────────────────
rf = safe_open(f"{src}/components/rnnt.safetensors", "np")
D = "stt_model.rnnt_decoder.prediction."
J = "stt_model.rnnt_joint."
r = lambda k: rf.get_tensor(k).astype(np.float32)

H = 640


def iofc(a):
    """PyTorch LSTM gate rows (i,f,g,o) → the ONNX iofc order lstmStep expects."""
    i, f, g_, o = (a[k * H : (k + 1) * H] for k in range(4))
    return np.concatenate([i, o, f, g_], axis=0)


tens = {
    "embed": r(D + "embed.weight"),  # [1025, 640], row 1024 = blank/SOS
    "l0_W": iofc(r(D + "dec_rnn.lstm.weight_ih_l0")),
    "l0_R": iofc(r(D + "dec_rnn.lstm.weight_hh_l0")),
    "l0_B": np.concatenate([iofc(r(D + "dec_rnn.lstm.bias_ih_l0")), iofc(r(D + "dec_rnn.lstm.bias_hh_l0"))]),
    "l1_W": iofc(r(D + "dec_rnn.lstm.weight_ih_l1")),
    "l1_R": iofc(r(D + "dec_rnn.lstm.weight_hh_l1")),
    "l1_B": np.concatenate([iofc(r(D + "dec_rnn.lstm.bias_ih_l1")), iofc(r(D + "dec_rnn.lstm.bias_hh_l1"))]),
    "encW": r(J + "enc.weight").T,  # [1024, 640]
    "encB": r(J + "enc.bias"),
    "predW": r(J + "pred.weight").T,  # [640, 640]
    "predB": r(J + "pred.bias"),
    "outW": r(J + "joint_net.2.weight").T,  # [640, 1025]
    "outB": r(J + "joint_net.2.bias"),
}
dman = {}
dblob = bytearray()
for k, a in tens.items():
    a2 = np.ascontiguousarray(a.reshape(-1), np.float32)
    dman[k] = {"dims": list(a.shape), "offset": len(dblob) // 4, "len": int(a2.size)}
    dblob += a2.tobytes()
open(f"{out}/decoder-fp32.bin", "wb").write(dblob)
json.dump(dman, open(f"{out}/decoder-fp32.manifest.json", "w"))
print(f"decoder+joint: {len(dman)} tensors, {len(dblob) // 1024 // 1024} MB; embed {list(tens['embed'].shape)} out {list(tens['outW'].shape)}")

# ── vocab: id → piece object (the asr-nemotron detok format) ─────────────────
pieces = json.load(open(f"{src}/rnnt_tokenizer/vocab.json"))
try:  # sanity: vocab.json order must equal the sentencepiece ids
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(f"{src}/rnnt_tokenizer/tokenizer.model")
    assert sp.get_piece_size() == len(pieces), (sp.get_piece_size(), len(pieces))
    mism = [i for i in range(len(pieces)) if sp.id_to_piece(i) != pieces[i]]
    assert not mism, f"vocab.json order != sentencepiece ids at {mism[:5]}"
    print(f"vocab: {len(pieces)} pieces, sentencepiece order verified")
except ImportError:
    print(f"vocab: {len(pieces)} pieces (sentencepiece unavailable — order unverified)")
vocab = {str(i): t for i, t in enumerate(pieces)}
vocab["1024"] = "<blank>"
json.dump(vocab, open(f"{out}/vocab.json", "w"), ensure_ascii=False)
print(f"wrote {out}/")
