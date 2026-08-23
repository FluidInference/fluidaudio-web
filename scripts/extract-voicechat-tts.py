# VoiceChat-11B TTS chain extractor: sliced safetensors → manifest+bin for the
# tts-voicechat engine (src/engines/tts-voicechat/). One manifest per bin with
# PER-TENSOR dtype ("f32" | "f16" | "u8") and BYTE offsets — unlike the older
# uniform-dtype manifests, so precision can be tuned per tensor from parity.
#
# Precision (see docs/VOICECHAT.md): f16 for the big backbone / CAS-encoder
# GEMMs; f32 for everything decision-critical — norms, MoG head (mixture argmax
# + PRVQ argmin margins), RVQ codebooks, codec convs, embeddings/fusion.
# Weight matrices are stored TRANSPOSED [in, out] for x@W row-major GEMV.
#
# Outputs (models-local/voicechat-tts/, gitignored):
#   tts.manifest.json + tts.bin        backbone + CAS + MoG + fusion/embeds + flags
#   codec.manifest.json + codec.bin    PRVQ codebooks + Latent2Wav decoder convs
#   tokenizer.json                     BPE vocab/merges/regex + char vocab + ids
#   config.json                        geometry, rope/eps/softcap, warmup arrays
#   golden.manifest.json + golden.bin  parity goldens (only with --golden)
#
#   /tmp/vc-tts-env/bin/python scripts/extract-voicechat-tts.py \
#       --src ~/Documents/models/voicechat-11b --out models-local/voicechat-tts \
#       --golden /tmp/voicechat-tts-golden.npz
import argparse
import json
import math
import os

import numpy as np
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.expanduser("~/Documents/models/voicechat-11b"))
ap.add_argument("--out", default="models-local/voicechat-tts")
ap.add_argument("--golden", default=None, help=".npz from voicechat-tts-reference.py")
ap.add_argument(
    "--backbone-dtype",
    default="f32",
    choices=["f16", "f32"],
    help="backbone + CAS GEMM storage. f32 is the default: with f16 the parity run is code-exact "
    "only through ~frame 22 before an MoG/PRVQ near-tie flips (1147/1550); f32 restores exactness.",
)
args = ap.parse_args()
GEMM_DT = args.backbone_dtype
os.makedirs(args.out, exist_ok=True)

tf = safe_open(os.path.join(args.src, "components/tts.safetensors"), "np")
cf = safe_open(os.path.join(args.src, "components/codec.safetensors"), "np")
T = lambda k: tf.get_tensor("tts_model.tts_model." + k)
C = lambda k: cf.get_tensor("tts_model.audio_codec." + k)


class Bin:
    """Sharded blob writer: node readFileSync and safe browser fetches cap out
    around 2 GiB, so tensors spill into <path>.<i>.bin shards (manifest entries
    carry a per-tensor shard index)."""

    MAX_SHARD = 1_600_000_000

    def __init__(self):
        self.man, self.shards = {}, [bytearray()]

    def add(self, name, a, dtype="f32"):
        a = np.ascontiguousarray(a)
        if dtype == "f16":
            payload = a.astype(np.float32).astype(np.float16)
        elif dtype == "u8":
            payload = a.astype(np.uint8)
        else:
            payload = a.astype(np.float32)
        blob = self.shards[-1]
        if len(blob) + payload.nbytes > self.MAX_SHARD:
            self.shards.append(bytearray())
            blob = self.shards[-1]
        while len(blob) % 4:
            blob.append(0)
        self.man[name] = {
            "dims": list(a.shape),
            "dtype": dtype,
            "bin": len(self.shards) - 1,
            "byteOffset": len(blob),
            "count": int(a.size),
        }
        blob.extend(payload.tobytes())

    def write(self, path):
        for i, blob in enumerate(self.shards):
            open(f"{path}.{i}.bin", "wb").write(blob)
        json.dump({"shards": len(self.shards), "tensors": self.man}, open(path + ".manifest.json", "w"))
        total = sum(len(b) for b in self.shards)
        print(f"{path}.*.bin: {len(self.man)} tensors, {len(self.shards)} shard(s), {total/1e6:.1f} MB")


tts = Bin()

# ── Gemma3 backbone (28 layers). GEMMs f16 transposed [in,out]; norms f32 ────
NL = 28
for L in range(NL):
    g = lambda s: T(f"backbone.layers.{L}.{s}")
    for nm, key in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")]:
        tts.add(f"B{L}_{nm}", g(key + ".weight").T, GEMM_DT)
    tts.add(f"B{L}_qn", g("self_attn.q_norm.weight"))
    tts.add(f"B{L}_kn", g("self_attn.k_norm.weight"))
    tts.add(f"B{L}_ln_in", g("input_layernorm.weight"))
    tts.add(f"B{L}_ln_postatt", g("post_attention_layernorm.weight"))
    tts.add(f"B{L}_ln_preff", g("pre_feedforward_layernorm.weight"))
    tts.add(f"B{L}_ln_postff", g("post_feedforward_layernorm.weight"))
    tts.add(f"B{L}_gate", g("mlp.gate_proj.weight").T, GEMM_DT)
    tts.add(f"B{L}_up", g("mlp.up_proj.weight").T, GEMM_DT)
    tts.add(f"B{L}_down", g("mlp.down_proj.weight").T, GEMM_DT)
tts.add("B_norm", T("backbone.norm.weight"))

# ── CAS encoder (1-layer t5gemma) ────────────────────────────────────────────
E = "embed_subword."
tts.add("cas_char_emb", T(E + "embed_tokens.weight"))  # [257, 1152] f32
for nm, key in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")]:
    tts.add(f"C0_{nm}", T(E + f"backbone.encoder.layers.0.{key}.weight").T, GEMM_DT)
for nm, key in [
    ("ln_preatt", "pre_self_attn_layernorm"),
    ("ln_postatt", "post_self_attn_layernorm"),
    ("ln_preff", "pre_feedforward_layernorm"),
    ("ln_postff", "post_feedforward_layernorm"),
]:
    tts.add(f"C0_{nm}", T(E + f"backbone.encoder.layers.0.{key}.weight"))
tts.add("C0_gate", T(E + "backbone.encoder.layers.0.mlp.gate_proj.weight").T, GEMM_DT)
tts.add("C0_up", T(E + "backbone.encoder.layers.0.mlp.up_proj.weight").T, GEMM_DT)
tts.add("C0_down", T(E + "backbone.encoder.layers.0.mlp.down_proj.weight").T, GEMM_DT)
tts.add("cas_norm", T(E + "backbone.encoder.norm.weight"))
tts.add("cas_proj", T(E + "proj_embedding.weight").T)  # [1152, 1152] f32
tts.add("cas_cont_emb", T(E + "subword_flag_emb.cont_emb.weight"))  # [2, 1152]
tts.add("cas_special_emb", T(E + "bos_eos_emb.special_emb.weight"))  # [3, 1152]
cont = T(E + "subword_flag_emb.is_continuation")  # [131073] i64
spec = T(E + "bos_eos_emb.special_flags")  # [131072] i64
tts.add("cont_flags", cont, "u8")
tts.add("special_flags", spec, "u8")

# ── fusion + embeddings + MoG head (all f32 — decision-critical) ─────────────
tts.add("bos_emb", T("bos_emb"))
tts.add("null_emb", T("null_emb"))
tts.add("embed_code", T("embed_code.weight").T)  # [512, 1152]
tts.add("gf_audio_w", T("gated_fusion_audio_text.audio_proj.weight").T)
tts.add("gf_audio_b", T("gated_fusion_audio_text.audio_proj.bias"))
tts.add("gf_text_w", T("gated_fusion_audio_text.text_proj.weight").T)
tts.add("gf_text_b", T("gated_fusion_audio_text.text_proj.bias"))
tts.add("gf_gate", T("gated_fusion_audio_text.gate"))
tts.add("gf_res", T("gated_fusion_audio_text.residual_scale").reshape(1))
tts.add("gf_norm", T("gated_fusion_audio_text.final_norm.weight"))
tts.add("aria_latent", tf.get_tensor("tts_model.audio_prompt_latents.Aria")[0])  # [37, 1152]

for i in range(3):
    m = lambda s: T(f"mog_head.mlp_stack.{i}.{s}")
    tts.add(f"M{i}_pre", m("pre_norm.weight"))
    tts.add(f"M{i}_post", m("post_norm.weight"))
    tts.add(f"M{i}_gate", m("mlp.gate_proj.weight").T)
    tts.add(f"M{i}_up", m("mlp.up_proj.weight").T)
    tts.add(f"M{i}_down", m("mlp.down_proj.weight").T)
tts.add("M_norm", T("mog_head.mlp_stack.3.weight"))
tts.add("M_logits", T("mog_head.proj_logits.weight").T)  # [1152, 1024]
tts.add("M_logs", T("mog_head.proj_logs.weight").reshape(-1))  # [1152]
tts.add("M_else", T("mog_head.proj_else.weight").T)  # [1152, 512]
# proj_mus kept in torch row layout [65536, 1152]: component i = rows 64i..64i+64
tts.add("M_mus", T("mog_head.proj_mus.weight"))
tts.add("M_lowmat", T("mog_head.low_mat"))  # [1024, 512, 64]
tts.write(os.path.join(args.out, "tts"))

# ── codec: PRVQ codebooks + Latent2Wav decoder ───────────────────────────────
codec = Bin()
mus = np.stack([C(f"prvq.mus_list.{i}") for i in range(31)], 0)  # [31, 1024, 512]
assert np.array_equal(mus, T("rvq_embs")), "codec PRVQ codebooks != tts rvq_embs"
codec.add("rvq_embs", mus)
# decoder layer order: 0 ConvT(512→1536,k9 s9) | 1-3 ConvNeXt1536 | 4 ConvT(1536→768,k7 s7)
# | 5-7 ConvNeXt768 | 8 ConvT(768→384,k7 s7) | 9-11 ConvNeXt384 | 12 Conv1d(384→18,k1)
for i in [0, 4, 8, 12]:
    codec.add(f"D{i}_w", C(f"decoder.layers.{i}.weight"))  # ConvT [in,out,k]; final [18,384,1]
for i in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
    d = lambda s: C(f"decoder.layers.{i}.{s}")
    codec.add(f"D{i}_dw_w", d("dwconv.weight"))
    codec.add(f"D{i}_dw_b", d("dwconv.bias"))
    codec.add(f"D{i}_ln_w", d("norm.weight"))
    codec.add(f"D{i}_ln_b", d("norm.bias"))
    codec.add(f"D{i}_pw1_w", d("pwconv1.weight")[:, :, 0].T)  # [C, 4C] for x@W
    codec.add(f"D{i}_pw1_b", d("pwconv1.bias"))
    codec.add(f"D{i}_pw2_w", d("pwconv2.weight")[:, :, 0].T)  # [4C, C]
    codec.add(f"D{i}_pw2_b", d("pwconv2.bias"))
codec.write(os.path.join(args.out, "codec"))

# ── tokenizer: BPE vocab/merges/regex + derived char vocab ───────────────────
from huggingface_hub import hf_hub_download

tok_json = json.load(open(hf_hub_download("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "tokenizer.json")))
vocab_map = tok_json["model"]["vocab"]  # token → id
id2tok = [None] * len(vocab_map)
for t, i in vocab_map.items():
    id2tok[i] = t
assert all(t is not None for t in id2tok)
singles = sorted([t for t in vocab_map if len(t) == 1], key=lambda t: vocab_map[t])
char_vocab = {c: i for i, c in enumerate(singles)}  # 256 byte-level chars
assert len(char_vocab) == 256

# verify the checkpoint-baked flags match a recomputation from token strings
cont_ref = np.array([0 if (t.startswith("Ġ") or t.startswith("▁") or t.startswith("<")) else 1 for t in id2tok] + [0])
assert np.array_equal(cont_ref, cont), "is_continuation flags mismatch vs checkpoint"
# Checkpoint quirk (intentional keep): the baked special_flags put the "EOS"
# embedding on <SPECIAL_12> (id 12 — the runtime text PAD), not "</s>" (id 2).
# BOSEOSEmbedding was constructed from the raw HF tokenizer (eos=<SPECIAL_12>)
# before NeMo's eos_token="</s>" override; the trained model therefore adds the
# EOS special embedding on every PAD frame. The torch reference reproduces this
# via load_state_dict, so the export ships the checkpoint values verbatim.
spec_ref = np.zeros(len(id2tok), np.int64)
spec_ref[vocab_map["<s>"]] = 1
spec_ref[vocab_map["<SPECIAL_12>"]] = 2
assert np.array_equal(spec_ref[: len(spec)], spec), "bos/eos special flags mismatch vs checkpoint"

pre = tok_json["pre_tokenizer"]["pretokenizers"][0]
assert pre["type"] == "Split" and tok_json["pre_tokenizer"]["pretokenizers"][1]["type"] == "ByteLevel"
json.dump(
    {
        "vocab": id2tok,
        "merges": [f"{a} {b}" for a, b in tok_json["model"]["merges"]],
        "ignoreMerges": bool(tok_json["model"].get("ignore_merges")),
        "splitRegex": pre["pattern"]["Regex"],
        "chars": singles,
    },
    open(os.path.join(args.out, "tokenizer.json"), "w"),
    ensure_ascii=False,
)

# ── engine config: geometry + warmup arrays (mirrors the torch reference) ────
warm_ids = [12] * 36 + [2]  # set_init_inputs: pad×36 + eos (see reference harness)
config = {
    "hidden": 1152,
    "heads": 16,
    "headDim": 72,
    "ffDim": 4608,
    "layers": NL,
    "eps": 1e-6,
    "attnScale": 256 ** -0.5,
    "ropeThetaLocal": 10000.0,
    "ropeThetaGlobal": 1000000.0,
    # HF Gemma3 layer_types with _sliding_window_pattern=6 (transformers 5.15.1)
    "globalLayers": [5, 11, 17, 23],
    # softcap 0: T5Gemma's config carries attn_logit_softcapping=50, but the HF
    # sdpa attention path (used at training time and by the torch reference)
    # silently drops softcapping — verified: golden CAS attention matches the
    # uncapped computation to 5e-6 and diverges by 4.1 with the cap applied.
    "cas": {"softcap": 0.0, "ropeTheta": 10000.0, "attnScale": 256 ** -0.5, "normalizer": math.sqrt(1152)},
    "latent": 512,
    "numQuantizers": 31,
    "codebook": 1024,
    "numPredictions": 1024,
    "lowRank": 64,
    "minLogStd": -4.0,
    "exponent": 3.0,
    "numIter": 8,
    # per-iteration quantizer counts of the 8-step unmasking schedule
    # (torch: ceil((1-linspace(0,1,9)[:-1]^3)^(1/3) * 31) diffs) — baked to keep
    # the JS side off float-precision cliffs at the ceil()
    "unmaskKs": [0, 0, 0, 1, 1, 3, 4, 22],
    "guidanceScale": 0.2,
    "topP": 0.95,
    "noiseScale": 0.001,
    "textBos": 1,
    "textEos": 2,
    "textPad": 12,
    "speechPad": 1024,
    "sampleRate": 22050,
    "wavToTokenRatio": 1764,
    "nFft": 16,
    "hop": 4,
    "codecChannels": [1536, 768, 384],
    "codecRates": [9, 7, 7],
    "warmSubwordIds": warm_ids,
    "warmSubwordMask": [False] * 35 + [True, True],
    "warmAudioMask": [False] * 36 + [True],
    "warmFrames": 37,
    "silenceTokens": tf.get_tensor("tts_model.codec_silence_tokens").tolist(),
}
json.dump(config, open(os.path.join(args.out, "config.json"), "w"))
print(f"tokenizer + config written")

# ── golden vectors for the parity gate (reuses the manifest+bin loader) ──────
if args.golden:
    z = np.load(args.golden)
    meta = json.loads(str(z["meta"]))
    # config warmup arrays must match what the reference actually used
    assert z["warm_subword_ids"].tolist() == warm_ids
    assert z["warm_subword_mask"].astype(bool).tolist() == config["warmSubwordMask"]
    assert z["warm_audio_mask"].astype(bool).tolist() == config["warmAudioMask"]
    assert z["silence_tokens"].tolist() == config["silenceTokens"]
    g = Bin()
    g.add("next_subword_ids", z["next_subword_ids"].astype(np.float32))
    g.add("codes_parity", z["codes_parity"].astype(np.float32))
    g.add("wav_parity", z["wav_parity"])
    g.add("warm_hidden_cond", z["warm_hidden_cond"])
    g.add("warm_hidden_uncond", z["warm_hidden_uncond"])
    g.add("step_hidden_cond", z["step_hidden_cond"])
    g.add("step_hidden_uncond", z["step_hidden_uncond"])
    g.add("cas_tokens", z["cas_tokens"].astype(np.float32))
    g.add("cas_embs", z["cas_embs"])
    g.add("codec_latents_parity", z["codec_latents_parity"])
    g.add("aria_latent_ref", z["aria_latent"])
    g.man["_meta"] = meta
    g.write(os.path.join(args.out, "golden"))
print(f"wrote {args.out}/")
