# VoiceChat-11B TTS torch reference harness — golden vectors for the tts-voicechat
# browser port. Runs the REAL checkpoint through the REAL NeMo Speech-repo module
# code (RVQEARTTSModel + RVQVAEModel imported by file path with tiny sys.modules
# stubs for the nemo package plumbing: fp32_precision / set_model_dict_for_partial_init
# / logging / AutoTokenizer-wrapper). The DuplexEARTTS *driver* (set_init_inputs
# speaker_name path + offline_inference loop) is transcribed 1:1 from
# Speech/nemo/collections/speechlm2/models/duplex_ear_tts.py (lines 993-1144,
# 1204-1269, 1303-1453) — it is orchestration only; all weight math runs in the
# imported module code.
#
# Two tracks:
#   • PARITY (deterministic): CFG kept ON (guidance_scale 0.2 — the shipped
#     default, and deterministic), top_p 0.95 kept (deterministic filter),
#     noise_scale 0, gumbel_like patched to zeros → mixture argmax-component mean.
#   • SAMPLED (realism): torch.manual_seed(0), config defaults
#     (noise_scale 0.001, gumbel sampling on).
#
# Invocation (from the fluidaudio-web repo root):
#   /tmp/vc-tts-env/bin/python scripts/voicechat-tts-reference.py \
#       --src ~/Documents/models/voicechat-11b --out /tmp/voicechat-tts-golden.npz
# Env: python 3.12, torch 2.12.1, transformers 5.15.1, numpy 2.5.2, librosa, omegaconf.

import argparse
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_grad_enabled(False)

p = argparse.ArgumentParser()
p.add_argument("--src", default=os.path.expanduser("~/Documents/models/voicechat-11b"))
p.add_argument("--out", default="/tmp/voicechat-tts-golden.npz")
p.add_argument("--text", default="Hello, do you know what color the sky is?")
p.add_argument("--frames", type=int, default=50, help="total generation frames (80ms each)")
p.add_argument("--debug-steps", type=int, default=6, help="steps with per-step hidden/latent goldens")
args = p.parse_args()
SRC = os.path.expanduser(args.src)
SPEECH = os.path.join(SRC, "Speech")

# ── nemo stubs so the module files import unmodified ─────────────────────────
def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


from contextlib import contextmanager


@contextmanager
def fp32_precision():
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.amp.autocast(device_type="cpu", dtype=torch.float32):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


def set_model_dict_for_partial_init(pretrained_dict, model_dict):
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(v, "shape") and v.shape != model_dict[k].shape:
            del pretrained_dict[k]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    return model_dict


class NemoTokenizerShim:
    """Mirrors the nemo AutoTokenizer surface consumed by ear_tts_model.py +
    the DuplexEARTTS driver (bos <s>, eos </s>, pad <SPECIAL_12> overrides)."""

    def __init__(self, name="nvidia/NVIDIA-Nemotron-Nano-9B-v2"):
        from transformers import AutoTokenizer as HFT

        self.tokenizer = HFT.from_pretrained(name, use_fast=True)
        self.bos_token, self.eos_token, self.pad_token = "<s>", "</s>", "<SPECIAL_12>"

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        id2v = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2v[i] for i in range(len(id2v))]

    def ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def text_to_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @property
    def bos(self):
        return self.tokenizer.convert_tokens_to_ids(self.bos_token)

    @property
    def eos(self):
        return self.tokenizer.convert_tokens_to_ids(self.eos_token)

    @property
    def pad(self):
        return self.tokenizer.convert_tokens_to_ids(self.pad_token)


for name in [
    "nemo",
    "nemo.collections",
    "nemo.collections.common",
    "nemo.collections.common.tokenizers",
    "nemo.collections.speechlm2",
    "nemo.collections.speechlm2.parts",
    "nemo.collections.speechlm2.parts.precision",
    "nemo.collections.speechlm2.parts.pretrained",
    "nemo.utils",
]:
    _mod(name)
sys.modules["nemo.collections.common.tokenizers"].AutoTokenizer = NemoTokenizerShim
sys.modules["nemo.collections.speechlm2.parts.precision"].fp32_precision = fp32_precision
sys.modules["nemo.collections.speechlm2.parts.pretrained"].set_model_dict_for_partial_init = (
    set_model_dict_for_partial_init
)


class _Log:
    def info(self, *a, **k):
        pass

    warning = info


sys.modules["nemo.utils"].logging = _Log()


def load_by_path(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


MOD = os.path.join(SPEECH, "nemo/collections/speechlm2/modules")
vae = load_by_path("ear_tts_vae_codec", os.path.join(MOD, "ear_tts_vae_codec.py"))
etm = load_by_path("ear_tts_model", os.path.join(MOD, "ear_tts_model.py"))

# ── build models from config.json + sliced safetensors ───────────────────────
from omegaconf import DictConfig
from safetensors import safe_open

cfg_all = json.load(open(os.path.join(SRC, "config.json")))
sg = cfg_all["model"]["speech_generation"]["model"]
tts_cfg = DictConfig(sg["tts_config"])
codec_cfg = DictConfig(sg["codec_config"])

tok = NemoTokenizerShim()
assert (tok.bos, tok.eos, tok.pad) == (1, 2, 12), (tok.bos, tok.eos, tok.pad)

tts = etm.RVQEARTTSModel(tts_cfg, tokenizer=tok).eval()
codec = vae.RVQVAEModel(codec_cfg).eval()


def load_st(path, prefix):
    sd = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix) :]] = f.get_tensor(k)
    return sd


tts_sd = load_st(os.path.join(SRC, "components/tts.safetensors"), "tts_model.tts_model.")
# RVQEARTTSModel overrides load_state_dict without returning the result — call
# the plain nn.Module implementation so we can assert coverage.
missing, unexpected = torch.nn.Module.load_state_dict(tts, tts_sd, strict=False)
# rvq_embs is registered by set_rvq_embs later; embed-side flag buffers come from ckpt
assert not [m for m in missing if m != "rvq_embs"], missing
assert not [u for u in unexpected if u != "rvq_embs"], unexpected  # registered via set_rvq_embs below

codec_sd = load_st(os.path.join(SRC, "components/codec.safetensors"), "tts_model.audio_codec.")
codec.load_state_dict(codec_sd, strict=True)

tts.set_rvq_embs(torch.stack([x.detach() for x in codec.prvq.mus_list], 0))
with safe_open(os.path.join(SRC, "components/tts.safetensors"), "pt") as f:
    assert torch.equal(f.get_tensor("tts_model.tts_model.rvq_embs"), tts.rvq_embs), "rvq_embs != codec prvq mus"
    aria_latent = f.get_tensor("tts_model.audio_prompt_latents.Aria").float()  # [1, 37, 1152]
    silence_tokens = f.get_tensor("tts_model.codec_silence_tokens").long()  # [31]
    control_codes = f.get_tensor("tts_model._control_codes").long()  # [pad,eos,bos] ids 1024/1025/1026... see below

CODEBOOK = tts_cfg.codebook_size  # 1024
NUM_Q = tts_cfg.num_quantizers  # 31
SPEECH_PAD = CODEBOOK  # 1024 (speech_pad_id: codebook+0? see DuplexEARTTS: pad=cb, eos=cb+1, bos=cb+2)
SPEECH_EOS = CODEBOOK + 1
SPEECH_BOS = CODEBOOK + 2
assert control_codes.tolist() == [SPEECH_BOS, SPEECH_EOS, SPEECH_PAD]
TEXT_BOS, TEXT_EOS, TEXT_PAD = tok.bos, tok.eos, tok.pad
SR = 22050
SPF = codec_cfg.wav_to_token_ratio  # 1764 samples/frame
PROMPT_SEC = 3.0

# cross-check the baked silence frame against a fresh codec encode of silence
sil_audio = torch.zeros(1, 10 * SR)
sil_pad = SPF * math.ceil(sil_audio.shape[1] / SPF)
sil_audio = F.pad(sil_audio, (0, sil_pad - sil_audio.shape[1]))
sil_codes, _ = codec.encode(sil_audio.unsqueeze(1), torch.tensor([sil_audio.shape[1]]))
from collections import Counter

combo, _ = Counter(tuple(r.tolist()) for r in sil_codes[0]).most_common(1)[0]
assert list(combo) == silence_tokens.tolist(), "baked codec_silence_tokens mismatch"

# ── set_init_inputs (speaker_name="Aria" path), duplex_ear_tts.py:993-1144 ───
prompt_audio_size = int(((PROMPT_SEC * SR) // SPF) * SPF)  # 65268 = 37 frames
speaker_audio = torch.zeros(1, prompt_audio_size)
prompt_audio = speaker_audio.clone()
prompt_audio[:, -int(SPF * 2) :] = 0
prompt_pad_frames = int(prompt_audio_size // SPF)  # 37

text_prompt = torch.tensor([TEXT_EOS], dtype=torch.long)
prompt_audio_text_pad = torch.full((prompt_pad_frames,), TEXT_PAD, dtype=torch.long)
prompt_audio_text_pad[-1] = TEXT_EOS
target_text_tokens = torch.cat([text_prompt, prompt_audio_text_pad]).unsqueeze(0)  # [1, 38]

pad_audio = torch.zeros(1, text_prompt.size(-1) * SPF)
target_audio = torch.cat([pad_audio, prompt_audio], dim=1)  # [1, 38*1764]
code, _ = codec.encode(target_audio.unsqueeze(1), torch.tensor([target_audio.shape[1]]))  # [1, 38, 31]

non_prompt_mask = torch.zeros_like(target_text_tokens)
non_prompt_mask[:, -2:] = 1
subword_mask = torch.zeros_like(target_text_tokens)
subword_mask[:, -3:] = 1
code[:, 0] = SPEECH_PAD
subword_ids = F.pad(target_text_tokens[:, 1:], [0, 1], value=0)
pos = non_prompt_mask.float().argmax(dim=1)  # 36
code[torch.arange(1), pos] = SPEECH_PAD

init_inputs = {
    "code": code[:, :-1],
    "audio_mask": non_prompt_mask.bool()[:, :-1],
    "context_hidden_state": None,
    "subword_ids": subword_ids[:, :-1],
    "subword_mask": subword_mask.bool()[:, :-1],
    "non_prompt_mask": non_prompt_mask.bool()[:, :-1],
    "audio_prompt_latent": aria_latent,
}

# ── next_subword_ids: [bos, tokens…, pad…, eos], one per 80ms frame ──────────
text_ids = tok.text_to_ids(args.text)
T = args.frames
assert T >= len(text_ids) + 2, "frames too small for the text"
next_subword_ids = torch.full((1, T), TEXT_PAD, dtype=torch.long)
next_subword_ids[0, 0] = TEXT_BOS
next_subword_ids[0, 1 : 1 + len(text_ids)] = torch.tensor(text_ids)
next_subword_ids[0, -1] = TEXT_EOS


# ── offline_inference (duplex_ear_tts.py:1303-1453 + 1204-1269) ──────────────
def run(generation_config, deterministic, capture=None):
    if deterministic:
        etm.gumbel_like = lambda t, eps=1e-8: torch.zeros_like(t)
    else:
        etm.gumbel_like = _orig_gumbel
    torch.manual_seed(0)

    ii = {k: v for k, v in init_inputs.items()}
    ii.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})
    outputs = tts(**ii)
    if capture is not None:
        h = outputs.hidden_states  # [2, 37, 1152] cond+uncond
        capture["warm_hidden_cond"] = h[0].numpy()
        capture["warm_hidden_uncond"] = h[1].numpy()
    code_t = ii["code"][:, -1:]  # skip_first_code_prediction_on_init=True
    past_key_values = outputs["past_key_values"]
    first_context_subword_id = ii["subword_ids"][:, -1].unsqueeze(-1)
    gen_codes = torch.zeros(1, T, NUM_Q, dtype=torch.long)

    step_h_cond, step_h_uncond = [], []
    for i in range(T):
        current_subword_id = next_subword_ids[:, i].unsqueeze(-1)
        prev_subword_id = first_context_subword_id if i == 0 else next_subword_ids[:, i - 1].unsqueeze(-1)
        # infer_codes_one_step: inference_force_speech_silence_on_eos=True
        prev_audio_tokens = code_t
        if (current_subword_id == TEXT_EOS).any():
            sil = silence_tokens.view(1, 1, -1).expand(prev_audio_tokens.shape)
            prev_audio_tokens = torch.where(current_subword_id.unsqueeze(-1) == TEXT_EOS, sil, prev_audio_tokens)
        out = tts(
            code=prev_audio_tokens,
            context_hidden_state=None,
            subword_ids=current_subword_id,
            subword_mask=torch.ones_like(current_subword_id, dtype=torch.bool),
            past_key_values=past_key_values,
            use_cache=True,
            guidance_enabled=True,
            generation_config=generation_config,
            ignore_eos_flag_stop=True,
        )
        code_t, past_key_values = out["codes"], out["past_key_values"]
        gen_codes[:, i] = code_t.squeeze(1)
        if capture is not None and i < args.debug_steps:
            step_h_cond.append(out.hidden_states[0, -1].numpy())
            step_h_uncond.append(out.hidden_states[1, -1].numpy())

    if capture is not None:
        capture["step_hidden_cond"] = np.stack(step_h_cond)
        capture["step_hidden_uncond"] = np.stack(step_h_uncond)

    # decode audio (offline path): replace control codes, batch codec decode
    codes = gen_codes.clone()
    sil = silence_tokens.view(1, 1, -1).expand(codes.shape[0], 1, -1)
    codes = torch.where(torch.isin(codes, control_codes), sil, codes)
    wav, wav_len = codec.decode(codes, torch.tensor([T]))
    return gen_codes[0], wav.squeeze(0).squeeze(0)


_orig_gumbel = etm.gumbel_like

GC_PARITY = {"num_iter": 8, "guidance_scale": 0.2, "top_p_or_k": 0.95, "noise_scale": 0.0, "eos_threshold": -3.0}
GC_SAMPLED = {"num_iter": 8, "guidance_scale": 0.2, "top_p_or_k": 0.95, "noise_scale": 0.001, "eos_threshold": -3.0}

capture = {}
codes_det, wav_det = run(GC_PARITY, deterministic=True, capture=capture)
codes_det2, wav_det2 = run(GC_PARITY, deterministic=True)
assert torch.equal(codes_det, codes_det2), "parity track is not deterministic!"
codes_smp, wav_smp = run(GC_SAMPLED, deterministic=False)

# CAS embeddings for every distinct conditioning token (browser-side unit oracle)
distinct = sorted(set(next_subword_ids[0].tolist() + init_inputs["subword_ids"][0].tolist()))
cas_ids = torch.tensor(distinct).unsqueeze(0)
cas_embs = tts.embed_subword(cas_ids, torch.ones_like(cas_ids, dtype=torch.bool))[0]

# codec unit oracle: dequantized latents for the generated codes
lat_det = codec.dequantize(codes_det.unsqueeze(0))[0]  # [T, 512]

meta = {
    "text": args.text,
    "text_ids": text_ids,
    "frames": T,
    "debug_steps": args.debug_steps,
    "gc_parity": GC_PARITY,
    "gc_sampled": GC_SAMPLED,
    "text_bos": TEXT_BOS,
    "text_eos": TEXT_EOS,
    "text_pad": TEXT_PAD,
    "speech_pad": SPEECH_PAD,
    "sample_rate": SR,
    "wav_to_token_ratio": SPF,
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__.split("+")[0],
}
np.savez_compressed(
    args.out,
    meta=json.dumps(meta),
    next_subword_ids=next_subword_ids[0].numpy(),
    warm_code=init_inputs["code"][0].numpy(),
    warm_subword_ids=init_inputs["subword_ids"][0].numpy(),
    warm_subword_mask=init_inputs["subword_mask"][0].numpy(),
    warm_audio_mask=init_inputs["audio_mask"][0].numpy(),
    aria_latent=aria_latent[0].numpy(),
    silence_tokens=silence_tokens.numpy(),
    control_codes=control_codes.numpy(),
    codes_parity=codes_det.numpy(),
    wav_parity=wav_det.numpy(),
    codes_sampled=codes_smp.numpy(),
    wav_sampled=wav_smp.numpy(),
    cas_tokens=np.array(distinct, dtype=np.int64),
    cas_embs=cas_embs.numpy(),
    codec_latents_parity=lat_det.numpy(),
    **capture,
)
d = wav_det.numpy()
print(f"parity: {T} frames → {len(d)} samples ({len(d)/SR:.2f}s), peak {np.abs(d).max():.3f}, rms {np.sqrt((d**2).mean()):.4f}")
print(f"codes_parity[0,:8] = {codes_det[0,:8].tolist()}")
print(f"codes_parity[last,:8] = {codes_det[-1,:8].tolist()}")
print(f"golden → {args.out}")
