# Kokoro Chinese — g2pW frontend port

**Goal:** match FluidAudio's KokoroAne-zh quality in the browser. kokoro-js does
Chinese via **espeak-ng** (no polyphone disambiguation) and only loads English
voices — verified broken (18 hanzi → 13.7 s, English voice). We bypass espeak and
feed our own IPA phonemes + a Chinese voice into the Kokoro model.

## Pipeline (port of misaki `zh.py`, the same G2P Kokoro-zh was trained on)

```
text → jieba segmentation
     → pypinyin (per word) + g2pW polyphone disambiguation → pinyin (TONE3)
     → pinyin_to_ipa (misaki transcription.py) + retone + tone-sandhi → IPA
     → Kokoro tokenizer (IPA→ids) + zh voice style vector
     → kokoro-js .model(input_ids, style, speed) → audio   (espeak bypassed)
```

## Exact sources to port (all identified)

| Piece | Source | Browser form |
|---|---|---|
| segmentation | `jieba` | [`jieba-wasm`](https://www.npmjs.com/package/jieba-wasm) |
| hanzi→pinyin dict | `pypinyin` | `pinyin-pro` (npm) or port pypinyin dict |
| polyphone | **g2pW** — `G2PWModel-v2-onnx` ([GitYCC/g2pW](https://github.com/GitYCC/g2pW)) | ONNX on ORT (WASM) + BERT tokenizer + `POLYPHONIC_CHARS.txt` (FluidAudio ships it at `kokoro-82m-coreml/ANE-zh/g2pw/`) |
| pinyin→IPA | misaki `transcription.py` (284 ln; INITIALS/FINALS IPA tables + TONE_MAPPING) | port to JS (finite table) |
| tone sandhi + retone | misaki `zh_frontend.py` + `tone_sandhi.py` + `zh.py` | port to JS |
| phoneme injection | kokoro-js exposes `.model` (StyleTextToSpeech2Model) + `.tokenizer` | build input_ids from IPA, skip espeak |
| zh voices | `onnx-community/Kokoro-82M-v1.1-zh-ONNX` (`zf_/zm_`) or FluidAudio `ANE-zh/voices/*.bin` | load style vector |

Reference files pulled to `/tmp/misaki_{zh,transcription,zh_frontend}.py`.

## Phases

1. ✅ **Phoneme-injection path** (foundational, DONE): kokoro-js exposes
   `generate_from_ids(input_ids)` — `generate()` is just `espeak → tokenizer →
   generate_from_ids`. So `tokenizer(phonemeString) → generate_from_ids` injects
   our own phonemes, skipping espeak. Verified via **lexicon-first English**:
   `us_lexicon_cache.json` phonemized a sentence → 3.42 s natural speech.
   Implemented in the engine (`synthFromPhonemes`, `lexicon.js`), espeak fallback
   for <95% coverage. Chinese reuses `synthFromPhonemes` with a g2pW phoneme string.
2. **pinyin→IPA port** (`transcription.py`): verify JS output byte-matches Python
   misaki over a syllable set (installable via `pip install misaki[zh]`).
3. **hanzi→pinyin** via pinyin-pro + **jieba-wasm**; wire tone-sandhi/retone.
4. **g2pW ONNX** for polyphones (replaces pinyin-pro's dictionary heuristic).
5. Voice packs + UI.

## Caveat
Chinese pronunciation quality is **not verifiable headless** (no zh ASR here) —
phases 2–4 must be checked against Python misaki output (unit) and in-browser by
ear. Phase 1 verifies mechanically (duration, token validity).
