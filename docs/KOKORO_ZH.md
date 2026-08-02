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
2. ✅ **pinyin→IPA** (DONE, better than porting): instead of reimplementing
   `transcription.py`, **precomputed the full table** from misaki's own
   `pinyin_to_ipa` + `retone` over every valid Mandarin syllable (1549 entries,
   `pinyin-ipa.json`) → the IPA is byte-exact to what Kokoro-zh trained on, zero
   port risk. `chinese-g2p.js` is a pure lookup.
3. ✅ **hanzi→pinyin** via `pinyin-pro` (tone-num, context polyphones, segments,
   nonZh passthrough). Verified: `你好世界` → `ni↓xau̯↓ʂɨ↘ʨje↘` (matches oracle);
   full sentence → 3.98 s natural audio (was 13.72 s broken). Wired into engine.
4. 🚧 **g2pW ONNX** — accuracy upgrade for polyphones (replaces pinyin-pro's
   dictionary heuristic). Optional; `G2PWModel-v2-onnx` + `POLYPHONIC_CHARS.txt`.
5. 🚧 **Native zh voices** (`zf_/zm_`) — kokoro-js loads only English voices for
   the zh model, so we currently synth with an English voice (correct Chinese
   phonemes, English timbre). Load the zf_ style vector directly to fix timbre.

Quality (pronunciation/prosody) still needs an ear/zh-ASR to judge; the pinyin→IPA
stage is verified exact, and audio length is now natural.

## Caveat
Chinese pronunciation quality is **not verifiable headless** (no zh ASR here) —
phases 2–4 must be checked against Python misaki output (unit) and in-browser by
ear. Phase 1 verifies mechanically (duration, token validity).
