# Browser tokenizer contract

This directory implements the Stage 1 tokenizer path for the two pinned Qwen
branches. It is browser/worker-safe TypeScript and has no runtime dependency on
Transformers, `tokenizers`, Python, WASM, or a third-party regular-expression
engine.

## Pinned assets

`loadPinnedAceTokenizer` accepts `tokenizer.json`, `tokenizer_config.json`, and
`chat_template.jinja` as strings or `Blob`s. It requires the exact byte lengths
and SHA-256 identities emitted from the pinned snapshots:

- text: `ACE-Step/Ace-Step1.5` at
  `19671f406d603126926c1b7e2adc169acbcade22`, under
  `Qwen3-Embedding-0.6B/`;
- planner: `ACE-Step/acestep-5Hz-lm-0.6B` at
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`.

The package manifest authenticates these files already. The loader repeats the
identity check because a tokenizer revision changes model inputs and therefore
counts as model math. It then validates the relevant serialized structure:
NFC normalization, the Qwen split expression, byte-level pre/post processing,
BPE vocabulary and merge ranks, added-token flags and IDs, padding/EOS values,
and the complete planner audio-code declaration.

Both branches contain the same 151,643-token base BPE. The loader retains one
shared JavaScript copy even when both tokenizers are constructed. Planner audio
codes are validated individually on load but represented at runtime by the
contiguous relation

`<|audio_code_n|> -> 151669 + n`, for `0 <= n < 65535`.

This avoids retaining 65,535 redundant token strings and lookup entries. Loading
still temporarily materializes one authenticated tokenizer asset (at most about
24 MB) for JSON parsing; it never mirrors a weight package or a multi-gigabyte
buffer.

## Behavior

`AceQwenBpeTokenizer` implements:

- NFC normalization and the pinned Qwen Unicode pre-tokenizer;
- GPT-2/Qwen byte-to-Unicode mapping and ranked BPE merges;
- exact added-token recognition, including non-special `<think>` tokens;
- the text encoder's `tokenizer.json` post-processor, which appends token
  `151643` when `addSpecialTokens` is enabled;
- planner behavior, which does not append that post token;
- decoding with the tokenizer's UTF-8 replacement behavior and optional
  special-token skipping;
- right truncation with reserved post-token space, and left/right batch
  padding with explicit attention masks.

`renderAceQwenChat` implements the exact no-tools portion of the pinned chat
templates used by ACE: system/user/assistant messages, reasoning extraction,
the open assistant generation turn, and `enable_thinking=false`. The ACE planner
product path uses system + user messages followed by that open assistant turn.

`formatAceTextEncoderCaptionInput` and `formatAceTextEncoderLyricsInput`
preserve the pinned SFT wrappers. In particular, each wrapper contains an
explicit `<|endoftext|>` and the text tokenizer post-processor appends its own
token `151643` as well. The lyric input therefore ends in two consecutive
`151643` IDs; that is upstream behavior, not padding to be removed.

## Validation and intentional gaps

`test/tokenizer.test.ts` contains Python `tokenizers==0.22.1` and
Transformers `4.57.1` vectors for Unicode, whitespace, normalization, chat
formatting, audio-code boundaries, truncation/padding, and every instruction,
caption, and lyric field in `golden/fixtures/`. Full asset tests run when the
ignored converter output under `model/files-reference/` is present; small
dependency-free core tests always run.

Stage 1 does not currently expose pair encoding, token offsets, token-type IDs,
tool schemas/calls, tool responses, or multimodal/non-string chat content. None
is used by the scoped text encoder or ACE planner generation path. Adding one
requires new upstream parity vectors rather than treating the browser output as
an oracle.
