# OPT-0082 — Production-BF16 compact semantic head and sampler

## Status

- Evidence: `inconclusive` (exact and median-positive; literal paired-win gate failed)
- Disposition: `benchmark-only`
- Production integration: not authorized or performed
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact constrained-domain elimination
- Allocation baseline: pushed `main` commit
  `2c062119eb36e4cdf8ae13c7275181857b016244`

No OPT-0082 code, harness, timing sample, package change, selector, or
production change existed when this experiment was allocated.

## Result

The current reference-BF16 Chrome/M3 developer screen passed every executed
raw-U32 retained-logit, filtered-logit, sampler/cursor, NaN, runtime-event, and
cleanup check. Across 36 balanced complete-token pairs, full-vocabulary median
wall was `379.15000009536743 ms` and compact median wall was `340.5 ms`, a
`1.1135095450671584x` median speedup and `38.65000009536743 ms` saving. All
three cache-position compact medians were lower and the aggregate saving
projects to `34.78500008583069 s` over 900 tokens.

Compact nevertheless won only `26/36` paired tokens, below the frozen `30/36`
gate. This developer screen also had no attached thermal trace. The literal
fast gate therefore failed. Per the declared stop rule, the 150-code
trajectory, product gate, and production selector were not run, and ordinary
production remains Arm A. The benchmark-only range/replay machinery is
retained because a materially changed exact sampler or planner kernel may
compound this verified mechanism under a new experiment; an unchanged rerun
is not authorized merely to seek a better sample.

Compact result: [`../results/OPT-0082/result.json`](../results/OPT-0082/result.json).
Raw receipt: ignored
`optimization/artifacts/OPT-0082/raw/developer-ab-receipt.json`, SHA-256
`95f705daef899fb8ebf27cfb1c47664cc8bcaaa75a91486408ba650329ef8556`.

## Hypothesis

The default semantic planner can emit only the 64,000 consecutive audio-code
tokens or the separately forced EOS token. Production nevertheless computes,
copies, reconstructs, masks, and samples all 217,204 vocabulary logits for
both CFG rows on every semantic step. Scoring only the exact admissible rows
and sampling them in ascending global-token order should preserve every
retained reference-BF16 logit, browser-v1 sampler operation, selected token,
and Philox cursor while removing most tied-head traffic and synchronous host
work.

This is the production-BF16 successor to OPT-0012. OPT-0012's raw-FP16 result
is mechanism evidence only; its trajectory is not the approved production
oracle and cannot authorize this integration.

## Frozen scope

Arm A is the current production reference-BF16 semantic M2 path:

- five complete tied-embedding head shards covering vocabulary rows
  `0..217203`;
- two complete FP32 logit rows read back and reconstructed in global-token
  order; and
- the accepted browser-v1 full-vocabulary sampler.

Arm B changes only semantic M2 head/readback/sampling:

- regular candidates are global IDs `151669..215668`;
- forced EOS is global ID `151645`;
- tied weights remain the authenticated five resident embedding shards;
- each selected row is read from its existing source-row-major BF16 storage;
- head products and accumulation remain FP32 and increasing-K;
- candidate logits are ordered by ascending global token ID before every
  stable tie break and categorical scan; and
- one random word is consumed for every emitted token, including forced EOS.

CoT M1, semantic guidance, prefill/decode transformer layers, KV caching,
tokenizer behavior, metadata FSM, softmax polynomial, top-p, temperature,
repetition penalty, package bytes, and every downstream model remain
unchanged. There is no approximate logit pruning, top-k substitution, sampler
change, quantization, new package, or fallback.

The ordinary production selector must remain Arm A until the integration gate
passes. Diagnostic selection must fail closed outside the exact
reference-BF16, two-row, browser-v1 semantic tuple.

## Static opportunity

For each ordinary semantic token, the full M2 head consumes 217,204 rows per
CFG row. The compact regular-code head consumes 64,000 rows per row, plus one
EOS row when the state requires it. The regular path therefore removes
153,204 rows, or 70.53461262223531% of full-head logical weight and logit
traffic. Full production readback is `2 * 217204 * 4 = 1,737,632` bytes before
alignment; the 64,000-code compact readback is `512,000` bytes before status
and alignment.

## Fast gate

Use one authenticated, already-loaded production planner owner and interleave
A/B work at short, middle, and long semantic cache positions. Setup, package
acquisition, upload, compilation, prefill construction, and untimed reference
generation are excluded from candidate timing but retained in the receipt.

Before any accepted timing:

1. Compare every compact conditional/unconditional FP32 logit raw-U32 against
   the corresponding global row from Arm A, including EOS, at all three cache
   positions. Require zero mismatches, no NaNs, and deterministic reruns.
2. From identical mapped bytes and random words, require identical allowed
   masks, CFG logits, top-p keep decisions, weights, selected global token,
   and next Philox cursor for regular and forced-EOS states.
3. Run at least one complete 150-code-plus-EOS semantic trajectory per arm.
   Require identical emitted tokens, semantic values, draw span, and all
   retained compact logits. CoT is outside this experiment.
4. Prove bounded cancellation, no post-abort submission/readback, balanced
   maps, and zero live buffers after idempotent cleanup.

Timing uses at least six balanced same-owner A/B or B/A pairs per cache
position after a nominal thermal start. Record complete token wall, model
wall, tied-head wall, readback/map/reconstruction, sampling wall, command
counts, bytes, and heartbeat gaps. Do not sum overlapping intervals.

Arm B is a material win only if all correctness/lifecycle gates pass and:

- B wins at least 30 of 36 same-round complete-token pairs;
- every cache-position B median is below its paired A median;
- aggregate complete-token speedup is at least `1.10x`; and
- the median per-token saving projects to at least 30 seconds over 900
  semantic decode tokens.

A large, consistent paired effect after a nominal start is decision-useful
with the complete thermal trace disclosed. Do not rerun unchanged work merely
to obtain a cooler-looking trace.

## Integration gate

A passing fast gate authorizes wiring Arm B into the exact default production
semantic M2 tuple. Before selection, run the committed planner unit/contract
tests and a package-native production-BF16 default-CoT planner trajectory with
identical caption/metadata, 150 semantic codes plus EOS, token sequence, draw
cursor, and downstream conditioning inputs. Then run one planner-enabled
product correctness gate through final latent, raw waveform, normalized WAV,
cancellation, lifecycle, and output release.

Product timing may be collected, but integration depends on exactness and the
fast-gate material win rather than a noisy multi-minute product A/B. Any
production mismatch retains Arm A and closes OPT-0082 without selection.

## Stop conditions

Stop without integration for any retained-logit/token/cursor mismatch,
unbounded resource behavior, missing target-browser evidence, aggregate
speedup below `1.10x`, projected 180-second saving below 30 seconds, or fewer
than 30/36 paired wins. Do not tune candidate-domain size, ordering, softmax,
or thresholds after timing begins under this ID.

## Authority

- Original planner attribution:
  [OPT-0010](OPT-0010-package-native-planner-token-profiler.md)
- Raw-FP16 mechanism evidence:
  [OPT-0012](OPT-0012-compact-semantic-head-sampling.md)
- Approved production behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
