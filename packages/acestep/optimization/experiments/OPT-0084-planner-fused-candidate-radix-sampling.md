# OPT-0084 — Planner fused candidate-domain radix sampling

## Status

- Evidence: `positive` for the standalone sampler and package-native compound
  actual-browser gates
- Disposition: `pending-integration`; complete trajectory and planner-enabled
  product gates remain mandatory
- Production integration: not yet authorized
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact browser-v1 sampler implementation
- Allocation baseline: pushed `main` commit
  `7a078faee2ba2644394663863f73f80e5544e358`

No OPT-0084 implementation, selector, harness, timing sample, or production
change existed when this experiment was allocated. A read-only design
prototype measured sorting mechanisms but did not modify repository code.

The standalone actual-browser sampler gate and the registered package-native
compact-head compound escalation later passed. Integration gates remain
mandatory.

## First-principles basis

The accepted sampler currently reconstructs, copies, masks, and scans logical
vectors as large as the 217,204-token vocabulary several times per draw. The
64,000-token semantic compact path still allocates roughly thirteen large
typed arrays, creates about 3.25 MB of transient storage, performs roughly two
dozen linear passes, and uses a boxed JavaScript `number[]` comparator sort.
OPT-0010 attributed 85.9--279.2 ms per package-native full-vocabulary sample;
OPT-0082 retained a material but position-variable complete-token saving after
reducing the domain.

Disallowed entries are negative infinity and contribute nothing to the max,
FP32 sums, sorted finite order, or categorical traversal. Therefore an
ascending-global-ID candidate vector can be exactly equivalent without
materializing omitted entries. The only order-sensitive exception is the
top-p cumulative walk, which must follow descending logit and ascending token
ID ties.

A read-only M3/Node diagnostic over 64,000 finite logits measured medians of
8.18 ms for the current boxed comparator sort and 0.88 ms for a stable four
pass eight-bit typed-array radix, with identical order. This is directional
design evidence, not a browser or production performance claim.

## Frozen candidate

Arm A is the current accepted `ace-browser-softmax-v1` full/compact pipeline.
Arm B is one reusable candidate-domain workspace owned by the sampling cursor:

- combined/filtered FP32 logits;
- one FP32 exponent/weight buffer;
- two Uint32 radix-index buffers;
- one Uint8 active/keep mask; and
- fixed 256-bin radix counts/offsets.

Finite candidate ordinals begin in ascending global token-ID order. Convert
each binary32 value to a descending unsigned float key, canonicalizing both
signed zeros to one key, then perform exactly four stable eight-bit LSD passes.
The result must be descending logit with ascending global-ID ties. NaN and
positive infinity remain rejected and negative infinity is omitted.

The fused path must preserve these exact orders and boundaries:

1. validate model logits and parameters before committing a draw;
2. CFG and repetition penalty retain the current explicit `Math.fround`
   boundaries;
3. top-k keeps every threshold tie;
4. top-p exponent generation and its FP32 normalization sum traverse
   ascending global token IDs;
5. only the top-p cumulative keep walk uses radix logit order;
6. temperature is applied after top-p;
7. final exponent generation and FP32 normalization sum again traverse
   ascending global IDs; and
8. categorical binary64 total/cumulative traversals, strict `>`, final positive
   guard, Philox word, and cursor commit remain unchanged.

Sparse CoT constraints are normalized to unique ascending global IDs before
gather. Input-order duplicates and range/additional-ID overlaps remain errors.
A singleton fast path may skip sorting and exponent work, but must still
validate the selected arithmetic, accepted oracle, parameters, word, and full
input contract before advancing the cursor. The metadata language raw-logit
winner is outside this experiment and stays unchanged.

No WASM SIMD, WebGPU sampler, parallel reduction, native transcendental,
relaxed arithmetic, head-layout change, GEMV change, scheduler change, or
model-math change belongs to the first candidate. WASM SIMD is revisitable
only if Arm B remains materially above 5--10 ms per dense draw after profiling.

## Correctness gate

Before timing, require old/new identity for:

- stable order, including signed zero, subnormals, equal-logit ties, and
  negative-infinity holes;
- top-k threshold membership and top-p keep membership/order;
- every final FP32 weight word, positive-candidate count, sampled global token,
  random word, draw index, and absolute cursor;
- semantic production-BF16 logits at prefill, early, middle, and late cache
  positions;
- CoT singleton, small-ID, caption-range, and all-token states;
- unordered sparse IDs, duplicate/overlap rejection, boundary top-p draws,
  forced terminal tokens, and invalid inputs; and
- allocation reuse, cancellation, cursor non-commit on failure, and cleanup.

The candidate may not weaken full-input validation merely because an invalid
value is outside the admitted domain.

## Performance gate

In target Chrome on the M3, compile/initialize both arms before timing and use
immutable retained production-BF16 logits. Run balanced interleaved pairs after
30 continuous nominal thermal seconds, retaining all observations and the
complete through-cleanup thermal trace.

Measure semantic 64,000-candidate draws and representative CoT singleton,
small, caption, and all-token states separately. Arm B must have no regressing
dense-state median, beat A in at least 14/16 aggregate paired rounds, improve
aggregate sampler wall by at least `1.50x`, and project at least 10 seconds of
default planner saving. Allocation counts/bytes are diagnostic, not a
substitute for fenced wall time.

## Package-native and compound escalation

A passing sampler may be combined, under this same materially changed
experiment, with OPT-0082's already exact compact production-BF16 semantic
head. Compare ordinary A against fused-full B and compact-head/fused-sampler C
on actual sequential M2 tokens at early, middle, and late positions. Require
all retained logits, filtered states, sampled tokens, and cursors exact; every
position's C median below A; at least 14/16 aggregate C/A pair wins; at least
`1.15x` aggregate complete-token speedup; and at least 40 seconds projected
saving over 900 semantic draws. This new sampler makes C materially different
from OPT-0082; an unchanged OPT-0082 rerun remains forbidden.

## Standalone actual-browser result

The exact harness at
`2aaa0454587f24ede91a62dcccb702ba9c06cf62`, using candidate seam
`245b5fe3347c370390eff990aa1ed45cb2b869ba`, passed on the target M3 in Chrome
151. All eight authenticated semantic/CoT correctness states matched every
logical filtered-logit and weight raw-U32 word, sampled token, Philox word,
positive count, and cursor. Cancellation, full-input validation, allocation
reuse, idempotent cleanup, and the through-cleanup nominal thermal trace also
passed.

Across sixteen balanced rounds and 224 exact timed arm samples, B won `16/16`.
The aggregate seven-state sampler median fell from `297.900` to `73.600 ms`
(`4.04755x`). The semantic early/middle/late medians fell from
`41.150/41.250/41.600 ms` to `9.950/9.800/9.600 ms`; the frozen projection is
`28.395 s` saved over 900 semantic draws. Every dense-state median improved.

The continuous trace contained 554 nominal observations at 1 s cadence, with
a `1,009 ms` maximum gap and no missing or non-nominal observations. The raw
browser receipt and trace are intentionally ignored; their hashes and sizes
are retained in
[`sampler-gate.json`](../results/OPT-0084/sampler-gate.json).

## Package-native compound actual-browser result

The frozen compound harness at
`3de2853ff85bc069782361ed2fba592ffdc93ef4`, using candidate seam
`245b5fe3347c370390eff990aa1ed45cb2b869ba`, passed on the same target M3 in
Chrome 151. One authenticated reference-BF16 owner executed actual sequential
M2 tokens at early, middle, and late cache positions. Duplicate fresh-prefill
branches proved the ordinary full-head and compact-head dispatches started
from identical state: every prefill word, retained semantic logit, next-token
full-vocabulary cache witness, filtered logit, sampling weight, emitted token,
Philox word, cursor boundary, mapped write status, cancellation checkpoint,
and lifecycle check was exact.

Across sixteen balanced A/B/C triples at all three positions, compact-head
plus fused-sampler C beat ordinary production A in `16/16` aggregate rounds.
The aggregate three-position complete-token median fell from `1,238.550` to
`1,025.200 ms` (`1.208106x`). Every position median improved:
`401.400 -> 337.900 ms` early, `383.200 -> 322.200 ms` middle, and
`422.550 -> 344.050 ms` late. The frozen per-position-median projection is
`60.900 s` saved over 900 semantic draws, clearing the registered `40 s`
compound threshold.

The continuous 347-observation trace began nominal and provided a fresh
32-observation nominal launch interval with `1,009 ms` maximum gap and no
missing observations. Thermal pressure became non-nominal after measurement
had started and reached level 2; the receipt explicitly discloses all 134
later non-nominal observations. Because A/B/C are interleaved within every
round and the registered rule requires nominality through the actual worker
start rather than throughout the long run, this is a passing, decision-useful
compound result, not a continuously nominal claim.

The complete browser receipt is retained in
[`compound-gate.json`](../results/OPT-0084/compound-gate.json), SHA-256
`bf7b1fe6fffd42627a9b15ffe53abfddcd9fc58d39ca47bbb2b978ddef7a715a`,
`480,803` bytes. Its raw thermal trace is intentionally ignored; the receipt
binds trace SHA-256
`488539927c1a9e26e92eb85f495d0ed7f13655fc3a12bc46e816b1489abb72fb`
and byte length `56,451`.

## Integration gate

A passing compound candidate becomes the production browser-v1 cursor path only after
focused unit/fuzz tests, one complete default-CoT trajectory, one complete
150-code-plus-EOS semantic trajectory, and one planner-enabled product gate.
Require identical emitted tokens, words, cursors, conditioning inputs, final
latent, raw waveform, normalized WAV, cancellation behavior, and resource
lifecycle. Exact identity requires no listening retest.

Stop without integration for any sampler/cursor mismatch, validation
weakening, dense-state regression, aggregate speedup below `1.50x`, or
projected saving below 10 seconds. Do not add WASM or alter browser-v1
arithmetic under this ID after timing begins.

## Authority

- Current sampler: [`planner-sampling.ts`](../../src/runtime/planner-sampling.ts)
- Planner attribution: [OPT-0010](OPT-0010-package-native-planner-token-profiler.md)
- Compact-head evidence: [OPT-0082](OPT-0082-planner-compact-production-head-sampling.md)
- Approved production behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
