# OPT-0083 — Planner low-row packed-BF16 GEMV

## Status

- Evidence: `negative` (correctness passed; frozen authoritative wall-bandwidth
  gate failed)
- Disposition: `abandoned`
- Production integration: not authorized or performed
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact-order planner dense-kernel ownership
- Allocation baseline: pushed `main` commit
  `552e977be6b1b5c8b01c346d4aeaa7f63c0edbf2`

No OPT-0083 kernel, selector, harness, timing sample, or production change
existed when this experiment was allocated.

## Result

The actual Chrome/M3 primitive screen is a literal negative stop. Both
candidates were raw-U32 exact across `610,344` output-word visits covering all
seven production GEMMs at M1 and M2, adversarial long-K cancellation, and the
physical tied-head tail at N20596. Deterministic repeats, complete writes,
guard regions, finite outputs, all `12/12` rejection contracts, bounded
cancellation, and the absence of uncaptured GPU errors or device loss all
passed.

After one complete seven-operation warmup for each arm and row count, the
screen ran 16 balanced complete-layer rounds over eight rotating
`31,457,280`-byte weight sets. Every primary occurrence used one compute pass,
command buffer, submit, and drain. Fenced wall was authoritative; GPU
timestamps were diagnostic only.

| Arm | Wall wins | Aggregate wall speedup | Diagnostic GPU speedup | Wall bandwidth | Diagnostic GPU bandwidth | Frozen outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B — direct | `16/16` | `3.605206087416953x` | `7.612403100775194x` | `10.917928005232424 GB/s` | `29.767441860465116 GB/s` | fail: wall bandwidth below `20 GB/s` |
| C — panel | `16/16` | `1.6358267683275975x` | `1.8389513108614237x` | `4.953902343604687 GB/s` | `7.191011235955058 GB/s` | fail: wall speedup and bandwidth |

Arm B beat A in every M1/M2 wall median and exceeded the frozen `14/16` win
and `1.75x` speedup requirements. It nevertheless sustained only
`10.917928005232424 GB/s` on the authoritative wall clock, below the declared
`20 GB/s` stop threshold. Its `29.767441860465116 GB/s` diagnostic GPU rate
cannot waive that failure. Arm C also missed the wall speed and bandwidth
requirements. No arm is selected, no package-native escalation is authorized,
and no OPT-0083 production integration was performed.

Lifecycle and thermal evidence were clean. The accepted run destroyed
`62/62` buffers, balanced `220/220` maps, left zero live buffers and bytes,
rejected both candidates after destroy, destroyed the device, and passed an
idempotent second cleanup call. Maximum live storage was `295,096,432` bytes.
The thermal trace contains `1,226` observations, had a maximum `1,018 ms` poll
gap, remained at level zero with no missing or non-nominal observations, and
covers cleanup.

The first registered attempt is retained separately as invalid measurement
evidence. Chrome returned a begin-equals-end timestamp before the first timing
sample completed, and the original harness incorrectly made that diagnostic
condition fatal. Cleanup was still exact (`62/62` buffers, `195/195` maps,
zero live storage). The follow-up changed the protocol so unavailable GPU
timestamps are nullable diagnostics while fenced wall remains authoritative;
it was not an unchanged retry seeking a favorable sample. The accepted run
recorded `48/48` valid primary timestamps and `35/36` valid per-shape diagnostic
timestamps, with the single unavailable timestamp explicitly attributed to
`webgpu-timestamp-end-not-greater-than-begin`.

Result: [`../results/OPT-0083/result.json`](../results/OPT-0083/result.json).
Accepted raw receipt: ignored
`optimization/artifacts/OPT-0083/raw/primitive-wall-gate.json`, SHA-256
`69822104f277df43d976a66e8e931db7e9d09621f20d5a56a6da4e05412bfa1e`,
`207,135` bytes. Invalid first attempt: ignored
`optimization/artifacts/OPT-0083/raw/invalid-first-timestamp-attempt.json`,
SHA-256
`aea1889c55b22dbd0634a902e460e49474f9920c1aa10a795f33421b2022328d`,
`1,492` bytes. Thermal trace: ignored
`optimization/artifacts/OPT-0083/raw/thermal-trace.jsonl`, SHA-256
`67a19a8e94c896457c81a95ef0ac32bf892c4da164c8f5b28321fe009d72fa00`,
`199,954` bytes.

Implementation history: registration
`7eec5a4f18ab002bb6f4fd6074cd6b42dbc21f2c`; candidate kernels
`1ddb65e751529936b3ef3cd48a6360386c7dd205`; browser harness
`be3e9a142bcf28db0a1669cbd6bb8f7393f0a8f5`; warmup attribution and
isolation `4289adcddd26a91e11399dbd870917b96be9c0a9` and
`818d0d3c2be7767db0369f47f4220a93f3f7a67f`; receipt clone correction
`9b96b9f73c70ac7f4aaaf62d6bcc7b4a43539d99`; nullable diagnostic timestamps
`211e08aa626b957f44e80b8139690922dd4e037d`.

Do not rerun the unchanged primitive or tune another geometry under OPT-0083.
The direct low-row mechanism may be revisited only under a newly allocated
experiment whose materially changed package-native graph measures real planner
token scheduling without a submit-and-drain boundary for each GEMM-layer
sample. Such work must declare new gates and must not cite OPT-0083 as having
passed. No follow-up ID is allocated here.

## Hypothesis

Planner decode has only one CoT row or two semantic-CFG rows, but the generic
M16 tiled correctness GEMM executes the complete K loop for 16 rows. A
planner-only kernel in which one lane owns one output column can preserve each
output's increasing-K reference-BF16/FP32 contraction while eliminating the
measured 16x/8x padded arithmetic. Cooperative K64 panels can retain contiguous
source-weight reads and reuse each weight for both CFG rows without changing
the package layout.

The goal is high effective model bandwidth, not DiT-like TFLOP/s. With one or
two rows, the planner is a matrix-vector workload and will become bandwidth
limited once avoidable arithmetic and synchronization are removed.

## Frozen arms

- **A — tiled control:** current `AceCorrectnessGemmKernel`, M16/N128/K16,
  source-row-major packed BF16, FP32 activations/outputs/accumulation.
- **B — direct low-row:** WG128, one lane per output column, direct sequential
  packed-BF16 loads, explicit increasing-K scalar FP32 accumulation, no shared
  weight panel.
- **C — transposed-panel low-row:** WG128/N128/K64. Each workgroup owns one
  N128 tile. Packed weights are cooperatively loaded from their existing
  source-row-major `[N,K/2]` order into a bank-padded transposed shared panel;
  one lane owns one output column and walks K pairs low then high.

Arm C shared storage is exactly `array<u32, 32 * 129>` for the K64/N128 weight
panel plus `rows * 64` FP32 activation values: 16,768 bytes for M1 or 17,024
bytes for M2. No subgroup/K reduction, `dot`, `fma`, FP16 accumulator,
reassociation, package repack, duplicate weights, bias, or changed output
storage is permitted.

The candidate kernels accept only `reference-bf16`, rows one or two, even K,
source-row-major weights, bias absent, and the target workgroup/storage limits.
They fail closed otherwise. Production prefill rows greater than two, raw-FP16,
text-encoder GEMMs, and every unsupported tuple retain their existing owner.

## Production shapes

Each of 28 planner layers contains seven bias-free dense operations:

| role | K | N | multiplicity |
| --- | ---: | ---: | ---: |
| query | 1024 | 2048 | 1 |
| key/value | 1024 | 1024 | 2 |
| attention output | 2048 | 1024 | 1 |
| gate/up | 1024 | 3072 | 2 |
| MLP down | 3072 | 1024 | 1 |

The layers stream 880,932,864 packed-weight bytes per token. The full tied head
adds 444,833,792 bytes; OPT-0082 may reduce the semantic head independently.
OPT-0010 measured 10,605,297,664 scheduled multiply-adds per token versus
662,818,816 logical M1 or 1,325,637,632 logical M2 multiply-adds.

## Fast primitive gate

Compile all arms before timing. Use the complete seven-operation layer mix for
both M1 and M2 with at least eight rotating distinct weight sets so a tiny hot
cache cannot masquerade as model bandwidth. Include every production shape
and the tied-head tail shape. Run balanced interleaved A/B/C orders after a
nominal thermal start.

Before timing, require on the target Chrome/M3 adapter:

- zero raw-U32 output differences versus A for finite production-like data,
  adversarial cancellation/long-K data, and every N tail;
- deterministic reruns, complete writes, intact guard regions, and no NaNs;
- short-binding, alias, wrong-profile, bias, row-count, and limit failures;
- balanced maps/destroys, idempotent cleanup, and bounded cancellation.

The selected candidate must beat A in every M1/M2 complete-mix median, win at
least 14/16 paired complete-mix rounds, reach at least `1.75x` aggregate layer
mix speedup, and sustain at least 20 GB/s effective packed-weight bandwidth.
Per-shape timings are diagnostic and do not override the complete mix.

## Package-native escalation

A passing primitive arm is routed behind an experiment-only planner selector.
Using one authenticated current reference-BF16 planner owner, compare A and the
candidate at one middle-cache M1 token and one middle-cache semantic M2 token.
Require identical full logits raw-U32, cache/token behavior, sampled token,
cursor, and lifecycle. Measure transformer-layer, head, readback, and complete
token walls. Require both layer intervals to improve, aggregate layer speedup
at least `1.50x`, and a projected saving of at least 60 seconds over the
roughly 1,010 default three-minute planner draws.

## Integration gate

A passing package-native arm may become the strict production planner decode
owner. Run the focused kernel/model/executor tests, a complete default-CoT M1
trajectory, a complete 150-code-plus-EOS semantic M2 trajectory, and one
planner-enabled product correctness gate. Require identical production-BF16
logits at declared checkpoints, emitted tokens, draw cursor, conditioning
inputs, final latent, raw waveform, normalized WAV, cancellation, and resource
lifecycle. Exact product identity requires no listening retest.

## Stop conditions

Stop without integration for any target-browser raw-U32 mismatch, incomplete
write, lifecycle failure, primitive speedup below `1.75x`, package layer
speedup below `1.50x`, projected three-minute saving below 60 seconds, or
effective bandwidth below 20 GB/s. Do not add further workgroup sizes, K tile
sizes, split-K schemes, or arithmetic variants under this ID after timing
begins.

## Authority

- Planner token attribution:
  [OPT-0010](OPT-0010-package-native-planner-token-profiler.md)
- Current GEMM oracle: [`gemm.ts`](../../src/webgpu/kernels/gemm.ts)
- Approved production behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
