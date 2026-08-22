# OPT-0085 — Planner depth-two completion epochs

## Status

- Evidence: `positive` (actual-browser correctness and performance gate passed)
- Disposition: `pending-integration-gates`
- Production integration: not yet authorized
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact FIFO scheduling, cancellation, and aliased-resource lifetime
- Allocation baseline: pushed `main` commit
  `7a078faee2ba2644394663863f73f80e5544e358`

No OPT-0085 selector, harness, timing sample, or production change existed
when this experiment was allocated. Its scheduler mechanism already exists for
the independently approved OPT-0080 DiT/VAE paths; planner use remains new and
unauthorized until this record's gates pass.

## Browser result

The isolated target-Chrome run at pushed commit
`85b20ae6c010d4f9b0f5bd28ae4bdbc70734428f` passed every frozen browser
gate. One authenticated reference-BF16 owner and one executor ran M1 CoT full,
M2 semantic full, M2 compact semantic, and forced-EOS paths. Every warmup and
all 16 balanced pairs reproduced the control's full or retained logits
raw-U32, cache/write status, sampled token, Philox word/cursor, and lifecycle.

The candidate retained all `34` full or `33` compact command buffers but
reduced true drains from `34/33` to `9/9` and cooperative idle turns from
`34/33` to `8/8`, with at most two outstanding commands. It won `16/16`
pairs and lowered every path median:

| Path | Control median | Candidate median | Saving |
| --- | ---: | ---: | ---: |
| M1 CoT full | `382.250 ms` | `293.900 ms` | `88.350 ms` |
| M2 semantic full | `353.050 ms` | `274.300 ms` | `78.750 ms` |
| M2 semantic compact | `311.150 ms` | `243.700 ms` | `67.450 ms` |
| M2 forced EOS | `252.450 ms` | `170.900 ms` | `81.550 ms` |

The aggregate median fell from `337.200 ms` to `259.500 ms`, a
`77.700 ms` saving and `1.29942x` speedup, projecting `78.477 s` over 1,010
draws. This exceeds the frozen `14/16`, `20 ms`, and `20 s` thresholds.
Cancellation stopped backfill from the first completion callback, published no
diagnostics or logits, preserved rejection identity, and settled submitted
work before double destroy. No WebGPU runtime event occurred.

The accepted thermal slice was continuously nominal: 46 pre-gate observations
over 45.004 seconds and 175 observations through cleanup over 174.030 seconds,
with a `1,006 ms` maximum gap and no missing or non-nominal observation.

Result: [`../results/OPT-0085/result.json`](../results/OPT-0085/result.json).
Ignored browser receipt SHA-256
`f1fb38eb2e94ebd36c5d637ef25e1f2df1691465b5d773b3fe8d566f36586abe`;
ignored thermal trace SHA-256
`d21ad99457c12246880f87b9b49c8fa50816ea4aab60847e1060a6e70bff079e`.

The positive browser gate authorizes only the declared trajectory and product
escalation. Production remains depth one until those exactness gates pass.

## First-principles basis

Every current full planner token submits and fully drains 33 singleton model
command buffers plus one readback command, requesting a real cooperative idle
after all 34. OPT-0010 measured 39.5--43.7 ms of explicit idle per token. The
accepted OPT-0080 scheduler can preserve every command buffer and cumulative
completion fence while keeping at most two submitted and forcing a true drain
after four completions.

One dynamic phase containing model quanta and the final readback changes the
full-token topology from 34 true drains/idles to 9 true drains and 8
nonterminal idles. A compact 32-model-command path changes 33/33 to 9/8. The
literal idle reduction alone projects roughly 30--33 ms per token, or about
30--34 seconds across approximately 1,010 default CoT plus semantic draws.
Those figures are mechanism projections, not measured claims.

## Frozen arms

- **A — depth one:** current planner loop, one singleton submission, cumulative
  fence, true drain, progress callback, and real idle before the next command.
- **B — depth two/epoch four:** existing `runLazyDepth2Epoch4` with one dynamic
  phase count `[quanta.length + 1]`. Factory indices below `quanta.length`
  encode the unchanged model singleton; the final index encodes the unchanged
  readback singleton.

B keeps every command, pass, dispatch, binding, arithmetic operation, FIFO
order, and fence. It submits at most two singleton buffers. The readback copy
may be submitted behind the tied head but mapping begins only after the
terminal fulfilled fence, so FIFO guarantees completed logits.

Do not coalesce commands, alter the epoch, add a depth-three arm, combine a
kernel/sampler change, overlap mapping, mutate aliased storage, or remove the
real idle under this ID. Counts remain dynamic rather than hard-coded.

## Static and failure gates

Deterministic tests must prove:

- exact FIFO encode/submit/completion/progress order and singleton submissions;
- maximum two outstanding command buffers;
- full `34 -> 9` drains and `34 -> 8` nonterminal idle turns;
- compact `33 -> 9` drains and `33 -> 8` nonterminal idle turns;
- readback is last and mapping cannot start until its terminal fence settles;
- fresh prefill clears cache/status only in command zero and decode never
  clears it;
- M1/M2 full and compact/EOS dispatches use the same scheduler topology;
- abort, callback failure, encode/submit/fence rejection, idle failure, and
  device loss stop backfill, settle submitted work, and delay phase/root
  destruction until terminal queue safety; and
- repeated destroy, map/unmap counts, queue lease, and all resources balance.

Diagnostics must distinguish requested/settled/rejected cumulative fences from
true queue drains and expose epochs, idle turns, maximum outstanding buffers,
and submitted commands without expanding the product API.

## Actual-browser gate

Using one authenticated current reference-BF16 planner owner, compare A/B on
one middle-cache M1 CoT token, one middle-cache M2 semantic token, the compact
semantic range, and forced EOS. Require identical full or retained logits
raw-U32, cache/write status, sampled token, Philox word/cursor, and lifecycle.

Run at least 16 balanced interleaved complete-token pairs after 30 continuous
nominal thermal seconds and retain the through-cleanup trace. B must have every
path median below A, win at least 14/16 aggregate pairs, save at least 20 ms in
the aggregate complete-token median, and project at least 20 seconds across
1,010 draws. Per-fence intervals overlap and must never be summed; complete
token wall and disjoint epoch walls are authoritative.

## Integration gate

A passing B may become the strict production planner scheduling owner. Run the
focused scheduler/model/executor tests, complete default-CoT and
150-code-plus-EOS trajectories, then one planner-enabled product correctness
gate. Require identical logits checkpoints, emitted tokens, draw cursor,
conditioning, final latent, raw waveform, normalized WAV, cancellation, and
resource lifecycle. Exact identity requires no listening retest.

Stop without integration for any output/cursor/cache mismatch, failure-path
regression, maximum outstanding count above two, median saving below 20 ms, or
projected saving below 20 seconds.

## Authority

- Integrated scheduler precedent: [OPT-0080](OPT-0080-dit-depth2-completion-epochs.md)
- Planner executor: [`planner-executor.ts`](../../src/webgpu/planner-executor.ts)
- Planner attribution: [OPT-0010](OPT-0010-package-native-planner-token-profiler.md)
- Approved production behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
