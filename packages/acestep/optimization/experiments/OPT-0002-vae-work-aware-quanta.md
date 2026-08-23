# OPT-0002 — Work-aware VAE cooperative quanta

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

The correctness VAE partitions every operation by the same 32,768-output cap.
`OPT-0001` showed that this produces 62,622 compute command buffers for one
256-latent-frame / 10.24-second output window, including 36,816 command buffers
for pointwise Snake and add operations. The requested one-millisecond idle
alone is 62.622 seconds before timer overshoot.

A fixed work-aware policy should remove most scheduling overhead without
changing any WGSL instruction, scalar invocation, source-order reduction,
operation dependency, queue drain, or real idle boundary:

- bound Conv1D and ConvTranspose quanta by a conservative maximum of
  234,881,024 multiply-accumulates, equal to the existing worst accepted
  quantum (`32,768 × 7 × 1,024`);
- additionally cap every quantum at 1,048,576 output scalars;
- derive the Conv1D output count from `kernelSize × inputChannels`;
- derive the ConvTranspose count from the safe maximum number of congruent taps
  per output,
  `ceil(kernelSize / (stride / gcd(stride, dilation))) × inputChannels`
  (the pinned dilation-one shapes simplify to `ceil(kernelSize / stride)`),
  while retaining complete logical output rows and every physical
  output-channel shard in one quantum; and
- use the output cap directly for Snake and add.

The exact production plan is 3,942 compute command buffers per maximum window,
15.8858× fewer than the uniform policy. Prior ledger work did not test this
variable: `OPT-0001` measured and localized it but changed no production code.

## Identity

- Baseline commit: `6281ca0000fa513d001252c4d4aee937bdbb007c`
- Candidate commit: `80f3c8bf550bd16fe18c64992627027972be18a7`
- Production bundle SHA-256: not produced; the committed TypeScript modules
  were served directly by Vite for this source-level browser experiment
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Benchmark harness commit: `80f3c8bf550bd16fe18c64992627027972be18a7`
- Execution profile: `reference-bf16-subgroups`; VAE weights and activations
  remain FP32
- Machine / GPU cores / memory: MacBook Air `Mac15,12`; Apple M3; 10 GPU cores;
  16 GB unified memory
- macOS build: macOS 26.5.2, build 25F84
- Chrome: Google Chrome Framework `151.0.7922.109`; reduced user agent
  `151.0.0.0`. No isolated command line was claimed.
- WebGPU adapter: non-fallback Apple `metal-3`, subgroup size 32, with the full
  feature and limit identity in the canonical result

## Change

Replace the scalar `quantumOutputElementCap` planning contract with an explicit
immutable work policy containing a maximum convolution-MAC budget and maximum
output-element budget. The ConvTranspose congruence bound conservatively falls
back to every kernel tap if WGSL `u32` intermediates could wrap. The change is
limited to the VAE cooperative planner, backend, tests, and benchmark harness.
Primitive shaders, weights, tensor layouts, operation graph, scheduler,
one-outstanding-command rule, drains, and one-millisecond intervals remain
unchanged.

The rollback is restoring the uniform 32,768-output planner. No public
baseline/candidate switch remains after the experiment decision.

## Correctness gate

- Oracle identity: approved Stage 1 reference profile and the authenticated
  VAE graph/package contracts at tag `stage1-approved-2026-08-13`.
- Required tests/tensor taps:
  - prove every operation's output domain is covered exactly once and in FIFO
    order under both policies;
  - prove every candidate convolution quantum stays within the MAC and output
    budgets;
  - prove ConvTranspose row bands keep all physical shards together and never
    expose a partial logical row to the next operation;
  - compare baseline and candidate complete-toy-graph output bit-for-bit on
    actual M3 WebGPU, including forced multi-range operations; and
  - retain existing VAE graph, package, backend, chunk, seam, lifecycle,
    cancellation, and device-loss tests.
- Declared tolerances: exact FP32 bit identity. Scheduling-only partitioning
  writes disjoint output scalars and cannot alter arithmetic order.
- Listening required and why: no, provided full output is bit-identical. Any
  arithmetic, layout, precision, fusion, or output mismatch changes the risk
  class and requires a new experiment and gate.
- Result: passed. The M3 complete toy graph covered 1,048,580 FP32 outputs,
  forced multiple ranges under both policies, retained both physical transpose
  parts per row band, and produced zero bit mismatches. All four
  production-shape representative A/B ranges were finite and bit-identical.

## Benchmark protocol

- Fixture, prompt/lyrics, duration, seed: no song initially. Use the exact
  authenticated 256-frame production workload plan, the complete deterministic
  toy decoder, and representative worst Conv1D/ConvTranspose/Snake/add shapes.
- Warmup policy: compile each unique pipeline once and execute one unmeasured
  warmup before paired timing.
- Thermal pre-gate and polling: 30 continuous nominal seconds, polled every
  1,000 ms, before attributed actual-GPU samples.
- Paired order: run uniform and work-aware policies in the same visible page in
  balanced AB/BA order; retain every sample, median, and range.
- Timing method: wall time around submit through completion fence, with encode,
  drain, actual idle wall, heartbeat, and command count separated. GPU
  timestamps remain diagnostic.
- Memory run method: exact range-control bytes and logical buffers from the
  planner; physical process-tree sampling is unnecessary unless the candidate
  unexpectedly increases memory.
- Cooperative scheduler topology: exactly one command buffer outstanding,
  full drain, then the production real one-millisecond queue-empty interval.
- Acceptance targets:
  - at least 10× fewer command buffers for the 256-frame production plan;
  - no candidate convolution quantum above 234,881,024 MACs or 1,048,576
    outputs;
  - candidate pointwise quantum drain below 50 ms preferred and 100 ms
    required on the M3 probe;
  - maximum animation/timer heartbeat gap below 50 ms and cancellation still
    available within 500 ms / one bounded quantum; and
  - no logical high-water increase.
- Escalation: do not run a full song. Run one real production VAE window only
  if the static plan plus paired representative-shape evidence leaves the
  decision uncertain or after the candidate is otherwise ready to accept.
- Exact commands:
  - `pnpm exec vite --host 127.0.0.1 --port 5177 --strictPort`
  - open
    `http://127.0.0.1:5177/test/browser/opt-0002-vae-quantum-ab.html`
    after the external thermal gate and run the visible-page probe once
  - `pnpm exec vitest run test/opt-0002-vae-quantum-ab-contract.test.ts test/opt-0002-vae-workload.test.ts test/vae-decoder-contract.test.ts test/vae-backend.test.ts test/opt-0001-vae-workload.test.ts`
  - `pnpm check`

## Results

The external pre-gate passed after 30.008 continuous nominal seconds across 31
observations. Its maximum 1.021-second polling gap was below the declared
1.250-second limit.

| Production-plan metric | Uniform baseline | Work-aware candidate | Delta |
| --- | ---: | ---: | ---: |
| Decoder quanta | 62,622 | 3,942 | -58,680 (15.8858× fewer) |
| Primitive dispatches | 62,702 | 3,988 | -58,714 |
| Command buffers including readback | 62,623 | 3,943 | -58,680 |
| Configured cooperative idle | 62,622 ms | 3,942 ms | -58,680 ms |
| Cached range-control bytes for 174/192/256-frame shapes | 39,012,400 | 2,491,184 | -36,521,216 |
| Accounted chunk-backend GPU bytes | 1,139,500,080 | 1,102,978,864 | -36,521,216 |

The candidate maximums were exactly 1,048,576 output scalars and 234,881,024
conservative convolution MACs, with zero budget violations. Arena, workspace,
weight, readback, and CPU high-water allocations did not increase.

The complete forced-multirange toy graph fell from 628 to 38 decoder quanta and
from 661 to 40 primitive dispatches. All 1,048,580 outputs were finite and
bit-identical (`0` mismatches; FNV-1a fingerprint `efecb8c1`). Its candidate
maximum drain was 8.5 ms.

| Warm paired M3 case | Outputs | Max MACs | Baseline → candidate CBs | Median wall | Candidate max drain | Max rAF / timer gap | Bit mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conv1D `conv1` | 524,288 | 234,881,024 | 16 → 1 | 38.7 → 6.5 ms | 6.8 ms | 22.6 / 10.9 ms | 0 |
| ConvTranspose block 3 | 458,752 | 234,881,024 | 14 → 1 | 39.5 → 7.65 ms | 10.4 ms | 40.4 / 11.4 ms | 0 |
| Snake block 1 | 1,048,576 | 0 | 32 → 1 | 69.8 → 1.0 ms | 1.1 ms | 41.5 / 10.9 ms | 0 |
| Add block 1 | 1,048,576 | 0 | 32 → 1 | 74.7 → 1.3 ms | 1.5 ms | 36.2 / 11.1 ms | 0 |

The heartbeat acceptance applies to those warmed, balanced, attributed
representative intervals. Whole-harness diagnostic maxima were 135.1 ms for
animation frames and 217.7 ms for the timer because that broader interval also
included allocation, shader compilation, the complete toy correctness run,
and result serialization; it is retained but is not an attributed warm timing.

Cancellation was not directly triggered in the browser harness. Acceptance of
the 500 ms / one-quantum target is an explicit inference: the slowest measured
candidate drain was 10.4 ms, production leaves only one command buffer
outstanding and then performs the mandatory 1 ms queue-empty idle, and the
existing backend tests prove that an abort during that interval prevents the
next submission after the current drain and idle finish. This is strong margin,
but it is not reported as a directly measured cancellation latency.

No full production VAE window or song was run. Exact output identity, complete
static coverage, four production-shape probes, and wide responsiveness margin
made escalation unnecessary. The raw result is retained outside Git at
`optimization/artifacts/OPT-0002/raw/paired-ab.json`, SHA-256
`082c5cca3f425c4b51995659468ef6b4de68a051be04b92669a4bba451182e54`.

## Evidence and disposition

- Evidence: `positive`; exact GPU output identity and every static work/memory
  invariant passed, the authenticated production plan reduced command buffers
  materially, and the scoped M3 responsiveness evidence has wide margin.
- Disposition: `integrated`; retain the work-aware policy as the production
  default.
- Result JSON: [canonical result](../results/OPT-0002/result.json)
- Implementing commit: `80f3c8bf550bd16fe18c64992627027972be18a7`
- Revert commit: not applicable
- Interactions with previous experiments: `OPT-0001` supplies the exact
  baseline workload and result schema.
- Revisit when: a supported device violates the pointwise or inferred
  cancellation envelope, or the authenticated VAE graph, primitive math,
  scheduler topology, or production shape set changes
- Follow-ups: keep convolution tiling, Snake/conv fusion, operation-native VAE
  kernels, precision changes, and scheduler idle sweeps in separate OPT IDs.
