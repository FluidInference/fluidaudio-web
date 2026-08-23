# OPT-0003 — Packed-BF16 FP32 subgroup GEMM

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; the arithmetic claim is scoped to bit identity with the
  accepted portable GPU kernel on the pinned Chrome/M3 target because WGSL
  permits contraction and reassociation

## Hypothesis

The accepted GEMM stages packed BF16 weights through a portable
M16/N128/K16 workgroup tile and reaches only about 0.21–0.51 diagnostic logical
TFLOP/s on the local M3. The adapter exposes fixed 32-lane subgroups, while the
audited Parakeet implementation demonstrates an ordinary-subgroup scalar GEMM
that gives each lane exclusive FP32 accumulators and uses subgroup broadcasts
to reuse activation scalars.

A benchmark-local direct-weight kernel with M32/N128/K32 tiles, WG128, FP32
activations and accumulators, and tile-major packed-BF16 weights should remove
the shared-memory weight staging and substantially improve the four weighted
production shapes without reducing precision. No prior experiment tests this
kernel/layout variable: `OPT-0001` measured the portable baseline and
`OPT-0002` changed only VAE scheduling.

## Identity

- Baseline commit: `16e680b2a92459a9ed6d7c4677cc6fc617914222`
- Candidate commit: `0fb193becf1ec359213bbc5f50ad7a9d04c272f8`
- Parakeet source commit: `7ee112738262a6f5a0efd2f150748a4087432fbb`
- Production bundle SHA-256: not recorded; the integrated runtime, package
  identity, and browser harness are authenticated by the commits below
- Original benchmark model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Benchmark harness commit: `0fb193becf1ec359213bbc5f50ad7a9d04c272f8`
- Integrated runtime commit: `a4e4ce4d2a2a74b7d9d0b1a05e7fd25343e9d404`
- Integrated package-identity commit:
  `68d7795c616c1520b1d97ddef9f9d3147ab3973e`
- Package-native integration harness commit:
  `bf6da81647814737e88c8e881da72e04887cba07`
- Integrated reference package manifest SHA-256:
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`
- Integrated browser: actual local Google Chrome `151.0.7922.138`; raw reduced
  user-agent Chrome version `151.0.0.0`
- Execution profile: `reference-bf16-subgroups`
- Machine / GPU cores / memory: MacBook Air `Mac15,12`; Apple M3; 10 GPU cores;
  16 GB unified memory
- macOS build: macOS 26.5.2, build 25F84
- Chrome and WebGPU identity: record from the paired browser result

## Change

First add an experiment-local generator and planner, without changing package
conversion or the production dispatch:

- M32/N128/K32, `@workgroup_size(128)`, four fixed 32-lane subgroups;
- each subgroup owns eight rows and every lane owns one `vec4<f32>` column
  vector for those eight rows;
- activation input and output remain row-major FP32;
- BF16 weight values are rearranged losslessly from logical `[N,K]` source
  order into physical `[N/128,K/32,32,128]` tiles and remain packed two per
  `u32`;
- every output accumulator's WGSL source visits scalar K in increasing order
  with explicit `acc = acc + a * b`, never subgroup/cross-lane partial sums;
  WGSL permits the implementation to reassociate and fuse this expression, so
  the correctness authority is actual portable-versus-candidate GPU identity
  on the pinned browser/adapter rather than a universal separate-rounding CPU
  claim; and
- bounded output ranges retain one command buffer outstanding, a full drain,
  and the real one-millisecond queue-empty interval.

The isolated candidate first established that the kernel/layout mechanism was
useful. Production integration subsequently selected one authenticated
converter-native tile-major layout for both implementations. The portable
fallback and the fixed-32 subgroup kernel now read the same physical weights,
so backend selection requires no runtime repack, second weight copy, or
persistent-residency increase. The subgroup backend remains fail-closed to the
reference-BF16 profile and reported fixed 32-lane subgroups.

## Correctness gate

- Oracle identity: approved Stage 1 reference profile plus the accepted
  portable GPU GEMM on the pinned Chrome/M3 target, with independent CPU
  oracles used to characterize permitted contraction and catch reassociation.
- Required tests/tensor taps:
  - prove the `[N,K]` to `[N/128,K/32,32,128]` scalar mapping is bijective and
    preserves every BF16 word, including signed zero, infinities, and NaNs;
  - compare subgroup and portable tiled GPU outputs over tails, bias/no-bias,
    adversarial contraction/cancellation, and all four M=2,250 production
    shapes; require GPU bit identity, retain both separate-rounding and
    contracted CPU diagnostics, and require the cancellation fixture to retain
    source-order behavior;
  - prefill every output with a non-finite sentinel, require complete finite
    writes, and retain full-domain fingerprints plus independent sentinels;
  - fail closed unless subgroup min/max are both 32 and the feature is enabled;
    and
  - retain exact range coverage, alias, lifecycle, scheduler, and portable
    fallback tests.
- Declared tolerance: bit identity between portable and subgroup GPU outputs on
  every finite deterministic fixture on the pinned Chrome/M3 target. WGSL
  explicitly permits reassociation and fusion; differences from a forced
  separate-rounding CPU emulation are retained as diagnostics and do not widen
  the candidate-versus-baseline tolerance. A candidate/portable GPU mismatch is
  a failure.
- Listening required: no for the benchmark-local candidate. Production
  integration also requires no listening only if repeated-block/final-latent
  evidence remains exact; otherwise a new listening gate is mandatory.
- Result: passed. All production-shape and adversarial target-browser GPU
  comparisons were bit-identical; the contracted CPU oracle also matched. The
  later package-native gate additionally produced identical portable/subgroup
  U32 bits for all 8,256 elements of the final latent after the complete
  24-layer, eight-evaluation DiT graph. A subsequent 12-second direct
  instrumental product run also reproduced the accepted Stage 1 Candidate A
  WAV exactly.

## Benchmark protocol

- Shapes at M=2,250: `[2048,2048]`, `[2048,1024]`, `[2048,6144]`, and
  `[6144,2048]`, retaining the exact logical work and output domains from
  `OPT-0001`.
- Warmup: compile/prepack outside measurement, run one unmeasured execution per
  kernel/shape, then clear/drain/idle.
- Thermal pre-gate: at least 30 continuous nominal seconds, polled every
  1,000 ms with the accepted tolerance.
- Paired order: four samples per candidate and baseline in balanced AB/BA order
  within one visible page; retain every sample, median, and range.
- Timing: wall time around encode/submit through completion fence, separately
  recording encode, submit, drain, explicit idle, logical TFLOP/s, and page
  heartbeat. GPU timestamps are unavailable in this fixed-feature probe and
  are not required; completion-fenced wall time is authoritative.
- Memory: record packed-weight bytes, any transient prepack bytes, logical
  high-water delta, and workgroup storage. The first checkpoint excludes
  prepack time from steady GEMM timing but reports it separately.
- Cooperative topology: no more than one command buffer outstanding; bounded
  ranges, drain, then real 1 ms idle before non-final work.
- Decision signals: exact target-GPU correctness, a credible paired positive
  delta, bounded responsiveness, and an integration path whose likely product
  value justifies its package/runtime cost. Earlier numerical predictions are
  retained in Git history as hypotheses, not vetoes.
- Escalation: do not run a song for the microbenchmark alone. Positive evidence
  advances to a one-layout package/runtime design, a miniature block, and one
  repeated production DiT layer/final-latent tap before production selection.
- Exact commands:
  - `pnpm exec vitest run test/opt-0003-subgroup-gemm.test.ts test/opt-0003-subgroup-gemm-ab-contract.test.ts`
  - `pnpm check`
  - serve the frozen `0fb193b` worktree and open
    `test/browser/opt-0003-subgroup-gemm-ab.html` in the in-app M3 Chrome
    after a continuously logged 30-second nominal thermal gate.

## Results

The first actual M3 run failed closed during the newly added adversarial
correctness preflight, before performance timing. Portable and subgroup GPU
outputs were bit-identical, but both differed from a forced
separate-rounding CPU emulation in exactly 1,088 outputs. WGSL section 15.7.5
permits this contraction/reassociation. The failed run is retained rather than
discarded:

- retained ignored result artifact:
  `optimization/artifacts/OPT-0003/raw/failed-preflight.json`
- result SHA-256:
  `787720da0081998c428892f2f1b789ac8f05e8cef43b3833de681b7563ec6946`
- retained ignored continuous thermal log:
  `optimization/artifacts/OPT-0003/raw/failed-preflight-thermal.jsonl`
- thermal-log SHA-256:
  `764ee84313ffb7ab36476cac4f17713b185487d802dd17affc0d7a911873d95d`
- thermal coverage: 72/72 nominal observations over 72.785 seconds, spanning
  the pre-gate, browser execution, failure, and immediate post-run state;
  maximum poll gap 1,040.172 ms.

This invalidated the original universal separate-rounding wording, not the
candidate: the candidate had zero bit mismatches against the accepted portable
GPU baseline. The gate above is corrected before any performance rerun.

A second attempt stopped before timing on a browser clock-quantization edge:
the requested real one-millisecond timer interval measured
`0.8999999761581421` ms and tripped an unnecessarily strict `>= 0.9` harness
assertion. The production timer request was unchanged; the harness now records
the observed wall interval without treating floating-point clock granularity as
a kernel failure. The failure and its transparently repaired single-stream
thermal log are retained under `optimization/artifacts/OPT-0003/raw/` with
SHA-256 values recorded in the machine-readable result.

The one decision-sufficient thermally valid diagnostic run then passed every
correctness and responsiveness gate:

- all four full production-shape outputs were finite, nonzero, completely
  written, and bit-identical between the portable and subgroup GPU kernels;
- row-tail plus BF16 bias and adversarial cancellation preflights passed;
  contracted CPU mismatches were zero, while the expected diagnostic
  separate-rounding mismatch count remained 1,088;
- weighted portable median: `0.4880478157` logical TFLOP/s;
- weighted subgroup median: `0.9251608162` logical TFLOP/s;
- median active-wall speedup: `1.893894876x`, with four of four in-page paired
  rounds won;
- maximum candidate drain: `58.600000024` ms, caused by the first
  `6144 -> 2048` candidate sample; later drains for that shape were at most
  `6.7` ms; and
- warmed paired maximum animation-frame/timer gaps: `35.0` / `57.8` ms.

The continuous external thermal log spans the pre-gate, page run, and
post-run state: 110/110 nominal observations over 110.084 seconds, maximum
poll gap 1,005.084 ms. The canonical result and thermal artifacts have SHA-256
`7fd4d64a7ba573a3c93bd8b97190dc2ec0fe54bbce234227d1edcbe1dccf8c4b`
and `0238fb4b3c41d106e920da368142ad28c455c0fd864db0b11bf9212eb4bb6690`.

Only one thermally independent page run was collected. Together with four
balanced in-page wins and a large consistent per-shape delta, it establishes a
credible positive direction and approximate magnitude. This original
thermally valid microbenchmark remains the sole performance authority; the
integration fixture below was intentionally not thermally gated and makes no
performance claim.

Production integration then passed an actual-Chrome package-native gate using
the converter-revision-4 reference manifest
`18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`:

- the harness authenticated and loaded 476 DiT tensors, including all 271
  tile-major GEMM weights, from 50 files with 3,150,917,888 resident bytes and
  3,150,917,760 logical tensor-payload bytes;
- heavyweight phases were sequential, with exactly one DiT phase resident;
- each backend ran the complete 24-layer graph for all eight denoising
  evaluations at batch 1, 129 latent frames, and one condition token;
- portable submitted and drained 634 command buffers with 633 explicit idle
  intervals; fixed-32 subgroups submitted and drained 826 with 825 idles;
  both reconciled all 249 logical graph quanta and completed eight evaluations;
- both paths used the same 3,170,141,952 accounted GPU bytes: 3,150,917,888
  resident weight bytes, 19,191,040 arena bytes, and a 33,024-byte readback;
  the bounded CPU plan was 173,876 bytes and the persistent layout-replacement
  delta was zero;
- every one of the 8,256 final-latent values was finite and nonzero, both
  outputs hashed to
  `71f98633ada680853ad9ef6ee3fccb40e7664da7f5795f6b0a68142803072bb7`,
  and the U32 mismatch count was zero; and
- the device reported no runtime events and remained healthy through sequential
  phase destruction and reload.

A subsequent actual-Chrome production-path run on harness commit
`bf6da81647814737e88c8e881da72e04887cba07` used the same revision-4 package,
the `reference-bf16-subgroups` profile, and cooperative scheduling for the
accepted 12-second direct instrumental request. It produced 576,000 stereo
frames with WAV SHA-256
`d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477`,
exactly matching accepted Stage 1 Candidate A. The tracked GPU peak remained
exactly 3,214,388,992 bytes.

That product run took 64,822.1 ms total, with 6,145.4 ms text encoding,
7,025.6 ms condition encoding, 14,239.3 ms DiT loading, 8,066.8 ms DiT
denoising, 3,094.8 ms VAE loading, 26,208.3 ms VAE decoding, and 25.2 ms WAV
encoding. It submitted 2,200 queue drains and requested 2,197 ms of cooperative
idle, versus Stage 1 Candidate A's 720,489.5 ms, 132,326 drains, and 132,323 ms
idle. The approximately 11.115x end-to-end change is a nonthermal directional
checkpoint for the combined integrated stack, not a performance result
attributable solely to OPT-0003. The original thermally valid microbenchmark
above remains OPT-0003's performance authority. No 180-second/full-song run was
performed.

The direct package-native gate did not spend another 3.15 GB reload on a
deliberate cancellation run. Cancellation evidence is therefore explicitly
compositional, not direct: the existing actual-Chrome ranged OPT-0003
cancellation probe passed, while the production DiT backend's cancellation,
drain-before-release, transferred-ownership, and cleanup contracts passed.
The ignored raw integration artifact is
`optimization/artifacts/OPT-0003/raw/package-native-dit-integration.json`,
SHA-256
`b15cb76304e65881b6dedd633b0d58dbb155efd873abd1c76339690213f36161`.
The ignored product receipt is
`optimization/artifacts/OPT-0003/raw/post-integration-direct-instrumental-12s-receipt.json`,
file SHA-256
`1cb73d06818062972d6b597ee61d40c25be19a7646378065fce721a7ca835d2d`,
and its receipt-body self SHA-256 is
`efdbce5e02cc148873fc33309032097edc28c3189a8d67e83116282ebac9e117`.

## Evidence and disposition

- Evidence: `positive`. The candidate preserved target-browser bits, won all
  four balanced paired rounds, and improved every important shape by
  1.83–1.98x. Its weighted median improved from 0.4880 to 0.9252 logical
  TFLOP/s, a 1.8939x active-wall gain. Missing an earlier prediction does not
  negate that useful result.
- Disposition: `integrated`. The one-layout converter/package contract,
  tile-major portable fallback, and fixed-32 subgroup production selection are
  landed. The complete tiny DiT graph passed exact final-latent, memory,
  scheduling, device-limit, and cleanup gates without a second resident weight
  layout or runtime repack.
- Result JSON: `optimization/results/OPT-0003/result.json`
- Implementing commits: benchmark candidate
  `0fb193becf1ec359213bbc5f50ad7a9d04c272f8`; runtime
  `a4e4ce4d2a2a74b7d9d0b1a05e7fd25343e9d404`; package identity
  `68d7795c616c1520b1d97ddef9f9d3147ab3973e`; integration harness
  `bf6da81647814737e88c8e881da72e04887cba07`
- Interactions: uses `OPT-0001` baseline shapes/profiler and leaves accepted
  `OPT-0002` VAE scheduling unchanged
- Revisit when: a materially different GEMM candidate is ready for a direct
  correctness and performance comparison.
- Follow-ups: continue measured optimization toward the sole end-to-end target.
  A future faster GEMM may supersede this implementation only after it is
  demonstrated.
