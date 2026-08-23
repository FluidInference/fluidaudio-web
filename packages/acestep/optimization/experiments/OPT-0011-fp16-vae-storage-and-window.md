# OPT-0011 — FP16 VAE storage and 512-frame window

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `approximate`; weight precision, activation storage, selected
  heavy-kernel arithmetic, and production window geometry may all change
  numerical output

## Hypothesis

Authenticated FP16 VAE weights and internal activation storage, with FP32
range-sensitive, nonlinear, reduction, and final-output islands, can unlock
native/subgroup FP16 heavy kernels while preserving the approved VAE quality
envelope. The halved activation element size can then expand the production
window from 256 to 512 latent frames with the existing 64-frame overlap and
without increasing the current largest workspace binding. The wider window
should reduce both kernel cost and duplicated overlap work.

This is deliberately a mixed-precision hypothesis. Literal all-FP16 VAE math
is out of scope. In the registered B/C arms every parameter and every internal
workspace boundary is stored as FP16, while FP32 accumulation, reductions,
Snake, and other nonlinear/range-sensitive islands may use FP32 registers
before explicitly rounding back to FP16 storage. The final raw waveform remains
FP32. Legitimate overflow may not be hidden by clamping, saturation, or silent
fallback. If evidence requires an FP32-stored parameter or internal activation
island, that is a new registered B/C variant whose exact package and workspace
bytes must be frozen before it runs; it cannot inherit the half-size accounting
below.

There is no numerical speed threshold. Predicted throughput, a fixed speedup,
or the three-minute product target cannot accept or reject this experiment by
itself. Correct, thermally valid evidence determines whether each mechanism is
useful.

## Identity

- Allocation baseline: clean pushed `main` commit
  `f07afbeb425157ca9c1eb4a9bc2102365ab7f616`
- ACE source revision:
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- ACE main model snapshot:
  `19671f406d603126926c1b7e2adc169acbcade22`
- Packed-BF16/FP32 oracle package manifest SHA-256:
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Experimental converter/package checkpoint:
  `bcb74ed6df6d4b77296f07fe059e8f23e4474830`
- Candidate `fp16-vae-experimental` revision-5 package manifest SHA-256:
  `5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3`
  (714,687 manifest bytes; 7,331,968,095 total package bytes). Two consecutive
  offline conversions produced this identical identity and each passed full
  installed-package hash verification. The candidate VAE payload is exactly
  145 unsharded FP16 records, 84,395,776 elements, and 168,791,552 bytes.
- Experimental runtime-profile contract checkpoint:
  `40b982b409146460571cb7e743ec1845cf239f0d`
- FP16 K7 Conv1D correctness-core checkpoint:
  `82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd`; source SHA-256
  `bdb1ce2732d8617f61132401ab01155163a4f4197e7c7b01eb550b8408553ceb`
- FP16 K7 Conv1D browser-harness checkpoint:
  `8648a4390b2decdb5bbbdc0c119d6562dfc8181a`
- FP16 K7 Conv1D correctness artifact SHA-256:
  `5dffca9c0f76012fa07305d1ff11eac32f56206d9280434bb0a1c639fd13e2d7`
  (48,213 bytes, ignored raw evidence)
- FP16 production-local Conv1D correctness-core checkpoint:
  `75f70f12bdb43ae33b9bd37391b7d49be5aa1704`; source SHA-256
  `fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310`
- FP16 production-local Conv1D browser-harness checkpoint:
  `1320051a2413e1f187143ac0f79958df9218b54f`
- FP16 production-local Conv1D correctness artifact SHA-256:
  `58c15ecf91926ed089f7c866217d46a5bc05a5971949ec7ffd2d48d248253593`
  (58,686 bytes, ignored raw evidence)
- FP16 pointwise correctness-core checkpoint:
  `dd36a04960f846e53c2fd948d67b9aa9ddced4f2`; source SHA-256
  `c801eb209132ed2705a3b7e7b742afd2a6b17855d257938b5df515b6285f3eab`
- FP16 pointwise browser-harness checkpoint:
  `1ab637aa3b174dcf3593beaa56fba6ce8ab4cd44`
- FP16 pointwise correctness artifact SHA-256:
  `22ff18f77d0ec154a45ca2a9dca39e8a4694b94662a176c145d8370307bd9d5c`
  (723,693 bytes, ignored raw evidence)
- FP16 ConvTranspose1D correctness-core checkpoint:
  `d2bf0819d0460f6bd60ebe0457eb091b45e7bf6a`; source SHA-256
  `ecad5f7e981c7310d73565cb15a95123d32725ed6bb41342f484235db3caadd5`
- FP16 ConvTranspose1D browser-harness checkpoint:
  `356f49b20841d0f051fcae3825a87c645c88c386`
- FP16 ConvTranspose1D correctness artifact SHA-256:
  `3c06879410036b42b70e3da408a1376900316defadf4515cbf58651a68962d68`
  (66,364 bytes, ignored raw evidence)
- FP16 Snake correctness-core checkpoint:
  `ae2106c9d5834a3cd5cb836cad484665752230e3`; source SHA-256
  `0e0cc8d1974e6f36942a98777e43c6b48b27c00a8cb0d912ff1f510be426601f`
- FP16 Snake browser-harness checkpoint:
  `59d94643def58a96c1cc081c177c91bd308fd83b`
- FP16 Snake correctness artifact SHA-256:
  `214767002480be3a6fe63e72fb5caaf5bff36f9ffb6cbf15d2f80687ad9190d0`
  (137,318 bytes, ignored raw evidence)
- Complete-window runtime/shader identities and their raw artifact hashes:
  pending and required in the eventual result
- Target machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB
  unified memory

The accepted packed-BF16/FP32 profile remains the numerical and listening
oracle throughout OPT-0011. A candidate manifest never replaces, mutates, or
relabels that oracle.

## Candidate boundary

Extend `model/convert.py`, the authenticated manifest, and the VAE-only runtime
behind an explicit experimental profile. The converter remains the sole
raw-source-to-browser-package entry point and must produce the FP16 package
deterministically, transactionally, and with complete consumed/excluded tensor
accounting. Large packages and raw evidence remain ignored.

Use a new `fp16-vae-experimental` package/profile identity and a
profile-specific converter revision. The approved revision-4 `reference` and
existing `fp16` package identities remain unchanged; do not globally bump
provenance or repurpose either package while OPT-0009 and OPT-0010 depend on
them. Authentication covers every contributing source tensor: in particular,
a weight-normalized output consumes both its `weight_v` and `weight_g`, even
when the logical output keeps `weight_v` as its canonical name.

The initially registered candidate may add:

- IEEE binary16 VAE parameter payloads authenticated back to every
  contributing source tensor and their exact deterministic FP32 weightnorm
  fusion followed by one round-to-nearest-even FP16 transform;
- FP16 decoder input and internal activation storage;
- native FP16 or fixed-32-subgroup heavy kernels selected fail-closed for the
  declared experimental M3 profile, without an undeclared arithmetic fallback;
- for every fixed-32-subgroup kernel, a declared `shader-f16` portable
  workgroup-memory counterpart with the same storage, precision-island,
  accumulation-order, range, and output contract, as required by the repository
  runtime contract; and
- explicit FP16-to-FP32 register boundaries for accumulation, reductions,
  Snake and other nonlinear/range-sensitive operations, followed by an
  explicit FP16 store at every internal workspace boundary, plus FP32 raw
  stereo waveform output.

Do not combine a sampler, DiT, planner, normalization, overlap amount,
cooperative scheduler, or unrelated VAE graph change with this experiment.
Kernel geometry may differ only where needed to realize the declared FP16
mechanism and must remain exactly attributable in the trace.

## Registered three-arm design

| Arm | Precision and kernels | Window | Purpose |
| --- | --- | --- | --- |
| A — FP32-256 oracle | Shipped VAE FP32 weights, activations, and arithmetic under the accepted packed-BF16/FP32 product profile | 256 latent frames, 64 overlap, 128-frame stride | Preserve the numerical/listening oracle and current production geometry |
| B — FP16-256 | Authenticated FP16 weights and internal activation storage, candidate FP16 heavy kernels, and declared FP32 islands | 256 latent frames, 64 overlap, 128-frame stride | Isolate precision, storage, package, and kernel effects while holding window geometry fixed |
| C — FP16-512 | Bit-identical package, precision-island map, kernels, scheduler policy, and arithmetic contract to arm B | 512 latent frames, 64 overlap, 384-frame stride | Isolate window geometry and overlap duplication by comparing C directly with B |

Arm A versus B is the precision/storage/kernel comparison. Arm B versus C is
the window comparison. Arm A versus C is useful product context but cannot be
used to attribute either mechanism alone. Any material B/C difference beyond
window sizes, derived range plans, and their unavoidable resource sizes
invalidates the three-arm attribution.

## Pinned leverage and memory invariants

The following values are pre-registered from the current graph audit. They are
arithmetic invariants to reconcile in static tests and browser evidence, not
measured results or performance promises.

| Quantity | A: FP32-256 | B: FP16-256 | C: FP16-512 |
| --- | ---: | ---: | ---: |
| Maximum activation elements per workspace | 62,914,560 | 62,914,560 | 125,829,120 |
| Bytes per internal activation element | 4 | 2 | 2 |
| Bytes per workspace | 251,658,240 | 125,829,120 | 251,658,240 |
| Three-workspace bytes | 754,974,720 | 377,487,360 | 754,974,720 |
| VAE parameter payload bytes | 337,583,104 | 168,791,552 | 168,791,552 |
| Decoder-input bytes | 65,536 | 32,768 | 65,536 |
| FP32 raw-output bytes | 3,932,160 | 3,932,160 | 7,864,320 |
| FP32 readback bytes | 3,932,160 | 3,932,160 | 7,864,320 |
| Named-buffer subtotal before controls/staging | 1,100,487,680 | 554,176,000 | 939,560,448 |

Thus an FP32 workspace is exactly `251,658,240` bytes at 256 frames, and an
FP16 workspace is exactly the same size at 512 frames. The authenticated VAE
parameter payload target is `337,583,104 -> 168,791,552` bytes. These values do
not waive alignment, range-control, upload, readback, or CPU-staging costs; the
complete account below must add those costs rather than treating the subtotal
as peak memory.

For a 180-second, 4,500-latent-frame decode, the unchanged 256/64 planner
decodes 8,936 latent-window frames. A 512/64 planner decodes 5,908. The static
geometry therefore removes 3,028 decoded frames, a `33.89%` reduction in total
decoded work. Duplicated contextual frames fall from 4,436 to 1,408, a
`68.26%` reduction in duplication specifically. Neither projection is
permission to claim the same wall-time reduction.

## Exact package and resource accounting gate

Before a performance run, retain and reconcile an exact inventory for every
arm:

- every required logical VAE tensor, source dtype/shape/element count, source
  byte span and hash, transformed FP16 byte span and hash, shard/container
  overhead, and total resident payload/file/upload bytes;
- every decoder input, each of the three disjoint workspaces, output, readback,
  range-control/uniform buffer, staging chunk, temporary conversion buffer,
  and owned CPU view, including size, alignment, usage, live interval, alias
  decision, and destruction event;
- logical requested bytes, actual aligned GPU allocation bytes, largest single
  buffer and storage binding, total simultaneously live GPU bytes, bounded CPU
  bytes, and physical browser-process high-water when production promotion is
  considered; and
- all window counts, latent/core/overlap frames, operation ranges, quanta,
  primitive dispatches, physical command buffers, submissions, drains, real
  cooperative idles, progress events, and final output/readback bytes.

Reconcile the inventory to manifest totals, allocator totals, runtime resource
tracking, and zero live owned resources after cleanup. Do not infer physical
memory from logical allocation totals.

The optimized profile fails closed before package loading or pipeline creation
unless `shader-f16` is present and every requested buffer, binding, workgroup,
and allocation fits the adapter/device limits. The M3 subgroup variant also
requires reported subgroup minimum and maximum both equal to 32. An unsupported
or over-limit candidate must not silently select FP32, shrink the window, split
an undeclared binding, or continue with partially loaded resources. Kernel
selection may choose the explicitly registered portable `shader-f16`
workgroup-memory counterpart when the fixed-32 subgroup contract is unavailable;
that counterpart is part of arms B/C rather than a compatibility escape hatch.
Static and browser traces must authenticate the selected kernel family and exact
shader identity for every operation, and must reject a subgroup kernel for which
the matching portable counterpart is absent.

## Frozen deterministic fixtures and acceptance rules

Before candidate code or evidence, OPT-0011 fixes its latent generator as
32-bit `xorshift32` with initial unsigned state `0x00110512`. For every scalar,
apply shifts `13`, `17`, and `5` with unsigned 32-bit truncation after the
left shifts and final update, then store the FP32 value
`((state >>> 8) / 8388608) - 1` in little-endian frame-major/channel-minor
order with 64 channels. The committed fixture contract must reproduce:

- the 256-frame, 16,384-scalar, 65,536-byte fixture SHA-256
  `55333d3ae4a0aca83dc1509b837c577f54646924e658e01e53889dc8a5a44875`;
  and
- the 1,024-frame, 65,536-scalar, 262,144-byte long fixture SHA-256
  `e8919adc02d83f2efcd60bcb6dec4f104628d2ed66742d0eddbffc6b0a481a14`.

The long fixture yields eight 256/64 windows and three 512/64 windows, so the
same 40.96 seconds of logical output contains first, interior, and tail windows
under both planners. Any fixture implementation, operation precision map, or
per-operation contraction allowance must be committed and hash-pinned before
the first candidate GPU result. The operation table may be stricter, but never
looser, than WGSL's declared arithmetic allowance plus the one explicit FP16
rounding at each registered storage boundary. Candidate-versus-declared-mixed-
precision CPU conformance requires zero unexpected classification or signed-
zero mismatches, exact FP16 boundary bits, and the table's frozen per-operation
bound.

Complete raw-waveform A/B acceptance is frozen before measurement as all of:

- every expected sample finite and written, with no clamp or saturation;
- normalized RMS error at most `0.003` and SNR at least `50 dB`;
- Pearson correlation at least `0.9999` per channel and jointly;
- relative RMS/energy drift at most `0.005`, relative peak drift at most
  `0.01`, and absolute DC-offset drift at most `0.001` times the greater of
  oracle RMS and `1e-6`; and
- maximum absolute error divided by the greater of oracle peak and `1e-6` at
  most `0.02`.

These bounds decide numerical escalation, not musical acceptance; fresh
instrumental and vocal listening remains mandatory. They may not be loosened
after observing candidate output under the same registered variant.

## Exact temporal-support gate

The decoder is purely local in latent time. Reverse propagation from every
one of the 1,920 output phases through final K7, the three sequential residual
K7 convolutions at dilations 9/3/1 in each block, transposed convolutions with
strides 2/4/4/6/10, and initial K7 confines support to one of `[-9,+8]`,
`[-8,+8]`, or `[-8,+9]` latent frames relative to the phase's base latent
index. The static contract must independently enumerate all output phases and
pin the maximum radius at 9.

Both candidate planners discard 64 contextual latent frames, far beyond that
radius. With the same FP16 package, precision map, kernels, and global input,
every retained B/C raw FP32 sample therefore has identical inputs and global
padding. The 1,024-frame long-fixture gate requires complete B/C U32 equality,
identical full-output hashes, and zero first/worst mismatches. Seam metrics are
retained as diagnostics only; any B/C bit difference is an indexing, range,
padding, or lifecycle failure and cannot be accepted by a tolerance.

## Primitive numerical gate

Authenticate the candidate package and freeze a machine-readable precision
map for all 88 graph operations before timing. For Conv1D K7/K1,
ConvTranspose1D, Add, Snake, conversions, and every distinct FP32 island:

- compare complete manageable fixtures against an independent FP32 CPU oracle
  and an independent CPU simulation of the declared FP16-storage/rounding and
  FP32-island contract;
- require exact FP16 storage bits at every observable FP16 boundary and exact
  FP32-island bits wherever the declared WGSL arithmetic contract permits it;
  otherwise retain the predeclared contraction-aware absolute/relative/ULP
  bound and identify every mismatch;
- report complete-output maximum absolute and relative error, RMS error,
  normalized RMS error, ULP distribution, SNR, output range, and first/worst
  mismatch location against the FP32 oracle;
- include signed zero, subnormal/normal transitions, FP16 finite-range edges,
  accumulation cancellation, high-gain learned parameters, long reductions,
  dilation/tail ranges, transpose-part boundaries, and values expected to
  overflow an unsafe literal-FP16 path; and
- require complete writes, guards, deterministic repeats, finite expected
  outputs, no unexpected non-finite value, no saturation/clamp, and no device
  event; and
- for every subgroup implementation, run the same manageable precision-map and
  complete-output gate through its portable `shader-f16` workgroup-memory
  counterpart, pin both shader identities, and require the selector/trace to
  distinguish the two paths without changing the registered B/C arithmetic
  contract.

The mixed-precision CPU contract is the kernel-conformance gate; FP32-oracle
error is the approximation evidence carried upward. A primitive that needs a
new FP32 island may be revised and rerun under the same experiment only if the
precision map and shader identities make that change explicit. Timing an arm
before its complete primitive gate passes is invalid.

### Completed production-local FP16 Conv1D checkpoint

The 2026-08-14 actual-Chrome production-local Conv1D run passed its declared
selected synthetic correctness gate, and its independent receipt audit returned
`GO`. Before execution the harness authenticated production core commit
`75f70f12bdb43ae33b9bd37391b7d49be5aa1704`, production core source SHA-256
`fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310`,
the frozen audited K7 authority at commit
`82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd` with source SHA-256
`bdb1ce2732d8617f61132401ab01155163a4f4197e7c7b01eb550b8408553ceb`,
and every generated production/authority shader hash recorded in the receipt.

The harness authenticated the exact B256 production shapes and graph-quantum
coordinates for nine selected graph cases: four K7 ranges and five K1 ranges
spanning all five distinct biased-K1 shapes. A tenth case was a 153-output K1
arithmetic fixture. It compared all 2,687,129 selected output bits between the
authority and production-local kernels on both the first execution and
deterministic rerun,
for 5,374,258 raw-U16 comparisons with zero mismatches. The six K1 cases also
passed complete selected-range, source-order FP32 CPU comparison for 2,392,217
outputs per arm, including signed zero, subnormal, cancellation, channel-tail,
output-tail, and binding-padding cases. QNaN prefills, adjacent and external
guards, complete finite writes, deterministic hashes, exact range coordinates,
and all execution counts passed.

Cancellation after the first drained real K1 graph quantum prevented the
second planned encode, submission, and readback. All 104 harness-owned buffers
were destroyed with zero live resources, destruction was idempotent,
intentional device loss was exactly `destroyed`, and heartbeat observation
continued through cleanup. The raw maximum animation-frame gap was
55,119.100000000006 ms (55.119 s), and the raw maximum timer gap was
24,751.399999976158 ms (24.751 s). Those long gaps are retained as raw
observations and explicitly do **not** support a responsiveness claim.

This result covers selected synthetic inputs and selected production-geometry
ranges only. It did not load the production package, execute a complete graph
or window, measure performance or thermal state, integrate a selector, compare
waveforms, or perform listening. The word `production-local` identifies the
kernel source and graph geometry, not production integration or release
approval.

### Completed FP16 pointwise checkpoint

The 2026-08-14 actual-Chrome pointwise run passed its declared primitive gate
and an independent receipt audit returned `GO`. Before execution the harness
authenticated core commit
`dd36a04960f846e53c2fd948d67b9aa9ddced4f2`, core source SHA-256
`c801eb209132ed2705a3b7e7b742afd2a6b17855d257938b5df515b6285f3eab`,
and the generated ingress/Add shader SHA-256 values
`750bdf07e86c2cfd639eb1217f11d35408d444c8dc5460ca067a6c6d656f7d16`
and `9998dbcc049a1795a0fb6df16e6d404f541d5cbc5d486515b869b4337a528eb5`.

The receipt covers both ingress fixtures and all 15 B256 Add graph operations
across their exact 348 production ranges, plus the 257-element Add arithmetic
fixture. Its 353 ranges compare 361,775,618 output elements on the first pass
and the same number on the deterministic rerun: 723,551,236 complete raw-bit
CPU comparisons in total, with zero mismatches. All 706 encode, submit, drain,
real queue-empty turn, and readback events reconcile. QNaN prefills, adjacent
canaries, external guards, binding padding, complete writes, finite-output
classification, and rerun hashes all passed.

Cancellation after the first fully drained real B256 Add range prevented the
second planned encode, submission, and readback. All 749 harness-owned buffers
were destroyed with zero live resources, destruction was idempotent, the
intentional device-loss reason was exactly `destroyed`, and heartbeat coverage
continued through cleanup. The maximum observed page animation-frame and timer
gaps were 100 ms and 113.20000004768372 ms; these are retained observations,
not thresholds.

This closes only the portable FP32-to-FP16 ingress and FP16-storage Add
primitive checkpoint. It contains no timing, thermal, waveform, listening,
production-selector, or complete FP16-256/FP16-512 window claim.

### Completed FP16 ConvTranspose1D checkpoint

The 2026-08-14 actual-Chrome ConvTranspose1D run passed its declared portable
primitive gate, and its independent receipt audit returned `GO`. Before
execution the harness authenticated core commit
`d2bf0819d0460f6bd60ebe0457eb091b45e7bf6a`, core source SHA-256
`ecad5f7e981c7310d73565cb15a95123d32725ed6bb41342f484235db3caadd5`,
and the six generated shader hashes frozen in the raw receipt.

The receipt authenticates all five canonical B256 ConvTranspose1D operations
and their exact 322-quantum B256 graph topology: range counts 46, 69, 69, 69,
and 69. The executed correctness sample selects one complete output-row range
from each operation, covering left padding, an interior two-tap contribution,
right padding, a two-workgroup time tail, and the longest-output tail. Those
five ranges contain 4,864 selected production outputs. A separate complete
918-output arithmetic fixture covers both padding boundaries, the 65-channel
input-chunk tail, the 9-channel output tail, and the output-time tail.

All 5,782 selected outputs passed on the first execution and deterministic
rerun: 11,564 complete raw-U16 CPU comparisons with zero mismatches. All 12
encode, submit, drain, real queue-empty turn, and readback events reconcile;
QNaN prefills, adjacent canaries, external guards, source padding, complete
finite writes, and rerun hashes passed. Cancellation after the first drained
real B256 range prevented the second planned encode, submission, and readback.
All 54 harness-owned buffers were destroyed with zero live resources,
destruction was idempotent, intentional device loss was exactly `destroyed`,
and heartbeat coverage continued through cleanup. The maximum observed page
animation-frame and timer gaps were 50.099999999999454 ms and
65.30000007152557 ms; these are observations, not thresholds.

This is a selected-range primitive correctness result. Authenticating the
322-quantum topology does not claim that every production output was executed
or compared, and it contains no timing, thermal, waveform, listening,
production-selector, quality, or complete FP16-256/FP16-512 window claim.

### Completed FP16 Snake checkpoint

The 2026-08-14 actual-Chrome Snake run passed its declared portable primitive
gate, and its independent receipt audit returned `GO`. Before GPU acquisition
the harness authenticated core commit
`ae2106c9d5834a3cd5cb836cad484665752230e3`, core source SHA-256
`0e0cc8d1974e6f36942a98777e43c6b48b27c00a8cb0d912ff1f510be426601f`,
harness commit `59d94643def58a96c1cc081c177c91bd308fd83b`, every generated shader hash,
and complete graph-topology SHA-256
`ec79060be88fba5d0a2579826f1ca50730dfba16410da09ffc048963f2623bf3`.

The receipt binds all 36 canonical B256 Snake operations and their exact 813
quanta, covering 844,627,968 topology elements. Six selected exact graph-range
fixtures cover all six production shape families; two complete manageable
arithmetic fixtures cover channel reuse, an odd 257-channel tail, signed zero,
subnormal values, round-to-nearest-even boundaries, finite-range overflow, and
binding padding. All 6,816,035 selected outputs passed on the first execution
and deterministic rerun: 13,632,070 complete raw-U16 CPU comparisons with zero
mismatches. QNaN prefills, adjacent and external guards, complete writes,
source binding padding, exact range coordinates, and deterministic hashes all
passed.

Cancellation after the first drained real B256 Snake range prevented the
second planned encode and submission and prevented all readback. All 85
harness-owned buffers were destroyed with zero live resources and a maximum of
eight live buffers; destruction was idempotent, intentional device loss was
exactly `destroyed`, and no runtime or uncaptured errors occurred. Animation
frame and timer counters remained live through cleanup, but the receipt records
them only as liveness signals and reports no wall-time gaps. It therefore makes
no responsiveness claim.

This is selected-range primitive correctness evidence only. It contains no
kernel-performance timing, thermal, waveform, listening, production-selector,
quality, or complete FP16-256/FP16-512 window claim.

## Full-window and long-latent numerical gate

Use a deterministic, nondegenerate 256-frame latent to compare complete raw
FP32 decoder output for A and B before normalization. Require exact output
length, complete writes, finite stereo samples, nonzero signal, distinct
channels, deterministic repeats, and no hidden clamp. Retain complete hashes
and maximum absolute/relative error, RMS/normalized RMS, SNR, correlation,
peak/energy/DC drift, and first/worst-error neighborhoods. The result must
pass every frozen A/B bound above and report the observed margin; it cannot
declare a looser envelope after observing data or promote on a primitive pass
alone.

Then use the frozen 1,024-frame latent, which crosses first, interior, and tail
windows under both geometries. It need not be a full song initially.
Compare A, B, and C raw waveforms before global normalization, retaining:

- every window/core/discard range and proof that output frames are covered
  exactly once;
- whole-waveform numerical metrics plus sample-level neighborhoods around
  every 256/64 and 512/64 stitch;
- seam value and first-difference discontinuities, local RMS/error, and each
  seam's position in the local 99.9th-percentile transient envelope; and
- complete B/C U32 comparison and full hashes, which must be identical under
  the exact temporal-support proof; and
- matched interior regions and seam neighborhoods as diagnostics for locating
  any forbidden B/C range/index/padding failure.

No C seam may introduce a non-finite value, coverage gap/duplication, channel
misalignment, or B/C bit difference. A difference is a failed geometry gate;
it is never accepted as approximate arithmetic or hidden in an aggregate seam
tolerance.

## Lifecycle, cancellation, and responsiveness gate

- Keep the DiT resident through all eight denoising evaluations, fully drain
  and destroy it, and only then acquire VAE heavyweight resources. Never make
  FP16 conversion by retaining a simultaneous FP32 VAE package or monolithic
  JavaScript/WASM mirror.
- Preserve one FIFO graph owner and the currently selected bounded cooperative
  policy unless a separately allocated experiment changes it. Trace every
  encode, submission, drain, completed real idle, and progress event.
- Probe cancellation after a fully drained physical batch and between VAE
  windows. Require no later encode, submit, readback, sink write, normalization,
  or output finalization; then require drain-before-release and idempotent
  destruction.
- Retain a worker/page heartbeat across authentication, upload, compilation,
  warmup, execution, readback, cancellation, and cleanup. Report raw maximum
  gaps without inventing a responsiveness threshold.
- Any validation error, uncaptured GPU error, device loss, incomplete write,
  count mismatch, post-cancel work, or leaked resource invalidates the affected
  run. Thermal evidence follows the owner-revised decision rule below and does
  not relax any correctness gate.

## Benchmark protocol

1. Land focused static/package/primitive contracts first. Run untimed A/B
   256-frame and B/C long-latent correctness gates before any reportable
   timing.
2. Authenticate and load only the VAE phase. Measure package read/hash,
   transformation if diagnostic, upload, allocation, compilation, and warmup
   separately from decoder execution; production packages must not transform
   weights on every generation.
3. Use symmetric warmup, completion-fenced wall time, balanced six-order
   A/B/C permutations, one outstanding command buffer, and the unchanged real
   cooperative idle. Begin attributed measurements only after at least 30
   continuous seconds at a documented nominal thermal state, poll through the
   sample and cleanup, and disclose every transition. A later non-nominal
   transition does not by itself invalidate a large and consistent balanced
   same-page comparison; marginal, directionally mixed, or variance-overlapped
   evidence remains inconclusive.
4. Time full production-shaped windows, not only isolated kernels. In addition,
   time complete B and C plans over the same frozen 1,024-frame logical output;
   this common-output aggregate, not a 256-window/512-window latency ratio,
   determines the measured geometry benefit. Retain every sample and order;
   report encode, submit-through-drain, completed idle, readback, total window
   and common-output wall, per-operation/family/kernel attribution, logical and
   scheduled work, window duplication, and all memory counts.
5. Repeat only when variance or a B/C attribution ambiguity can change the
   decision. Do not repeat an unchanged benchmark solely to obtain an
   all-nominal trace, do not publish only the fastest sample, and do not use the
   static 33.89% work reduction as a measured speedup.
6. No full song is required for the initial primitive, window, seam, or timing
   decision. A 180-second run remains a later sustained-memory/performance and
   release gate after the candidate has passed its higher-level quality gates
   and is credible against the product budget.

The experiment may retain separate positive or negative conclusions for FP16
storage/kernels and for the 512-frame geometry. A failed C arm does not erase a
useful B result, and a faster C arm cannot rescue an unacceptable B precision
contract.

## Production promotion and listening gate

OPT-0011 is not production-approved by numerical or timing evidence alone.
Before selecting either FP16 arm in production:

- integrate it behind an explicit profile with authenticated package/runtime
  identities and rerun the complete VAE subsystem and raw-waveform gates;
- generate fresh end-to-end instrumental and vocal candidates from the pinned
  requests, seeds, sampler, DCW, planner behavior, and corresponding fresh
  packed-BF16/FP32 oracle runs;
- validate raw waveforms before the ordinary bounded `-1 dBFS` normalization,
  then validate final WAV structure, peak, channels, seams, and hashes; and
- obtain explicit owner listening approval for both the instrumental and vocal
  candidates. Prior Stage 1 approval does not transfer to changed VAE math.

The fresh vocal gate must be long enough to contain actual vocals; the accepted
12-second vocal crops do not qualify. No approximate FP16 candidate becomes
the shipped profile until both listening decisions pass.

## Main risks

- Native FP16 accumulation may overflow or amplify error inside deep K7 and
  transposed-convolution chains. FP32 accumulation or an additional FP32
  activation island may be required even when stored operands are representable.
- Snake is range- and quality-sensitive. Its low isolated profiler share is no
  reason to force it into FP16; retain FP32 unless direct evidence establishes
  a safe narrower contract.
- Larger windows reduce duplicated overlap but double valid work and individual
  activation domains. A kernel may become less responsive, exceed a device
  dimension, or need revised bounded ranges even though the largest storage
  binding is unchanged.
- Candidate package savings can be offset by duplicate staging or conversion
  buffers. Only the complete live-interval inventory can establish peak-memory
  improvement.
- A 512-frame window changes stitch locations and boundary context. Whole-window
  aggregate error can hide a localized audible seam, so every seam remains a
  first-class numerical and listening diagnostic.

## Evidence and disposition

- Evidence conclusion and rationale: `pending`. The deterministic revision-5
  candidate package exists and passes complete package authentication. The
  first actual-Chrome FP16 primitive checkpoint is also positive: scalar-oracle
  and portable-workgroup K7 Conv1D were bit-identical across 3,682,122 outputs
  per execution, including the full 80-range C1024 production block and the
  1,048,577-output two-range cancellation fixture; deterministic reruns,
  12,271 CPU classifications per arm, guards, cancellation, and 81/81 buffer
  cleanup all passed. The raw artifact is 48,213 bytes with SHA-256
  `5dffca9c0f76012fa07305d1ff11eac32f56206d9280434bb0a1c639fd13e2d7`.
  A later selected synthetic production-local Conv1D checkpoint is positive:
  ten selected K7/K1 cases compared 2,687,129 outputs per execution across the
  first run and deterministic rerun, for 5,374,258 production-versus-authority
  raw-U16 comparisons with zero mismatches. Its 58,686-byte raw artifact has
  SHA-256
  `58c15ecf91926ed089f7c866217d46a5bc05a5971949ec7ffd2d48d248253593`,
  and independent audit disposition `GO`. The observed 55.119-second rAF and
  24.751-second timer gaps explicitly do not support responsiveness.
  The subsequent actual-Chrome pointwise checkpoint is positive as well: both
  ingress fixtures and all 15 B256 Add operations across the exact 348 graph
  ranges plus the arithmetic fixture passed 723,551,236 first/rerun raw-bit
  comparisons with zero mismatches, complete guards and writes, deterministic
  hashes, cancellation, heartbeat-through-cleanup, and 749/749 buffer cleanup.
  Its 723,693-byte raw artifact has SHA-256
  `22ff18f77d0ec154a45ca2a9dca39e8a4694b94662a176c145d8370307bd9d5c`,
  and independent audit disposition `GO`. The actual-Chrome ConvTranspose1D
  checkpoint is positive too: it authenticated all five canonical operations
  and the exact 322-quantum topology, then passed 11,564 first/rerun raw-U16
  comparisons over 5,782 selected outputs, including its complete 918-output
  arithmetic fixture, with zero mismatches, deterministic hashes, guards,
  cancellation, heartbeat-through-cleanup, and 54/54 buffer cleanup. Its
  66,364-byte raw artifact has SHA-256
  `3c06879410036b42b70e3da408a1376900316defadf4515cbf58651a68962d68`,
  and independent audit disposition `GO`. The actual-Chrome Snake checkpoint
  is positive as well: it authenticated all 36 canonical operations and their
  exact 813-quanta/844,627,968-element B256 topology, then passed 13,632,070
  first/rerun raw-U16 CPU comparisons over 6,816,035 selected outputs with zero
  mismatches. Its cancellation, guards, deterministic reruns, 85/85 buffer
  cleanup, and heartbeat-through-cleanup liveness checks passed. The heartbeat
  counters are liveness-only and support no responsiveness claim. Its
  137,318-byte raw artifact has SHA-256
  `214767002480be3a6fe63e72fb5caaf5bff36f9ffb6cbf15d2f80687ad9190d0`,
  and independent audit disposition `GO`. These are correctness-only primitive
  results with no timing, thermal, quality, responsiveness, or selector claim. No complete
  FP16-256 or FP16-512 decoder window, waveform comparison, performance result,
  selector integration, or listening decision exists yet, so the experiment
  remains pending and is not production-ready.
- Code disposition and rationale: `benchmark-only`; allocation authorizes the
  bounded three-arm experiment but no production selection.
- Partial result JSON: [canonical schema-v2 evidence](../results/OPT-0011/result.json).
  Its experiment-level conclusion is `inconclusive` because it freezes only
  completed package and primitive evidence while the registered A/B/C gates
  remain pending.
- Interactions: builds on OPT-0008's exact attribution that K7 Conv1D dominates
  the shipped VAE window, uses OPT-0009's FP16 hardware/kernel calibration as
  input, and leaves every integrated exact optimization plus the
  packed-BF16/FP32 oracle intact.
- Revisit/close when: arms A/B/C have complete authenticated package, numerical,
  seam, lifecycle, cancellation, memory, responsiveness, and thermally valid
  window evidence. Production disposition additionally requires fresh
  instrumental and vocal end-to-end waveform/listening gates.

## Closeout — superseded umbrella

The literal A/B/C window experiment never completed, so its experiment-level
evidence remains inconclusive. The package and primitive receipts above remain
valid partial benchmark evidence, while the narrower exact-packed and
revision-7 VAE work is owned by OPT-0028 and OPT-0054/0066/0072. No OPT-0011
profile or window was selected for production; close this umbrella as
superseded without reinterpreting its partial receipt as a complete gate.
