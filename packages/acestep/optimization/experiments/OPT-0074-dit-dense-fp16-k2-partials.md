# OPT-0074 — DiT dense FP16 K2 partials

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `reordered-rounding`
- Allocation baseline: pushed `main` commit
  `f3eb9fafc406676681b2892e08890d51a8301a6d`

## First-principles basis

The accepted three-minute product spends `54,986.2 ms` in DiT denoising and
`32,586.7 ms` in VAE decode. OPT-0043 established that dense GEMM is primarily
GPU-kernel-bound: its original production-weighted exact arm placed
`90.6326%` of submit-to-drain wall inside timestamped compute passes. An
unpersisted allocation-time diagnostic was directionally consistent but is not
evidence for this record. The bound result below independently timestamps all
three frozen arms and remains the only new OPT-0074 performance evidence.

The exact OPT-0009 owner converts every FP16 activation and weight operand to
FP32 and updates each FP32 accumulator once per K element. OPT-0032 instead
forms one native-FP16 `dot(vec4)` per four consecutive K elements, widens once,
and updates FP32 state once per K4. It improved the production-weighted
primitive wall by `1.404996x`, but all-K4 trajectory integration failed the
final-latent maximum-error gate (`0.9955760 > 0.25`). Retaining exact arithmetic
only for the K6144 down projection improved that maximum to `0.6395987` but did
not close the gate. K8/K16 worsened performance, dynamic INT8 failed its
adversarial gate, and exact cooperative geometries did not provide enough
headroom.

K2 is the unmeasured point between exact scalar FP32 updates and K4 partials.
For each output and increasing K4 storage group, compute the first and second
consecutive `dot(vec2<f16>)` independently, widen each result immediately, and
add both to the FP32 running accumulator in source order. This halves the FP32
dependency-chain length versus exact accumulation while bounding each local
FP16 reduction to one addition rather than K4's three-element reduction tree.
It reuses the authenticated revision-8 physical K4 layout, so the primitive can
isolate arithmetic without inventing another several-gigabyte package.

The mechanism is worth testing only if it yields seconds, not a cosmetic
microbenchmark win. The current exact repeated-dense projection is tens of
seconds across eight evaluations; a credible `1.15x` weighted primitive win is
large enough to project multiple seconds of product saving. Anything smaller
stops before package or trajectory work.

## Frozen implementation direction

1. Add a benchmark-only fixed32/WG128 `M32 x N128` kernel beside OPT-0032.
   Keep its four subgroups, eight rows per subgroup, four adjacent outputs per
   lane, barrier-free ownership, FP32 input/output storage, and revision-8
   `[N/128,K/4,output4,lane32,K4]` FP16 weight layout.
2. Within every increasing K4 group, broadcast one `vec4<f16>` activation per
   owned row. For each of four outputs, compute K0/K1 as one `dot(vec2<f16>)`,
   widen and add it to the FP32 accumulator, then compute K2/K3, widen and add
   it. Do not retain an FP16 running accumulator, reassociate across K2 pairs,
   use split-K, or change output materialization.
3. The primitive harness compares exact OPT-0009, K2, and the known K4 arm on
   the four M2250 shapes and the signed-zero, cancellation, finite-range, and
   K6144 adversarial fixtures. It requests WebGPU timestamps and records GPU
   and fenced wall separately in a balanced order.
4. No source under the production selector, converter, manifest, package
   identity, runtime graph, sampler, attention, VAE, or demo default changes in
   the primitive phase. A losing or numerically ineligible kernel is removed
   or retained benchmark-only with its negative evidence.

## Primitive correctness and performance gates

1. Prove layout indexing/inversion, exact shape planning, complete guarded
   writes, tail rows, deterministic K2 reruns, finite outputs, canaries,
   fixed-subgroup capability rejection, idempotent cleanup, and zero uncaptured
   GPU errors.
2. Compare all `25,344,000` production-shape outputs and the complete OPT-0032
   adversarial set against OPT-0009. Retain the OPT-0032 full thresholds:
   NRMSE at most `0.02`, SNR at least `34 dB`, Pearson at least `0.999`, maximum
   absolute error at most `0.25`, and no non-finite output. Retain its
   adversarial thresholds of NRMSE at most `0.05`, SNR at least `26 dB`,
   Pearson at least `0.995`, and maximum absolute error at most `0.5`.
3. K2 must not be worse than K4 on aggregate full-shape or adversarial NRMSE,
   SNR, maximum absolute error, or finite/class behavior. This is a mechanism
   check, not permission to weaken the later trajectory gate.
4. After one untimed warmup per shape and at least 30 continuous seconds at
   thermal level 0, retain every sample in balanced exact/K2/K4 orders. K2 must
   beat exact fenced wall on all four production shapes, reach at least
   `1.15x` on the `4/2/2/1` weighted wall, and save at least `25 ms` per
   weighted evaluation mix. GPU timestamps must show that the gain occurs
   inside the compute pass rather than being a queue artifact. The harness
   conservatively applies the same `1.15x` and `25 ms` materiality thresholds
   to weighted timestamped GPU time and requires a GPU and wall win on every
   shape. It also requires all six paired rounds to agree and the K2 and exact
   sample ranges not to overlap, both per shape and for the weighted mix; a
   mixed or variance-overlapped result remains inconclusive.

Failure of any correctness, lifecycle, thermal-provenance, or performance gate
closes this direction as negative or inconclusive without trajectory work.

## Escalation gate if the primitive passes

Allocate no new experiment merely to continue the same frozen K2 mechanism.
Bind it to the existing authenticated revision-8 package only in a diagnostic
profile, then compare exact and K2 at every one of the eight sampler taps and
the final latent on the immutable M2250/C98 request. Require deterministic
repeats, no non-finite/class failure, NRMSE at most `0.02`, SNR at least
`34 dB`, Pearson at least `0.999`, and maximum absolute error at most `0.25`.
Use a balanced one-evaluation or bounded graph timing screen before any full
product timing.

A passing trajectory still authorizes no production selection. Because K2
changes dense reduction rounding, promotion requires the ordinary 12-second
instrumental raw-waveform/output gate, the accepted vocal/listening gate, an
authenticated replace-not-duplicate package identity, bounded memory and
cancellation evidence, and a material Generate-to-WAV win. If those gates pass,
integration must occur under a distinct production identity and retain exact
OPT-0009 as the oracle rather than a silent fallback.

## Benchmark protocol

- Machine: local `Mac15,12`, Apple M3 10-GPU-core, 16 GiB unified memory.
- Browser target: Codex in-app Browser Chromium 151 WebGPU surface with
  `shader-f16`, fixed-32 `subgroups`, and standard `timestamp-query`; record
  the exact user agent, OS build, adapter features, and requested limits.
- Primitive fixtures: OPT-0032 deterministic four-shape and adversarial inputs.
- Warmup: compilation, correctness, deterministic rerun, then one untimed
  dispatch per arm/shape before the thermal gate.
- Thermal rule: distinct raw trace, at least 30 continuous nominal seconds,
  current final observation at launch, and disclosure through cleanup under
  the owner-revised PLAN protocol.
- Timing: balanced order with one command buffer, one timestamp pair, one
  submit, and one matching drain per sample; no readback inside timing.
- Memory: exact tracker reconciliation and zero live bytes after repeated
  destroy.

## Results

The benchmark-only candidate and harness were frozen and pushed at
`3ad45dade32cb5d53c37a685ef1f2ba01427fef0`. The Codex in-app Browser reported
Chrome `151.0.0.0`, Apple `metal-3`, fixed-32 subgroups, and standard
`timestamp-query`. The thermal trace contained `329` observations, no
non-nominal value, and a maximum `938 ms` poll gap. Launch followed `43.2 s`
after READY was observed, the last pre-launch level-0 sample was `868 ms` old,
and polling continued beyond cleanup.

Correctness passed. K2 and K4 each compared all `25,344,000` production-shape
outputs plus `17,408` adversarial outputs against exact OPT-0009. K2 was
deterministic, finite, complete, and better than K4 on every frozen aggregate
metric:

| Aggregate | Arm | NRMSE | SNR dB | Pearson | Max abs | Class changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | K2 | `0.0002116480` | `73.4877` | `0.9999999796` | `0.00990677` | `1,636` |
| full | K4 | `0.0003114215` | `70.1330` | `0.9999999524` | `0.01443172` | `2,484` |
| adversarial | K2 | `0.0000237415` | `92.4898` | `0.9999999997` | `0.00708961` | `1` |
| adversarial | K4 | `0.0000358939` | `88.8996` | `0.9999999994` | `0.01056182` | `2` |

The aggregate speed mechanism was material but the frozen consistency gate did
not pass. The production-weighted exact/K2/K4 means were respectively
`249.3500 / 216.7167 / 177.6667 ms` fenced wall and
`223.9911 / 184.3637 / 150.5362 ms` timestamped GPU. Thus K2's mean speedups
were `1.15058x` wall and `1.21494x` GPU, saving `32.6333 / 39.6274 ms`; median
speedups were `1.20116x / 1.27534x`. All six weighted GPU rounds won, but the
six wall ratios were `1.30154, 1.18835, 1.21355, 1.15341, 1.26984, 0.89985`.
K2 and exact ranges overlapped, and the final round contained paired losses on
two shapes. Under the predeclared rule this is directionally mixed and
variance-overlapped evidence, hence **inconclusive**, not a genuine win.

Trajectory, package, waveform, listening, and production integration were
therefore not attempted. The diagnostic mean would project about `6.266 s` of
eight-evaluation dense saving, but that projection has no performance or
integration authority. The benchmark-only kernel remains isolated from every
production selector. Cleanup reconciled all `50` buffers to zero live bytes,
repeated destroy was idempotent, and peak tracked benchmark storage was
`388,230,688` bytes.

The complete `72` timing samples, per-case numerics, shader hashes, and cleanup
receipt are in [the browser receipt](../results/OPT-0074/browser-receipt.json).
The compact decision record is [result.json](../results/OPT-0074/result.json),
with its [thermal gate](../results/OPT-0074/thermal-gate.json) and
[thermal trace](../results/OPT-0074/thermal-trace.json).

## Authority and interactions

- [OPT-0009 exact FP16/FP32 production dense owner](OPT-0009-fp16-gemm-calibration.md)
- [OPT-0032 K4 primitive](OPT-0032-dit-dense-fp16-k4-partials.md)
- [OPT-0037 all-K4 trajectory failure](OPT-0037-dit-k4-layout-trajectory-integration.md)
- [OPT-0038 K8/K16 negative screen](OPT-0038-dit-dense-fp16-k8-k16-partials.md)
- [OPT-0043 timestamp utilization profile](OPT-0043-webgpu-timestamp-utilization-profile.md)
- [OPT-0056 selective exact-down trajectory failure](OPT-0056-dit-selective-k4-exact-down-projection.md)
- [OPT-0058 dynamic INT8 negative screen](OPT-0058-dit-dense-int8-dp4a.md)
- [Final current production result](../results/OPT-0073/result.json)

Revisit after a negative result only if a future standardized WebGPU matrix
primitive or materially different arithmetic/dataflow mechanism changes the
dependency-chain cost; do not repeat an unchanged K2 geometry or weaken its
gates.
