# OPT-0068 — Native Metal/MPS/MLX M3 ceiling gate

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: alternate native execution backend and changed floating-point reduction
  order; no browser, package, model, sampler, or production-default change

## First-principles basis

The measured browser runtime is primarily GPU-kernel-bound, not stalled on the
host. OPT-0043 placed about `90.6-90.9%` of dense submit-through-drain wall
inside timestamped compute passes, so deleting every observed outside-pass
millisecond would provide only about `1.10x`. OPT-0067 independently measured
an exact attention reuse win, but its complete eight-evaluation projection is
only `5.8284 s`. Neither result identifies enough submission overhead to close
the product target.

The stock Chrome adapter exposes ordinary fixed-32 subgroups, `shader-f16`,
and packed 4x8 integer dot products, but no cooperative/SIMD-group matrix type
or matrix multiply-accumulate instruction. Native Metal on the same M3 can use
`simdgroup_matrix` MMA through MPS or MLX. A separately authenticated
same-machine reference measured:

- MPS `MPSMatrixMultiplication`: `3.214901556 TFLOP/s` GPU and
  `3.208685434 TFLOP/s` commit-to-completion wall over a production-weighted
  five-shape mix;
- MLX `0.32.0`: `3.0098765 TFLOP/s` on the measured QKV shape; and
- the final tuned scalar Parakeet WGSL path: `2.726526 TFLOP/s`.

The comparison must stay honest. MPS was pure `C=A*B`, while Parakeet already
uses native-FP16 contraction plus production epilogues. MPS is only
`1.179120x` above that final Parakeet WGSL rate, not a generic `2x` browser
speedup. ACE has more headroom because its accepted FP16-input/FP32-accumulate
dense kernels sustain about `1.61-1.84 TFLOP/s`; the approximate K4 screen
reached `1.89251 TFLOP/s` GPU but changes reduction semantics. Native MMA must
therefore be measured on ACE's actual shapes, boundaries, and data before a
port is justified.

For the 180-second direct request, the counted core is approximately:

| path | repeated dense | attention | two-window VAE convolution | total counted core | impossible `3.2149 TFLOP/s` floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| eight evaluations | `50.7343 TFLOP` | `4.7829 TFLOP` | `22.52 TFLOP` | `78.04 TFLOP` | `24.3 s` |
| five evaluations | `31.7089 TFLOP` | `2.9893 TFLOP` | `22.52 TFLOP` | `57.22 TFLOP` | `17.8 s` |

Those are unattainable arithmetic lower bounds, not predictions. Attention,
convolution, nonlinearities, package loading, command construction, output,
and fanless thermal throttling cannot all run at the MPS GEMM rate. A bounded
native gate is the smallest credible way to determine whether the observed
matrix capability yields product margin rather than another synthetic peak.

## Frozen implementation direction

1. Keep the browser runtime and accepted packages unchanged. Add a standalone
   benchmark-only native owner outside the production worker. Do not select it
   from the browser, change a manifest, or add a silent fallback.
2. Use the exact `Mac15,12` Apple M3 with 10 GPU cores and 16 GiB unified
   memory, macOS `26.5.2` build `25F84`. Record compiler, SDK, MPS, MLX, Metal
   family, OS, and executable/source SHA-256 identities.
3. First measure the four ACE M2250 dense shapes and registered multiplicities
   `4/2/2/1`:
   `K2048->N2048`, `K2048->N1024`, `K2048->N6144`, and
   `K6144->N2048`. Use authenticated actual revision-7 weights and deterministic
   actual activations, not zero-filled throughput fixtures. Report contraction,
   bias/epilogue, explicit materialization, command, and synchronization walls
   separately.
4. Compare MPS and MLX against the unchanged exact WebGPU FP16-input/
   FP32-accumulate owner. Native operands may be FP16 only with FP32 MMA
   accumulation. Native-FP16 accumulation remains forbidden by OPT-0009's
   signed-zero, overflow, cancellation, and long-K failures.
5. If the dense gate is positive, implement one complete 24-layer evaluation
   in MLX or custom Metal while retaining the pinned shift-3 Euler/DCW inputs,
   online-softmax semantics, explicit rounding/materialization boundaries, and
   bounded cancellation points. Library fusion may not erase an observable
   storage boundary without a distinct quality experiment.
6. Independently measure complete revision-7 C512 and C2314 VAE windows using
   native convolution/transpose-convolution plus custom kernels where library
   semantics differ. Never materialize dense attention scores. Keep one DiT
   owner, destroy it, then create one VAE owner as in the browser architecture.
7. MLX lazy graphs must be evaluated at bounded phase/evaluation boundaries.
   MPSGraph compiled graphs, constants, temporaries, and pipeline caches count
   toward peak unified memory. Do not retain browser and native weight copies
   simultaneously to make a benchmark convenient.

## Correctness and quality gates

Validate in this order:

1. For every dense shape, compare all outputs against the independent
   FP16-input/FP32-accumulate CPU contract and the accepted WebGPU owner. Retain
   signed-zero, cancellation, finite-range, long-K, tail, complete-write,
   canary, determinism, and non-finite screens. Report ULP/class differences,
   NRMSE, SNR, Pearson, maximum absolute and relative error; do not require raw
   identity from a different reduction tree.
2. For the complete native evaluation, require the same immutable M2250/C98
   inputs and capture every selected layer boundary plus the final evaluation
   latent. Compare against the pinned browser oracle and require deterministic
   repeats, finite/class stability, correct route counts, and no dense score
   matrix allocation.
3. For C512 and C2314, validate package tensors, each kernel family, all former
   rounding boundaries, complete raw waveform, seams, overlap/stitch,
   cancellation, and zero live resources. Revision-7 K4 remains approximate
   versus the scalar quality oracle and retains OPT-0066/OPT-0044 gates.
4. A complete native stack must then pass the accepted 12-second direct
   instrumental and 30-second planner-vocal blinded listening gates with
   explicit owner approval. A five-evaluation native candidate additionally
   requires OPT-0065's separate schedule approval.

No primitive metric, MPS conformance, MLX agreement, or attractive waveform
metric substitutes for the listening boundary.

## Thermal and performance gate

- Finish compilation, package authentication, allocation, and warmup before
  each accepted arm. Use balanced independently cooled WebGPU/native/native/
  WebGPU arms with at least 30 continuous seconds at thermal level 0 and a
  trace through completion and destruction.
- Retain raw GPU and fenced wall samples. Separate matrix pass, epilogue,
  complete evaluation, complete VAE window, initialization, and cleanup.
- Dense must improve the exact WebGPU weighted wall by at least `1.40x` and
  sustain at least `2.40 TFLOP/s` including required materialization and
  epilogues in both paired directions before whole-graph work is allocated.
- A complete evaluation must improve by at least `1.25x`, and native C512 plus
  C2314 decoder walls by at least `1.25x`, with no unrelated phase regression
  above `2%` after absolute deltas are considered.
- Run a separate continuous repeated native workload for at least 60 seconds.
  Report level transitions and throughput by interval; the final third must
  retain at least `80%` of the first nominal third.
- Before product escalation, a source-to-WAV budget using measured native
  components must project the approved eight-evaluation path at no more than
  `55 s`, or an independently listening-approved five-evaluation path at no
  more than `45 s`. These thresholds reserve margin for first-use variance and
  passive cooling; they are not product claims.

A pass authorizes only a complete native prototype and its declared quality
gates. Under one minute is established only by a fully cached and a clearly
labelled first-use 180-second Generate-to-WAV run on the pinned machine, with
thermal trace, bounded memory, durable output, and explicit owner approval.

## Historical product architecture if the gate passes

This direction was registered before the product scope was frozen to stock
browser WebGPU/WASM. It is retained as provenance only and does not authorize
a native prototype, companion, `WKWebView` owner, or product selection.

The preferred product boundary is a signed macOS compute owner using
Swift/Objective-C++ with MPSGraph and custom Metal, while retaining the current
browser UI in `WKWebView`. A standalone browser may instead use an
authenticated localhost companion. Only request metadata, progress, and the
final durable file URL cross that boundary; model tensors and the full waveform
do not. Static M2250/C98 graphs should be precompiled, and the existing
DiT-resident-then-destroy-before-VAE phase boundary remains mandatory.

## Authority

- [OPT-0009 accumulation calibration](OPT-0009-fp16-gemm-calibration.md)
- [OPT-0043 timestamp utilization profile](OPT-0043-webgpu-timestamp-utilization-profile.md)
- [OPT-0059 two-window geometry](OPT-0059-vae-c2378-short-projection-gate.md)
- [OPT-0066 revision-7 VAE quality gate](OPT-0066-vae-revision7-dual-k4-quality-gate.md)
- [OPT-0067 isolated attention screen](OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md)
- pinned Parakeet reference `../parakeet.wgsl` at
  `7ee112738262a6f5a0efd2f150748a4087432fbb`
- same-machine native reference report
  `/Users/hamza/Documents/parakeet-web/bench/NATIVE_GEMM_REFERENCE_RESULTS.md`

No code, native executable, browser, GPU, package, production selection, or
quality claim occurred when this experiment was registered.

## Closeout — 2026-08-15

The benchmark-only Phase-1 source was audited at repository HEAD
`115d0857715b6aad215752e8e3b0f9e1635f580c`. Its bounded source scope is the
13 files under `benchmark/opt-0068-native-metal/`: the Swift package and four
Swift implementation files, one Swift test file, the MPS and MLX fixture/
runner documentation, two Python tools plus one Python test, the fixture JSON
schema, and the local ignore file. Generated `.build` contents and
`__pycache__` are not evidence.

The documented static audit passed after removing a stray application-level
`--` from the Swift command examples:

- `swift test` passed all 7 Swift Testing tests;
- the release Swift package built successfully;
- `ace-opt-0068-mps --mode describe` completed through the early-return path,
  before package authentication or `MTLCreateSystemDefaultDevice()`;
- both Python entry points passed CPython 3.13 bytecode compilation;
- all 3 Python unit tests passed; and
- `fixture.schema.json` parsed successfully with `jq empty`.

No authentic six-activation/nine-output M2250 fixture exists. Therefore no
`inspect`, `correctness`, `measure`, or `sustained` mode was run; no Metal
device, MPS operation, MLX import, thermal trace, timing sample, receipt, or
result JSON was created. The same-machine MPS/MLX/Parakeet references above
remain ceiling/mechanism evidence only and establish no ACE speedup.

The experiment closes `inconclusive` / `benchmark-only`. Native execution and
selection are explicitly outside the stock-browser WebGPU/WASM product scope,
so this harness is frozen as non-production reference material. It authorizes
no implementation, runtime/default change, or further native work.
