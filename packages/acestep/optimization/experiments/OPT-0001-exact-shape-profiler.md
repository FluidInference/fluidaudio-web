# OPT-0001 — Exact-shape Stage 2 profiler

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

A production-shaped, low-overhead browser profiler can separate useful GPU
work, command encoding, submission/fence latency, mandatory cooperative idle,
pipeline compilation, package I/O/upload, and CPU readback without repeatedly
running full songs. Exact DiT GEMM shapes, one complete DiT layer, one
authenticated 256-latent-frame VAE window, and planner decode boundaries should
identify the first optimization targets and provide reproducible baselines.

No earlier ledger entry addresses this; this is the first Stage 2 experiment.

## Identity

- Baseline commit: `eb7e802df8dc025d9323cbd580b7e67a67c9f56c`
- Candidate commit: `bbc961121b379b314c8929b16ae37eb292cde3cb`
- Production bundle SHA-256: not applicable; the measurement-only harness was
  served directly from the candidate source tree
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Benchmark harness commit: `bbc961121b379b314c8929b16ae37eb292cde3cb`
- Execution profile: `reference-bf16-subgroups`; raw FP16 is diagnostic only
- Machine / GPU cores / memory: MacBook Air `Mac15,12`; Apple M3; 10 GPU cores;
  16 GB unified memory
- macOS build: macOS 26.5.2, build 25F84
- Chrome version and command line: the running framework was
  `151.0.7922.109` (reduced user agent `151.0.0.0`). The diagnostic samples
  used the ordinary nonisolated user profile; no command line is attributed.
- WebGPU adapter features and limits: captured from the benchmark result; the
  profiler reports Apple/Metal 3, fixed 32-lane subgroups, `subgroups` and
  `shader-f16` adapter support, exposed 4 GiB-minus-four-byte buffer/binding
  limits, and 32 KiB exposed workgroup storage. The harness requests only its
  largest exact fixture binding and the kernel's 9 KiB workgroup requirement.

## Change

Add benchmark-only browser harnesses, workload accounting, and a
machine-readable result writer. The retained profiler covers:

- exact DiT GEMMs at M=2,250 for H→H, H→1,024, H→6,144, and 6,144→H;
- an authenticated static 256-latent-frame VAE window with exact
  per-operation-family and representative-quantum aggregates;
- allocation/preparation, pipeline compilation, encode, submit, queue drain,
  explicit idle, readback, checksum, and sentinel validation as distinct GEMM
  measurements; and
- a strict schema, duplicate-key-safe parser, canonical serializer, provenance
  identity, thermal pre-gate, raw-sample summary, command counts, heartbeat,
  and logical GPU-byte accounting.

The planned full synthetic layer, planner, package-I/O, cancellation, and
external process-tree probes were deliberately left for decision-relevant
follow-ups; they are not claimed by the retained OPT-0001 code.

Instrumentation aggregates inside the benchmark/worker. It must not emit one
message per VAE quantum or alter the production execution path. Rollback is
deleting the benchmark-only files; no model/package or kernel math changes are
part of OPT-0001.

## Correctness gate

- Oracle identity: approved Stage 1 baseline at tag
  `stage1-approved-2026-08-13`, plus existing independent CPU/vector fixtures.
- Required tests/tensor taps: current GEMM and VAE contract suites; exact-shape
  deterministic sentinels; finite/nonzero/full-domain checks; command-count and
  timing-field invariants; actual M3 WebGPU execution in the selected profile.
- Declared tolerances: bit identity where the benchmark reuses production
  dispatches; existing profile-specific primitive tolerances for independent
  CPU comparisons.
- Listening required and why: no. Measurement-only code does not alter
  production model math or output.
- Result: passed. All 25,344,000 outputs in every actual-M3 execution were
  finite and nonzero, and every independently recomputed range-boundary
  sentinel was bit exact. The static VAE workload report is pinned by exact
  production-planner counts and a deterministic serialized-report hash.

## Benchmark protocol

- Fixture, prompt/lyrics, duration, seed: no song for microbenchmarks. Exact
  production tensor shapes use deterministic nondegenerate inputs and the
  approved package identity. Subsystem fixtures record their own seed.
- Warmup policy: compile pipelines and populate required cache once; no hidden
  full-generation warmup. Report compile/upload separately.
- Thermal pre-gate and polling: nominal for at least 30 s; poll every 1,000 ms.
- Paired order: baseline-only reproducibility first; future A/B uses balanced
  AB/BA ordering.
- Timing method: authoritative wall time around submit through final completion
  fence; timestamp queries diagnostic only when enabled.
- Memory run method: runtime logical allocation accounting plus isolated Chrome
  process-tree footprint outside the timed inner loop.
- Cooperative scheduler topology: report both useful GPU work and the exact
  production drain/idle policy. Diagnostic packed submissions must be labeled
  and must not silently become the production default.
- Exact commands:

  ```sh
  pnpm exec vitest run test/benchmark-result.test.ts \
    test/opt-0001-gemm-profiler-contract.test.ts \
    test/opt-0001-vae-workload.test.ts test/gemm-contract.test.ts \
    test/vae-backend.test.ts test/vae-decoder-contract.test.ts
  pnpm check
  pnpm exec vite --host 127.0.0.1 --port 5176 --strictPort
  ```

  The browser harness was run from
  `http://127.0.0.1:5176/test/browser/opt-0001-gemm-profiler.html` by its
  explicit **Run profiler** control. Thermal pressure was polled once per
  second with `notifyutil -g com.apple.system.thermalpressurelevel`; each of
  the four retained Chrome observations followed 30 continuous nominal
  seconds.
- Samples: at least three thermally valid samples for reportable baselines;
  retain every sample and median/range.

## Results

The retained exact-shape GEMM observations are deliberately classified as
**diagnostic, nonisolated, and not a close A/B baseline**. One measured
execution per shape produced substantial foreground/GPU-contention variance:

| Aggregate metric | Raw samples | Median | Range |
| --- | --- | ---: | ---: |
| active wall, ms | 312.5, 556.6, 595.4, 689.3 | 576.0 | 376.8 |
| cooperative wall, ms | 353.9, 597.1, 635.3, 730.0 | 616.2 | 376.1 |
| logical active TFLOP/s | 0.452985, 0.254326, 0.237752, 0.205365 | 0.246039 | 0.247620 |
| logical cooperative TFLOP/s | 0.399994, 0.237075, 0.222820, 0.193915 | 0.2299475 | 0.206079 |
| compile + bind, ms | 375.0, 95.7, 90.6, 78.8 | 93.15 | 296.2 |

An earlier in-app diagnostic reached 0.508835 logical active TFLOP/s. The
post-commit real-Chrome smoke reached 0.256399 TFLOP/s. These observations are
not pooled with the four samples. The robust conclusion is architectural, not
a close timing claim: the current exact GEMM is a portable workgroup kernel,
uses no subgroup operations, and is materially below the 1.1–1.5 TFLOP/s
Stage 2 planning gates even at the best observed diagnostic rate.

The authenticated static VAE report for one 256-latent-frame / 10.24-second
output window found:

- 88 operations, 2,051,997,696 output elements, 623,639,753,728 valid MACs,
  and 1,247,279,507,456 convolution FLOPs;
- 62,622 decoder command buffers, 62,702 primitive dispatches, one readback
  command buffer, and 62,623 queue drains;
- 62,622 ms of *requested* one-millisecond idle before timer overshoot; and
- 38 representative quantum classes under the uniform 32,768-output cap.

This is exact workload accounting, not an empirical decoder wall-time result.
It nevertheless localizes an avoidable scheduling problem: the same output cap
assigns wildly different work to pointwise and convolution operations, while
Snake and add alone create 36,816 command buffers per window.

The originally proposed full synthetic layer, planner boundaries, package I/O,
and physical-memory probes were not run in OPT-0001. GEMM and VAE accounting
already decide the first two production experiments, and the owner's cadence
authority favors postponing broader probes until a candidate can change a
decision. Future kernel A/B pages must execute balanced pairs in one page to
reduce the focus/contention variance seen here.

## Evidence and disposition

- Evidence: `positive`. The schema, exact-shape GPU harness, and authenticated
  VAE workload profiler passed their measurement-only correctness gate and
  localized the first production experiments without changing model math.
- Disposition: `integrated`; the measurement infrastructure is retained, while
  no production kernel or inference math changed.
- Result JSON: [`../results/OPT-0001/result.json`](../results/OPT-0001/result.json)
- Implementing/revert commits:
  `bbc961121b379b314c8929b16ae37eb292cde3cb`
- Interactions with previous experiments: none
- Revisit when: a production candidate permits balanced in-page A/B, an
  isolated process-tree measurement becomes decision-relevant, or a subsystem
  winner warrants escalation.
- Follow-ups: allocate work-aware VAE quantum sizing as `OPT-0002`, then an
  experiment-local packed-BF16/FP32 subgroup GEMM as a separate OPT ID. Do not
  fold either production change into this measurement experiment.
