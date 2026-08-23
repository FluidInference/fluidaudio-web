# OPT-NNNN — Short title

## Status

- Evidence: `pending`, then `positive`, `negative`, or `inconclusive`
- Disposition: `benchmark-only`, `pending-integration`, `integrated`,
  `superseded`, or `abandoned`
- Date:
- Author/agent:
- Risk class: `exact`, `reordered-rounding`, or `approximate`

## Hypothesis

State the mechanism, affected production path, expected benefit, and why prior
ledger entries do not already answer it.

Expected performance is a hypothesis, not an automatic acceptance threshold.

## Identity

- Baseline commit:
- Candidate commit:
- Production bundle SHA-256:
- Model manifest SHA-256:
- Reference fixture manifest SHA-256:
- Benchmark harness commit:
- Execution profile:
- Machine / GPU cores / memory:
- macOS build:
- Chrome version and command line:
- WebGPU adapter features and limits:

## Change

Describe the one primary variable, affected kernels/files, resource/lifetime
changes, and rollback plan.

## Correctness gate

- Oracle identity:
- Required tests/tensor taps:
- Declared tolerances:
- Listening required and why:
- Result:

## Benchmark protocol

- Fixture, prompt/lyrics, duration, seed:
- Warmup policy:
- Thermal pre-gate and polling:
- Paired order:
- Timing method:
- Memory run method:
- Cooperative scheduler topology:
- Exact commands:

## Results

List every baseline and candidate sample. Report medians and ranges, per-stage
latency, effective throughput where meaningful, logical GPU high-water mark,
isolated Chrome-tree physical peak, responsiveness, cancellation latency, and
thermal validity.

## Evidence and disposition

- Evidence conclusion and rationale:
- Code disposition and rationale:
- Result JSON:
- Implementing/revert commits:
- Interactions with previous experiments:
- Revisit when:
- Follow-ups:
