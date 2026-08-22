# Optimization program

This directory is the durable memory of the ACE-Step WebGPU optimization
program. Read this file, `BASELINE.md`, `LEDGER.md`, and every related
experiment record before changing performance-sensitive code.

## Authorization state

**Stage 2 was explicitly authorized by the repository owner on 2026-08-13.**

The approved browser artifacts, owner decision, and remaining unclaimed native
and release-hardening evidence are recorded in `LISTENING_CANDIDATE.md`. The
frozen quality baseline is recorded in `BASELINE.md`; experiments start only
from its pushed checkpoint and never silently widen its quality envelope.

Correctness and memory-safety architecture—online attention, phase staging,
bounded uploads, activation liveness, chunked VAE decoding, and cooperative
submission—belongs to Stage 1. Performance-driven precision changes, changed
accumulation order, approximations, aggressive fusion, tile sweeps, and
quantization do not.

## Working method

1. Search `LEDGER.md`, `experiments/`, and Git history for the idea.
2. Allocate the next never-reused `OPT-NNNN` ID in `LEDGER.md`.
3. Copy `EXPERIMENT_TEMPLATE.md` to
   `experiments/OPT-NNNN-short-name.md` and complete the pre-experiment fields.
4. Start from a pushed baseline commit and isolate the mechanism well enough to
   understand the evidence.
5. Pass the declared correctness gate before timing.
6. Follow the thermal and benchmark protocol in `PLAN.md`.
7. Save the schema-valid small result to `results/OPT-NNNN/result.json`.
8. Record every sample and the evidence, including negative and inconclusive
   work.
9. Record evidence separately from code disposition; a positive isolated
   result may remain pending integration while other bottlenecks are explored.
10. Link integrated implementation and result commits from the ledger and
    update the production baseline only after applicable correctness and
    listening gates pass.

IDs, failures, and superseded results are never deleted. Large audio, shader
traces, and raw profiles stay in the ignored `artifacts/` tree; commit their
hashes and reproduction commands when they matter.

## Evidence and disposition vocabulary

Evidence describes what the measurement established:

- `positive`: a credible useful improvement at the measured boundary;
- `negative`: the hypothesis was refuted, regressed the system, or produced no
  worthwhile net benefit; and
- `inconclusive`: the protocol could not distinguish the result or was invalid.

Disposition describes where the code currently lives:

- `benchmark-only`: measurement infrastructure or an intentionally isolated
  candidate;
- `pending-integration`: positive evidence whose production work or
  higher-level validation is unfinished;
- `integrated`: retained in the production path after applicable correctness
  and listening checks;
- `superseded`: replaced by a demonstrated incompatible improvement; and
- `abandoned`: no further work is currently justified.

Predicted speedups and throughput numbers are hypotheses, never automatic
vetoes. A useful result does not become negative because it missed a speculative
target.

No public A/B switches are retained merely to preserve experiment code.

## Benchmark escalation

The owner has delegated optimization and benchmark-cadence decisions to the
agent. Prefer the cheapest authoritative boundary: microbenchmark,
authenticated subsystem fixture, short end-to-end run, then full 180 s run.
Repetition scales with uncertainty rather than a universal sample count.
Short 12–30 s production generations are ordinary decision tools and need not
be artificially avoided when they provide clearer evidence. The 180 s run is
the intentionally infrequent milestone and remains mandatory before release
even though its scalar pre-optimization execution was explicitly deferred.
