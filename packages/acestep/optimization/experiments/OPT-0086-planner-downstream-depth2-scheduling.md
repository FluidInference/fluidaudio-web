# OPT-0086 — Planner-enabled downstream depth-two scheduling

## Status

- Evidence: `pending`
- Disposition: `benchmark-only` until the declared exact product gate passes
- Production integration: not yet authorized
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact selector widening and FIFO resource lifetime
- Allocation baseline: pushed `main` commit
  `1ddb65e751529936b3ef3cd48a6360386c7dd205`

No OPT-0086 selector, harness, result, or production change existed when this
experiment was allocated.

## First-principles basis

OPT-0080's production DiT and C2314 VAE depth-two/four-completion policies
preserve every singleton command buffer, dispatch, binding, arithmetic
operation, FIFO dependency, and completion fence. They are already integrated
and product-exact for the authenticated direct request tuple. The current
selectors nevertheless return depth one whenever planner mode is enabled.
That exclusion records the scope of the completed evidence; it is not a
measured dependency of either downstream scheduler.

Planner mode changes conditioning values and the pinned low/high DCW scalers,
but it does not change the DiT command topology, sampler evaluation count,
resident packages, VAE window geometry, or FIFO correctness argument. Keeping
the downstream policies off therefore pays the same avoidable queue-drain and
idle overhead after planner execution has finished.

OPT-0080 measured a 7,437.1 ms DiT full-graph saving at M2250 and reduced each
C2314 VAE window from 556 to 139 true drains. Those are causal mechanism
budgets, not an OPT-0086 planner-enabled performance claim.

## Frozen change

Arm A is current planner-enabled production: downstream DiT and VAE selector
policies are absent solely because `plannerMode !== "disabled"`.

Arm B removes only that planner-mode exclusion when every existing positive
identity check still passes:

- reference manifest and reference-BF16 subgroup execution profile;
- revision-7 OPT-0009 dense package/profile;
- OPT-0070 production attention profile;
- production eight-evaluation Turbo sampler/DCW topology;
- cooperative scheduling with no diagnostic/capture seam;
- revision-7 OPT-0072 VAE package/runtime, C2378 window profile, and exact
  C2314 production-window selector; and
- no benchmark policy override.

All existing negative cases remain negative. The planner itself is outside
this experiment and retains whichever separately integrated kernel, sampler,
and scheduling owners are current. No model math, package, graph command,
epoch, queue depth, VAE window, DCW scaler, or public API changes.

## Static gate

Tests must prove selection for both direct and enabled-planner requests on the
exact tuple, while rejecting each wrong manifest/profile/package/window,
nonproduction sampler schedule, noncooperative mode, diagnostic seam,
capture/override, and unsupported VAE remainder. Existing direct selection
must remain unchanged.

The scheduler's established abort, fence-rejection, progress-callback,
device-loss, queue-lease, aliased-resource, and destruction tests must remain
green. Planner resource draining/destruction must complete before DiT loading,
exactly as before.

## Planner-enabled product gate

Use one deterministic planner-enabled request whose duration exercises the
production DiT policy and at least one exact C2314 VAE window. Run forced Arm A
then forced Arm B in fresh workers, followed by one seam-free ordinary
production Arm B and one ordinary cancellation arm. This is a correctness and
selector gate; timing may be captured as decision context but is not required
to re-prove OPT-0080's scheduler mechanism.

Require:

- identical complete CoT/semantic emitted tokens, Philox cursor, resolved
  caption/language/metadata, conditioning tensors, and DCW configuration;
- raw-U32 identity for every retained DiT evaluation tap and final latent;
- raw-U32 identity for full VAE waveform and every seam region;
- byte-identical normalized WAV and stable output metadata;
- exact expected singleton submission/fence counts, depth-two epoch/drain/idle
  counts, maximum outstanding two, and ordinary production selector receipt;
- bounded heartbeat, no post-abort progress/output, balanced maps/resources,
  worker termination, and no automatic retry; and
- the ordinary production WAV equal to forced Arm B without a diagnostic seam.

Because all output-affecting values must be identical, no listening gate is
required. Stop without integration for any mismatch, selector broadening,
cancellation/lifecycle failure, or topology discrepancy.

## Integration

If the product gate passes, widen only the two internal production selectors
to accept planner-enabled requests on the same authenticated tuple. Record the
observed downstream timing separately from total planner improvement so this
experiment is not credited for planner kernel/sampler gains.

## Authority

- Scheduler mechanism and direct product evidence:
  [OPT-0080](OPT-0080-dit-depth2-completion-epochs.md)
- Pipeline selectors: [`webgpu-pipeline.ts`](../../src/runtime/webgpu-pipeline.ts)
- Approved planner behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
