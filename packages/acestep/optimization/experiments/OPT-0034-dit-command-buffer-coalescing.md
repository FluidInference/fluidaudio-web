# OPT-0034 — DiT command-buffer coalescing

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: exact scheduling-only change
- Receipt: [`optimization/results/OPT-0034/result.json`](../results/OPT-0034/result.json),
  SHA-256
  `6c87e7f07f24b3eb2b1f515c153c873ab41c6b1a6df3c8f00bb57ef99ffa8a38`

## Hypothesis

The authenticated OPT-0018 M2250 trace drains `2,553` graph command buffers.
Their submit-through-drain sum is `62,148.2 ms`, while the complete DiT
generation stage is `73,072.6 ms`. Production also requests one real
millisecond of queue-empty time after every non-final command buffer. Encoding
several consecutive physical graph quanta into one FIFO command buffer can
remove most drain, event-loop, encoder, and requested-idle overhead while
leaving every dispatch, binding, dependency, and model operation unchanged.

This is the DiT analogue of OPT-0027, whose exact VAE batch64 arm reduced a
C512 outer window by `2,275.05 ms`. It is not a kernel throughput claim and it
does not alter model precision.

## Frozen mechanism

- Keep one outstanding command buffer and drain it before encoding/submitting
  the next one.
- Compare the shipped batch of one physical graph quantum with batches of
  eight and sixteen consecutive physical quanta.
- Preserve the exact global FIFO sequence. Each dispatch is encoded once, in
  the same order, into a compute pass; no operation is fused, dropped,
  reordered, or duplicated.
- Preserve abort checks and progress at each drained batch. Report the maximum
  operations and observed wall per batch so the cancellation bound is explicit.
- Keep the final-latent readback separate. Do not change weights, graph math,
  kernels, sampler/DCW, queue depth, or any non-DiT stage.
- Use only stock Chrome WebGPU/WASM and ordinary advertised adapter limits.

## Gate

First prove structural accounting for all `2,553` physical graph quanta and
raw-U32 identity of the final latent. Then run one thermally gated M2250
batch1/batch8/batch16 comparison, recording graph and stage wall, command
buffers, drains, requested idle, and maximum batch duration. A candidate is
positive only if it is exact, clean, and improves complete DiT stage wall by
at least `1.10x`; choose the fastest arm whose observed cancellation bound is
acceptable. Listening is unnecessary because execution order and arithmetic
remain unchanged. Integration and an end-to-end song timing are separate
follow-ups.

## Actual-Chrome result

The stock-Chrome/M3 run at core and harness commit
`ab88972a561f05d1cc2a3a11658f49795091f680` completed all three arms in the
frozen order `batch1 -> batch8 -> batch16`. The compact receipt preserves its
literal harness outcome:

- `status: negative`;
- `decision: negative-below-complete-stage-speed-gate`;
- fastest reported candidate `batch8`;
- complete generation-to-checkpoint ratio `0.9360015938x`, below the frozen
  `1.10x` gate; and
- no production selection, integration, VAE run, audio run, or under-60-second
  claim.

That literal one-run gate outcome is not rewritten. The experiment-level
performance conclusion is nevertheless **inconclusive**, because fixed-order
sustained thermal drift is too large to attribute the observed ordering to
batch size.

### Exactness and lifecycle

The scheduling mechanism passed its authoritative exactness and cleanup
boundaries:

- both candidates matched batch1 across all `288,000` final-latent raw U32
  words, for `576,000` comparisons and zero mismatches;
- all three final latents had SHA-256
  `1812a085f48b7879212633c7193dda08ec2854852a492ce661262c5e6be98f4c`,
  were entirely finite and nonzero, and had identical maximum absolute value;
- every arm reconciled the same `2,553` physical graph quanta, `6,833`
  primitives, `26,840,955,355,136` scheduled multiply-adds, descriptor-table
  SHA-256, separate final readback, and eight denoise evaluations;
- the private checkpoint sentinel matched once per arm, the DiT was destroyed
  before checkpoint publication, pipeline cleanup completed before the next
  arm, no post-DiT/VAE progress occurred, and final backend disposal completed;
  and
- the run used stock WebGPU, one outstanding command buffer, and a drain after
  every batch. Listening remains unnecessary because the final latent is
  raw-bit identical and execution order/arithmetic did not change.

### Proven topology and cancellation tradeoff

Batching removed the intended host scheduling topology exactly:

| Arm | Graph command buffers / drains | Requested graph idle | Maximum observed batch drain |
| --- | ---: | ---: | ---: |
| batch1 | `2,553` | `2,552 ms` | `329.2 ms` |
| batch8 | `320` | `319 ms` | `578.4 ms` |
| batch16 | `160` | `159 ms` | `1,301.6 ms` |

Thus command-buffer/drain and requested-idle reductions are real, while the
measured cancellation boundary grows materially, especially for batch16.
Those facts do not establish a wall-time win.

### Why wall-time evidence is inconclusive

The three fixed-order graph walls increased monotonically:

| Arm | Graph wall | Submit-through-drain sum | Generation to checkpoint |
| --- | ---: | ---: | ---: |
| batch1 | `123,094.6 ms` | `119,212.7 ms` | `129,668.7 ms` |
| batch8 | `128,405.3 ms` | `127,896.3 ms` | `138,534.7 ms` |
| batch16 | `134,193.9 ms` | `133,936.5 ms` | `143,533.6 ms` |

Most importantly, this run's unchanged batch1 graph wall was
`1.980662x` the prior exact OPT-0018 batch1 authority (`123,094.6 ms` versus
`62,148.2 ms`). That approximately twofold cross-run slowdown is far larger
than the scheduling deltas being tested. Independent non-candidate work also
drifted during the fixed order: text encoding was `1,428.1 / 2,268.2 /
1,813.0 ms`, condition encoding was `2,462.2 / 3,724.3 / 3,802.7 ms`, and DiT
load was `2,538.4 / 3,953.8 / 3,553.6 ms` for batch1/8/16. These measurements
are evidence of a changing sustained-run environment, not effects the batching
mechanism can cause.

The one truthful pre-run thermal observation was level zero after a
`30,053 ms` wait, as authorized by the owner. It says nothing about relative
thermal state across the subsequent `411,740.9 ms` (`6.86` minute) fixed-order
comparison. Consequently, neither the apparent `0.936x`/`0.903x` regressions
nor a hypothetical win from this ordering can be assigned confidently to
batch size.

## Decision and revisit condition

Retain the exactness, topology, lifecycle, and cancellation evidence. Do not
integrate batch8 or batch16 from this receipt, and do not describe either as
faster or intrinsically slower. Reject an unchanged rerun of the same nearly
seven-minute `1 -> 8 -> 16` protocol: it cannot separate mechanism from
sustained thermal/order drift and would spend substantial GPU time without
improving identifiability.

Revisit only with a shorter thermally comparable design, such as repeated
balanced/interleaved batch1/batch8/batch16 graph segments, or representative
layer/evaluation slices that retain the real command encoding, drain, idle,
and dependency topology. Alternate order and repeat each arm closely enough
to estimate order/thermal drift. Preserve raw-bit checks and report the same
maximum-batch cancellation bound. Only a positive balanced short result should
authorize a narrowly paired full-M2250 confirmation.
