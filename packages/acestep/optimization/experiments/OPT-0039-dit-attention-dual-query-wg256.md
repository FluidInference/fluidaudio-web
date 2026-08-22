# OPT-0039 — DiT fixed-WG256 dual-query attention

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: exact ascending-key FP32 online-softmax arithmetic

## Hypothesis

Query8 maps eight WG256 subgroups to two GQA heads × four adjacent query
tokens. OPT-0030's query16 reused K/V across twice as many streams but widened
the group to WG512 and regressed; OPT-0033 cut barriers within WG256 but also
regressed. Neither result tests query16 reuse without a wider workgroup.

Keep WG256 and assign each subgroup two query tokens for the same query head:
the ordinary token and token+4. Each lane holds two four-component query/value
vectors and two FP32 online max/denominator states. For every ascending key,
load K/V once, barrier, execute the two subgroup reductions and online updates
in fixed token order, then barrier. A workgroup now covers two heads × eight
tokens, halving workgroups, K/V loads, and barriers versus query8 while avoiding
WG512 occupancy.

## Gate

Add an isolated fixed32 owner; leave query8 production unchanged. Require
bit-exact equality for all `4,608,000` M2250 F32 outputs, finite complete writes,
canaries, deterministic cleanup, and one balanced nominal timing. Continue
only at `>=1.25x` versus query8. No layer, product, or production claim follows
from the primitive screen.

## Result

The isolated candidate passed. Across `13,824,000` F32 comparisons, query8,
dual-query16, and their repeats were bit-exact, deterministic, finite, complete,
and left every canary untouched. The fixed-WG256 candidate halved the
workgroup count (`4,504 -> 2,256`) and nearly halved K/V scalar loads and
barrier events (`1.99645x` reductions for both) while retaining the same
1,024-byte workgroup-storage footprint.

One balanced four-round timing followed the nominal one-check protocol: a
30.055-second idle, then one thermal-level-0 observation before launch. Median
query8/dual-query16 wall was `132.2 -> 97.4 ms`, a `1.3572895x` speedup, which
clears the frozen `1.25x` primitive gate. All eight created buffers were
destroyed exactly once, including two before timing and six at final cleanup,
and the device was destroyed.

This is positive benchmark-only evidence. It does not authorize production
integration or a product-speed claim. Graph/layer/trajectory integration and
its correctness and performance gates require an explicit follow-up under a
new experiment ID.

Receipt: [`../results/OPT-0039/result.json`](../results/OPT-0039/result.json),
SHA-256 `ffc0a27929e903c3366f5b1c8da4c945cf2f0f8a17134d73a96c315227b41ac9`.
