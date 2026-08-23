# OPT-0062 — DiT quad-query full-attention production integration

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: exact FP32 ascending-key online-softmax ownership and runtime-profile
  routing change

## First-principles basis

OPT-0061 qualified fixed-WG256 quad-query32 over both current query8 and the
OPT-0039 dual-query primitive while preserving every output bit. Its
authoritative same-page medians were `133.94999998807907 ms` for query8 and
`77.25 ms` for quad-query, a `1.7339805823699557x` speedup. Applied only to the
96 full-self-attention calls in the frozen eight-evaluation M2250 graph, the
primitive delta is `5,443.199998855591 ms` of planning arithmetic, not a graph
claim.

This experiment prospectively supersedes OPT-0045's incomplete dual-query
integration direction with the stronger qualified quad-query owner. OPT-0045
and its OPT-0039 authority remain immutable history; their records, evidence,
and incomplete status are not rewritten or treated as quad-query evidence.

## Frozen integration direction

- Add the distinct fail-closed runtime profile
  `opt-0062-fixed32-quad-query32-full-self-v1`, kernel set
  `opt-0062-query8-plus-quad-query32-full-self-v1`, and quad kernel ID
  `opt-0062-fixed32-quad-query32-full-self-v1`. Do not reuse or rename the
  OPT-0045 dual-query identities.
- Route quad-query only for labels matching exactly
  `ace-dit-eval-E-layer-L-self-full-attention`, where `E` is `0..7` and `L` is
  one of the twelve odd full-attention layers
  `{1,3,5,7,9,11,13,15,17,19,21,23}`. Require exactly `96` unique quad routes.
- Every selected operation must have exact shape
  `B1/Hq16/Hkv8/Q2250/KV2250/D128`, mode `full`, fixed subgroup size `32`,
  workgroup size `256`, `1,128` workgroups, `1,024` bytes of workgroup storage,
  and the authenticated OPT-0061 quad-query WGSL identity
  `7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6`.
  A selected-label or selected-shape mismatch must throw; it must never fall
  back to query8, dual-query, or another attention owner.
- Retain query8 unchanged for exactly `96` sliding-self-attention routes and
  `192` cross-attention routes. Preserve masks, valid lengths, RoPE, head
  transforms, projections, FP32 ascending-key online max/denominator/weighted
  state, K/V staging, barriers, dispatch order, sampler/DCW, dense profile,
  package bytes, scheduling, progress, and cancellation.
- Keep production query8 as the public default until every gate below passes.
  This ID does not combine OPT-0055's six-evaluation schedule or OPT-0056's
  selective dense profile.

## Gates

1. **Profile and route contract.** Freeze new runtime-profile, kernel-set, and
   kernel IDs. Exhaustively enumerate all `384` attention calls and prove
   `96` quad full-self, `96` query8 sliding-self, `192` query8 cross, zero
   query8-other, zero duplicate/missing labels, and zero unintended quad routes.
   Reject every near-miss evaluation, layer, suffix, mode, and dimension.
2. **Actual-layer bit identity.** On authenticated production weights and
   deterministic inputs, compare query8 and quad-query for every selected layer
   output as raw U32. Require zero mismatches, non-finite values, unwritten
   outputs, or canary changes, plus deterministic reruns.
3. **Trajectory and final latent.** Run the exact canonical direct 180-second
   C98/M2250 request with the pinned eight-evaluation shift-3 Euler/DCW-double
   sampler. Capture every evaluation latent without adding timed GPU work and
   require raw-U32 identity at all eight taps and the final
   `288,000`-element latent. Authenticate request, conditioning, package,
   profile, descriptor, and final-latent hashes.
4. **Lifecycle and failure paths.** Prove one FIFO graph owner, unchanged queue
   depth/submission topology, drain-before-release, bounded cancellation, no
   post-abort encode/submit/map/callback, idempotent destruction, device-loss
   propagation, and zero live buffers/bytes after success, cancellation, and
   injected failure.
5. **Realized slice and full-graph performance.** After correctness passes, run
   a short balanced query8/quad layer-or-evaluation slice under the repository
   thermal protocol. Retain every raw sample; both paired directions must
   improve full-self wall and median full-self speedup must reach `1.30x`.
   Then run complete query8/quad M2250 graphs in both forward and reverse paired
   directions, each beginning after its own nominal gate and retaining its
   through-cleanup thermal trace. Separately report full-self family, total
   graph, complete DiT stage, command/drain, requested-idle, readback, and
   residual walls. Quad must reduce graph and stage wall in both directions,
   retain at least `1.30x` aggregate full-self speedup, and save at least
   `3,000 ms` from aggregate graph wall without any non-attention family
   regression above `2%`. Directionally mixed or variance-overlapped evidence
   is inconclusive, not a pass. The OPT-0061 `5.4432 s` eight-evaluation and
   `4.0824 s` six-evaluation figures remain labeled primitive arithmetic, never
   measured graph savings.
6. **Production boundary.** A pass may select the exact quad profile without a
   listening repeat because all actual-layer, every-step, and final-latent words
   must be identical. It still authorizes no six-evaluation, selective-dense,
   full-product, or under-one-minute claim; those combinations require their own
   gates and the final nominal 180-second Generate-to-WAV measurement.

## Result

### Correctness and scope

The production integration screen was exactly positive for correctness. Across
all `96` selected routes it compared `442,368,000` actual-layer U32 words with
zero mismatches, zero non-finite outputs, and zero changed canaries. All eight
evaluation taps and the final latent were raw-U32 exact, and the independent
quad repeat was exact. The route census was exactly `96` quad full-self, `96`
query8 sliding-self, `192` query8 cross, and zero other or unintended routes.
Each arm used one FIFO graph owner; the arms were sequential and
non-overlapping; drain-before-release and complete backend/device disposal
passed; and identity capture added zero submits and zero drains. The common
final-latent SHA-256 was
`1812a085f48b7879212633c7193dda08ec2854852a492ce661262c5e6be98f4c`.

The run changed no production default, dense profile, package bytes, scheduler,
VAE, or audio path. It makes no six-evaluation, selective-dense, listening,
full-product, or under-one-minute claim.

### Exact retained performance samples

The fixed order was forward query8, forward quad, reverse quad, reverse query8.
Values below are the exact millisecond values stored in the authoritative
receipt; no median or fastest-sample substitution was made.

| Direction / arm | Full self | Eval-0 full-self slice | Graph | DiT stage | Command/drain | Requested idle | Readback | Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward / query8 | `26071.699999928474` | `1531.3000000715256` | `154996.59999990463` | `157285.80000007153` | `150936.80000019073` | `2552` | `2.9000000953674316` | `1507.7999997138977` |
| forward / quad | `17896.700000166893` | `2214.5` | `156630` | `161019` | `152653.79999935627` | `2552` | `7.5000001192092896` | `1424.2000006437302` |
| reverse / quad | `17352.600001215935` | `998.3999999761581` | `151389.20000004768` | `153612.60000002384` | `147417.6000019312` | `2552` | `7.600000023841858` | `1419.5999981164932` |
| reverse / query8 | `28943.300001740456` | `3603.899999499321` | `165918.69999992847` | `170172.39999997616` | `161941.70000302792` | `2552` | `2.600000023841858` | `1424.9999969005585` |

The full-self owner improved in both directions: `1.4567881229324595x`
forward and `1.6679517766624214x` reverse, for the receipt's aggregate
`1.56074021326695x`. Aggregate graph savings were
`12896.099999785423 ms`. Those aggregates do not pass the frozen graph gate:
the forward quad arm regressed graph by `1633.4000000953674 ms`, DiT stage by
`3733.1999999284744 ms`, and the evaluation-0 slice from
`1531.3000000715256` to `2214.5 ms`. The reverse arm saved
`14529.49999988079 ms` of graph and `16559.799999952316 ms` of stage and
improved the slice from `3603.899999499321` to `998.3999999761581 ms`.
Forward non-attention drift also exceeded the frozen `2%` limit, including
cross-cache `+134.82637070720997%`, feed-forward `+17.972572335693648%`, and
output `+30.213567792866214%`. The receipt's maximum non-attention ratio was
`2.6785714171674786`, from a very small reverse precompute bucket. Exact family
vectors remain retained in the receipt.

### Thermal evidence

Both directions began after valid nominal gates, but most of each multi-minute
comparison then ran at pressure levels 1-2:

- Forward gate: start `1786848198296`, completion `1786848228898`, `35`
  observations, `907 ms` maximum gap, zero non-nominal observations. Its trace
  ended at `1786848678004` with `534` observations, `916 ms` maximum gap, and
  `469` non-nominal observations; trace SHA-256
  `6a02c2021953e84f9e01306add23407a86c99d709b49379ec268f6dc19c48481`.
  Exact transitions were `1786848228898:0`, `1786848256807:1`,
  `1786848277506:2`, `1786848529508:1`, `1786848531302:2`,
  `1786848532201:1`, `1786848625797:2`, and `1786848631199:1`.
- Reverse gate: start `1786848748508`, completion `1786848779115`, `35`
  observations, `906 ms` maximum gap, zero non-nominal observations. Its trace
  ended at `1786849230918` with `537` observations, `910 ms` maximum gap, and
  `475` non-nominal observations; trace SHA-256
  `e7008dc519a7bdb9392b60a8a6656e8000ba9153aeb2ec2b87ced479f074def9`.
  Exact transitions were `1786848779115:0`, `1786848804312:1`,
  `1786848827714:2`, `1786848950114:1`, `1786848961813:2`,
  `1786849077015:1`, `1786849078821:2`, `1786849079720:1`,
  `1786849138220:2`, `1786849142715:1`, `1786849143618:2`, and
  `1786849157116:1`.

An earlier setup attempt supplied an otherwise nominal gate too late for the
page's five-second handoff limit. It reached correctness-ready but performed no
timed GPU work. Its stopped trace had `65` nominal observations, `910 ms`
maximum gap, and SHA-256
`a99eb114a05881e868a86aaafe1e1ab1232bb5811b0627a6c4ea1f075aa16196`;
it is setup provenance, not a fifth sample.

### Decision and first-principles revisit

The correctness result is positive, while the literal full-graph performance
gate is negative and the underlying graph-level conclusion is inconclusive.
The owner itself is materially faster, but forward graph/stage/slice
regressions, unrelated-family drift, and sustained pressure levels 1-2 prevent
attributing the directionally mixed complete-graph result to the owner alone.
Production therefore keeps query8 as the default. Quad remains benchmark-only
and is not integrated.

A revisit must isolate the causal question with short balanced/interleaved
full-self measurements or bounded graph/layer slices, each pair beginning after
a separate cool nominal gate. Do not repeat the unchanged four-arm,
multi-minute full M2250 graph as the next experiment. Only a stable short screen
should authorize one narrow full-graph confirmation.

## Authority

- Primitive record: [OPT-0061](OPT-0061-dit-attention-multi-query-wg256.md)
- Primitive receipt: [result.json](../results/OPT-0061/result.json), SHA-256
  `aa94b429d026d8e2093589b8664be24dbd64ffc14f51160ced4682521a3b95e6`
- Integration receipt: [result.json](../results/OPT-0062/result.json), SHA-256
  `cf7db9c02496e216af79bde68387d32955f6eb57649b245e290c62332707db03`
- Failed setup provenance:
  [failed-thermal-handoff.json](../results/OPT-0062/failed-thermal-handoff.json),
  SHA-256
  `e5ec259ed606dce2ff4bf34f1c1f8171e9100177e326d5bd6b1206d308dc4ccc`
- Preserved prior integration direction:
  [OPT-0045](OPT-0045-dit-dual-query-attention-integration.md)

Registration itself preceded implementation and GPU work. The authenticated
screen used core and harness commit
`aa46aa5f4f7b1f8ca15453c336418b9cfe471348`; its result does not select a
production profile.
