# OPT-0081 — DiT typed-F16 dense inputs plus packed-weight multicast

## Status

- Evidence: `inconclusive` overall; arm B was positive through the actual MLP
  chain but failed the representative-layer directional/material wall gate;
  arm C remains a literal diagnostic non-pass
- Disposition: `benchmark-only`
- Authorized next step: none; OPT-0081 is closed on this browser/GPU
- Production integration: not warranted or authorized
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `raw-U32-exact producer storage boundary and dense ownership`
- Allocation baseline: pushed `main` commit
  `bbe180bf7feb59272a5d5f7afbafb3877afee416`

No candidate, harness, actual-GPU run, timing sample, package change, runtime
selector, or production change existed when this experiment was allocated.

## Hypothesis

The current repeated-layer dense owner reads FP32 graph activations and rounds
each value to FP16 inside every dense consumer. The value has already been
computed by a GPU producer and all nine dense routes use it only as an FP16
operand. Folding that one existing cast into the producer's final store can
therefore retain the exact dense operand bits while halving both the stored
role and every repeated activation request. Combining that typed-F16 input
with OPT-0078's materially different WG256 packed-weight-multicast geometry
can remove independent activation and weight traffic without changing the
converter-native weights, source-order FP32 accumulation, or FP32 outputs.

This is a bandwidth, residency, and private-state experiment, not an
arithmetic-precision experiment. Whether the M3/Chromium implementation is
limited enough by those terms to produce a material wall-time win remains an
actual-GPU question.

## Frozen production inventory

Only the following six M2250 activation roles may change storage type. Their
nine consumers are the complete repeated-layer dense inventory and reconcile
to the production `4/2/2/1` shape multiplicities.

| producer role | producer operation | exact consumers | elements | F32 bytes | typed-F16 bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `selfModulated` | self AdaLN | self query `2048→2048`; self key and value `2048→1024` | 4,608,000 | 18,432,000 | 9,216,000 |
| `selfMergedAttention` | self head merge | self output `2048→2048` | 4,608,000 | 18,432,000 | 9,216,000 |
| `crossNormalized` | cross RMSNorm | cross query `2048→2048` | 4,608,000 | 18,432,000 | 9,216,000 |
| `crossMergedAttention` | cross head merge | cross output `2048→2048` | 4,608,000 | 18,432,000 | 9,216,000 |
| `mlpModulated` | MLP AdaLN | gate and up `2048→6144` | 4,608,000 | 18,432,000 | 9,216,000 |
| `gatedActivation` | SwiGLU | down `6144→2048` | 13,824,000 | 55,296,000 | 27,648,000 |
| **total** | | **nine consumers** | **36,864,000** | **147,456,000** | **73,728,000** |

The role set is closed. In particular, residual-spine tensors, layer inputs
and outputs, RMSNorm statistics, attention Q/K/V/head plumbing, modulation
tables, caches, sampler/DCW buffers, VAE buffers, and dense outputs stay FP32.
No additional role may be converted after timing begins or under this ID.

The exact consumer shapes and production multiplicities are:

| M2250 dense shape | route multiplicity | routes |
| --- | ---: | --- |
| `K2048/N2048` | 4 | self query/output; cross query/output |
| `K2048/N1024` | 2 | self key/value |
| `K2048/N6144` | 2 | MLP gate/up |
| `K6144/N2048` | 1 | MLP down |

## Exact storage and arithmetic boundary

Candidate producers retain their current FP32 computation. Their final
assignment alone becomes an explicit `f16` cast stored into a scalar typed
`array<f16>` binding. Candidate dense owners read that same scalar
`array<f16>` with the same logical row/K index. There is no conversion
dispatch, post-producer copy, persistent duplicate, `vec2`/`vec4` input,
packed-U32 activation, scalar/vector sweep, quantization scale, clamp, or
fallback.

Arm A's existing expression is FP32 activation load, conversion to native
FP16, native-FP16 weight operand, widening to FP32, multiplication, and
increasing-K `acc = acc + product`. Arms B and C move only the identical FP16
rounding to the final producer store; the dense load is already native FP16.
All arms retain the converter-native packed-FP16 weight bytes, identical
increasing-K iteration, FP32 multiplication/accumulator behavior, and
row-major FP32 dense output. `dot`, `fma`, reassociation, split K, K2/K4
partials, FP16 accumulation, subgroup matrix extensions, and changed package
bytes are forbidden.

The experiment is raw-U32 exact by construction only if the target compiler
and implementation honor that boundary. The gates below must prove it; this
paragraph is not evidence.

## Static traffic and memory accounting

The exact M32/N256/K32 workgroups, M2250 tail, and `4/2/2/1` multiplicities
schedule `6,816` workgroups and `133,412,421,632` multiply-adds per layer and
evaluation for every arm. The frozen source-request model is:

| weighted per layer/evaluation | A: F32 current | B: F16 current | C: F16 multicast | excluded unchanged F32 OPT-0078 |
| --- | ---: | ---: | ---: | ---: |
| activation requests | 2,084,569,088 B | 1,042,284,544 B | 1,042,284,544 B | 2,084,569,088 B |
| weight requests | 33,353,105,408 B | 33,353,105,408 B | 8,338,276,352 B | 8,338,276,352 B |
| total operand requests | 35,437,674,496 B | 34,395,389,952 B | 9,380,560,896 B | 10,422,845,440 B |
| input element storage | 4 B | 2 B | 2 B | 4 B |
| geometry | M32/N256/K32/WG128 | M32/N256/K32/WG128 | M32/N256/K32/WG256 multicast | M32/N256/K32/WG256 multicast |
| FP32 accumulators per lane | 64 | 64 | 32 | 32 |
| declared workgroup storage | 0 B | 0 B | 16,384 B | 16,384 B |

The excluded column is accounting context only. There is no fourth timing arm
and no unchanged F32 replay of OPT-0078.

Across the six owned roles, the candidate graph arena changes exactly
`147,456,000 → 73,728,000` bytes. The complete authenticated graph arena
must therefore change exactly `674,815,488 → 601,087,488` bytes, a
`73,728,000`-byte reduction, with no extra cast buffer and no increased
largest binding.

Typed inputs remove `1,042,284,544` modeled activation-request bytes and
`73,728,000` producer-store bytes per layer/evaluation, or
`1,116,012,544` bytes in total. Across `24 * 8` layer evaluations that is
`214,274,408,448` bytes. The two MLP roles (`mlpModulated` and
`gatedActivation`) account for `706,904,064` bytes per layer/evaluation,
`63.34%` of this combined typed-boundary saving. These are static request
counts, not cache-miss, bandwidth, utilization, or speed claims.

## Frozen three-arm mechanism

1. **A — current control.** Use the immutable OPT-0009
   M32/N256/K32/WG128 owner with FP32 activation storage and its current
   load-time FP16 cast.
2. **B — typed-input causal arm.** Use new scalar-typed-F16 producer and dense
   owners with exactly OPT-0009's M32/N256/K32/WG128 output ownership,
   subgroup broadcasts, native weight layout, and increasing-K arithmetic.
3. **C — typed-input plus multicast candidate.** Use the same scalar-typed-F16
   producer boundary as B with OPT-0078's materially different
   M32/N256/K32/WG256 eight-subgroup packed-weight multicast: four rows and
   eight columns per lane, 32 FP32 accumulators, one 16 KiB packed-U32 weight
   tile, and exactly two uniform workgroup barriers per K32 tile.

Arm B isolates the storage/input-read mechanism. Its classification is not a
precondition for C: an exact but sub-material B does not block C. B may be
considered alone for integration only if its own frozen timing gate passes.
C must independently beat both A and B, so it cannot inherit B's gain while
adding a non-material multicast owner.

Implementation must add, rather than mutate historical controls:

- `src/webgpu/kernels/dit-f16-dense-input-producers.ts` for the six exact
  scalar-F16 final-store producer variants;
- `src/webgpu/kernels/dit-dense-f16-input.ts` for arm B; and
- `src/webgpu/kernels/dit-dense-f16-input-weight-multicast.ts` for arm C.

The frozen allocation-baseline SHA-256 identities are
`a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3`
for `src/webgpu/kernels/dit-dense-fp16.ts` and
`1a8907e9c24d12ddd61e58d55e467051ea5def92db975be47d808f2a31318f1d`
for `src/webgpu/kernels/dit-dense-fp16-weight-multicast.ts`. Those files are
immutable A and historical-OPT-0078 authorities under this experiment.

The primitive browser owner must be new
`test/browser/opt-0081-dit-f16-dense-input-multicast.ts` with a matching HTML
page. Do not retrofit an OPT-0009 or OPT-0078 historical harness.

## OPT-0078 revisit-boundary reconciliation

OPT-0078 closed with this exact boundary:

> Do not repeat this unchanged benchmark or relax its threshold. A materially different tile/dataflow, browser/compiler, or target GPU requires a new experiment ID; this result authorizes no inference-speed or product claim.

It did not authorize an unchanged timing retry. This experiment does not run
that unchanged candidate: the excluded fourth column above records it only
for accounting. Arm C changes every dense activation binding from FP32
storage with a load-time cast to a scalar typed-F16 producer boundary, halves
modeled activation requests, removes `73,728,000` arena bytes, and uses a new
owner under the new OPT-0081 ID. That is a materially changed operand-storage
and traffic dataflow while retaining OPT-0078's output tile and weight
multicast. No browser/compiler or target-GPU change is claimed or needed for
this reconciliation.

## Static and actual-GPU correctness gate

Before timing:

1. Reconcile the six role element counts, nine unique consumer labels, four
   shapes, `4/2/2/1` multiplicities, `6,816` workgroups, scheduled/valid
   multiply-adds, binding byte ranges, arena slots, and the traffic table
   above. Reject any additional or missing route.
2. Prove B has the same M32/N256/K32/WG128 output owner and physical packed
   weight index as A. Prove C has OPT-0078's exact WG256 output ownership,
   1,024-record cooperative weight fill, uniform two-barrier lifecycle, and
   physical packed index. All three arms must cover the partial M2250 row
   tile exactly once.
3. On the actual target GPU, execute all six real producer variants. Compare
   every candidate scalar-F16 stored bit against the bit produced by applying
   the exact A load-time FP16 conversion to the corresponding F32 producer
   value. Require finite production outputs, exact signed-zero behavior, no
   missing store, and intact U16 prefix/suffix/adjacent canaries.
4. For all four full M2250 shapes, compare A, A rerun, B, B rerun, C, and C
   rerun over every FP32 output U32. Require zero mismatches, deterministic
   repeats, complete qNaN-prefill overwrite, finite-class identity, intact
   U32 canaries, and explicit first/last valid and tail-row coverage.
5. Retain bounded signed-zero, subnormal/normal boundary, alternating
   cancellation, maximum-finite FP16, and rounding-boundary fixtures. Require
   raw U16 producer-boundary and raw U32 dense-output identity rather than a
   numerical envelope.
6. Require zero validation errors, uncaptured GPU errors, and device losses;
   balanced created/destroyed/mapped resources; drain-before-release;
   idempotent destroy; setup-failure cleanup; and rejection after destroy.

Any mismatch, unwritten word, changed arithmetic/layout, alias, capability
failure, or lifecycle failure stops before timing.

## Fixed M2250 primitive performance gate

Compile, upload, run correctness, and warm all arms before the thermal
boundary. Begin timing only after at least 30 continuous seconds at accepted
thermal level 0. Use an external absolute-cadence 1,000 ms thermal trace from
before the launch gate through cleanup, with no stale sample or gap over
1,500 ms and every observation nominal.

Run exactly eight rotated three-arm rounds on one fresh capable device. The
arm order is fixed as `ABC, CBA, BCA, ACB, CAB, BAC, ABC, CBA`; this balances
the relative order of every pair `4/4`. Rotate the four shape orders one
position per round so every shape occupies every position exactly twice. Each
sample contains the same dense dispatch, command buffer, submit,
timestamp-query pair, matching drain, and output range. Producer execution,
allocation, upload, compilation, readback, hashing, and serialization are
outside the timed interval. Retain all raw arm/shape/round GPU and fenced-wall
durations, exact orders, pair ratios, thermal samples, and cleanup evidence.
There is one timing run and no unchanged retry.

Define the complete score as
`4*T2048x2048 + 2*T2048x1024 + 2*T2048x6144 + T6144x2048` and the MLP score
as `2*T2048x6144 + T6144x2048`. Report mean and median for every shape and
score on both timestamped GPU and fenced wall.

C qualifies for MLP escalation only if all literal conditions hold:

- every shape's C mean and median is strictly faster than both A and B on GPU
  and wall;
- C wins at least seven of eight same-round complete-score pairs against A
  and separately at least seven of eight against B on GPU and wall;
- C/A complete-score mean speedup is at least `1.12x` on GPU and wall;
- C/A complete-score mean saving is at least `20.8334 ms` on GPU and wall,
  projecting at least `4,000 ms` over `24 * 8` layer evaluations;
- C/B complete-score mean saving is at least `10.4167 ms` on GPU and wall,
  projecting at least `2,000 ms`; and
- each comparison's wall saving is within `0.75x..1.25x` of its timestamped
  GPU saving.

B's standalone classification is causal and non-blocking for C. B qualifies
alone only if every shape mean and median beats A, it wins at least seven of
eight complete-score pairs, its complete-score mean speedup is at least
`1.05x`, its mean GPU and wall saving is at least `10.4167 ms` (a two-second
graph projection), and its wall/GPU saving ratio is within `0.75x..1.25x`.

Thermal invalidity, wall/GPU disagreement, or mixed evidence is
`inconclusive`. Any mismatch is `negative`. A literal timing miss is a
non-pass; do not waive a threshold after observation, substitute a fastest
sample, add a fourth arm, sweep geometry/vector width, or repeat unchanged
timing.

## Authoritative primitive result

The single authoritative actual-GPU run is the complete, raw-sample-preserving
[primitive receipt](../results/OPT-0081/primitive.json), byte-for-byte copied
from `optimization/artifacts/OPT-0081/raw/primitive-result.json`. Its SHA-256 is
`8cb2c7c30dec7d179729d5644608fb1f0b9ad5b0478ba92125476936644c775c`
and its byte length is `193,665`. It identifies registration
`70a5e4a29c5455ec00a4b757dcdf5cdcc70a5e91`, allocation baseline
`bbe180bf7feb59272a5d5f7afbafb3877afee416`, and implementation checkpoint
`312d67024978a64b77d2563dd9386b4328f17d33`. The generated-shader aggregate
SHA-256 is
`c5c9a02d77ca6191fd78620ce7d2bde7fc20b9ef16ddac921a9d503653e530b1`;
the arm A/B/C source SHA-256 identities are respectively
`a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3`,
`7dcf10f751641aedf0f1cc811938559a1db04513ba3e66c09f7de81121b8c72a`,
and `66bc8e1ce01253d8a50683b1c1393a295cc62192d21cc0f876701025d099900d`.

Arm B passed its standalone primitive gate. The exact complete-score B/A
receipt metrics are:

| clock/statistic | A (ms) | B (ms) | B/A speedup | saving (ms) | paired wins | `24 * 8` projection (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU mean | 212.000768 | 173.768704 | 1.2200169715255513x | 38.23206399999998 | 8/8 | 7,340.556287999996 |
| GPU median | 203.29267199999998 | 176.06246399999998 | 1.1546621998883306x | 27.230208000000005 | — | — |
| wall mean | 227.35000026226044 | 186.54999974370003 | 1.218708124227367x | 40.80000051856041 | 8/8 | 7,833.600099563599 |
| wall median | 219.70000052452087 | 190.04999959468842 | 1.1560115811263654x | 29.65000092983246 | — | — |

The mean wall/GPU saving ratio was `1.0671670909151134`. Every B shape mean
and median was faster than A on both clocks:

| shape | A/B GPU mean (ms) | A/B GPU median (ms) | A/B wall mean (ms) | A/B wall median (ms) |
| --- | ---: | ---: | ---: | ---: |
| `2048→2048` | 17.178624 / 13.565952 | 16.154624 / 13.074432 | 19.312500059604645 / 14.937499970197678 | 17.300000071525574 / 14.049999952316284 |
| `2048→1024` | 6.758400000000001 / 6.18496 | 6.06208 / 5.373952 | 8.087499976158142 / 7.762499988079071 | 7.25 / 7.099999904632568 |
| `2048→6144` | 43.630592 / 35.536896 | 44.335104 / 35.2256 | 44.98750001192093 / 36.549999952316284 | 45.60000002384186 / 36.10000002384186 |
| `6144→2048` | 42.508288 / 36.061184000000004 | 41.484288 / 36.601856 | 43.950000047683716 / 38.17499998211861 | 42.25 / 38.549999952316284 |

The primitive MLP-weighted score also retained material B/A evidence: mean
GPU was `129.769472 → 107.13497600000001 ms` (saving `22.634496 ms`,
`1.2112708365193454x`, projecting `4,345.823232 ms`) and mean wall was
`133.92500007152557 → 111.27499988675117 ms` (saving
`22.6500001847744 ms`, `1.2035497659656362x`, projecting
`4,348.800035476685 ms`). This remains primitive evidence; it does not replace
the actual-chain gate below.

Arm C is a literal non-pass. Although C beat A materially, it did not
independently beat B: complete-score C/B mean saving was only
`1.9988480000000095 ms` GPU and `2.949999749660492 ms` wall, with `4/8` GPU
and `5/8` wall paired wins and a `1.475849964409739` wall/GPU mean-saving
ratio. C lost both GPU mean and median for `2048→1024`, and lost every GPU and
wall mean/median condition for `2048→6144`, against B. Thus `cPassed=false`,
`bStandalonePassed=true`, and the receipt selects B for diagnostic-only MLP
follow-up; it authorizes no production integration, package change,
trajectory/listening claim, or unchanged retry.

Correctness covered all six actual producers and all four full M2250 dense
shapes. All `36,864,000` producer U16 values matched independent F32-to-F16
casts, including signed-zero and guard checks. The receipt records
`25,344,000` compared U32 output words per complete comparison set, 24
shape/comparison cases, all A/A, A/B, B/B, A/C, C/C, and B/C raw-U32
comparisons exact, five bounded adversarial fixtures, deterministic reruns,
intact tails/canaries, and zero uncaptured GPU errors or device losses.

The accepted absolute-cadence thermal trace has SHA-256
`0c77b6874e1cfd25ef0cffc4d510fbaff724df17101a2c0e2d67a560f0acbd45`
and byte length `10,798`. It contains 101 observations, all nominal, with a
maximum `1,004 ms` poll gap and covers cleanup. Cleanup destroyed all `66/66`
buffers, balanced all `164/164` maps, reached zero live buffers/bytes/maps,
rejected B, C, and producer use after destroy, destroyed the device, and was
idempotent. Maximum tracked live bytes were `442,374,176`.

Separately, an untimed launch-gate preflight produced ignored receipt
`optimization/artifacts/OPT-0081/raw/primitive-launch-receipt.json` (SHA-256
`1ad14c3d2496f9d49ab47833663e8bc705810a520457d90639370dcdffa60adc`,
`1,469` bytes) and trace
`optimization/artifacts/OPT-0081/raw/thermal-trace.jsonl` (SHA-256
`8b8734724ddf226ec3742e25e8aa10e304b729c6d8c9e4da14d2f66a0b762899`,
`11,878` bytes). The strict parser rejected that attempt at
`thermal-launch`; it ran no timing samples, selected no arm, authorized
nothing, and cleaned `66/66` buffers to zero live resources. It is not a
performance run, is not pooled with the authoritative samples, and is not an
unchanged timing retry.

## MLP-to-product escalation

Only a passing primitive may proceed. Before any MLP performance source is
written, the first screen is frozen literally as follows:

- **Arms and actual operations.** At exact M2250, A executes the current F32
  `mlpModulated` AdaLN producer, current gate and up dense owners, current
  SwiGLU producer, and current down owner. B executes the corresponding actual
  typed-F16 final-store producers and the OPT-0081 WG128 typed-input owners. C
  executes the same actual typed-F16 producers with the OPT-0081 WG256
  multicast owners. No synthesized activation upload, preconversion dispatch,
  copy, duplicate candidate activation, or isolated dense-only substitute may
  stand in for either producer.
- **Exact boundaries and outputs.** Before timing, compare every
  `mlpModulated` and `gatedActivation` candidate U16 to an independent exact
  cast of A's corresponding F32 producer output. Compare every gate, up, and
  down FP32 output word A/A, A/B, B/B, A/C, C/C, and B/C as raw U32. Require
  exact signed zeros, deterministic repeats, finite complete outputs,
  qNaN-prefill overwrite, first/last/tail coverage, intact adjacent guards,
  and zero validation errors, device losses, or live resources after
  idempotent cleanup. Gate and up use the exact `2048→6144` weights and down
  uses the exact `6144→2048` weights; arithmetic and output storage remain the
  frozen increasing-K FP32/F32 path.
- **Literal timing panels.** Compile, upload, run the complete correctness
  audit, and symmetrically warm all arms before READY. Each measured arm
  occurrence runs four fixed panels: actual `mlpModulated→gate`, actual
  `mlpModulated→up`, actual `gatedActivation→down` from validated actual
  gate/up results, then the complete actual
  `mlpModulated→gate/up→SwiGLU→gatedActivation→down` chain. Producer execution
  is inside every panel's timed interval. Each panel uses one command buffer,
  one submit, one timestamp-query pair, one matching drain, and one fenced-wall
  sample; allocation, compilation, readback, hashing, and serialization remain
  outside. Retain every raw panel/arm/round GPU and wall duration.
- **Balanced fixed run.** Use one fresh capable device, the same accepted
  30-second nominal thermal launch gate and trace-through-cleanup contract, and
  exactly eight rounds in the fixed arm order
  `ABC, CBA, BCA, ACB, CAB, BAC, ABC, CBA`. The four panel order above is fixed
  in every arm occurrence. There is one MLP timing run and no unchanged retry.
- **B standalone gate.** B must be strictly faster than A in the mean and
  median for gate, up, down, and complete chain on both GPU and wall; win at
  least `7/8` same-round pairs for every panel on both clocks; and retain at
  least `10.4167 ms` complete-chain mean saving on GPU and wall, projecting at
  least `2,000 ms` over `24 * 8`. Its complete-chain mean wall saving divided
  by GPU saving must be within `0.75x..1.25x`. Every conjunct is literal and
  there is no post-observation waiver.
- **C diagnostic gate and current stop.** Were C primitive-qualified, it could
  advance only by beating both A and B in every gate/up/down/chain GPU and wall
  mean and median, winning at least `7/8` same-round pairs for every panel
  against each arm on both clocks, and saving at least `15.625 ms` per-layer
  complete-chain mean on GPU and wall versus A and separately versus B. Each
  comparison must project at least `3,000 ms` over `24 * 8` and have a
  `0.75x..1.25x` mean wall/GPU saving ratio. The authoritative primitive did
  not qualify C, so even a diagnostic MLP observation cannot retroactively
  select or advance C under OPT-0081. Production selection remains B-only.

### Authoritative actual-MLP result

The single authoritative actual-MLP run is the complete, raw-sample-preserving
[actual-MLP receipt](../results/OPT-0081/actual-mlp.json), copied byte-for-byte
from `optimization/artifacts/OPT-0081/raw/mlp-result.json`. Its SHA-256 is
`92c27035c18ecddd32ebc6a15e8e732f14f36437ee471cbbc0f698d3ab107bfd`
and its byte length is `207,391`. The receipt binds registration
`606d1e29f56867bfda637c117b58778c634c4ee9`, B-only correction and
implementation base `0f13bcc486569819df7587349b8b1e049b924ccd`, and harness
checkpoint `436355ff16fb971d11a959e99e1550abc6186480`; commit
`608bdbca56a428fa243842368631754a62ee67dc` bound that harness checkpoint into
the receipt before the run. Its generated-shader aggregate SHA-256 is
`080fff1d8c115c8d748d93e4f62d4285456fd1e9e7fc7a327799398a5e6e97c3`.

Arm B passed every literal panel and complete-chain conjunct. The table reports
the exact eight-round receipt statistics in milliseconds; the tracked receipt
retains every raw sample and the unrounded JavaScript values.

| panel/clock | mean A → B | mean speedup | mean saving | median A → B | median speedup | paired wins | `24 * 8` projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate GPU | 47.423488 → 36.962304 | 1.2830230496453898x | 10.461184 | 36.306944 → 32.636928 | 1.1124497991967872x | 8/8 | 2,008.547328 |
| gate wall | 48.912500 → 38.587500 | 1.2675736967738478x | 10.325000 | 37.550000 → 34.850000 | 1.077474893711236x | 8/8 | 1,982.400003 |
| up GPU | 40.108032 → 37.855232 | 1.0595109283704827x | 2.252800 | 36.929536 → 33.619968 | 1.0984405458089668x | 7/8 | 432.537600 |
| up wall | 41.625000 → 39.100000 | 1.064578006801791x | 2.525000 | 38.400000 → 35.550000 | 1.0801687805027818x | 7/8 | 484.800013 |
| down GPU | 42.582016 → 41.107456 | 1.035870864886409x | 1.474560 | 41.123840 → 36.765696 | 1.1185383244206772x | 7/8 | 283.115520 |
| down wall | 43.437500 → 42.062500 | 1.0326894509545685x | 1.375000 | 41.900000 → 37.650000 | 1.1128818026567189x | 7/8 | 264.000006 |
| chain GPU | 123.052032 → 108.068864 | 1.138644633110976x | 14.983168 | 112.885760 → 101.351424 | 1.1138053669576462x | 7/8 | 2,876.768256 |
| chain wall | 124.150000 → 108.737500 | 1.1417404304511203x | 15.412500 | 113.950000 → 102.050000 | 1.1166095061335404x | 7/8 | 2,959.200010 |

The gate/up/down/chain mean wall-to-GPU saving ratios were respectively
`0.9869819723925511`, `1.1208274438765575`, `0.9324815740304354`, and
`1.028654290844513`. Thus B retained faster GPU and wall means and medians for
all four panels, at least `7/8` paired wins per panel, the required
`10.4167 ms` complete-chain saving on both clocks, and the required saving
agreement. `bStandalonePassed=true` and the only authorized follow-up is the
arm-B representative-layer diagnostic.

C remains diagnostic-only, non-passing, and nonselectable. Its chain beat A,
but versus B it saved only `4.857856 ms` GPU and `4.725000 ms` wall, projected
only `932.708352/907.200010 ms`, and won only `6/8` pairs on each clock. It
also lost gate GPU median (`32.702464 ms` versus B's `32.636928 ms`) and won
only `3/8` gate GPU and `4/8` gate wall pairs. These miss the literal
`15.625 ms`, all-mean/median, and `7/8` C/B requirements; independently, C
never passed the primitive and cannot be selected under OPT-0081.

Correctness ran the actual M2250 order
`mlpModulated → gate → up → gatedActivation → down` as A/A, B/B, and C/C. All
`73,728,000` candidate boundary U16 words matched independent casts. Gate, up,
and down comprise `32,256,000` U32 words per complete comparison set and all
18 checkpoint comparisons—`193,536,000` compared instances—were raw-U32
exact. Deterministic reruns, direct B/C identity, signed zeros, finite complete
outputs, qNaN overwrite, rows 2240–2249, adjacent guards, and first/last words
all passed, with zero uncaptured GPU errors or device losses.

The accepted trace `optimization/artifacts/OPT-0081/raw/mlp-thermal-trace.jsonl`
has SHA-256
`27ab8f4c8dd2e338fca7f89672acb3bfec70aab6e852900604684083070f19ea`
and byte length `10,476`. Its 98 observations were all nominal, its maximum
poll gap was `1,013 ms`, and it covers cleanup. Cleanup destroyed `16/16`
buffers, balanced `132/132` maps, reached zero live buffers/bytes/maps,
rejected B, C, and producer use after destroy, destroyed the device, and was
idempotent. Maximum tracked live bytes were `388,861,984`. The run used the
fixed arm and panel orders, included actual producers inside every timed panel,
and performed no unchanged timing retry.

### Representative-gate preparation evidence (untimed)

Three preparation-only browser attempts stopped before READY and therefore
consumed no accepted timing. They are retained as harness-development evidence,
not candidate performance samples:

1. Receipt SHA-256
   `591b3069fb6b541d4f4f9fe4ba2ea2b79b652585cea4243f57942422ae7ea84d`,
   byte length `6,719`, exposed that typed-array `GPUQueue.writeBuffer` sizes
   had been passed in bytes instead of U32/U16 elements. The fix uses exact
   `/4` and `/2` element counts and has full- plus partial-chunk regressions.
2. Receipt SHA-256
   `158d54931dff33a585855ebf981c3d8412bbf119b41a3812c6eed404a43c59fe`,
   byte length `9,188`, exposed reuse of the production OPT-0070 route
   registry across correctness targets. Each target now owns one fresh
   correctness runtime, with at most one live, exact compile-failure cleanup,
   and terminal `66/66` runtime-owner destruction.
3. Receipt SHA-256
   `4c4da12a155ab089a15a26ed3b9f7d67bfedac1a61b1d40f11d8432db230f51b`,
   byte length `83,981`, completed all 32 checkpoint groups and proved A/B
   exact at corresponding occurrence indices, but A/A and B/B repeats changed
   because aliased activation-arena state was not restored between independent
   snapshots. It had zero validation/uncaptured errors, zero device loss,
   balanced `311/311` graph buffers and `128/128` maps, and zero live resources
   after cleanup. This is a harness-correctness failure, not an OPT-0081 math
   mismatch.

The third finding requires every independent correctness snapshot to start
from the exact same post-precompute graph state. The bounded fix snapshots both
activation arenas GPU-to-GPU after precompute, restores the applicable arena
before every A1/A2/B1/B2 checkpoint, restores both once more on exit, and then
destroys the snapshots before cancellation, warmup, READY, or timing. No arena
is mirrored through JavaScript or WASM, and none of these snapshot/copy commands
can enter the frozen 28-command timed slice. A fresh untimed preparation must
prove all A/A, A/B, and B/B gates before the one accepted timing remains
authorized.

### Frozen arm-B representative-layer diagnostic

Only arm A and primitive/actual-MLP-qualified arm B participate. A is the
current FP32-activation OPT-0009 dense path. B is the exact
`opt-0081-six-dense-input-f16-storage-v1` six-role scalar-F16 storage profile
with the OPT-0081 WG128 typed-input dense owner. Arm C, a new geometry, a
conversion dispatch, duplicate candidate activation storage, and any package-
weight change are forbidden.

Use the authenticated canonical direct fixture from OPT-0067/OPT-0080: request
SHA-256 `031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f`,
main manifest SHA-256
`18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
and revision-7 exact-dense manifest SHA-256
`d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`.
The executed shape is M2250/C98 at evaluation 0. Execute the actual production
graph prefix through `ace-dit-eval-0-layer-1`: the ordinary 25-command
condition/cross-cache precompute, evaluation-0 timestep and input projection,
then `ace-dit-eval-0-layer-0` with sliding self attention and
`ace-dit-eval-0-layer-1` with full self attention. Layer 1 must consume the
actual layer-0 output from the same arm; it may not be tested from an uploaded
or control-derived substitute. The canonical conditioner may execute only
during excluded setup to produce and authenticate the fixture's condition and
context tensors; no conditioner work is part of the selected command slice or
accepted timing. Do not execute layer 2, output projection, sampler/DCW, a
later evaluation, VAE, or planner work.

Retain authority from the exact token arrays and conditioner outputs handed to
the diagnostic owner, and require the established canonical hashes: text-token
SHA-256
`8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2`,
lyric-token SHA-256
`b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764`,
condition-F32-LE SHA-256
`102308c377139c80b034acd38d90d3c81c7272eb9aa077cd9fbd66f47100c49b`,
and context-F32-LE SHA-256
`22c66fd3f3c80d1cd4c6c7ffbf3f20f65a5fc822d89aa0ed4b5729a92c1a66c0`.
The independently created setup-failure and accepted-run invocations must
retain identical counts and hashes. Hashing is excluded setup work.

Both arms retain the production OPT-0070 attention owner and
`opt-0080-depth2-epoch4` scheduler. The selected slice is physical graph
commands 25 through 52 inclusive: one timestep, one input projection, 11
physical commands for the sliding layer, and 15 for the full/quad layer, for
exactly 28 command buffers and 28 completion fences. It has seven completion
epochs/true drains, six real idle turns, maximum outstanding two, and maximum
pending descriptors two. There is no layer-boundary drain: the fourth epoch
contains command 37, the final layer-0 command, and commands 38 through 40, the
first three layer-1 commands, exactly as in the production evaluation phase. A
and B must have identical logical labels, physical-command count and order,
epoch boundaries, submit/fence/drain/idle counts, attention routes, and
progress order. B may differ only at the twelve selected producer stores and
eighteen selected dense consumers. Reconcile the full authenticated arena as
exactly `674,815,488 → 601,087,488` bytes, with no cast/copy buffer or larger
binding.

The diagnostic harness may drive the existing exported OPT-0080 depth-two
scheduler directly over these bounded graph-prefix sequences. The ordinary
product/backend selector must omit or reject the OPT-0081 storage profile and
the public production default must remain unchanged throughout this gate.

Before timing, use an untimed correctness owner and correctness-only snapshot
copies. Descriptor and tap-capture commands are correctness-only, outside the
28 timed commands, may add no operation to the timed descriptor sequence, and
may not stand in for a producer. For each of A/A, A/B, and B/B deterministic
comparisons, retain and check both layers' complete checkpoints:

- all six candidate boundaries per layer—`selfModulated`,
  `selfMergedAttention`, `crossNormalized`, `crossMergedAttention`,
  `mlpModulated`, and `gatedActivation`—for `73,728,000` U16 words across the
  two layers in each B run, each bit-equal to an independent exact cast of the
  corresponding A FP32 producer value;
- all nine dense outputs per layer—self query, key, value, and output; cross
  query and output; MLP gate, up, and down—for `110,592,000` U32 words across
  the two layers in each run; and
- both complete layer outputs, `9,216,000` U32 words per run.

Require zero raw-bit mismatches, exact signed zeros, deterministic repeats,
finite complete production outputs, complete qNaN-prefill overwrite,
first/last and rows 2240–2249 coverage, intact prefix/suffix/adjacent guards,
zero validation or uncaptured GPU errors, and no device loss. Correctness
readback, comparison, hashing, allocation, compilation, upload, fixture reset,
and serialization are outside accepted timing. The timed path has no
measurement-only submit, drain, map, readback, descriptor/tap capture, or
timestamp query.

Compile, upload, finish correctness, execute the 25-command precompute, and
symmetrically warm one selected-slice occurrence per arm before READY. On one
fresh capable device, begin accepted timing only after at least 30 continuous
seconds at documented thermal level 0. Use an external absolute-cadence
1,000 ms trace from before that launch gate through final cleanup, with no
stale launch sample or polling gap above 1,500 ms. Disclose every later
transition under PLAN.md; do not repeat unchanged work merely to obtain an
all-nominal trace.

Run exactly eight paired rounds in this fixed order:
`AB, BA, BA, AB, AB, BA, BA, AB`. Every occurrence begins from the same
immutable latent, timestep, controls, and precomputed C98 caches; timestep and
input projection overwrite the arm's hidden input before its two actual
layers. A profile switch or fixture reset is permitted only after a terminal
true drain, and A/B work may never overlap on the queue. For every occurrence,
retain authoritative fenced wall from immediately before evaluation command
25 through the true drain after command 52, all seven disjoint epoch walls,
ordinary submit/fence/drain/idle counts, and the exact order. Overlapping per-
fence intervals are non-additive and may not be summed. Timestamp or family
attribution, if retained without changing submissions, is diagnostic only;
fenced slice wall is the decision authority. There is one timing run and no
unchanged retry.

Define forward rounds as the four `AB` rounds and reverse rounds as the four
`BA` rounds. B qualifies for the complete-evaluation gate only if every
correctness, route, arena, topology, thermal-start, and lifecycle condition
passes and all of the following wall conditions hold:

- B's overall eight-sample mean and median slice wall are strictly below A's;
- B wins at least seven of eight same-round pairs, including at least three of
  four forward and three of four reverse pairs;
- mean paired saving `mean(A_wall - B_wall)` is strictly positive separately
  in the forward and reverse subsets; and
- the mean paired saving in each direction is at least `31.25 ms`. Because the
  slice contains two of the 24 layers,
  `31.25 * (24 / 2) * 8 = 3,000 ms`; this is a representative-layer
  projection, not a measured evaluation or product saving.

Before accepted timing, make the already-registered variance condition exact:
for each four-round direction, calculate the ordinary sample standard deviation
of the paired savings with denominator `n - 1`, then the lower endpoint of the
two-sided 95% Student-t confidence interval as
`mean - 3.182446305284263 * sample_sd / sqrt(4)`. Both directional lower
endpoints must be at least `31.25 ms`. Retain the raw four paired savings, both
sample standard deviations, and both lower endpoints. This is the literal
operationalization of the registered rule that variance may overlap neither
zero nor the material floor; it is frozen before the one accepted browser
timing run and is not a post-result reclassification.

The `3,000 ms` directional floor is the literal meaning of “no loss of the
candidate's material projection” and aligns the already-frozen next-stage
one-evaluation threshold. A directionally mixed result, a miss in either
order, variance overlapping zero or the material floor, invalid thermal
provenance, topology drift, or wall-boundary inconsistency is inconclusive and
stops escalation; do not substitute a fastest sample or pool only the
favorable order.

Use one FIFO graph owner at a time. Require drain before every reset, profile
switch, snapshot map, or release; no host mutation of aliased graph storage
while work may read it; balanced created/destroyed buffers and mapped/unmapped
ranges; zero live buffers, bytes, maps, pending descriptors, callbacks, and
leases after success; idempotent destruction; and rejection after destroy.
Run an untimed candidate cancellation/failure preflight with one successor
already submitted, require no backfill or later progress after observation,
settle all submitted fences before release, preserve the original error, and
complete cleanup within `1,000 ms`. Setup-failure and device-loss paths must
also release every owned resource. The cancellation preflight reuses the one
resident B arm and its already-drained precompute; it may not allocate a second
candidate arena. Retain zero temporary graph buffers/bytes/owners and let the
required post-preflight B warmup overwrite the resident state before READY.
A representative-layer pass authorizes
only the independently cooled complete-evaluation A/B and B/A gate; it does
not select production.

Subject to that screen, escalation remains in this exact order:

1. **Actual MLP chain.** Passed only for B under the authoritative receipt
   above. C remains diagnostic-only and nonselectable.
2. **Representative layers.** Run only the frozen arm-B representative-layer
   diagnostic above. The authoritative run below completed as a non-pass, so
   step 3 was not authorized or executed.
3. **One complete evaluation.** Compare independently cooled A/B and B/A
   complete 24-layer M2250/C98 evaluation slices. Require selected dense and
   layer outputs raw-U32 exact, the evaluation result exact and deterministic,
   both directions faster, and at least `3,000 ms` projected eight-evaluation
   wall saving after unrelated-family deltas.
4. **Full trajectory.** Run all eight pinned shift-3 Euler/DCW-double
   evaluations. Require all eight sampler taps and the final latent raw-U32
   exact, deterministic repeat, bounded cancellation/failure behavior, one
   FIFO owner, unchanged OPT-0070 attention and OPT-0080 scheduling, exact
   `674,815,488 → 601,087,488` arena accounting, and zero live
   buffers/bytes/maps after cleanup.
5. **Product and selection.** Run the declared short direct product gate and
   require identical final latent, complete raw pre-normalization waveform
   U32, VAE seams, normalized WAV bytes/hash, metadata, cancellation, and
   lifecycle. Then run the repository's required cumulative production and
   thermal confirmation before selecting only the proven arm and exact six
   roles. No planner, long-duration, under-one-minute, or release-speed claim
   follows from the primitive and actual-MLP screens alone.

No new subjective listening comparison is required only if raw-U16 boundary,
every retained raw-U32 checkpoint, final latent, complete raw waveform, seams,
and WAV bytes remain exact throughout. The first raw mismatch stops this
experiment; an approximate or quality-envelope continuation requires a new
quality-affecting experiment with its own numerical and listening gates.

## Authoritative representative-layer result

The single accepted representative-layer timing is preserved in the compact
[result](../results/OPT-0081/representative-layers.json). The browser receipt
blob has SHA-256
`eddb46919d5f281d64c3babe4f4de7d68eabb70d955e19d0f690ba10739270de`
and byte length `208,588`. The ignored preserved copy adds one terminal LF and
therefore has SHA-256
`43741ab16e041ce996515e6c8a9f753ef4a94ca937bff6b7b6fe73b4847605da`
and byte length `208,589`. The continuous raw thermal JSONL has SHA-256
`55b2e0099c6aa8cf880a929ce35e8a6848f609caca9423f7d1aa616112a0807d`
and byte length `37,258`.

Correctness, arena, topology, thermal, cancellation, and lifecycle all passed.
A/A, A/B, and B/B had zero raw mismatches across every retained producer
boundary, dense output, and layer output. The candidate arena was exactly
`674,815,488 → 601,087,488` bytes with no timed cast/copy buffer. All 28
commands, seven epochs/drains, six idle turns, maximum depth two, and command
labels matched. Cleanup balanced `507/507` buffers, `128/128` maps, and
`66/66` runtime owners, reached zero live resources, and passed setup-failure,
cancellation, device-loss, idempotence, and post-destroy gates.

The timing result was directionally mixed and missed the frozen material gate:

| metric | observed | required |
| --- | ---: | ---: |
| A/B mean wall | `548.4625 / 519.7500 ms` (`1.055243x`) | B faster |
| A/B median wall | `538.5000 / 523.6500 ms` (`1.028359x`) | B faster |
| all paired wins | `6/8` | `≥7/8` |
| forward wins | `4/4` | `≥3/4` |
| reverse wins | `2/4` | `≥3/4` |
| forward mean saving | `24.0500 ms` | `≥31.25 ms` |
| reverse mean saving | `33.3750 ms` | `≥31.25 ms` |
| forward 95% lower endpoint | `-0.8357 ms` | `≥31.25 ms` |
| reverse 95% lower endpoint | `-42.6067 ms` | `≥31.25 ms` |
| forward eight-evaluation projection | `2,308.8 ms` | `≥3,000 ms` |
| reverse eight-evaluation projection | `3,204.0 ms` | `≥3,000 ms` |

The trace began before READY, then truthfully recorded `0 → 1 → 2 → 1 → 0`
during preparation/cooling. Launch used a fresh 87-observation all-level-0
suffix with `1,010 ms` maximum gap and occurred `898 ms` after its completion.
The complete 346-observation trace had `1,013 ms` maximum gap, zero missing
samples, covered cleanup, and ended at level 0. Later thermal transitions were
disclosed as required and did not alter the failed directional wall decision.

The authoritative disposition is
`inconclusive-directional-or-material-wall-evidence`. Complete-evaluation,
full-trajectory, product, package, and production-selection work did not run.
The exact six-role implementation and its positive primitive/MLP evidence stay
in the benchmark ledger, but production retains FP32 activation storage. No
unchanged timing retry is authorized.

## Authority and stop boundary

- Current exact dense owner:
  [OPT-0009](OPT-0009-fp16-gemm-calibration.md)
- Historical multicast mechanism/result:
  [OPT-0078](OPT-0078-dit-dense-weight-tile-multicast.md)
- Latest dense multicast negative/inconclusive context:
  [OPT-0079](OPT-0079-dit-dense-decoded-half-tile-multicast.md)
- Current production scheduling:
  [OPT-0080](OPT-0080-dit-depth2-completion-epochs.md)
- Thermal/reporting protocol: [`PLAN.md`](../../PLAN.md)

A non-pass closes the exact scalar-F16 producer boundary with the frozen B/C
owners on this browser/GPU. Do not retry it with a vector binding, conversion
dispatch, unchanged F32 OPT-0078 arm, relaxed materiality, or a selectively
reported MLP-only score. A materially different producer/storage format,
dense geometry, arithmetic, browser/compiler, or target GPU requires a new
experiment ID and ledger entry.
