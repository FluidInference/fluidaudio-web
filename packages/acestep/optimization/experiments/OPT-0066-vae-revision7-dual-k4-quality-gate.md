# OPT-0066 — Revision-7 dual-K4 arithmetic-oracle and quality gate

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: two bounded FP16-K4 partial mechanisms change arithmetic relative to
  the revision-6 scalar-FP32 VAE oracle

## Why a new experiment is required

The frozen OPT-0054/OPT-0057 C512 preparation did not reach READY and no timed
work ran. Its six complete untimed executions established:

- both revision-6 scalar runs reproduced accepted SHA-256
  `893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47`;
- both revision-7 runs were deterministic at
  `84a908b8aa9cc6656b967226de69f51046c109065d38ec07599d8d2165c93564`;
- candidate versus scalar passed the frozen numerical envelope: joint NRMSE
  `0.0015684427168221327`, SNR `56.090626762434 dB`, Pearson
  `0.9999987699871865`, and relative maximum error
  `0.0038859529685460943`;
- topology, finite output, sequential ownership, idempotent destruction, and
  runtime-event gates passed; but
- candidate versus the declared native-layout K7-only K4 oracle differed in
  `1,966,027 / 1,966,080` U32 output words and
  `2,544,784 / 3,932,160` U16 words.

The failure receipt is
[`../results/OPT-0057/failed-setup-raw-oracle-diagnostic.json`](../results/OPT-0057/failed-setup-raw-oracle-diagnostic.json),
SHA-256 `364273a17eedc5553fd6bb1b2cf9b9908fd1924d3d89d8e57f7478ba8122794f`.
It is setup/correctness evidence only.

Offline inverse-layout verification then proved all `145 / 145` revision-7
VAE logical tensors bit-identical to revision 6. The converter and package
bytes are not the cause. The frozen full-output oracle was incomplete: it gave
selected K7 operations OPT-0024 K4 arithmetic, but retained revision-6
OPT-0040 scalar ConvTranspose arithmetic while the candidate used OPT-0048 K4
partials for blocks 1-4. The earlier OPT-0048 dyadic fixture happened to be
raw-exact; FP16 dot4 partials widened into FP32 are not generally the same
arithmetic as an increasing-Cin scalar-FP32 reduction.

Changing that oracle after observing failure would weaken OPT-0054/0057.
OPT-0066 therefore starts a distinct quality-affecting direction. It treats
both selected K7 and selected ConvTranspose K4 partials as approximate versus
the scalar revision-6 oracle while preserving a separate raw identity gate for
physical layout and routing.

## Frozen direction

- Reuse the authenticated revision-7 package bytes only after verifying
  manifest SHA-256
  `36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7`
  and `716185` manifest bytes. Do not regenerate or relabel its bytes.
- Add a distinct fail-closed runtime/diagnostic profile identity. The
  misclassified OPT-0054 identity and the revision-6 production default remain
  unselected.
- Build a complete same-arithmetic oracle from authenticated revision-6
  logical weights: selected K7 uses native-layout OPT-0024 K4; selected
  ConvTranspose blocks 1-4 use benchmark-packed OPT-0048 K4 with the same
  channel/row routes as revision 7; all other owners remain literal OPT-0040.
  The oracle and candidate must never own GPU weights simultaneously.
- Candidate and same-arithmetic oracle must differ only in physical package
  layouts and the corresponding indexers. K7/ConvTranspose reduction order,
  dot width, partial precision, FP32 running state, bias, FP16 stores, graph,
  window, scheduling, input, and every other owner must match.
- Separately retain revision-6 OPT-0040 as the exact scalar-arithmetic quality
  oracle. Never describe raw equality to the same-arithmetic arm as equality
  to production math.

## Gates

1. Prove exhaustive converter/TypeScript index and inverse bijections and
   authenticate all `145` logical tensors. On actual package weights, compare
   bounded native-versus-packed slices for every selected label before GPU
   execution.
2. Run two complete C512 executions each for scalar revision 6, the complete
   revision-6 same-arithmetic oracle, and revision 7. Require deterministic
   repeats and raw-U16/U32 complete-output identity between the latter two.
   Preserve output hashes, first mismatch, full counts, topology, canaries,
   finite/class scans, one-owner-at-a-time lifecycle, and failure cleanup.
3. Compare revision 7 separately against scalar revision 6 under OPT-0044's
   unchanged joint/left/right waveform envelope. The already observed setup
   metrics are diagnostic evidence, not a substitute for the corrected gate.
4. Only after READY, make one level-0 `notifyutil` observation after at least
   30 idle seconds and run scalar/candidate/candidate/scalar. Require in both
   paired directions:
   - homogeneous K7 wall improves;
   - homogeneous ConvTranspose wall improves;
   - complete decoder and outer-window walls do not regress.

   Median K7 speedup must reach `1.50x` and median ConvTranspose speedup
   `1.30x`. Retain every raw sample and attribute mixed batches separately.
5. If C512 passes, run the bounded OPT-0059 short C512/C2314 geometry screen,
   then one C4500 correctness/lifecycle waveform pass against scalar revision
   6. Require deterministic output, finite samples, clean seams/tails,
   under-4-GB peak residency, cancellation, and exactly-once cleanup.
6. Because both K4 mechanisms change arithmetic, production selection still
   requires OPT-0044 trajectory evidence, accepted 12-second instrumental and
   30-second planner-vocal listening comparisons, and explicit owner approval.

No timing retry, production selection, long-song projection, or under-one-
minute claim is authorized until the complete same-arithmetic oracle reaches
READY and every later gate passes.

## C512 result — 2026-08-15

The authenticated result is
[`../results/OPT-0066/result.json`](../results/OPT-0066/result.json), SHA-256
`3062c4ca30e346fc3a4d0bd8e7dcf1258c76d021e808c47c516c27c963c71b63`.
It reports `status: passed` and decision
`positive-authenticated-dual-k4-c512-quality-gate-passed` for core and harness
commit `acf8dd6c4e4ee02f5556bd2495a67a853fb4aed5` on the 10-GPU-core,
16-GiB `Mac15,12`, macOS `26.5.2` build `25F84`, and stock Chrome
`151.0.7922.138`. The C512 fixture contained `32,768` FP32 elements / `131,072`
bytes and authenticated as
`eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899`.

### Package and corrected raw gate

- Revision 6 authenticated as manifest
  `94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`,
  `715,301` bytes, converter revision 6. Revision 7 authenticated as
  `36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7`,
  `716,185` bytes, converter revision 7. Each had 145 tensor records, seven
  VAE weight files, and `168,791,552` resident weight bytes.
- The bounded pre-device proof authenticated all 145 logical tensors in each
  manifest and compared all 12 selected K7 plus four selected ConvTranspose
  layouts: `35,880,960` U16 words, zero mismatches, and null first mismatch for
  every entry. The four revision-6-derived ConvTranspose K4 arrays retained
  for the oracle occupied exactly `15,335,424` bytes.
- Correctness ran scalar/scalar, complete-same-arithmetic-oracle/oracle, then
  candidate/candidate. Reaching READY, and therefore the persisted passing
  timed receipt, required both full candidate-versus-complete-oracle
  comparisons to be raw-U32 and raw-U16 exact across the fixed
  `1,966,080`-U32 / `3,932,160`-U16 output. This closes the incomplete-oracle
  failure without reclassifying OPT-0054/0057.
- The final timed receipt did not serialize the preparation receipt's
  per-comparison mismatch objects or the complete-oracle output hash. Those
  missing fields are not reconstructed here. It did retain deterministic
  timed reloads: both scalar arms hashed
  `893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47`
  and both candidates hashed
  `84a908b8aa9cc6656b967226de69f51046c109065d38ec07599d8d2165c93564`.
  Each timed output versus its untimed reference had zero mismatches over all
  `1,966,080` U32 and `3,932,160` U16 words, with both first-mismatch indices
  null.

All four timed outputs contained `1,966,080` finite, nonzero FP32 samples;
first and last frames were finite. Scalar peak/min/max were
`0.9584733843803406 / -0.9420424103736877 / 0.9584733843803406`; candidate
peak/min/max were
`0.9587961435317993 / -0.9416432976722717 / 0.9587961435317993`.

### Separate scalar-arithmetic quality result

Both persisted candidate-versus-scalar comparisons produced exactly the same
OPT-0044 joint/left/right metrics and passed every bound:

| channel | count | NRMSE | SNR dB | Pearson | relative RMS drift | relative energy drift | relative peak drift | relative DC drift | relative maximum error | maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint | 1,966,080 | `0.0015684427168221327` | `56.090626762434` | `0.9999987699871865` | `0.0000011019845389320056` | `0.000002203967863527171` | `0.000336742946354641` | `0.0000020724293023343167` | `0.0038859529685460943` | `0.0037245824933052063` |
| left | 983,040 | `0.0015342902552414296` | `56.28184946581954` | `0.9999988229094495` | `0.0000005961118658326742` | `0.0000011922233762732618` | `0.000336742946354641` | `0.0000022695852403901888` | `0.003504940606690768` | `0.003359392285346985` |
| right | 983,040 | `0.0016094903523344473` | `55.86623244468659` | `0.9999987047577804` | `0.0000017247262737493206` | `0.000003449449572753394` | `0.0003765079594383436` | `0.0000018595691962373533` | `0.004171502913630876` | `0.0037245824933052063` |

The joint control/candidate RMS values were
`0.16239398088862908 / 0.16239380193297293`, peaks were
`0.9584733843803406 / 0.9587961435317993`, and means were
`-0.0004063913983484597 / -0.0004060548483039434`. This is a numerical pass
against scalar revision 6, not raw equality to production arithmetic.

### Thermal protocol and raw timing samples

The first prepared page correctly rejected timing. Its receipt is
[`../results/OPT-0066/failed-thermal-preflight.json`](../results/OPT-0066/failed-thermal-preflight.json),
SHA-256 `78fe992709013c5f567a7247968a4f56245f791185bfa15dd33c0a2fe5bd414b`.
It reached READY at epoch ms `1786851048944`, made exactly one `notifyutil`
observation after `49,498 ms`, observed thermal-pressure level 1, did not click
the timing button, performed no timed GPU work, and released the page.

The fresh successful gate waited from epoch ms `1786851366206` through
`1786851512206`, exactly `146,000 ms`, then made one level-0 observation.
Timing used scalar/candidate/candidate/scalar, one outstanding command buffer,
64 quanta per command buffer, 123 decoder command buffers, 124 total submits
and matching drains, and `123 ms` of requested queue-empty idle in every arm.
The raw samples were:

| order | arm | owner setup ms | homogeneous K7 submit-through-drain ms | homogeneous ConvTranspose submit-through-drain ms | decoder submit-through-drain ms | outer-window wall ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | scalar | `270.10000002384186` | `3996.300000190735` | `387.39999997615814` | `6419.600000023842` | `6610.900000095367` |
| 2 | candidate | `244.89999997615814` | `2045.9000002145767` | `184.89999997615814` | `3779.7000004053116` | `3979.899999976158` |
| 3 | candidate | `195.5` | `2047.9000005722046` | `207.39999985694885` | `3725.5000001192093` | `3917.5` |
| 4 | scalar | `207.30000007152557` | `3990.2999999523163` | `397.40000009536743` | `6445.599999785423` | `6637` |

Forward/reverse K7 speedups were
`1.9533212765881018x / 1.9484838121184553x`; ConvTranspose speedups were
`2.095186587485729x / 1.9161041483580923x`; decoder speedups were
`1.6984416750894102x / 1.7301301837549794x`; and outer-window speedups were
`1.6610718862622103x / 1.6941927249521378x`. Every paired-direction condition
passed. Aggregate median speedups were:

- K7: `1.95090136269679x`, above the required `1.50x`;
- ConvTranspose: `2.0005098149513043x`, above the required `1.30x`;
- complete decoder: `1.7141715076094104x`;
- outer window: `1.6775014561925903x`.

The homogeneous measurements covered 52 batches / 3,328 quanta for K7 and
six batches / 384 quanta for ConvTranspose in each arm. The complete topology
still reconciled all 4,090 K7 quanta (`3,360` selected and `730` native), all
644 ConvTranspose quanta, 819 K1, 1,611 Snake, 690 Add, and one ingress
quantum. Each graph had 88 operations, 7,854 graph quanta, and 7,855 sequence
quanta; the remaining mixed-family batches were retained separately rather
than prorated into either family.

### Lifecycle and disposition

GPU ownership stayed sequential with a peak of one live owner. The run created
and destroyed four scalar owners, two complete same-arithmetic owners, and four
candidate owners: `10 / 10`, zero live owners, every backend destruction
idempotent, and no runtime events. Final cleanup passed before and after device
destruction, destroyed all weight phases, backends, activation/control/readback
resources and the device context, and took `0.10000002384185791 ms`.

This is positive authenticated C512 evidence for the revision-7 dual-K4
mechanism, but it remains benchmark-only. The receipt explicitly records
`productionDefaultChanged: false`, `productSelectionAuthorized: false`,
`listeningApprovalStillRequired: true`, and `under60SecondClaim: false`.
It authorizes only the registered OPT-0059 bounded C512/C2314 geometry screen,
followed on success by the C4500 correctness/lifecycle waveform and the frozen
instrumental/planner-vocal listening gates. It does not authorize revision-7
production selection, a long-song speedup, a listening conclusion, or an
under-one-minute product claim.
