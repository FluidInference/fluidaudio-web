# OPT-0005 — Reduction-chunked FP32 VAE Conv1D tile

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; the arithmetic claim is target-browser bit identity with
  the accepted scalar FP32 GPU kernel on the pinned Chrome/M3 target because
  WGSL permits contraction and reassociation

## Hypothesis

`OPT-0004` established that a 16-time by eight-output-channel FP32 tile is a
useful Conv1D direction, but its complete input halo and current-tap weight
slice scale with the full input-channel count. Its measured 128-channel shape
uses 15,904 bytes of workgroup storage, and the same layout exceeds the local
32 KiB limit above 264 input channels. Extending a complete halo to dilation
three or nine would increase storage further.

A distinct tile can keep kernel tap outermost and split only the inner input
channel reduction. For each `(kernel, input-channel chunk)` pair, it stages the
16 current-tap input positions and the eight current-tap weight rows, visits
the chunk's channels in increasing order, and then advances to the next chunk.
This retains the scalar `kernel -> input channel` arithmetic order while making
workgroup storage independent of the operation's total input channels and
dilation. It may therefore cover all production K7 Conv1D operations rather
than only the lower-channel dilation-one subset.

This is a composable optimization hypothesis, not a predicted minimum speedup
or an automatic acceptance threshold.

## Identity

- Allocation baseline: pushed `main` commit
  `e90f22741a1a81564a70bf73299f64157799e6c1`
- Candidate and harness commit:
  `75c810783668b2013e69c4852e1bf55349d2bbc3`
- Production integration commit:
  `31e8ef7f385b4c3b21180b356ca2d89ec00a7099`
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Execution profile: `reference-bf16-subgroups`; VAE package tensors and
  arithmetic remain FP32
- Package layout: unchanged converter-native Conv1D weights
  `[output_channel,kernel,input_channel]`
- Scheduling baseline: integrated `OPT-0002` work-aware FIFO ranges, one
  command buffer outstanding, full drain, and a real queue-empty interval
- Machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory; capture exact OS, Chrome, and WebGPU adapter identity in any result

## Production coverage and work

For a same-length K7 convolution with symmetric padding `3 * dilation`, the
accepted scalar shader executes
`inputChannels * outputChannels * (7 * frames - 12 * dilation)` valid MACs.
Applying that formula to the authenticated 256-latent-frame decoder graph gives
the following proposed coverage:

| Operations | Frames | Channels | Dilations | Count | Valid MACs |
| --- | ---: | ---: | --- | ---: | ---: |
| `decoder.conv1` | 256 | 64 -> 2,048 | 1 | 1 | 233,308,160 |
| block 0 residual `conv1` | 2,560 | 1,024 -> 1,024 | 1, 3, 9 | 3 | 56,207,867,904 |
| block 1 residual `conv1` | 15,360 | 512 -> 512 | 1, 3, 9 | 3 | 84,516,274,176 |
| block 2 residual `conv1` | 61,440 | 256 -> 256 | 1, 3, 9 | 3 | 84,546,945,024 |
| block 3 residual `conv1` | 245,760 | 128 -> 128 | 1, 3, 9 | 3 | 84,554,612,736 |
| block 4 residual `conv1` | 491,520 | 128 -> 128 | 1, 3, 9 | 3 | 169,111,781,376 |
| `decoder.conv2` | 491,520 | 128 -> 2 | 1 | 1 | 880,800,768 |

The 17 K7 operations total 480,051,590,144 valid MACs: 87.520381% of all
548,502,631,424 Conv1D MACs and 76.975784% of all 623,639,753,728 decoder
convolution MACs including ConvTranspose. They account for 2,045 of the 2,459
accepted `OPT-0002` Conv1D quanta. The excluded Conv1D work is the 15 K1
residual `conv2` operations, totaling 68,451,041,280 valid MACs.

By dilation, the proposed K7 domain contains 160,816,296,960 valid MACs across
seven dilation-one operations, 159,668,371,456 across five dilation-three
operations, and 159,566,921,728 across five dilation-nine operations. The six
512/1,024-input-channel operations alone contribute 140,724,142,080 valid MACs;
their two dilation-one members contribute 46,960,476,160. These are exact
static work counts, not performance projections.

## Candidate design

The first geometry to test is deliberately small and portable:

- `@workgroup_size(16, 8, 1)`; one invocation exclusively owns one output for
  one time and one output channel;
- a fixed 64-input-channel reduction chunk;
- a logical `16 x 64` current-tap input slice stored channel-major with padded
  time stride 17: `64 * 17 = 1,088` FP32 values, or 4,352 bytes;
- a logical `8 x 64` current-tap weight slice stored output-channel-major with
  padded input-channel stride 65: `8 * 65 = 520` FP32 values, or 2,080 bytes;
- 1,608 FP32 values and 6,432 bytes of total workgroup storage, independent of
  total input channels and dilation; and
- native `[output_channel,kernel,input_channel]` package weights, with no
  prepack, duplicate package layout, or model conversion change.

For comparison, chunk sizes 32 and 128 would use 3,232 and 12,832 bytes with
the same padding scheme. Chunk 64 is the initial balance: it leaves ample room
under the M3's 32 KiB limit without imposing the doubled synchronization count
of chunk 32. The storage limit alone could admit five 6,432-byte workgroups,
but no occupancy claim is made without measurement because registers, thread
limits, and the implementation's resource allocation also matter.

The shader source order must be:

1. initialize the private accumulator once from bias or positive zero;
2. visit K in increasing order;
3. within each K, visit 64-channel chunks in increasing order;
4. cooperatively load that K/chunk's 16 input positions and eight weight rows;
5. reach a uniform workgroup barrier;
6. for an active output and valid source tap only, visit the chunk's real input
   channels in increasing order with `sum = sum + input * weight`; and
7. reach a second uniform barrier before either shared tile is overwritten.

Concatenating the chunk-local channel loops is exactly the scalar input-channel
order for each K. Every lane must execute every K/chunk load and both barriers;
output, channel, range, and padding tails may predicate arithmetic and the final
write, but may not return around a barrier. An invalid padding tap must skip
the complete arithmetic loop rather than multiply staged zero placeholders,
which is observably different for signed zero and NaN. A final partial channel
chunk must likewise exclude placeholder lanes from arithmetic.

At production channel counts 64, 128, 256, 512, and 1,024, each K has 1, 2, 4,
8, and 16 chunks. A K7 workgroup therefore executes 14, 28, 56, 112, or 224
barriers respectively. This synchronization cost and the repeated per-K input
loads are central measurements, not hidden overhead.

The candidate reuses complete-channel, batch-bounded row ranges from
`OPT-0002`. Every current production K7 quantum is row-aligned under the
accepted work policy: its row counts are 256 for 64 -> 2,048, 32 for
1,024 -> 1,024, 128 for 512 -> 512, 512 for 256 -> 256, 2,048 for
128 -> 128, and 131,072 for 128 -> 2. Unsupported shapes, non-row-aligned
ranges, device limits, or unsafe U32 staging/index bounds must fail closed to
the scalar kernel. `OPT-0004` remains a measured candidate and may remain the
better low-channel selector; this experiment does not assume one universal
kernel before a direct overlap comparison.

## Correctness gate

- Authority: zero scalar-versus-candidate GPU bit mismatches on the pinned
  Chrome/M3 target. Independent CPU indexing, contraction, and cancellation
  sentinels diagnose source-order errors but do not replace the GPU authority.
- Exercise complete manageable multi-range graphs for dilation 1, 3, and 9;
  left/right padding; bias and no bias; multiple batches; time, output-channel,
  workgroup, range, and partial 64-channel tails; and both sides of a channel
  chunk boundary.
- Include signed zero, mixed magnitudes, NaN-protected padding, a
  contraction-sensitive value, and a cancellation case that differs if K or
  global input-channel order is reassociated in source.
- Prefill scalar and candidate outputs independently with a non-finite U32
  sentinel. Require complete finite, nondegenerate writes, zero full-domain GPU
  bit mismatches, matching fingerprints, and independent CPU edge/range
  sentinels.
- Prove exact-once output coverage, no batch/range crossover, native weight
  indexing, dilation address calculation, channel-tail masking, and all U32
  additions/products before any actual-GPU timing.
- Validate device capability, workgroup-storage and binding limits, alias
  rejection, compilation failure cleanup/retry, destruction during compile,
  bounded cancellation, and no submission after an abort.
- Listening is unnecessary while the primitive and every later integrated
  boundary remain bit-identical. Any numerical change requires a new risk
  declaration and the appropriate subsystem, waveform, and listening gates.

## Sparse benchmark protocol

- Start with static planner/source tests and one complete manageable GPU
  correctness graph. Do not run a song or full VAE window.
- Use one bounded production range to compile and screen chunk 64 on the
  highest-channel dilation-one geometry. Only compare chunk 32 or 128 if the
  initial result leaves synchronization/occupancy attribution genuinely
  uncertain; do not perform an automatic geometry sweep.
- If correctness and the screen are credible, run one thermally valid balanced
  scalar/candidate paired page for the complete production
  `block-0-res-1-conv1` operation (2,560 frames, 1,024 channels, dilation one),
  whose working buffers are about 48 MiB and whose 18,777,899,008 valid MACs
  make it decision-relevant without a long decoder or song.
- Validate dilation three and nine on complete manageable graphs. Time at most
  one bounded high-channel dilation-nine production range unless dilation
  unexpectedly changes the result; its arithmetic and storage topology are
  otherwise the same.
- On the overlapping 128-channel dilation-one geometry, perform one cheap
  direct range comparison against both scalar and `OPT-0004`. Use that evidence
  to retain a hybrid selector or a unified chunked kernel rather than assuming
  either outcome.
- Compile, allocate, and upload outside timing; warm every compared kernel
  symmetrically. Retain all balanced paired samples, encode/submit/drain/idle
  components, valid logical throughput, workgroup storage, buffer bytes,
  heartbeat gaps, and per-range maxima.
- Preserve the accepted cooperative topology: one command buffer outstanding,
  full drain, real queue-empty interval between non-final ranges, and an abort
  probe that prevents every later submission after the active quantum drains.
- Require a continuously nominal external thermal pre-gate and poll through
  the run and immediate post-run state. One controlled run may establish a
  clear direction; repeat only if variance or attribution can change the
  decision.
- Judge the experiment by exactness, measured direction, resource cost, and a
  credible integration path toward the repository's single end-to-end target.
  There is no per-experiment speed threshold. Escalate to an integrated block,
  decoder boundary, waveform, or song only when that evidence can change a
  production decision.

## Main risks

- Synchronization scales with `7 * ceil(inputChannels / 64)`: the 1,024-channel
  shape reaches 224 barriers per workgroup. Barrier and shared-memory overhead
  could outweigh the storage reuse even though the candidate now fits.
- Unlike `OPT-0004`'s one-time complete input halo, this design reloads 16
  current-tap inputs for every K. It reduces storage and still reuses each input
  across eight output channels and each weight across 16 times, but increases
  global input traffic relative to the whole-halo tile.
- Padded shared strides are a bank-conflict hypothesis, not an Apple hardware
  guarantee. Actual occupancy and bank behavior must be inferred from measured
  target-browser results, not the storage ceiling alone.
- A chunk loop can accidentally change arithmetic order, include tail
  placeholders, or place a barrier behind a nonuniform predicate. Any such bug
  invalidates the exact risk class.
- Dilation and padding arithmetic adds new U32 overflow surfaces at the last
  staged time. Range planning must validate the actual tiled addresses, not
  only logical output ends.
- The chunked kernel may win high-channel shapes while losing to `OPT-0004` at
  128/256 channels. A fail-closed shape selector is acceptable; premature
  replacement of a measured winner is not.

## Results

The benchmark-local chunk-64 kernel passed its declared target-GPU correctness
gate and produced a stable positive full-operation result on the production
block-0 dilation-one shape. Compile, allocation, upload, and deterministic
input construction remained outside timing. Each timed path kept one command
buffer outstanding, drained after every range, and requested a real queue-empty
idle between the 79 non-final ranges.

| Full `block-0-res-1-conv1`, dilation 1 | Scalar | Chunked | Ratio |
| --- | ---: | ---: | ---: |
| Median active wall | 691.400000244379 ms | 401.14999997615814 ms | 1.723544809386692x |
| Median wall including queue-empty idle | 787.5 ms | 495.0499999821186 ms | 1.5907484093090491x |
| Median logical throughput | 27.16179131379093 GMAC/s | 46.81269988365762 GMAC/s | 1.7234761633666364x |
| Paired wins | — | 4 / 4 | — |

This page executed all 80 production ranges and 18,777,899,008 valid MACs per
sample. The chunked active samples were 418.0999998450279,
404.10000014305115, 383.3000003695488, and 398.19999980926514 ms. The scalar
samples were 815.8999996185303, 672.6999999284744, 698.1000004410744, and
684.7000000476837 ms. The first scalar sample was slower than the other three,
but every same-round comparison still favored the candidate. The candidate's
largest single range drain was 10.199999988079071 ms, and its maximum observed
animation-frame and timer gaps were 6.7000000000000455 and
11.599999964237213 ms.

The full-operation readback checked first, middle, and last complete range
slices: 98,304 FP32 outputs had zero scalar-versus-chunked bit mismatches and
all independent CPU sentinels were bit exact. This is not a claim that every
one of the 2,621,440 full-operation outputs was read back. The complete
manageable preflight did compare every output for dilation one, three, nine,
bias/no-bias, batches, padding, 63/64/65-channel tails, and arithmetic-order
discriminants; all four cases were fully finite, bit identical, and CPU
sentinel exact. The cancellation probe submitted and drained its active range
and prevented every later range submission.

A bounded middle production range at dilation nine also passed a 32,768-output
exact slice and won all four pairs. Its scalar and chunked median active times
were 8.699999988079071 and 4.5999999940395355 ms, a
1.8913043476852442x speedup. This is one range, not a complete dilation-nine
operation.

The direct 128-channel dilation-one overlap page compared scalar, `OPT-0004`,
and `OPT-0005` over all 262,144 outputs and found zero bit mismatches for both
candidates. Its timing is deliberately classified as ambiguous: scalar ranged
6.5–29.19999998807907 ms, `OPT-0004` ranged 6–32.10000002384186 ms, and
`OPT-0005` ranged 5.300000011920929–32.69999998807907 ms. The large,
cross-kernel outliers make the apparent medians unsuitable for choosing the
overlap winner. Production integration should therefore preserve the already
proven `OPT-0004` priority wherever it is eligible and select `OPT-0005` only
where `OPT-0004` fails closed. A later stable integrated measurement may
revisit that conservative policy.

All four pages had continuous external nominal thermal coverage spanning the
pre-gate, run, and immediate post-run state:

| Page | External span | Observations | Maximum gap | Non-nominal |
| --- | ---: | ---: | ---: | ---: |
| dilation-one range screen | 146.74246262502857 s | 147 | 1.0040905839996412 s | 0 |
| full block-0 dilation one | 71.52881466702092 s | 72 | 1.003572916961275 s | 0 |
| dilation-nine range | 63.863770207972266 s | 64 | 1.00496887502959 s | 0 |
| 128-channel overlap | 84.8038889580057 s | 85 | 1.005010791006498 s | 0 |

The in-page nominal pre-gates were respectively 34.005, 35.003, 35.005, and
35.002 seconds, with 35/36/36/36 observations and zero non-nominal states.
Only one thermally valid page was run per mode, so independent-run variance is
not quantified. The initial dilation-one single-range screen had extreme
scalar drift (57.89999997615814–544.3000000119209 ms); it established only a
positive direction and is not used to quantify the retained speedup. No full
VAE window or song was executed.

Raw ignored artifacts and SHA-256 identities:

- `screen-d1.json`:
  `ef177d738f0dced36369a10d29f01e373a55911e46ae3db6cc549a03471d0b6d`
- `screen-d1-thermal.jsonl`:
  `79e56d163f3e8f37885186cb40b451e7cc3fa653ba3ab0223afa27f9ff7b7e73`
- `block0-d1.json`:
  `94f9a88498a242916ccf16cac7cc23511edb09e767d1eadad5bdb8c315b34e96`
- `block0-d1-thermal.jsonl`:
  `49eecf03e50ed1b9423fed67d4237a51f6c176eca105ce4efd16fded0e384335`
- `screen-d9.json`:
  `ed88ca5d5d54094b03d4609e91aad52d0af2b0adf42a55c59fdcc3875941b5ce`
- `screen-d9-thermal.jsonl`:
  `afdb24312cc1009a4a7ee62ce15a47749875aec8a3af74841b0deda6ec8e8158`
- `overlap-d1.json`:
  `990549573ca3b4c357d35d410cd7c82409c1a7b9378d39620a121ad1f02a0553`
- `overlap-d1-thermal.jsonl`:
  `3c6c61e1029e4d6d427f2c0d8b3e0bc5b3d597c7d228439e9d9038ab1eb1392a`

Production commit `31e8ef7f385b4c3b21180b356ca2d89ec00a7099`
integrates the channel-chunked kernel through a fail-closed hybrid selector.
It preserves the conservative policy derived above: `OPT-0004` is tried
first, then `OPT-0005`, then the portable scalar kernel. The converter-native
weights, package identity, FP32 arithmetic order, quantum policy, FIFO owner,
drain policy, and real queue-empty interval are unchanged.

For the canonical 256-latent-frame graph, the production selector assigns 365
of 2,459 Conv1D quanta to `OPT-0004`, 1,680 to `OPT-0005`, and the remaining
414 K1 quanta to the portable kernel. Thus `OPT-0005` fills the proven
high-channel and dilation 1/3/9 remainder without displacing `OPT-0004` on its
eligible overlap.

An actual-Chrome complete C136 -> C128 one-block decoder comparison then ran a
forced-portable profile against the production default. It exercised all
three production families in the optimized path: seven tiled quanta for
`block-0-res-1-conv1` and `conv2`, 15 channel-chunked quanta for `conv1`,
`block-0-res-2-conv1`, and `block-0-res-3-conv1`, and 18 portable quanta for
the three K1 residual `conv2` operations. The forced-portable control assigned
all 40 quanta to the scalar kernel. All 12 final FP32 outputs were U32
bit-identical, and the optimized output's maximum absolute error against the
independent CPU oracle was 0.000354766845703125. Listening remains unnecessary
because the integrated boundary is exact. This is still not a full VAE window,
waveform, or song benchmark.

The ignored integrated gate artifact is
`integrated-decoder-correctness.json`, SHA-256
`26c4f9f35eb8e255cd5a8fd3972d7fc46d7543bfd030f732a44a9fc05210d643`.

## Evidence and disposition

- Evidence conclusion and rationale: `positive`. The complete production
  high-channel dilation-one operation was exact on checked range slices, won
  every pair, and reduced both active and cooperative wall time materially;
  the separate dilation-nine range was also exact and positive.
- Disposition: `integrated`. Production commit `31e8ef7` uses the declared
  fail-closed `OPT-0004`-first hybrid selector, and the actual-Chrome complete
  decoder boundary was bit-identical to the forced-portable control while
  exercising tiled, channel-chunked, and portable paths.
- Result JSON: `optimization/results/OPT-0005/result.json`
- Candidate and harness commit:
  `75c810783668b2013e69c4852e1bf55349d2bbc3`
- Production integration commit:
  `31e8ef7f385b4c3b21180b356ca2d89ec00a7099`
- Interactions: extends the positive `OPT-0004` mechanism without changing its
  retained evidence; holds `OPT-0002` package layout, range policy, and
  cooperative scheduling fixed
- Revisit when: a stable integrated overlap measurement can change the
  conservative `OPT-0004`-first policy, or a safer and faster
  reduction-preserving geometry is derived
- Follow-ups: retain the hybrid selector while broader optimization continues;
  run a VAE-window, waveform, or song only when it can change an end-to-end
  production decision
