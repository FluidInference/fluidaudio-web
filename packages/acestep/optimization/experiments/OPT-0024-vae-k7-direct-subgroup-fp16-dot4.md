# OPT-0024 — VAE K7 direct-subgroup FP16 dot4 partials

## Status

- Evidence: `positive` at the frozen primitive and complete C512 subsystem
  gates
- Disposition: `benchmark-only`; activation-trajectory and 12-second product
  waveform escalation is authorized, but no production selection or
  integration is authorized
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `approximate FP16-partial reduction`; operands and the dot result
  are FP16, each four-Cin partial is widened once, and only the accumulator
  across increasing four-Cin groups remains FP32
- Allocation baseline: pushed `main` commit
  `c902bc77d888dc31b47275baf531962bbb348fa4`
- Current production owner:
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`
- Candidate implementation and browser primitive gate: present after
  registration
- C512 result: present; production selection: absent

## Hypothesis and why this is next

The current authenticated C4500 production VAE capture takes
`161,392.39999997616 ms`. Homogeneous K7 command buffers are its largest
directly measured family at `59,993.59999811649 ms`, ahead of ConvTranspose at
`42,401.00000369549 ms` and K1 at `25,772.300002217293 ms`. OPT-0023 did not
measure GPU utilization or pure GPU execution, so neither low utilization nor
memory bandwidth is treated as an observed cause. It did establish a large
enough wall-time owner for a concrete K7 mechanism to clear the repository's
`10,000 ms` credibility floor.

The shipped subgroup kernel already has useful output ownership: it is
barrier-free, keeps native O-K-I weights, and broadcasts one input over 128
output channels. Its inner loop nevertheless advances Cin one scalar at a
time. Every lane performs four scalar FP16 weight loads, eight scalar subgroup
broadcasts feed the owned rows, and eight FP32 `vec4` accumulator additions
occur per scalar-Cin step. The candidate freezes one different reduction
mechanism: advance Cin in groups of four, load native contiguous operands as
`vec4<f16>`, compute FP16 dot4 partials, explicitly widen each four-output
partial vector once, and add it once to the existing FP32 accumulators.

If a representative weighted K7 screen reaches `1.25x`, carrying that ratio
to the measured long K7 family would reduce it to about `47,994.88 ms`, a
planning saving of about `11,998.72 ms`. That arithmetic motivates the gate;
it is not a projected or measured product speedup. The candidate must prove an
actual C512 saving and then meet three explicit C4500 thresholds before any
long saving is claimed.

Adaptive scheduling is not a stronger current mechanism on the available
evidence. OPT-0006's historical bounded 16-range screen improved batch 16 only
about `2.6%` beyond the integrated batch 8 while increasing the maximum
observed drain from `39.3 ms` to `69.2 ms`. That older shape/profile result is
not current C4500 performance evidence. OPT-0023 also leaves
`17,658.19999921322 ms` of within-decode residual unsplit, so it cannot assign
at least `10,000 ms` of that residual to scheduling. K7 dot4 is therefore the
stronger directly attributable mechanism to screen next; no scheduling saving
or final priority is claimed without a separately registered measurement.

## Frozen current authority

Fail closed unless the implementation and every result authenticate:

- ACE source `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`, ACE main-model
  snapshot `19671f406d603126926c1b7e2adc169acbcade22`, and converter-v5
  manifest SHA-256
  `5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3`;
- current runtime profile
  `opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1`, kernel set
  `opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1`, and
  precision-map SHA-256
  `4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85`;
- exactly 145 authenticated native-layout FP16 VAE tensor records in seven
  files, totaling `84,395,776` elements and `168,791,552` resident bytes;
- `shader-f16`, `subgroups`, fixed subgroup minimum/maximum `32`, the actual
  browser/OS/Mac/adapter identities, and all requested device limits; and
- the complete current C4500 result authority below.

Current source and result identities at allocation are:

| Authority | SHA-256 |
| --- | --- |
| `src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts` | `7d218516d6b2c8d6e3332a53101be5fdeae1142096c442433915bfa58941ce32` |
| `src/webgpu/kernels/vae-conv1d-fp16.ts` | `fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310` |
| `src/webgpu/vae-fp16-decoder.ts` | `06f76b31b122c3e794142dc9d4058d31ff4760fbcb6b85465cc33671616d8873` |
| `src/webgpu/vae-fp16-backend.ts` | `d5440efdc9be32a72613988859dac113daca659023276bc08220edf1135e719d` |
| `src/webgpu/vae-fp16-profile.ts` | `eeb063ddada1027f6a3ebd352997500fc5dd1e8bc6d7b61800c8f55b9f33baf9` |
| `optimization/experiments/OPT-0023-vae-c4500-production-family-profile.md` | `250c8503686e3ef988c819d80ee3ba468abf4b2dd029e7e8e05d013798ff084f` |
| `optimization/results/OPT-0023/result.json` | `dc7261b16d1db6cd24820ba5220ef594a9d6e9e7ade60fa4045eaac68777d935` |

The OPT-0023 browser receipt is `34,172` bytes with SHA-256
`6454a37243849ec9838d998abd9ca478d4b3720aa9b3edd4f497d152bda92d5c`.
Its complete thermal trace is `59,950` bytes with SHA-256
`ecd7eded7f17dd9a5a585b8859e30a942a573840e365500a718d3d67e0a64161`.
Its source authority covers 33 files / `846,140` bytes with aggregate SHA-256
`6946795009d3e578a00e2c5bc355b0b2d8fb4b9607fa542eafff4cbcbbba7daf`.
Any candidate core, harness, static contract, package-facing profile, receipt,
or thermal trace must add its own hash before it can become evidence.

## Candidate boundary

Change only the 16 biased, FP16-output K7 Conv1D operations in each decoder
window: `conv1` and the 15 residual `conv1` operations. Leave the final
no-bias, FP32-output `conv2` on
`ace-vae-fp16-fixed32-subgroup-k7-conv1d-v1`. K1, ConvTranspose, Snake, Add,
ingress, windowing, scheduling, readback, stitching, normalization, and all
DiT/planner math remain unchanged.

Freeze exactly one candidate geometry:

- WG128 containing four fixed-32 subgroups, zero workgroup storage, and zero
  workgroup barriers;
- each subgroup owns the shipped `8` rows by `128` output-channel tile;
- each lane owns eight rows by four output channels as eight live
  `vec4<f32>` accumulators, or 32 FP32 scalars;
- the four subgroups cover a `32`-row by `128`-channel workgroup tile exactly
  as the shipped kernel does;
- require `Cin % 4 == 0` and `Cout % 128 == 0`; the candidate shapes have Cin
  in `{64, 128, 256, 512, 1024}` and Cout in
  `{128, 256, 512, 1024, 2048}`; and
- bind the unchanged native NLC input and native O-K-I weight bytes as aligned
  `array<vec4<f16>>` views. Do not repack or duplicate them.

For increasing `kernel = 0..6`, then increasing
`cin4 = 0..Cin/4-1`, execute exactly:

1. Source lanes `0..7` each load the owned row's one contiguous input
   `vec4<f16>` when that row is valid.
2. Broadcast those eight vectors with constant subgroup source IDs `0..7`.
3. Every lane directly loads four native contiguous-Cin `vec4<f16>` weight
   vectors, one for each owned output channel.
4. For each valid owned row, construct
   `vec4<f16>(dot(x,w0), dot(x,w1), dot(x,w2), dot(x,w3))`, explicitly widen
   that vector once to `vec4<f32>`, and add it once to the row's FP32
   accumulator.
5. Begin each accumulator with the corresponding FP16 bias widened to FP32,
   and explicitly round the final store to FP16.

Invalid padded rows remain skipped; do not add artificial zero partials. K and
R4-group order remain increasing, but the four products inside each WGSL
`dot(vec4<f16>, vec4<f16>) -> f16` are a different, lower-precision reduction.
No raw-bit, algebraic equivalence, cross-browser bit identity, contraction, or
unspecified compiler-precision claim is permitted.

There is no fallback under an existing kernel ID. A passing candidate receives
a new benchmark kernel ID and explicit profile/kernel-set identity. Existing
portable, OPT-0011, and OPT-0015 profiles remain byte-for-byte and
selection-for-selection unchanged until every promotion gate passes.

## Frozen implementation and API plan

Implement the candidate as one isolated owner in
`src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts`, with kernel ID
`ace-vae-fp16-opt-0024-direct-native-oki-fp16-dot4-k7-conv1d-v1`. Do not
replace or conditionally alter the shipped fixed32 owner.

Mirror the existing ownership API:

- `create(device, capability)` requires `shader-f16`, `subgroups`, fixed
  subgroup minimum/maximum `32`, WG128 limits, and a live device;
- `createDispatch(label, shape, existingBindings, "float16", range)` reuses
  the existing native input/weight/bias/output/range bindings and
  `planAceFp16VaeConv1dSubgroupRange`; and
- fail closed for non-K7, absent bias, FP32 output, `Cin % 4 != 0`,
  `Cout % 128 != 0`, invalid ranges, misaligned/undersized/overlapping
  bindings, or unsupported device limits.

An explicit OPT-0024 benchmark profile routes exactly the 16 biased
FP16-output K7 operations to this owner. Final no-bias FP32 `conv2`, K1,
ConvTranspose, Snake, Add, ingress, and every other operation retain their
current shipped owners. Cache keys must separate candidate/shipped pipelines
and every relevant shape/bias/output-storage dimension; rejected compilation
promises are evicted; destroy is idempotent; and create/dispatch after destroy
is rejected. Static contracts bind this file, generated WGSL, profile routing,
and unchanged-owner source hashes before any GPU result.

## Why this is not OPT-0017 again

OPT-0017 used WG256, adaptive 32/64-row tiles, shared R32 input and transposed
weight panels, millions of barriers, a packed K-I-O weight payload, and FP16
operands explicitly widened before an FP32 dot4. It regressed to
`0.5255788875168637x` and is not being retried.

OPT-0024 retains the shipped WG128 ownership, direct native O-K-I layout, zero
workgroup memory, zero barriers, and zero repack bytes. Its changed mechanism
is the native FP16 dot result followed by one widening and one FP32 add per R4
group. Do not add an OPT-0017 panel, packed layout, R32 tile, or alternate
shared-memory arm under this ID.

OPT-0014 is not repeated: OPT-0024 creates no packed/duplicated K-I-O layout,
and OPT-0023's measured `59,993.59999811649 ms` long K7 owner also supplies the
new decision value that OPT-0014 named as a revisit condition. OPT-0016 closed
nearby exact-order microtiles and required any changed-rounding follow-up to
use a new ID plus declared risk and tensor, trajectory, waveform, and listening
gates. This registration satisfies that seam with OPT-0024 and the explicit
`approximate FP16-partial reduction` class; it does not reopen OPT-0016's tile
sweep.

Exploration of MLX and ONNX kernels and experimental WGSL translations is
brainstorming context only. Neither framework supplies this WebGPU production
kernel, neither is an implementation source for OPT-0024, and no result from
that exploration is accepted as ACE package, target-browser, numerical, or
performance evidence. This registration makes no MLX/ONNX reuse or parity
claim.

## Exact long-scope topology

For the OPT-0023 `C448 + 10 x C512 + C340` sequence, the candidate scope is:

| Tier | Biased operations | Graph ranges | Physical workgroups | Logical MACs |
| --- | ---: | ---: | ---: | ---: |
| `conv1` | 12 | 24 | 2,960 | 5,420,613,632 |
| C1024 residual K7 | 36 | 5,541 | 44,328 | 1,300,947,271,680 |
| C512 residual K7 | 36 | 8,310 | 132,936 | 1,951,420,907,520 |
| C256 residual K7 | 36 | 8,310 | 265,860 | 1,951,420,907,520 |
| C128 residual K7 | 72 | 24,927 | 1,595,160 | 5,854,262,722,560 |
| **Candidate total** | **192** | **47,112** | **2,041,244** | **11,063,472,422,912** |

The unchanged final FP32 `conv2` contributes another 12 operation executions,
92 ranges, 354,480 physical workgroups, and `20,327,301,120` logical MACs.
The candidate must not absorb those rows into its performance score or change
their generated shader, binding, dispatch, arithmetic, or output bits.

## Static instruction, broadcast, and traffic accounting

Static contracts must independently enumerate every candidate operation and
range for C300, C512, C448, and C340, then derive the long aggregate rather
than matching hard-coded totals alone. For the exact long candidate scope:

| Candidate dynamic quantity | Exact count |
| --- | ---: |
| Physical workgroups | 2,041,244 |
| Workgroup x `(K, Cin/4)` instances | 675,299,072 |
| Validity-respecting input `vec4<f16>` loads | 21,603,155,968 |
| Native O-K-I weight `vec4<f16>` loads | 345,753,124,864 |
| Logical input bytes read | 172,825,247,744 |
| Logical weight bytes read | 2,766,024,998,912 |
| Logical operand bytes | 2,938,850,246,656 |
| Logical operand bytes/MAC | 0.2656354293 |
| Subgroup-level vector-broadcast collectives | 21,609,570,304 |
| Broadcast invocation calls | 691,506,249,728 |
| FP16 dot4 calls | 2,765,203,963,904 |
| FP32 `vec4` accumulator adds | 691,300,990,976 |
| Workgroup storage / barriers / repack bytes | 0 / 0 / 0 |

Against the same shipped output ownership, scalar input-load instructions,
scalar weight-load instructions, and scalar subgroup-broadcast collectives are
exactly four times the corresponding candidate vector counts. The shipped
FP32 `vec4` multiply and add each occur `2,765,203,963,904` times; the
candidate replaces that arithmetic with `2,765,203,963,904` FP16 dot4 calls
and `691,300,990,976` FP32 vector accumulator adds.

The logical operand-byte total is essentially unchanged. The hypothesis is
about aligned vector instructions, fewer broadcast instructions, a shorter
FP32 dependency chain, and native FP16 dot execution—not a fourfold bandwidth
reduction. These are static instruction counts, not compiler ISA counts, GPU
utilization, cache transactions, occupancy, throughput, or performance
claims. Report generated WGSL and, if diagnostic compiler evidence is
available, keep it separate from the decision's fenced wall time.

Persistent and transient GPU allocation deltas attributable to the candidate
must both be zero. Any new persistent weight layout, scratch/partial buffer,
workgroup panel, split-K storage, or full-window readback retained for timing
is outside OPT-0024.

## C300 primitive correctness gate

Before package allocation or a production-facing profile:

1. Pin the generated WGSL, kernel ID, fixed subgroup capability, WG128
   geometry, constant source lanes, `vec4<f16>` alignment/indexing, all
   ownership guards, `Cin/4` divisibility, output stores, bindings, pipeline
   cache separation, compile-failure eviction, destruction, and post-destroy
   rejection.
2. Reuse OPT-0017's deterministic periodic FP16 input and bias patterns and
   per-operation native O-K-I weight generator, but do not reuse its packed
   layout, repack, cooperative kernel, or timing result. The current shipped
   fixed32 kernel in the same page is the only primitive oracle.
3. Cover all 16 biased C300 K7 operations at exact first, interior, and tail
   ranges: 48 probes and `6,299,648` unique raw-U16 values. Execute the
   candidate twice and compare each execution with its same-run shipped
   control, for exactly `12,599,296` raw-U16 candidate/control comparisons.
   Candidate rerun hashes must also match. `conv1` has only two graph ranges,
   so freeze its established third probe explicitly: first graph probe
   `base=0, count=524,288`; synthetic-interior probe
   `base=290,816, count=32,768`, source `synthetic-interior`; and tail graph
   probe `base=524,288, count=90,112`. Every other first/interior/tail probe is
   sourced from its corresponding graph range.
4. Require every output finite and written over a qNaN prefill, unchanged
   leading/trailing guards and adjacent canaries, valid output ranges, and no
   uncaptured GPU error or device loss.
5. Per probe and in aggregate require NRMSE at most `0.001`, SNR at least
   `60 dB`, Pearson correlation at least `0.99999`, and maximum absolute error
   divided by the greater of control peak and `1e-6` at most `0.01`.
6. Report compared words, mismatch counts, maximum/mean/RMS error, NRMSE, SNR,
   Pearson correlation, relative maximum error, complete FP16 ULP
   distribution, signed-zero differences, ranges, first/worst locations, and
   deterministic hashes. Thresholds may not be loosened after output is seen.

This is approximate arithmetic. Passing the tensor envelope is necessary but
does not authorize package selection, waveform acceptance, listening, or a
production default.

## C300 representative performance qualification

After primitive correctness, compile and symmetrically warm only the shipped
and one frozen candidate geometry. Time these four unchanged representative
operations at their exact C300 first/interior/tail ranges:

| Tier | Operation | Exact C300 score weight | Long-scope context |
| --- | --- | ---: | ---: |
| C1024/d1 | `block-0-res-1-conv1` | 282 | 5,541 ranges |
| C512/d3 | `block-1-res-2-conv1` | 423 | 8,310 ranges |
| C256/d1 | `block-2-res-1-conv1` | 423 | 8,310 ranges |
| C128/d9 | `block-4-res-3-conv1` | 1,269 | 24,927 ranges |
| **Score total** | | **2,397** | **47,088 ranges** |

Use four submit-through-matching-drain samples per arm per tier. Freeze each
tier's eight-entry order as shipped/candidate, candidate/shipped,
candidate/shipped, shipped/candidate so each arm occupies each paired position
twice. Retain every sample; score the median of each arm/tier and weight the
four tier medians by the exact C300 weights
`282 / 423 / 423 / 1,269`, totaling `2,397`. The long-scope counts are
decision context only; edge-window rounding makes them non-identical weights,
so do not silently substitute them into the frozen C300 qualification score.

Qualify only if:

- the candidate median wins in all four tiers;
- the dominant C128 tier reaches at least `1.20x`; and
- the complete C300-weighted representative score reaches at least `1.25x`
  versus shipped fixed32.

The 24 long `conv1` ranges remain compile/correctness coverage, not timing
weight. If any performance condition fails, stop before authenticating the
production package, creating a candidate profile, running a complete decoder,
or changing any production selector. Do not tune chunk size, ownership, dot
precision, or tier-specific variants after seeing the result.

## Authenticated C512 subsystem gate

Only a qualifying primitive may receive a new explicit benchmark profile. Use
`createAceOpt0011LatentFixture(512)`: exactly 32,768 FP32 values / 131,072
little-endian bytes, SHA-256
`eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899`,
generator `xorshift32-13-17-5-high24-symmetric-f32-v1`, seed `0x00110512`.
This is a deterministic synthetic latent decoded by the authenticated current
VAE package through the production backend/profile. It is not package-native,
model-produced, or a real DiT latent, and no stale OPT-0011 output hash is an
oracle. The same-run current OPT-0015 profile is the oracle.

Run two complete paired control/candidate C512 decodes in balanced AB/BA order
after symmetric compile/warmup: shipped then candidate, followed by candidate
then shipped. Each execution retains the unchanged 88-operation graph, 7,854
graph quanta, one ingress, 7,855 dispatches, 982 decoder command buffers, one
readback command buffer, 983 submissions/drains, batch eight, and production
idle placement. The K7 family retains 500 pure batches / 3,999 pure quanta.
Report every family, mixed, complete decoder, readback, map, idle, and
outer-window wall; do not prorate mixed batches. Each complete raw output must
contain exactly `1,966,080` interleaved FP32 values / `7,864,320` bytes before
the waveform envelope is evaluated.

Require:

- complete candidate waveform correctness and deterministic candidate repeats
  under the frozen raw-waveform envelope below;
- candidate pure-K7 wall at least `1.25x` faster than its paired control and at
  least `1,000 ms` lower in **each** AB and BA order;
- candidate complete decoder submit-through-drain no slower than its paired
  control in either order; and
- candidate complete outer C512 decode/readback wall no slower than its paired
  control in either order.

The `1,000 ms` C512 condition makes ten repeated C512 windows a concrete
`10,000 ms` mechanism floor. It is still a bounded-fixture inference, not a
C4500 measurement or end-to-end claim. A failure stops before short-product,
listening, C4500, or production-default escalation.

## Activation-trajectory and waveform quality gates

Because the candidate changes reduction rounding, timing cannot select it.
After the C512 gate passes:

1. In a correctness-only diagnostic, compare shipped and candidate output at
   every one of the 16 changed K7 activation boundaries in a complete C512
   decode. Read back one bounded checkpoint at a time after a drain; never
   retain all activations or include diagnostic copies in performance timing.
   Require complete finite FP16 writes, deterministic candidate hashes, and
   no new saturation. Report errors, ULPs, ranges, and first/worst locations,
   but treat these internal taps as localization diagnostics only: there is no
   historically justified propagation envelope for them, and no tap threshold
   may be invented or relaxed after output is seen.
2. Run the pinned 12-second direct instrumental product request with the
   candidate profile as the only quality-affecting change. Retain raw U32
   identity at all eight DiT denoise-step taps and the complete final latent.
   This is an isolation/trajectory requirement: OPT-0024 may change only the
   subsequent VAE decode.
3. Feed the identical immutable final latent to current-control and candidate
   VAE decodes. Compare all `1,152,000` raw FP32 stereo samples before ordinary
   `-1 dBFS` normalization. Require exact length, complete finite nonzero
   writes, distinct channels, deterministic repeats, no clamp/saturation, and
   no coverage gap, duplicate, channel shift, or seam corruption. The complete
   C512 and 12-second raw waveforms—not internal activation taps—are the
   authoritative VAE numerical gates.
4. Require normalized RMS error at most `0.003`, SNR at least `50 dB`, Pearson
   correlation at least `0.9999` per channel and jointly, relative RMS/energy
   drift at most `0.005`, relative peak drift at most `0.01`, absolute
   DC-offset drift at most `0.001` times the greater of control RMS and
   `1e-6`, and relative maximum absolute error at most `0.02`.
5. Report complete waveform hashes and error neighborhoods, each window/core/
   discard range, seam metrics, final normalized WAV structure/frames/peak/
   channels/hash, memory high-water, all runtime errors, and cleanup. Raw-bit
   waveform identity is not required; the envelope cannot be changed after
   seeing candidate output.

At every applicable chunk boundary in the C512, direct-instrumental, and vocal
waveforms, apply this frozen seam test independently to both channels. For
`seamFrame`, define candidate `valueJump = abs(candidate[seamFrame, channel] -
candidate[seamFrame - 1, channel])`. Set
`startFrame = max(1, seamFrame - 1_920)` and
`endFrameInclusive = min(outputFrameCount - 1, seamFrame + 1_920)`. For every
integer frame in that inclusive interval, collect
`abs(control[frame, channel] - control[frame - 1, channel])`, sort ascending,
let `N` be the count, and select nearest-rank p99.9 at index
`max(0, ceil(N * 0.999) - 1)`. Require candidate `valueJump` no greater than
that same-location control threshold for each channel. Separately report the
diagnostic second-difference jump
`abs((candidate[seamFrame + 1] - candidate[seamFrame]) -
(candidate[seamFrame - 1] - candidate[seamFrame - 2]))`; it is not the
acceptance threshold. A subjective post-observation judgment or newly chosen
window is forbidden. The Stage 1 rule is bound at allocation to
`optimization/BASELINE.md` SHA-256
`f82d82d917455696503bed30457109ea07d61874027e434238a41ab45c3f778b`
and `LISTENING_CANDIDATE.md` SHA-256
`07886191bd50d73427c9273c9dd318b849bfad8f1fc0f3f5d99e45647cf177e8`.

The accepted packed-BF16/FP32 profile remains the ultimate numerical and
listening oracle under `PLAN.md`. The same-run OPT-0015 control is the direct
mixed-FP16 regression oracle for isolating this candidate. State clearly which
comparison supports each conclusion; do not substitute browser output for an
upstream golden.

## Fresh listening gate

Numerical acceptance does not establish musical acceptance. After all earlier
quality gates pass:

1. Generate fresh level-matched current-control and candidate outputs for the
   pinned 12-second direct instrumental request and obtain explicit owner
   listening approval.
2. Generate fresh level-matched current-control and candidate outputs for the
   pinned default-CoT planner vocal request, long enough to contain actual
   vocals—at least 30 seconds—and obtain explicit owner listening approval.
   A 12-second vocal crop does not qualify. Require exact U32 identity between
   the two arms at all eight DiT denoise-step taps and the complete final
   latent, then decode that same immutable final latent through both VAE arms.
   Planner or DiT trajectory drift invalidates the vocal pair rather than being
   attributed to OPT-0024.
3. Keep prompt, lyrics, seed, duration, sampler/DCW, planner behavior,
   metadata/defaults, normalization, and encoding fixed within each pair.
   Compare each complete raw waveform before normalization under the exact
   waveform envelope frozen above; the 30-second-or-longer default-CoT vocal
   pair is an authoritative numerical waveform gate as well as a listening
   gate. Then validate final WAV structure, channels, peak, seams, and hashes
   before presenting either pair.

Prior Stage 1 approval does not transfer to changed K7 math. The candidate
cannot become a shipped/default profile until both fresh owner decisions pass.

## Definitive C4500 VAE-only gate

Only after primitive, C512, trajectory, waveform, lifecycle, and both listening
gates pass, run complete current-control and candidate VAE-only
`C448 + 10 x C512 + C340` sequences in paired AB/BA order under the complete
OPT-0023 production protocol and deterministic C4500 latent SHA
`d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971`:
current then candidate, followed by candidate then current. Keep all graph,
batching, submission, drain, idle, mapping, stitching, OPFS, memory, progress,
and cleanup behavior unchanged. Do not run a full product or 180-second DiT
under this gate.

In **each** paired order the candidate must save at least `10,000 ms` versus
its fresh current control in homogeneous K7 submit-through-drain, complete
decoder submit-through-drain, and complete stream through raw-sink finish. In
addition, each candidate sequence must meet all three frozen absolute ceilings:

| Long metric | OPT-0023 current authority | Candidate maximum | Required reduction |
| --- | ---: | ---: | ---: |
| Homogeneous K7 submit-through-drain | 59,993.59999811649 ms | 49,993.59999811649 ms | 10,000 ms |
| Complete decoder submit-through-drain | 143,453.1000008583 ms | 133,453.1000008583 ms | 10,000 ms |
| Complete stream through raw-sink finish | 161,392.39999997616 ms | 151,392.39999997616 ms | 10,000 ms |

The ceilings are frozen against the authenticated OPT-0023 capture; the paired
savings come from fresh same-screen controls. Authenticate browser, package,
source, topology, thermal, and output identities and disclose every per-window
drift. If fresh current drift and the frozen ceilings disagree about the
decision, classify the result as inconclusive rather than selecting the more
favorable comparator or retrying. Passing both the paired and absolute gates
supports a measured current-machine C4500 saving for this profile. It does not
establish an end-to-end song speedup or an under-60-second result.

## Thermal and no-retry protocol

Every reportable performance screen uses an absolute-cadence external
`1,000 ms` logger reading
`notifyutil -g com.apple.system.thermalpressurelevel`. Start it before
compile/warmup and retain it through validation and cleanup. After symmetric
warmup require one fresh nominal gate with duration at least `30,000 ms`, at
least `floor(duration / 1,000) + 1` observations, maximum adjacent gap at most
`1,250 ms`, zero missing/non-nominal observations, and attributed launch no
more than `5,000 ms` after gate completion.

Persist the complete trace, command, schema, indices, timestamps, initial and
final levels, transitions, missing counts, maximum gap, byte length, and hash.
Once the first timed candidate/control dispatch begins, retain that run and do
not retry unchanged work for cooler pressure or a better number. A later
transition is disclosed and interpreted with balanced same-page ordering; a
marginal, mixed, or variance-overlapped result becomes inconclusive rather
than being selectively rerun. Preflight rejection before any timed dispatch is
setup evidence, not a sample, and must still be recorded.

No report may compare a cool historical number with a heated candidate as if
it were paired evidence, publish only the fastest sample, estimate GPU time
from wall time, or infer utilization from command-buffer attribution.

## Lifecycle, cancellation, and memory gates

The candidate must preserve:

- existing bind-group ownership, immutable range controls, FIFO graph order,
  batch eight, one-outstanding-command-buffer policy, drain-before-idle, and
  no `queue.writeBuffer` mutation of aliased storage while recorded work may
  read it;
- zero candidate persistent bytes, zero candidate scratch bytes, zero repack,
  and unchanged production steady/high-water memory outside bounded
  correctness-only readbacks;
- cache separation between shipped and candidate pipelines, eviction after a
  rejected compilation promise, exact created/destroyed buffer counts,
  complete unmap, drain before release, idempotent destruction, post-destroy
  rejection, and zero live bytes;
- cancellation after a fully drained decoder batch and between VAE windows,
  with no later encode, submit, readback, sink write, normalization, or output
  finalization; and
- worker/page progress and heartbeat through authentication, compilation,
  warmup, execution, mapping, validation, cancellation, and cleanup, reporting
  raw maximum gaps without inventing a responsiveness claim.

Any validation error, non-finite value, incomplete overwrite, guard/canary
change, uncaptured GPU error, device loss, count mismatch, post-cancel work,
resource leak, changed production selector, or changed noncandidate kernel
invalidates the affected gate.

## Evidence escalation and stop rule

Execute in this order and do not skip a failed seam:

1. static shader/topology/accounting/lifecycle contracts;
2. C300 48-probe primitive numerical gate;
3. one frozen C300 representative timing screen;
4. authenticated balanced C512 subsystem gate;
5. complete C512 activation trajectory and 12-second product waveform gates;
6. fresh instrumental and qualifying vocal listening; and
7. one definitive paired C4500 VAE-only gate.

Record a negative result and stop this geometry if any static safety contract,
primitive numerical bound, representative timing condition, C512 numerical or
performance condition, activation/denoise trajectory requirement, waveform
bound, lifecycle/cancellation condition, listening decision, or exact C4500
threshold fails. Record an inconclusive result rather than retrying when
variance or thermal evolution overlaps a decision boundary.

Do not add a chunk2/chunk8 arm, retile rows or output channels, use FP32 dot,
accumulate successive dot partials in FP16, add workgroup panels, pack weights,
split K, add cross-thread output reduction, bring final FP32 `conv2` into scope,
loosen a quality threshold, or retry the unchanged OPT-0017 tile. Any such
mechanism requires a new never-reused experiment ID.

A primitive or C512 pass can at most make the candidate
`positive / pending-integration`. Default production integration requires all
quality and listening decisions plus the C4500 gate. Even a complete OPT-0024
pass authorizes no 180-second product run by itself: re-profile the combined
stack, re-rank the remaining owners, and run the full song only when the
combined projection makes the under-60-second target credible.

## Registration nonclaims

At registration there is no candidate shader, package profile, browser result,
numerical comparison, listening decision, integration, measured speedup,
utilization result, 180-second generation, or under-60-second claim. Static
counts and threshold arithmetic are planning evidence only. The next available
experiment ID is `OPT-0025`.

## Primitive result — 2026-08-15

The sole timed C300 screen passed the frozen primitive correctness and timing
gate in stock Chrome 151 on the Apple M3 Metal adapter. It used `shader-f16`
and fixed 32-lane `subgroups`; no experimental feature, WebNN, native Metal
API, or production package was used.

All 48 first/interior/tail probes completed finite qNaN-overwriting writes with
unchanged guards and canaries. Both candidate executions were deterministic.
Across `12,599,296` same-run candidate/control FP16 comparisons, aggregate
NRMSE was `0.0001774657859663131`, SNR was `75.01770726325253 dB`, Pearson
correlation was `0.9999999842612771`, relative maximum absolute error was
`0.0005091649694501018`, maximum absolute error was `0.000030517578125`, and
signed-zero differences were zero. All frozen numerical thresholds passed.

The balanced four-sample medians were:

| Tier | Shipped median | Candidate median | Speedup |
| --- | ---: | ---: | ---: |
| C1024/d1 | `6.550000011920929 ms` | `3.649999976158142 ms` | `1.7945205629330503x` |
| C512/d3 | `3.949999988079071 ms` | `1.5999999642372131 ms` | `2.468750047730283x` |
| C256/d1 | `3.6499999165534973 ms` | `1.3500000834465027 ms` | `2.7037034747695543x` |
| C128/d9 | `2.3500000834465027 ms` | `1.899999976158142 ms` | `1.2368421647026937x` |

The exact C300-weighted score improved from `8,044.05006891489` to
`4,688.24998319149` weighted milliseconds, or `1.7157894945352221x`. Every
tier won, C128 cleared its `1.20x` floor, and the weighted result cleared the
`1.25x` floor. Applying the primitive ratio only as planning arithmetic to
OPT-0023's `59,993.59999811649 ms` long K7 family gives
`34,965.59466600985 ms`, a potential `25,028.005332106644 ms` reduction. This
is not a C512 or C4500 measurement.

The reportable gate followed one 31-second fresh nominal slice: 32 observations,
zero missing/non-nominal observations, maximum adjacent gap `1,004 ms`, and
`4,391 ms` gate-to-launch. The complete logger remained nominal through
cleanup. The actual screen issued 96 timed dispatches across 32 matching
submit/drain events and was not retried. Two earlier form submissions were
rejected before the first timed dispatch (one stale launch and one 29,999 ms
slice caused by observation jitter); neither produced a performance sample.

The result authorizes only the frozen C512 subsystem escalation. It does not
authorize a production package, waveform/listening work, C4500, integration,
or a full-product claim. The compact durable receipt is
`optimization/results/OPT-0024/result.json`.

## Complete C512 subsystem result — 2026-08-15

The one authorized stock-Chrome subsystem gate passed. The same authenticated
revision-6 package (`94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`,
`715,301` manifest bytes, `168,791,552` resident weight bytes) and OPT-0028
exact-packed graph served both arms. K1 and ConvTranspose were therefore the
same exact kernels in both arms. The only dispatch-owner difference was the
16 biased K7 operations: `4,082` C512 quanta used direct FP16 dot4 in the
candidate, while the final unbiased FP32-output K7 retained the shipped owner
for its eight quanta. The graph retained 88 operations, 7,855 sequence quanta,
983 submissions/drains, batch eight, one outstanding command buffer, and the
production one-millisecond queue-empty interval.

One complete shipped warmup followed by one complete candidate warmup passed
before timing. The external simple thermal gate then waited 30 seconds, read
`com.apple.system.thermalpressurelevel = 0` at
`2026-08-15T20:04:51Z`, and launched the single AB/BA action without a retry.

The four complete arm samples were:

| Order | Arm | Pure K7 submit/drain | Complete decoder submit/drain | Outer decode/readback wall |
| --- | --- | ---: | ---: | ---: |
| AB | shipped | `4,857.099998950958 ms` | `7,083.399999141693 ms` | `8,389.100000023842 ms` |
| AB | candidate | `2,221.6999992132187 ms` | `4,316.299998044968 ms` | `5,605 ms` |
| BA | candidate | `2,243.9999985694885 ms` | `4,338.799996495247 ms` | `5,622.5 ms` |
| BA | shipped | `4,754.300000309944 ms` | `6,915.700000286102 ms` | `8,212.799999952316 ms` |

The balanced two-sample medians were `4,805.699999630451 ms` shipped versus
`2,232.8499988913536 ms` candidate for pure K7, a `2.1522717612094677x`
speedup and `2,572.8500007390976 ms` saving. Complete decoder median improved
from `6,999.549999713898 ms` to `4,327.549997270107 ms`; outer C512 median
improved from `8,300.949999988079 ms` to `5,613.75 ms`. Both AB and BA cleared
the `1.25x`/`1,000 ms` K7 conditions and made both complete intervals faster.
Scaling only the directly measured K7 ratio to OPT-0023's long K7 bucket is
planning arithmetic of `27,874.54682971965 ms`, or a possible
`32,119.053168396844 ms` saving; it is not a C4500 measurement.

Both shipped runs were raw-U32 identical and both candidate runs were raw-U32
identical across all `1,966,080` output elements. Every output was finite,
nonzero, sentinel-free, unclamped, and stereo-distinct. The shipped and
candidate hashes were respectively
`893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47`
and
`11db972c1c8f4bdc42eea1f300bf5f78cccf7b33072dc9f0af3cef2a5538b93a`.
The joint candidate/control comparison produced NRMSE
`0.0018405635694897268`, SNR `54.70098356541297 dB`, Pearson
`0.9999983061725438`, relative RMS drift `0.000008171561893529713`, relative
energy drift `0.000016343057012592563`, relative peak drift
`0.00004844372210715888`, relative DC drift `0.0000013881631696440805`, and
relative maximum absolute error `0.008118983595985577`. Left and right channel
SNR were `54.88156839515197 dB` and `54.48852055506643 dB`; every frozen C512
waveform threshold passed.

No browser error or runtime event was observed and the timed button disabled
after the one action. The result authorizes only the next declared activation
trajectory and 12-second product waveform gates. It does not authorize a
default selector, listening decision, C4500 claim, or production integration.
Durable receipts are
`optimization/results/OPT-0024/c512-subsystem.json` (18,365 bytes,
SHA-256 `dd85a8d2cdb4c6a252b7b4616f47b267ae2b8bd396f74c860c891e0f4baf2a69`)
and `optimization/results/OPT-0024/c512-thermal.json` (521 bytes, SHA-256
`fce5777fc6d0c8b76190c0d57557f93a19c2491b3b330528f2c175b0745a0a9c`).
