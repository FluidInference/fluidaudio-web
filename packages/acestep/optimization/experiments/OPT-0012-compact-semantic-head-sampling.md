# OPT-0012 — Compact semantic head and candidate-domain sampling

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; the candidate changes which provably disallowed tied-head
  rows are computed and represented, but not the arithmetic or FP16 result of
  any retained row, the accepted browser-v1 sampling semantics, the emitted
  token sequence, or the Philox cursor relative to the unchanged authenticated
  raw-FP16 arm A at the same planner state

Allocation correction, 2026-08-14, before the first candidate browser run:
OPT-0010 proved that the authenticated raw-FP16 planner does not reproduce the
listening-approved packed-BF16 semantic trajectory. The packed-BF16 receipt and
semantic-code hash below remain historical listening evidence, not the numeric
oracle for this raw-FP16-only experiment. Arm A is the exact authority for B
and C; no evidence or result may claim otherwise.

## Hypothesis

During the two-row M2 semantic phase, every ordinary draw admits exactly the
64,000 audio-code token IDs `151669..215668`. Computing and reading the other
153,204 vocabulary rows cannot affect that draw. A state-specific tied head can
therefore score only the two exact source-shard intersections, read the results
back in a compact layout, and sample in ascending global-token-ID order while
preserving the accepted `ace-browser-softmax-v1` result bit for bit. The final
forced terminal draw is a different phase: it computes only EOS token `151645`
and still consumes the identical categorical word.

This experiment isolates the GPU/head/readback mechanism from the CPU sampling
mechanism. It makes no speed threshold or end-to-end target claim. In
particular, OPT-0010's projected 132–140 ms per semantic token was unmeasured;
OPT-0012 either measures the real contribution or leaves that projection
unconfirmed.

## Identity

- Allocation baseline: clean pushed `main` commit
  `f5e8e5db0b88a9a44dc96b73319183114daf136a`
- ACE-Step source revision:
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- Planner source revision:
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`
- Raw-FP16 package manifest SHA-256:
  `c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b`
- Historical listening-approved packed-BF16 default-CoT receipt SHA-256:
  `554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678`
- Historical listening-approved packed-BF16 30-second semantic-code SHA-256
  (not an OPT-0012 acceptance value):
  `42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be`
- OPT-0010 frozen result:
  `optimization/results/OPT-0010/result.json`
- Target machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB
  unified memory
- Frozen screening core commit:
  `740a16813a29ad89cdd18cc4a0cc49967458ac77`
- Frozen screening harness commit:
  `0795fc596fc10a27de8acaa1f20eee8202977634`
- Screening browser/OS/adapter identity, source hashes, and raw artifact hashes:
  retained in the frozen receipts registered below
- Final corrective core commit:
  `f73380bceebdd5568d93908a67ff33cea2b7d8f0`
- Final corrective harness commit:
  `eb6d8cc4c1d1db8f4fc75f0cd836ceed8a12daa8`
- Final corrective core source SHA-256:
  `53fb06aa1ec54c7dc4003731c7d360aac06110715e601451bee8a22256236fdf`
- Browser/OS/adapter identity and every authenticated source and shader hash:
  bound by the final raw receipts and
  [`../results/OPT-0012/result.json`](../results/OPT-0012/result.json)

The tied embedding is the LM-head weight and remains fully resident. The
traffic reductions below are per-draw logical head traffic, not package-size or
resident-memory reductions.

## Candidate boundary and fail-closed states

Implement only a benchmark-local M2 semantic head/readback path and compact
sampler. Keep the authenticated `raw-fp16` package, source-row-major tied
embedding shards, two CFG rows in conditional-then-unconditional order,
FP16 head outputs, existing tiled GEMM arithmetic, all transformer layers and
cache math, the FIFO graph owner, the one-outstanding-command-buffer policy,
and the real post-drain cooperative idle policy unchanged.

Do not combine this experiment with low-row GEMV, FP16 accumulation, weight
repacking, command-buffer batching, GPU-resident sampling, cache changes,
different constraints, or any package/converter change. A reduction in tied
head physical dispatches, quanta, copies, drains, and idles that follows
directly from removing head shards is part of the restricted-head mechanism;
it must be reported exactly and must not be labeled command batching.

The candidate is eligible only for the authenticated M2 semantic FSM in one of
these two explicit states:

1. `regular-code`: exactly 64,000 allowed global IDs `151669..215668`, EOS
   excluded, two CFG rows, temperature `0.85`, guidance scale `2`, top-k `0`,
   top-p `0.9`, repetition penalty `1`, and accepted
   `ace-browser-softmax-v1`; or
2. `forced-eos`: exactly global ID `151645`, two CFG rows, after the requested
   number of regular codes has been accepted.

Every other mode, row count, allowed-token shape, range, package/layout
identity, or FSM state rejects before candidate head encoding. M1 CoT must take
the unchanged shipped full-head/full-vector path; it may never select,
approximate, or silently fall back through this M2-only candidate.

## Registered three-arm protocol

| Arm | Head and readback | Sampling representation | Attribution |
| --- | --- | --- | --- |
| A — shipped full | Existing five-shard, 217,204-row head and full FP16 readback | Existing full 217,204-entry logical vectors and unchanged browser-v1 sampler | Performance and exactness authority |
| B — restricted/full-vector | Exact regular-code or forced-EOS rows only, with the compact readback layouts below | Reconstruct two 217,204-entry FP32 vectors, fill every omitted entry with exact `-inf`, place retained rows at their global IDs, and run the unchanged sampler | A versus B isolates restricted GPU head, transfer, host FP16 decode, and compact-readback/reconstruction cost while retaining the full-vector sampler |
| C — restricted/compact | Bit-identical head and readback path to B | Sample directly over the compact candidate domain while carrying exact global IDs and global tie order | B versus C isolates removal of full-vector reconstruction/scans and the compact CPU sampler |

B and C must use the same retained-FP16 implementation, buffer layout, head
dispatches, and readback commands. Their attributable CPU comparison replays
both samplers over the exact same mapped bytes and common GPU receipt; separate
equivalent B/C executions may be retained as end-to-end observations but do not
establish CPU attribution. A versus C is useful combined context but does not
attribute the two mechanisms separately.

## Exact row plan

The regular-code domain is the inclusive global range `151669..215668`, count
64,000. In the frozen five-shard tied embedding it has exactly two disjoint,
ascending intersections:

| Shard | Global first row | Local first row | Row count | Global last row |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 151,669 | 4,213 | 44,939 | 196,607 |
| 4 | 196,608 | 0 | 19,061 | 215,668 |

The forced-EOS plan is separate and contains exactly one row:

| Shard | Global first row | Local first row | Row count | Token |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 151,645 | 4,189 | 1 | EOS |

The regular plan never computes EOS. The forced-EOS plan never computes the
64,000 regular rows. Prefix rows, the gap between EOS and the first audio code,
and tail IDs `215669..217203` are disallowed in both compact representations.

## Frozen work and traffic accounting

The head shape uses two CFG rows, hidden width 1,024, source-row-major FP16
weights, and FP16 logits. Values in this section are static accounting, not
performance evidence.

### Regular-code draw

| Quantity | A: full head | B/C: restricted head | Exact saving |
| --- | ---: | ---: | ---: |
| Logical head weight traffic | 444,833,792 B | 131,072,000 B | 313,761,792 B |
| Logical multiply-adds | 444,833,792 | 131,072,000 | 313,761,792 |
| Scheduled multiply-adds | 3,558,866,944 | 1,050,673,152 | 2,508,193,792 |
| Raw FP16 logit payload | 868,816 B | 256,000 B | 612,816 B |
| Write-status payload | 8 B | 8 B | 0 B |
| Mapped readback allocation | 869,120 B | 256,256 B | 612,864 B |
| Copy commands, including status | 6 | 3 | 3 |

The two restricted slices contribute exactly:

| Slice | Weight bytes / logical MA | Scheduled MA | Raw two-row logits |
| --- | ---: | ---: | ---: |
| Shard 3, local `4213`, count `44939` | 92,035,072 | 738,197,504 | 179,756 B |
| Shard 4, local `0`, count `19061` | 39,036,928 | 312,475,648 | 76,244 B |

The benchmark must derive scheduled work by calling the production tiled-GEMM
planner for shapes `[2,1024,44939]` and `[2,1024,19061]` and summing its output
ranges. The frozen plans contain 352 and 149 workgroups respectively. The
benchmark must assert the values above and retain every derived range and
workgroup count. A copied constant or an assumed fixed padding multiplier is
not sufficient evidence.

The regular compact readback is frozen as two whole, shard-major, row-major
blocks. Shard 3 copies 179,756 bytes to destination offset `0`; shard 4 copies
76,244 bytes to destination offset `179756`. Both offsets and lengths satisfy
WebGPU's four-byte `copyBufferToBuffer` alignment. The raw-logit region therefore
ends at `256000`; the two-row write status occupies offset `256000`, length
8; and the allocation is aligned to 256 bytes at exactly `256256`. There is no
inter-shard padding. Logical candidate order is nevertheless global-ID-major
within each CFG row: the CPU view maps shard 3 followed by shard 4 for the
conditional row, then the same two spans for the unconditional row.

### Forced-EOS draw

| Quantity | A: full head | B/C: EOS-only head | Exact saving |
| --- | ---: | ---: | ---: |
| Logical head weight traffic | 444,833,792 B | 2,048 B | 444,831,744 B |
| Logical multiply-adds | 444,833,792 | 2,048 | 444,831,744 |
| Scheduled multiply-adds | 3,558,866,944 | 2,097,152 | 3,556,769,792 |
| Raw FP16 logit payload | 868,816 B | 4 B | 868,812 B |
| Write-status payload | 8 B | 8 B | 0 B |
| Mapped readback allocation | 869,120 B | 256 B | 868,864 B |
| Copy commands, including status | 6 | 2 | 4 |

The EOS-only scheduled work must likewise be derived from the production
tiled-GEMM planner for `[2,1024,1]`; the frozen plan contains one workgroup. Its
one weight row produces two FP16 logits. The raw four bytes copy to offset `0`,
the two-row write status occupies offset `4`, length 8, and the whole mapped
allocation is exactly 256-byte aligned at `256` bytes. Raw logit bytes, status
bytes, used bytes, and alignment padding remain distinct in every trace and
comparison.

## Static and kernel correctness gate

Before any timing:

- authenticate the package, every source file that defines the planner/head,
  the five embedding shard ranges, the source-row-major layout, and every
  candidate shader and host-sampling source;
- independently enumerate the regular and EOS intersections from global
  ranges, prove exact ascending coverage with no gap or overlap, and reject an
  off-by-one end, wrong shard, wrong local row, or unexpected vocabulary size;
- prove each weight binding byte offset and length from shard metadata, not
  from an unauthenticated literal, and prove all source and destination ranges
  stay within their owned buffers;
- derive every tiled-GEMM output range, tail lane, workgroup count, scheduled
  multiply-add, tied-head quantum, command buffer, drain, idle, copy, status
  offset, map span, and allocation byte from production planners;
- prefill output and guard regions with discriminating sentinels, require every
  expected FP16 logit and status word to be written, and require every prefix,
  inter-buffer guard, and tail sentinel to remain unchanged; and
- run first/last rows of each intersection, the shard-3/shard-4 boundary, EOS,
  odd FP16 row tails, both CFG rows, forced termination, and explicit invalid
  prefix/gap/tail/range/state cases.

For every admitted global row and both CFG rows, B/C's retained FP16 logit bits
must equal A exactly. Report the complete compared count, hashes, zero mismatch
count, and first/worst mismatch fields even when null. Approximate float
agreement, equal sampled tokens alone, or candidate-versus-candidate agreement
cannot pass this gate.

## Full-vector and compact-sampler equivalence

For a regular draw, arm B constructs two FP32 vectors of length 217,204. Every
admitted entry must have the exact U32 value obtained by decoding A's retained
FP16 logit; every other entry, including EOS, must be the exact negative-
infinity bit pattern `0xff800000`. For the terminal draw, only global ID
`151645` is finite and every other entry is that same exact `-inf`. Compare the
complete B vectors against independently masked A vectors as U32, including
all omitted prefix, gap, and tail entries.

Arm C orders candidates by ascending global ID and maps local result `i` to
global token `151669 + i` for regular draws or to `151645` for the terminal
draw. Stable sorting and ties remain descending-logit then ascending **global**
token ID. CFG, final FSM masking, repetition penalty, top-k, accepted browser-v1
top-p, temperature, deterministic software exp, FP32 sum/storage order, and
categorical traversal must visit the same finite values in the same order and
round at the same boundaries as B. Omitted `-inf` entries contribute no
arithmetic and cannot change the finite traversal order.

Compare B and C at every observable sampling boundary: CFG logits, final mask,
top-k threshold and keep set, top-p global-ID keep set, temperature-scaled U32
values, softmax weight U32 values, positive-candidate count, categorical word,
selected global token, draw index, and draw end. Include adversarial equal-logit
ties across the shard boundary, top-k and top-p cutoff ties, all-zero/subnormal
weight tails, dominant first/last candidates, one positive candidate, and the
forced-EOS case. C must still request and consume the same Philox word for a
one-candidate draw.

## Package-native token and trajectory gate

Use the exact OPT-0010 M2 short, middle, and long cache cases and their frozen
decode token, cache length/capacity, sampler word, and cursor state. Across A,
B, and C require:

- complete retained FP16-logit identity for both rows;
- B's complete reconstructed-vector U32 identity;
- exact token ID, word, positive-candidate count, draw index, and draw end;
- exact cache append, valid-length controls, progress, and deterministic repeat;
  and
- complete dispatch/quantum/copy/map/status/resource reconciliation.

The exact OPT-0010 raw-FP16 arm-A sample authorities are frozen as follows:

- short: token `192370`, word `2004582350`, positive-candidate count `16`,
  draw `125..126`;
- middle: token `156326`, word `503673048`, positive-candidate count `1`,
  draw `185..186`; and
- long: token `155832`, word `3288166745`, positive-candidate count `1`,
  draw `258..259`.

Then execute a complete raw-FP16 30-second semantic trajectory independently
through A, B, and C: 150 regular audio-code draws followed by one forced-EOS
draw at the true post-code-150 state. M1 CoT is unchanged and supplies the same
resolved initial inputs and continuous cursor state to all three M2 arms.
Require exact A/B/C equality at every draw for the 150 code IDs, sampler words,
positive-candidate counts, cursor receipts, terminal EOS, final draw, and
serialized audio-code text. Record and freeze the newly observed raw-FP16
semantic-code SHA-256 in the result. Execute two independent complete
trajectories per arm; require every arm to repeat exactly and all six traces to
match draw by draw, then fail at the first divergence. Do not compare that hash
or token sequence to the historical packed-BF16 value. A shorter 12-second
deterministic trajectory may be retained as an additional diagnostic but cannot
replace this raw-FP16 30-second gate.

This exact experiment does not require a new listening judgment. It is
benchmark-only and cannot itself promote production code.

## Thermal benchmark protocol

1. Compile, authenticate, allocate, upload, and build correctness references
   outside timing. Perform symmetric untimed warmups for A, B, and C, then pass
   the continuous nominal thermal gate from `PLAN.md`; external logging must
   span every timed sample and cleanup.
2. At each frozen short, middle, and long M2 cache point, run all six arm
   permutations (`ABC`, `ACB`, `BAC`, `BCA`, `CAB`, `CBA`). Each arm therefore
   has six completion-fenced samples and appears twice in every within-triplet
   position. Every observation uses a fresh equivalent prefill/cache and the
   same input token, sampler word, and cursor state.
3. Retain total token wall time and non-overlapping named intervals for layers,
   tied-head encode/submit-through-drain/idle, readback copy/drain/idle/map,
   FP16 host decode, full-vector construction, CFG/constraints, top-k, top-p,
   temperature/softmax, categorical selection/global mapping, and callback.
   Do not subtract overlapping clocks or publish only the fastest sample.
4. A versus B is the restricted-head/readback comparison. B versus C is the
   CPU representation/sampler comparison and must additionally replay both
   samplers over the exact same retained compact bytes and word outside GPU
   timing. Report A versus C only as combined context.
5. Retain the complete raw-FP16 30-second trajectory timings as benchmark-scale
   raw-FP16 context, not a listening-approved product receipt.
   Correctness requires the two registered complete executions per arm. If
   those executions are used for a reportable performance comparison, balance
   additional full trajectories before interpreting drift; do not present one
   fixed-order A/B/C pass as thermal comparative evidence.
6. Report every raw sample, median/range, same-round wins, thermal observation
   and maximum poll gap, maximum single drain, worker/page heartbeat gaps,
   memory/resource high-water, and all exact work/topology/readback counts.

There is no numerical speed threshold. Evidence may be positive, negative, or
inconclusive based on exactness, balanced attributable measurements, the
disclosed thermal trace, and observed stability. A fast result cannot excuse a
correctness, responsiveness, or lifecycle failure.

On 2026-08-14 the owner revised the thermal decision rule after the corrective
run passed its 30-second nominal pre-gate but became non-nominal during the
sample. For this experiment and subsequent balanced screens, the external trace
remains mandatory disclosure, but an in-run transition is a caveat rather than
an automatic rejection when the within-page effect is large and consistent.
Do not repeat an unchanged run solely to obtain continuous nominality. This
does not relax correctness, cancellation, lifecycle, quality, or listening
gates, and marginal or mixed timing remains inconclusive.

## Cancellation, lifecycle, and responsiveness gate

- Probe cancellation after a fully drained pre-head quantum, after the
  restricted head has drained and completed its real idle but before readback,
  and during a multi-token semantic trajectory. Require no later encode,
  submit, copy, map, reconstruction, sampling, callback, cursor advance, cache
  append, or finalization beyond the last completed boundary.
- Drain before releasing any mapped or GPU resource. Reconcile every created,
  mapped, unmapped, and destroyed buffer; require zero live owned resources,
  idempotent cleanup, post-destroy rejection, and no runtime/device-loss event.
- Preserve the dedicated worker, one FIFO graph owner, at most one outstanding
  command buffer, real queue-empty intervals, bounded cache/resource ownership,
  and explicit progress. Never mutate aliased graph storage while recorded work
  may read it.
- Retain worker and page heartbeat gaps. The synchronous CPU sampling gap
  observed by OPT-0010 is a motivation for C, not permission to omit or relabel
  responsiveness evidence.

## Main risks

- A globally contiguous candidate range crosses a physical shard boundary and
  each shard stores both CFG rows. A shard-major mapped buffer can be mistaken
  for a row-major 64,000-candidate vector unless the four logical spans are
  mapped explicitly.
- Changing GEMM column count changes tile tails and command topology. Retained
  rows must still use the identical inner reduction and source weights, and
  scheduled work must come from the production planner.
- Compact sampling can accidentally change stable tie order or FP32 reduction
  order even when selected tokens usually agree. Internal U32 comparisons and
  adversarial boundary ties are mandatory.
- EOS is excluded on regular draws and forced only after all requested codes.
  Combining it with the regular head changes both accounting and the admitted
  state contract.
- The complete tied embedding remains resident for input embedding. Logical
  weight-traffic savings must not be reported as package or memory savings.
- CPU instrumentation can perturb short sampling intervals. Completion-fenced
  total wall, balanced orders, same-byte B/C replay, and retained observer
  overhead are all required.

## Evidence and disposition

- Evidence conclusion: `positive`
- Code disposition: `benchmark-only`
- Production integration: not performed; this allocation remains
  benchmark-only
- Final result JSON:
  [`../results/OPT-0012/result.json`](../results/OPT-0012/result.json)
- Integration priority: parked until the direct-generation target's complete
  FP16 VAE and proven DiT integration work permits planner work to resume

## Corrective-pass registration, 2026-08-14, before implementation

The first complete screening pass is frozen rather than replaced:

- balanced timing receipt
  `optimization/artifacts/OPT-0012/raw/compact-semantic-head-timing.json`,
  SHA-256
  `b261b892021328debfc9da1d26687844ba7f859944c7126e682ec7ffa0e0dddf`;
- continuous thermal receipt
  `optimization/artifacts/OPT-0012/raw/compact-semantic-head-timing-thermal.jsonl`,
  SHA-256
  `982951ebe45a4320307f73a954cdfcdf12a2421d8375f5226d1d8be3f9b539f8`;
  and
- six-run trajectory receipt
  `optimization/artifacts/OPT-0012/raw/compact-semantic-head-trajectory.json`,
  SHA-256
  `6468fb1d71e1d3f8e7309aa2217fc30dc2264b12f3f21a8b4c59d31bbc9486a8`.

The external receipt covers the timed interval and cleanup with 4,079 nominal
observations, zero non-nominal observations, and a maximum poll gap of
`1089.688042004127` ms. The joined screening comparison is therefore thermally
valid. The exact total-wall medians and same-round wins are below; win cells
are `left/right/tie` in the named comparison.

| Cache case | A median (ms) | B median (ms) | C median (ms) | A/B wins | B/C wins | A/C wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 456.14999997615814 | 586.6500000059605 | 477.4000000357628 | 6/0/0 | 0/6/0 | 5/1/0 |
| middle | 462 | 598.3500000238419 | 504.25 | 6/0/0 | 0/6/0 | 5/1/0 |
| long | 499.2000000178814 | 590.2000000178814 | 523.3000000119209 | 6/0/0 | 1/5/0 | 5/1/0 |

Correctness screening is **GO**: the retained FP16 bits, reconstructed full
vectors, sample/cursor boundaries, adversarial sampler cases, cleanup, and all
six complete A/B/C trajectories are exact. Performance screening is
nevertheless negative for the combined A-to-C candidate: C has the slower
median and loses five of six same-round comparisons to A at each cache point.
There is no speed threshold and this observation does not change the pending,
benchmark-only disposition.

Attribution is **NO-GO**. The post-timing same-byte receipts prove that the full
and compact samplers produce the same result from each immutable mapped-byte
payload, but they contain no separate full-sampler and compact-sampler timing
intervals. Separate B and C GPU executions cannot establish the registered CPU
attribution. Source inspection also identifies the corrective target:
`aceOpt0012Float16BitsToNumber` allocates a one-element `Uint32Array` and a
`Float32Array` view for every decoded scalar. A regular compact readback invokes
that allocation pair 128,000 times, and the timing receipt records this host
decode cost inside the candidate observations.

The following corrective pass is frozen before code changes. It remains inside
OPT-0012's existing allocation and three-arm boundary:

1. Replace only the per-scalar typed-array conversion with an allocation-free,
   exact FP16-to-FP32 expansion. Before using it in an arm, compare its expanded
   Float32 U32 result in the target browser with the current scalar
   implementation for every one of the 65,536 input bit patterns
   `0x0000..0xffff`. Benchmark current and allocation-free implementations over
   that complete domain in balanced `AB`/`BA` order and retain every raw
   interval and exact-output receipt.
2. Separately time the full-vector and compact-sampler CPU replays over the
   exact same immutable mapped bytes and sampler word. Use balanced
   full-then-compact/compact-then-full (`AB`/`BA`) order; retain the byte SHA-256
   before and after every pair, raw total intervals, FP16 decode,
   reconstruction, constraints, top-k, top-p, temperature/softmax, categorical
   mapping, selected token, cursor, and every existing U32 equivalence
   boundary. GPU work is common and excluded from this attribution comparison.
3. Repeat the original short/middle/long six-permutation three-arm timing under
   continuous thermal coverage through cleanup. Preserve the receipts above
   byte for byte; write the corrective browser and thermal receipts to new
   artifact paths, retain every raw sample, and apply the original correctness,
   cancellation, responsiveness, topology, and lifecycle gates.

Provided the exhaustive target-browser 65,536-pattern U32 comparison passes
and the corrective source delta is authenticated as conversion/timing-only,
the frozen six-run trajectory receipt remains admissible joined correctness
evidence and is not rerun; it retains its original core/harness identities and
the final result must name this equivalence join. Failure of either condition
requires all six trajectories under the corrective identities.

This corrective pass has no numerical speed threshold, grants no production
integration, and allocates no new experiment ID. At registration time,
OPT-0012 remained `pending` and `benchmark-only`; the next available ID
remained `OPT-0013`.

### Target-browser NaN-oracle amendment (2026-08-14)

The first target-browser corrective launch stopped before package acquisition:
the frozen legacy scalar conversion produced exhaustive U32 SHA-256
`d033d83af36297bbdbc929e17b9987e65682e66a4cd9748116d0fc1c9c053f44`,
while the initially canonicalized allocation-free arm produced
`b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf`.
All 511 differences were negative signaling-NaN quiet bits. A narrowly
corrected core then matched the first pass, but the next authenticated launch
proved that the same legacy function changes with target V8 tiering: the fixed
`legacy -> candidate` pass produced `d033...`, while the immediately following
reverse-order legacy pass produced `b636...`. The allocation-free output was
deterministic. The second failure receipt is retained at
`optimization/artifacts/OPT-0012/raw/corrective-failed-unstable-nan-oracle.json`
with SHA-256
`24ef921ae175859e3e2ef2447ee701b1ab768c9f7e3e6412b137bb595ad3debc`.

This demonstrates that legacy raw signaling-NaN U32 words are not a stable
oracle. It does not relax finite-value correctness or any actual model-output
gate. Before the next code change, the corrective acceptance rule is amended
as follows:

1. For all 63,490 non-NaN binary16 patterns, both implementations must equal
   the independently expanded binary32 U32 word exactly, including both signed
   zeros, every subnormal and normal, and both infinities. The ordered compact
   non-NaN U32 SHA-256 is
   `680bbc22915f61aa1bbfc7265bc3882a6aa42d299bfd2c571807196e5544de2e`;
   any mismatch is NO-GO.
2. For all 2,046 binary16 NaNs, a legacy output may be only the exact expanded
   NaN word or that word with binary32 quiet bit `0x00400000` set. Sign,
   exponent, and every payload bit other than the quiet bit must remain exact,
   and the result must classify as NaN. Quiet binary16 NaNs therefore still
   have one exact accepted word. Any sign loss, payload change, non-NaN result,
   or other-bit difference is NO-GO.
3. The allocation-free candidate uses one deterministic representation: every
   NaN is quieted while preserving sign and payload. Its complete-domain raw
   U32 SHA-256 must be `b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf`
   on every invocation, with zero signaling and 2,046 quiet outputs. Setting
   only the quiet bit on every accepted legacy NaN must produce the same full
   hash; the canonical NaN-only SHA-256 is
   `32f37d24c421f50695da47516d20cffa27c96fb2be53d1ac6f88cfbc1cec9039`.
4. Before package acquisition, run exactly one `legacy -> candidate` pass and
   one `candidate -> legacy` pass. Retain order, raw and canonical hashes,
   signaling/quiet counts by input class and sign, first raw difference, sign
   mismatches, payload-excluding-quiet mismatches, and disallowed-word
   mismatches. Do not repeat adaptively until a preferred legacy hash appears.
   Candidate outputs must be word-for-word identical across the two calls;
   legacy raw hashes are diagnostics, not pass criteria.
5. The late six-pair converter screen retains its registered balanced order and
   raw intervals. Validate outputs after both arms in each pair with the same
   envelope, retain every legacy raw hash and class count, and describe the
   result only as a fixed-schedule target-browser allocation/JIT diagnostic.
   It is not evidence of portable legacy NaN payloads or a stable JIT tier.
6. Every authenticated regular and forced-EOS package readback used by A/B/C,
   replay, or trajectory evidence must contain zero binary16 NaNs. Any actual
   model NaN is NO-GO and cannot use the synthetic NaN envelope.

The earlier conditional trajectory join is not available: the exact exhaustive
raw-U32 clause failed, and the frozen trajectory artifact did not retain a
complete binary16 NaN census. All six complete trajectories must therefore be
rerun under the final corrective core and harness identities before OPT-0012
can close. The primary six-order A/B/C timing and same-immutable-byte replay
remain the performance authorities and are otherwise unchanged.

## Final corrective evidence and closeout, 2026-08-14

The final target-browser evidence passed an independent receipt audit. The
authenticated identities are corrective core
`f73380bceebdd5568d93908a67ff33cea2b7d8f0`, harness
`eb6d8cc4c1d1db8f4fc75f0cd836ceed8a12daa8`, allocation
`f5e8e5db0b88a9a44dc96b73319183114daf136a`, core-source SHA-256
`53fb06aa1ec54c7dc4003731c7d360aac06110715e601451bee8a22256236fdf`,
and raw-FP16 package-manifest SHA-256
`c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b`.
The run used Chrome `151.0.7922.138` on macOS `26.5.2` build `25F84`,
MacBook Air `Mac15,12`, Apple M3 with 10 GPU cores and 16 GiB unified memory.
The raw receipts retain the exact adapter features/limits, authenticated source
map, and three candidate shader hashes.

### Exactness, trajectory, cancellation, and lifecycle

Each corrective timing receipt independently compared 384,002 retained FP16
words for each of B and C against A and 1,737,632 reconstructed full-vector U32
words for each candidate arm, with zero mismatches. Each receipt retained 54
actual-package NaN censuses spanning 12,427,344 binary16 words with zero NaNs.
The exhaustive converter gate covered all 65,536 binary16 patterns, including
63,490 non-NaNs and the registered 2,046-NaN envelope. The deterministic
allocation-free result had SHA-256
`b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf`
on both pre-package calls with zero mismatches.

The required final trajectory receipt is
`optimization/artifacts/OPT-0012/raw/compact-semantic-head-corrective-trajectory.json`,
SHA-256
`03246b9f8c8654cf5f22159b5b8bbd45689e1d9cfc3a2bb6dec6c422af492c4c`.
Two independent executions of each of A, B, and C produced the same 150 codes
plus forced EOS, every per-draw token/word/count/cursor receipt matched from
draw `109` through final draw end `260`, and every arm self-repeated exactly.
The newly frozen raw-FP16 semantic-code SHA-256 is
`08c69f3d598bea591754948b831684e67e879fca730b11ba796e7547a0f798fb`;
the serialized audio-code text SHA-256 is
`f9aec6e269424585028aae47ad5582fbda5139ef9961642774f0ce0b4c58f62e`.
Its 906 actual-package censuses inspected 207,991,224 binary16 words with zero
NaNs.

Trajectory cancellation rejected with `AbortError`, left the sampling cursor
at `118`, performed no sample call in the cancelled invocation, published no
later callback or cache state, and performed no later submit, drain, map,
allocation, cursor advance, callback, or finalization. Trajectory cleanup
destroyed all 4,954 tracked buffers, balanced all 2,869 maps/unmaps, left zero
live resources, rejected post-destroy prefill/decode, and observed no runtime
error. Each timing run likewise destroyed all 7,997 tracked buffers, balanced
all 362 maps/unmaps, left zero live resources, made executor destruction
idempotent, and observed no runtime error.

### Decision-useful balanced timing

Both corrective runs passed a 45,005 ms, 46-observation nominal pre-gate. The
owner then codified the practical thermal decision rule in commit `8f0d13b`:
retain and disclose the through-cleanup trace, but accept a large, consistent,
balanced within-page effect despite a later transition rather than repeat an
unchanged run solely for continuous nominality.

The first timing and thermal receipts are, respectively,
`corrective-failed-nonnominal-timing.json` at SHA-256
`8a3bb770ad5f05f6b3f4cde9a2fdb4a3bf2373930d3d97a93a05ee15e6d119ad`
and `corrective-failed-nonnominal-timing-thermal.jsonl` at SHA-256
`09e2ed9492cce7279e5b415cace2fe64ee93c629714daed77aed2c817860d492`.
The external trace's maximum monotonic poll gap was
`1050.9930829284713` ms. It first transitioned from level 0 to level 1
15,625 ms after timed start and later reached level 2; the joined interval
through cleanup contains 309 observations, 237 non-nominal.

The unchanged rerun receipts are
`corrective-failed-nonnominal-rerun-timing.json` at SHA-256
`089a3bae9f0a09cdc88bdcc75f1d073e6b54d3640fc32971826b432084d09415`
and `corrective-failed-nonnominal-rerun-timing-thermal.jsonl` at SHA-256
`f83abcda338261aad2822b6cc3401b2dff70162c444cebb5d7f876a709de6004`.
Its maximum monotonic poll gap was `1032.6179170515388` ms. Its first
non-nominal observation arrived 32,005 ms after timed start; the joined
interval through cleanup contains 273 observations, 189 non-nominal.

The filenames remain historical: `failed-nonnominal` describes the earlier,
stricter continuous-nominality classification and does not mean the receipts
failed the later owner decision rule. Neither run is represented as
continuously all-nominal or release-quality thermal evidence.

| Run | Cache | A median (ms) | B median (ms) | C median (ms) | C/A wins | C/B wins |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | short | 294.94999998807907 | 278.44999998807907 | 269.80000001192093 | 6/6 | 4/6 |
| 1 | middle | 300.94999998807907 | 302.60000002384186 | 286.10000002384186 | 5/6 | 4/6 |
| 1 | long | 551.9499999880791 | 520.5999999642372 | 420.10000002384186 | 6/6 | 4/6 |
| 2 | short | 287.69999998807907 | 262.94999998807907 | 238.50000005960464 | 6/6 | 6/6 |
| 2 | middle | 357.94999998807907 | 347.39999997615814 | 320.25 | 4/6 | 4/6 |
| 2 | long | 384.80000001192093 | 374.3499999642372 | 334.39999997615814 | 5/6 | 5/6 |

C beat A in all six medians and 32/36 same-round comparisons. C beat B in all
six medians and 27/36 same-round comparisons. A-to-B alone was marginal and
noisy: B won five of six medians and 24/36 pairs, but the middle median in run
1 regressed and several individual deltas were small or reversed. The positive
conclusion therefore belongs to the complete restricted-head plus compact-
sampling arm C; it does not independently promote B's restricted-head effect.

The registered same-immutable-byte attribution is stronger and isolates CPU
representation/sampling from GPU work:

| Run | Cache | B median (ms) | C median (ms) | C saving |
| ---: | --- | ---: | ---: | ---: |
| 1 | short | 80.89999997615814 | 44.55000001192093 | 44.9320% |
| 1 | middle | 99.29999995231628 | 57.80000001192093 | 41.7925% |
| 1 | long | 95.54999995231628 | 53.15000003576279 | 44.3747% |
| 2 | short | 73.3500000834465 | 41.8999999165535 | 42.8766% |
| 2 | middle | 61.69999998807907 | 33.89999997615814 | 45.0567% |
| 2 | long | 53.19999998807907 | 30.099999964237213 | 43.4211% |

C won all 36/36 immutable-byte pairs; every before/after byte hash was equal,
and the six median savings span 41.8–45.1%. The allocation-free converter won
all 12/12 late balanced pairs. Its two median comparisons were
27.30000001192093 → 0.30000007152557373 ms and
21.900000035762787 → 0.19999998807907104 ms, each more than 98.9% lower.

### Final disposition

OPT-0012 closes **positive / benchmark-only**. The exact restricted M2 head
plus compact candidate-domain sampler is a credible integration candidate, but
no production path was changed, no product listening or end-to-end speed claim
is made, and the mechanism's contribution to the three-minute direct target is
unmeasured. Per the integration-first priority, park this planner integration
until the complete FP16 VAE and proven DiT paths are integrated and the direct
target permits planner work to resume. The historical screening, rejected
pre-gate, and NaN-instability artifacts remain registered unchanged in the
final result rather than being overwritten or renamed.
