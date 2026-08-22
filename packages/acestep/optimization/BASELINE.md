# Accepted Stage 1 quality baseline

## Authorization and checkpoint

- Owner approval date: 2026-08-13
- Decision: the SHA-bound instrumental and vocal browser outputs sound correct;
  Control E sounds great, with great vocals beginning at approximately 20 s.
- Authorization: Stage 2 measured optimization explicitly authorized.
- Approved checkpoint commit: `fbf831d94dff0220e91341c2781c4d4aea09f1c8`
- Annotated tag: `stage1-approved-2026-08-13`
- Runtime commit that generated Control E:
  `99f3e90c442c311cd9bfb02cab2434484ca2312c`
- Evidence/receipt commit:
  `ab22054da650a9eb441cbd52642f54adc011000e`
- Approval record: `LISTENING_CANDIDATE.md`
- Automatic Control E receipt:
  `LISTENING_CANDIDATE_DEFAULT_COT_RECEIPT.json`

This freezes the product quality/listening authority. It does not claim native
upstream tensor or waveform identity. Authenticated CUDA/XPU native captures
remain unavailable external parity evidence and are required before any formal
upstream-equivalence claim.

## Model and fixture identity

- ACE source: `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- ACE main snapshot: `19671f406d603126926c1b7e2adc169acbcade22`
- Planner snapshot: `148d8ea0225bdab342ee1ae3a354275ccd60ca80`
- Parakeet implementation reference:
  `7ee112738262a6f5a0efd2f150748a4087432fbb`
- Reference model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- `golden/MANIFEST.json` file SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Execution profile: `reference-bf16-subgroups`
- Weight storage: packed BF16 decoded to FP32 arithmetic
- Sensitive reductions and VAE arithmetic: FP32
- Scheduling profile: cooperative, one command buffer outstanding, drain, then
  a real queue-empty interval before every non-final command.

## Listening-approved artifacts

| Artifact | Role | SHA-256 | Owner result |
| --- | --- | --- | --- |
| `ace-step-browser-reference-direct-instrumental-12s-c0ffee.wav` | direct instrumental | `d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477` | accepted |
| `ace-step-browser-reference-planner-vocal-control-30s-badc0de.wav` | fixed-metadata planner vocal | `da57f30a8a10f744e4f3848cfdf71e385a3878792948093d857b5f9be4760f00` | amazing vocals |
| `ace-step-browser-reference-planner-vocal-default-cot-30s-badc0de.wav` | default-CoT product vocal | `b702e10a2bae1b4a99bb10e54c92b663635a872b33d3aea58e3ba7fd4d739c14` | sounds great; vocals at ~20 s |

The 12 s vocal samples are not accepted vocal gates. Control E demonstrates
that its generated instrumental-intro trajectory enters vocals after their crop.

## Mechanical and numerical quality envelope

- Output: stereo, 48,000 Hz, IEEE float WAV, 32-bit samples.
- All accepted WAV samples finite and nonzero; left and right channels differ.
- Peak: `0.8912509083747864`, approximately -1 dBFS.
- All applicable 5.12 s overlap-discard seams through 25.60 s stay below their
  local 99.9th-percentile transient envelope.
- Control E conditioning: 53 lyric rows, zero non-finite condition/context
  values, condition SHA-256
  `bf1fdf4ecf31e150baa481527592e231f4095f969b9302d5781b2b66903c873f`,
  context SHA-256
  `ce5aa5c3f93896ad2a8131b0bd8c97a388633b5a3c2e13950b95ba99987173ea`.
- Control E semantic codes: 150 U32 values, SHA-256
  `42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be`.
- Receipt-body SHA-256:
  `e4f4b1b3dd77983fe8e64460880715a763c35713d1ac0babaf86329e3e754102`.

Exact-output changes must preserve declared bit identities. Reordered-rounding
or approximate changes must declare tolerances at the smallest useful boundary,
pass every applicable primitive/layer/subsystem/final-latent comparison, and
repeat human listening whenever they alter precision, sampler behavior,
attention/normalization math, VAE math, or another quality-sensitive path.

## Directional untuned timings

These are correctness-run observations, not thermally controlled Stage 2
benchmarks and not valid close A/B baselines.

| Fixture | Total | Planner | DiT load | DiT | VAE | Logical GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct vocal, 12 s | ~718.5 s | n/a | included in legacy total | not separately retained | dominant | 3,222,785,792 B |
| Fixed-metadata planner vocal, 30 s | 1,700.8 s | included in legacy total | included | included | dominant | 3,287,899,392 B |
| Default-CoT planner vocal, 30 s | 1,853.3 s | 163.7 s | 14.35 s | 48.04 s | 1,607.79 s | 3,295,681,792 B |

Control E recorded 351,964 queue drains and 351,961 ms of requested cooperative
idle. VAE accounts for 86.75% of its wall time and nearly all submissions.

## Current integrated exact checkpoint

After integrating `OPT-0002`, `OPT-0003`, `OPT-0004`, `OPT-0005`, and
`OPT-0006`, frozen commit `bf6da81647814737e88c8e881da72e04887cba07`
reran the accepted direct instrumental 12 s fixture through the complete
reference product path. The regenerated revision-4 manifest was authenticated
at SHA-256
`18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`.

The final WAV SHA-256 was exactly the accepted Stage 1 value
`d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477`.
This closes the short end-to-end numerical gate without another listening
round. The run is a nonthermal integrated checkpoint, not a reportable A/B and
not attribution to any one experiment.

| Metric | Stage 1 fixture | Current integrated fixture |
| --- | ---: | ---: |
| Generate-to-WAV wall | 720,489.5 ms | 64,822.1 ms |
| Queue drains | 132,326 | 2,200 |
| Requested cooperative idle | 132,323 ms | 2,197 ms |
| Tracked GPU peak | 3,214,388,992 B | 3,214,388,992 B |

The combined stack is approximately 11.115x faster on this exact short path,
with no tracked-GPU high-water increase. Current stage walls were 6,145.4 ms
text encoder, 7,025.6 ms condition encoder, 14,239.3 ms DiT load, 8,066.8 ms
DiT denoise, 3,094.8 ms VAE load, and 26,208.3 ms VAE decode. These directional
numbers now guide the next bounded experiment; the 180 s target remains unrun.

The retained ignored receipt is
`optimization/artifacts/OPT-0003/raw/post-integration-direct-instrumental-12s-receipt.json`
with file SHA-256
`1cb73d06818062972d6b597ee61d40c25be19a7646378065fce721a7ca835d2d`.

## Open evidence and release gates

- Physical Chrome-process high-water has not been measured; only logical GPU
  allocations are available.
- The formal 180 s run on the 16 GB M3 has not completed. Static maximum-length
  arena accounting and bounded-window contracts pass, but an empirical run is
  still required before release.
- On 2026-08-13 the owner explicitly approved deferring that empirical run until
  exact Stage 2 work makes it practical. The deferral is not a pass.
- Stock Chrome 151 has not rerun the complete production pipeline; actual-GPU
  kernel harnesses and listening runs used the Codex in-app Chromium surface.
- Native CUDA/XPU BF16 production taps and matched native audio are unavailable;
  no upstream numerical-equivalence claim is made.
- The raw-FP16 product path is not approved: learned lyric channel 259 exceeds
  FP16 finite range. Do not clamp it or treat it as the quality baseline.

## Stage 2 product target

The headline goal is a warm-cache three-minute song in under 60 s on the local
10-GPU-core, 16 GB M3. Measure direct and default-CoT planner paths separately.
The interval begins at Generate and ends when the final WAV is ready; it includes
per-generation package reads/hashing/upload, compilation, model execution, VAE,
normalization, and OPFS finalization. First download/cache population is reported
separately. Quality, reliable completion on the 16 GB machine, useful
cancellation, and browser/computer responsiveness remain mandatory.

## Benchmark authority and escalation policy

The owner delegated benchmark cadence and optimization decisions to the agent
on 2026-08-13. Spend full-generation time only when it can change a decision:

1. iterate with exact-shape microbenchmarks and authenticated primitive/window
   fixtures;
2. promote only strong candidates to one-layer, one-evaluation, or subsystem
   tests;
3. use short end-to-end generations to verify integrated winners; and
4. reserve 180 s generations for major convergence, memory, sustained-thermal,
   or release checkpoints.

Comparisons follow the owner-revised thermal protocol in `PLAN.md`: retain
every sample, start only after 30 continuous nominal seconds, and disclose the
complete through-cleanup trace without repeatedly chasing an all-nominal run.
This authority changes cadence, not correctness or disclosure: positive,
negative, and inconclusive evidence stays in the ledger, quality-sensitive
changes retain their declared listening gate, and no unrun long fixture is
described as passing.
