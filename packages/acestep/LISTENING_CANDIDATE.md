# Stage 1 short-fixture listening candidate

Status: **approved by the repository owner; Stage 2 optimization is explicitly
authorized as of 2026-08-13**. The owner judged both 30-second planner controls'
vocals excellent, including the default-CoT product control whose vocals begin
around 20 seconds. Both 12-second vocal requests are unsuitable vocal gates;
Control E demonstrates that this accepted trajectory enters after their crop.

This packet records the first complete browser outputs from the correctness
pipeline. It is a short-fixture human gate, not a claim that every formal Stage
1 exit criterion is closed. Native upstream checkpoint/audio captures and a
three-minute browser high-water run remain pending.

## Environment and trust root

- Date: 2026-08-13
- Frozen candidate commit:
  `77a5297c82c189f07f231538cbdc9d65867d4286`
- Diagnostic controls C/D ran from base commit
  `5d998346cbc2c93c63ddf010c6d4e006504cb2bd` with uncommitted trace
  instrumentation. Exact runtime-file hashes are recorded in
  `LISTENING_CANDIDATE_RECEIPTS.json`.
- Default-CoT control E ran from clean commit
  `99f3e90c442c311cd9bfb02cab2434484ca2312c`; its job-bound automatic receipt
  is preserved verbatim in `LISTENING_CANDIDATE_DEFAULT_COT_RECEIPT.json`.
- Host: MacBook Air `Mac15,12`, Apple M3 with 10 GPU cores, 16 GB unified memory
- OS: macOS 26.5.2 (25F84)
- Browser surface: Codex desktop in-app Chromium WebGPU browser, Dawn/Metal
- Separately installed stock Chrome: 151.0.7922.138; not yet rerun there
- Scheduling: `cooperative`, one command buffer outstanding, queue drain, then
  a real queue-empty interval before every non-final command
- Execution profile: `reference-bf16-subgroups`
- Package profile: packed BF16 weights decoded to FP32 activations; FP32 VAE
- Manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- ACE source commit:
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`

The WAV files are intentionally outside Git. The local listening paths below
are the artifacts whose hashes and metrics were recorded.

## Candidate A — direct instrumental

- Local file: `~/Downloads/ace-step-browser-reference-direct-instrumental-12s-c0ffee.wav`
- SHA-256:
  `d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477`
- Prompt: `Warm analog synth arpeggios over a restrained breakbeat, rounded
  electric bass, airy pads, instrumental, detailed stereo production.`
- Effective lyrics: `[Instrumental]`
- Duration: 12 seconds
- Seed: `0000000000c0ffee`
- Metadata: 104 BPM, D minor, time signature 4
- Planner: disabled
- DCW: double/Haar, low/high 0.05/0.02
- Wall time: 720,489.5 ms
- Tracked GPU peak: 3,214,388,992 bytes
- Queue drains / requested cooperative idle: 132,326 / 132,323 ms

## Candidate B — planner plus supplied vocal lyrics

- Local file: `~/Downloads/ace-step-browser-reference-planner-vocal-12s-badc0de.wav`
- SHA-256:
  `e3a3256e9521f3995bf1a9e5adc2b64f7718e67fd6f7105393e8826b332a897a`
- Prompt: `Bright neo-soul with elastic bass, clipped rhythm guitar, crisp
  pocket drums, warm keys, and a confident mezzo-soprano vocal.`
- Lyrics:

  ```text
  [Verse]
  Open the curtains, let the whole day in
  Dust in the sunlight starts to spin

  [Chorus]
  We found the rhythm under our feet
  Turn up the room and follow the beat
  ```

- Duration: 12 seconds
- Seed: `000000000badc0de`
- User BPM/key/time/language: unset; normal constrained CoT planner resolved them
- Planner: 0.6B, default thinking and two-row CFG semantic generation
- DCW: double/Haar, low/high 0.02/0.06
- Wall time: 1,024,269.3 ms
- Tracked GPU peak: 3,230,781,184 bytes
- Queue drains / requested cooperative idle: 138,283 / 138,280 ms

## Diagnostic control C — direct vocal, planner disabled

- Local file:
  `~/Downloads/ace-step-browser-reference-direct-vocal-control-12s-badc0de.wav`
- SHA-256:
  `d12463ea9d98fc86f9b83f8ada33fc8b46a68f1f1e89e217b906c9fb6e70eb79`
- Prompt, lyrics, seed, and 12-second duration: identical to Candidate B
- Instrumental: false
- Planner: disabled
- Metadata: 104 BPM, D minor, time signature 4, vocal language `en`
- Wall time: approximately 718,538 ms
- Tracked GPU peak: 3,222,785,792 bytes
- Queue drains / requested cooperative idle: 132,334 / 132,331 ms
- Lyric tokens: 53, all valid; SHA-256
  `04dacb6a616f7685c299bb7c76267286acee4f5f40af36a6596d26e911553915`
- Lyric-condition rows: 53; SHA-256
  `0413ca461a72bee7cfbc20d86687b77ec111f2654db85faba32ad64ae4e8a5ce`
- Full condition/context non-finite counts: 0 / 0
- Owner listening: **no vocals**

This control removes the planner semantic-cover route while preserving the
shared lyric tokenizer, eight-layer lyric encoder, packed condition, DiT
cross-attention, sampler, and VAE path.

## Diagnostic control D — 30-second planner vocal

- Local file:
  `~/Downloads/ace-step-browser-reference-planner-vocal-control-30s-badc0de.wav`
- SHA-256:
  `da57f30a8a10f744e4f3848cfdf71e385a3878792948093d857b5f9be4760f00`
- Prompt, lyrics, and seed: identical to Candidate B
- Duration: 30 seconds
- Instrumental: false
- Planner: enabled; 150 semantic codes
- Bound metadata: 104 BPM, D minor, time signature 4, vocal language `en`
- Generated CoT phase: skipped because all core metadata was supplied; the
  recorded `<think>` block is the synthesized Phase-2 metadata prefix, not
  generated reasoning
- Resolved caption: unchanged and still explicitly requests a confident
  mezzo-soprano vocal
- Semantic-code SHA-256:
  `4af8265ae6f48606ed1a8878109295ec65c1988f42aa2c6323f911e8af070d3f`
- Wall time: 1,700,753.5 ms
- Tracked GPU peak: 3,287,899,392 bytes
- Queue drains / requested cooperative idle: 348,202 / 348,199 ms
- Lyric tokens: 53, all valid, with the same token and lyric-condition hashes
  as control C
- Full condition/context non-finite counts: 0 / 0
- Owner listening: **amazing vocals**

This is a tighter duration/planner control than the original candidate: the
fixed metadata avoids a generated caption/language rewrite, while planner
semantic conditioning and its full-strength upstream-compatible cover context
remain enabled. The complete 150-code sequence, full requests, runtime-file
identities, and both controls' tensor/WAV records are preserved in
`LISTENING_CANDIDATE_RECEIPTS.json` for identity checking and reproduction.
The historical file was manually assembled from the job-scoped trace and
independently verified WAV; future demo runs emit one automatic WAV-bound
receipt.

## Product control E — 30-second planner vocal with default CoT

- Local file:
  `~/Downloads/ace-step-browser-reference-planner-vocal-default-cot-30s-badc0de.wav`
- SHA-256:
  `b702e10a2bae1b4a99bb10e54c92b663635a872b33d3aea58e3ba7fd4d739c14`
- Prompt, lyrics, seed, duration, instrumental=false, and planner defaults:
  identical to control D
- User BPM/key/time/language: all unset
- Generated CoT phase: **ran**; sampling cursor advanced from `0` to `260`
- Planner result: 100 BPM, B minor, time signature 2, language `unknown`
- Resolved caption: `A clean electric guitar plays a gentle, melodic chord
  progression ... This short, looping instrumental piece feels like a
  thoughtful intro or a mellow interlude.`
- Semantic codes: 150; SHA-256
  `42c83500063bf85d7856940620f7d8e7b97307e9584cd9ebd03e0b7ae7b8a3be`
- Lyric tokens: 53; condition/context non-finite counts: 0 / 0
- Wall time: 1,853,333.1 ms
- Tracked GPU peak: 3,295,681,792 bytes
- Queue drains / requested cooperative idle: 351,964 / 351,961 ms
- Automatic receipt-body self-checksum (canonical JSON body, excluding the
  checksum field):
  `e4f4b1b3dd77983fe8e64460880715a763c35713d1ac0babaf86329e3e754102`
- Preserved receipt-file SHA-256:
  `554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678`
- Owner listening: **sounds great; vocals begin around 20 seconds and sound
  great**

This run closes the setup ambiguity: Instrumental was off, the supplied lyrics
were present, all optional metadata was empty, and the constrained CoT phase
actually executed. It also exposes a concrete planner behavior that can explain
an instrumental excerpt without a lyric-path failure: generated CoT rewrote an
explicit vocal request into an instrumental intro/interlude. The lyric branch
remained finite and fully populated, and local supporting ASR recognizes the
supplied opening lyric beginning at 20.32 seconds. Human listening remains the
acceptance authority; it places the audible onset at approximately 20 seconds,
consistent with the supporting ASR onset of 20.32 seconds.

## Vocal-path audit

- The original Candidate B browser-operation record was retrospectively
  verified immediately before Generate and after completion: Instrumental was
  off, Planner was on, the exact lyrics were present, metadata fields were
  empty, and the CoT stage ran. This rules out the obvious Instrumental,
  Planner, and lyrics-field mistakes; it is not a general proof of every UI
  state.
- Static comparison found no path that drops or zeros lyrics. The request,
  canonicalization, planner downstream fields, lyric tokens/mask, eight lyric
  encoder layers, packed condition, and DiT cross-attention match the pinned
  source structure.
- The planner semantic route also matches pinned upstream behavior: nonempty
  semantic hints turn the request into cover mode, replace the source/context
  latents, and remain active at cover strength 1.0 for all eight denoising
  evaluations.
- That makes a semantic trajectory representing an instrumental intro a real
  possibility. Twelve seconds is legal within the 10–600-second range, falls
  within upstream's 10–20-second short-clip guidance, and is below the
  30–60-second duration described as stable for short songs. It is therefore a
  credible confounder, not proof that the implementation is correct.

## Mechanical validation

All five artifacts independently passed their applicable checks:

- canonical 44-byte RIFF/WAVE header, IEEE-float format tag 3;
- stereo, 48,000 Hz, 32-bit float;
- the three 12-second files contain exactly 576,000 frames / 4,608,044 bytes;
- each 30-second control contains exactly 1,440,000 frames / 11,520,044 bytes;
- each 12-second file has 1,152,000 finite, nonzero samples, and each
  30-second file has 2,880,000 finite, nonzero samples;
- distinct left and right channels;
- zero samples at or above absolute 1.0;
- peak `0.8912509083747864`, the expected -1 dBFS target; and
- no numerical discontinuity at any applicable 5.12-second overlap-discard
  boundary through 25.60 seconds. Every seam jump was below its local
  99.9th-percentile transient envelope.

No device loss, uncaptured GPU error, macOS thermal warning, performance
warning, or memory-throttled page was observed during these completed runs.
Tracked GPU bytes exclude opaque browser/GPU-process overhead and are therefore
not a complete system high-water measurement.

## Supporting vocal-presence analysis

An offline, local Parakeet-TDT 0.6B v3 greedy ASR pass was run over controls C,
D, and E. This is supporting evidence only: singing ASR can miss or hallucinate
words. The owner subsequently confirmed its C/D binary vocal-presence results
by ear; E remains pending.

- Control C, direct 12 seconds: empty transcript and zero emitted speech
  segments.
- Control D, planner 30 seconds: one 0.16–21.84 s segment at sentence
  confidence 0.899. It recognized the supplied opening as `Open the curtains
  let the day in`; the subword confidences for `Open the curtains` were about
  0.99–1.00. It also recognized a degraded version of the chorus beginning
  near 13.76 seconds.
- Control E, default-CoT planner 30 seconds: two segments spanning
  20.32–28.32 seconds at confidences 0.965 and 0.950. It recognized the
  supplied opening as `Open the curtains for the whole day` and a degraded
  form of the following line. The raw ASR JSON SHA-256 is
  `71ee62267c643346b7106b7bb65e64e42233215f37d00f216b985fb03b699758`.

This strongly supports vocals being present essentially from the beginning of
control D and absent or unrecognizable in control C; the owner confirmed that D
has amazing vocals and C has none. Control E is different: its generated CoT
explicitly asks for an instrumental intro/interlude, and ASR does not find the
supplied lyrics until 20.32 seconds. Together those are strongly consistent
with a default planner trajectory that defers ASR-recognizable supplied lyrics
beyond a 12-second crop, but ASR cannot rule out an earlier audible vocal that
it missed. Owner listening must establish the human onset. Exact ASR
model/input/result identities are in the two receipt files.

## Known differences and open gates

- Codex cannot judge musical quality. The owner must listen for prompt match,
  lyric intelligibility, section continuity, stereo image, unexpected silence,
  noise/distortion, repeated or missing material, and audible VAE seams.
- The pinned native reference harness requires a supported CUDA/XPU BF16 host;
  no matched native waveform or production checkpoint taps were available on
  this M3. The browser outputs are not treated as their own numerical oracle.
- Stock Chrome 151 and the formal three-minute memory fixture remain to be run.
- Correctness kernels are intentionally untuned. The current 12-second VAE
  emits about 131,000 drained command buffers; timer/idle overhead dominates.
  Scheduling coarsening and optimized convolution belong in Stage 2 after
  listening approval.
- A raw-FP16 diagnostic run completed the direct fixture, but the vocal fixture
  failed closed before DiT at valid lyric token 32, channel 259. The learned
  layer-7 MLP output is about 306,000 in FP32, beyond FP16's 65,504 finite
  maximum. Clamping is forbidden; raw FP16 is not a listening-candidate profile.

## Owner decision

Recorded 2026-08-13:

- Candidate A: musically accepted for this short-fixture checkpoint.
- Candidate B: rejected because the full 12-second excerpt sounded
  instrumental despite supplied verse/chorus lyrics and a vocal caption.
- Control C: rejected as a vocal sample because its full 12 seconds contained
  no vocals.
- Control D: accepted for vocal presence and quality; the owner described its
  vocals as amazing. Together with the finite, fully populated conditioning
  receipts, this confirms that the reference-BF16 browser pipeline can produce
  high-quality vocals.
- Control E: accepted for vocal presence and quality. The owner reported that
  it sounds great, with great-sounding vocals beginning around 20 seconds. Its
  trace records an instrumental-intro caption, and supporting ASR places
  recognizable supplied lyrics at 20.32 seconds.
- The original browser-operation record was retrospectively checked: its
  ready-state snapshot immediately before Generate and completed snapshot both
  show Instrumental off, planner on, exact lyrics intact, and optional metadata
  empty. A configuration toggle did not cause this result; that record remains
  supporting audit evidence rather than durable artifact provenance.
- Stage 1's product browser correctness and human-listening gate is closed.
  Static comparison found no lyric-dropping browser mismatch, and the accepted
  default-CoT control confirms that the 12-second clips ended before this
  trajectory's planned vocal entrance. This approval does not claim native
  upstream tensor/audio identity.
- The controls are hypothesis-narrowing rather than a strict A/B: planner mode
  changes DCW, and the longer run also fixes metadata/language, skips generated
  CoT, changes latent length, and samples a new semantic trajectory. The result
  supports duration/global trajectory as the practical explanation for the
  failed short clips; it does not isolate which changed factor caused it.
- The supporting ASR places Control D's vocals at about 0.16 seconds, so this
  is not evidence of a vocal merely arriving after a 12-second intro. Asking
  for 30 seconds changed the generated arrangement from its beginning.
- The required 30-second missing-metadata/default-CoT planner run is complete
  and owner-approved. The repository owner explicitly authorized Stage 2 on
  2026-08-13 with a target of generating a three-minute song in under one
  minute on this M3. Native CUDA/XPU captures, the stock-Chrome rerun, and the
  pre-optimization three-minute high-water fixture remain unclaimed. Native and
  stock-Chrome checks are retained as parity/release-hardening evidence. On
  2026-08-13 the owner explicitly approved deferring the empirical three-minute
  run until exact Stage 2 work makes it practical; it remains mandatory before
  release and is not represented as passing.
