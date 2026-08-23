# OPT-0070 — Exact quad-attention and C2378 production promotion

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Risk: larger bounded VAE allocation and changed command topology; no model
  arithmetic, sampler, package bytes, or waveform change

## First-principles basis

Two exact mechanisms are stranded behind frozen gates that measured the wrong
causal boundary:

1. OPT-0061's fixed-WG256 quad-query owner is raw-U32 identical to query8.
   OPT-0062 then compared `442,368,000` actual-layer U32 words with zero
   mismatches and reproduced all eight sampler taps and the final latent.
   OPT-0067's independently cooled evaluation slice measured
   `1.5808922936x` aggregate full-self speedup and projects
   `5,828.4 ms` saving over eight evaluations. Its literal non-pass came from
   per-family `2%` checks on unrelated sub-millisecond-to-tens-of-milliseconds
   buckets even though aggregate non-full-self work also improved.
2. C2378 is the minimum two-window cover for C4500 at overlap 64. OPT-0035
   proved deterministic identity across all `17.28 million` waveform samples,
   correct seams, one-owner lifecycle, and bounded memory. OPT-0059 later
   measured a same-profile C4500 projection of `38.4331 -> 29.4480 s`
   (`1.305117x`, `8.9851 s` saving) with exact shape outputs and a
   `3,667,109,696`-byte actual-C2314 guarded peak. The general C2378 backend
   allocates for the nominal maximum window and previously observed a
   `3,758,347,792`-byte peak; that larger number is the production ceiling.
   Its literal non-pass came from comparing family
   buckets whose pure/mixed batch composition necessarily changes with shape;
   the complete decoder and projected output wall were positive in both
   directions.

Neither result may be relabelled after the fact. This new ID promotes only the
demonstrated mechanisms under correctly targeted combined gates. The control
remains the current revision-7 exact dense/query8 plus revision-6 OPT-0028
VAE/C512 production selection. The candidate changes only full-self attention
ownership and VAE window geometry.

## Frozen integration direction

- Add a production-authenticated attention identity for the exact OPT-0062
  quad owner. Route only the exact 96 full-self M2250 operations through quad;
  retain query8 for all sliding and cross attention.
- Remove the diagnostic checkpoint requirement only for this production
  identity. Keep the optional actual-layer oracle/counter buffers absent from
  ordinary generation, and preserve every sampler/graph order and drain rule.
- Extend the exact revision-6 OPT-0028 VAE configuration with one explicit
  C2378 production window identity. Do not silently accept arbitrary sizes.
- Request `maxBufferSize` and `maxStorageBufferBindingSize` from the real C2378
  maximum plan rather than the C512 constant. Retain a single DiT owner,
  destroy it, then create a single VAE owner; no simultaneous heavy-model
  residency is permitted.
- Keep batch64, overlap 64, bounded normalization, durable WAV output,
  cancellation, progress, and the current eight-evaluation sampler unchanged.
- Do not select revision-8/selective dense or revision-7 approximate VAE under
  this ID. Those require their own quality evidence and later combined
  profile.

## Correctness and lifecycle gates

1. Statically reconcile exactly 96 quad full-self routes, unchanged
   sliding/cross routes, all graph quanta, and absence of diagnostic-only
   buffers/submissions in production.
2. Reuse the actual-layer raw-U32 authority and add a production-path result
   proving all eight evaluation taps and final latent remain exact.
3. Authenticate the exact revision-6 package and compare C512/C2378 raw
   pre-normalization waveform, stitched samples, seams, final WAV, and
   deterministic repeats. Require zero mismatches; this experiment has no
   numerical tolerance.
4. Prove C2378 device limits before allocation, exact workspace/binding sizes,
   peak live bytes at most `3,758,347,792`, one heavy owner, cancellation
   at a bounded command-buffer boundary, idempotent destroy, and zero live
   resources after success and failure.
5. Run a short complete production request that exercises more than one C512
   window and require byte-identical output versus the control.

## Performance gate

- Use balanced query8/C512 versus quad/C2378 complete arms after nominal starts
  and retain raw per-stage, graph, decoder, outer, cleanup, and thermal data.
- Quad must improve the full-self aggregate in both directions and save at
  least `4,500 ms` when projected across eight evaluations. Judge unrelated
  families by their aggregate absolute wall; do not repeat OPT-0067's invalid
  per-bucket percentage rule for sub-millisecond noise.
- C2378 must improve complete decoder and outer/output projections in both
  directions, save at least `7,000 ms` on C4500, and not regress normalized
  decoder work by more than `2%`.
- The combined short product wall must improve in both directions with no
  conditioner, package, finalization, or lifecycle regression above `2%` after
  absolute deltas are considered.

A pass authorizes production selection of these two exact mechanisms. It does
not approve revision-8 K4 dense arithmetic, revision-7 VAE K4 arithmetic,
fewer denoising evaluations, or an under-one-minute claim. The final cumulative
stack still requires its own short output check and one complete 180-second
Generate-to-WAV measurement.

## Authority

- [OPT-0061 quad primitive](OPT-0061-dit-attention-multi-query-wg256.md)
- [OPT-0062 complete quad integration](OPT-0062-dit-quad-query-attention-integration.md)
- [OPT-0067 evaluation slice](OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md)
- [OPT-0035 two-window C2378](OPT-0035-vae-c2378-two-window.md)
- [OPT-0059 short C2378 projection](OPT-0059-vae-c2378-short-projection-gate.md)

No production edit, browser/GPU work, package change, or performance claim
occurred when this experiment was registered.

## Closeout — integrated

Checkpoint `78d7346` added the authenticated OPT-0070 quad-attention and
C2378/overlap64 production identities, production device limits, protocol and
diagnostic reconciliation, and selected them in the demo tuple. Checkpoints
`ee2a5c9` and `75302f4` then made the attention owner shape-safe: only exact
M2250 full-self operations use the measured quad owner, while every other
product shape and sliding/cross attention uses the exact query8 production
owner without diagnostic buffers. The code and focused static evidence support
positive/integrated status. The cumulative checks are now complete: the final
OPT-0073 receipt used the public OPT-0070 attention and C2378 identities, two
actual C2314 windows, and produced the 180-second WAV in
`94,774.80000007153 ms` with no device loss. See the
[cumulative result](../results/OPT-0073/result.json).
