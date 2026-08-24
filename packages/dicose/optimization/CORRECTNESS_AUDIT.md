# Upstream correctness audit

This audit compares the browser runtime with the official DiCoSe source at
commit `a1dc0a41ad2b1829674a60ea74e74edfd1509083`. The released checkpoints used
by both sides are byte-identical:

- deterministic: `8087fbdcbc63f11f3ee305ef042cf42a42a5802e8a76678997f6448cb45256f5`
- consistency-distilled: `d25035bed7294a227fcb0f1ea691a0d1b8452ef76bde0e411c2b75536acf13da`

The performance measurements in the optimization ledger still describe GPU
cost, but measurements made before this audit are not output-quality evidence.
The previous acceptance checks compared optimized WebGPU paths with other
local paths and broad signal-energy envelopes; correlated implementation bugs
could therefore pass.

## Confirmed semantic defects

### Shared GELU was too steep

The WebGPU erf approximation evaluated `erf(x)` where GELU requires
`erf(x / sqrt(2))`. This affected every deterministic feed-forward block and
the CD time/mapping/adapter graph. The first large upstream divergence appeared
at the first feed-forward GELU despite strong agreement immediately before it.

The WGSL implementation now scales the erf input by
`0.7071067811865476`. Against a genuine upstream float32 execution on the
included fixture, the corrected deterministic waveforms measure:

| Stem | NRMSE | SNR | Cosine |
| --- | ---: | ---: | ---: |
| drums | 0.000350 | 69.12 dB | 0.99999994 |
| bass | 0.000526 | 65.57 dB | 0.99999986 |
| other | 0.000433 | 67.27 dB | 0.99999994 |
| vocals | 0.002745 | 51.23 dB | low-energy stem |

The browser reference gate also compares 30 sampled intermediate tensors, so
this class of shared activation error now fails close to its origin.

### CD time conditioning was off by 1000×

The sampler transports sigma as `250 * log(sigma)`. Upstream
`EDMPrecond.forward` reverses that transport and passes `log(sigma) / 4` to the
BS-RoFormer. The browser passed the transported value directly. At the released
one-step sigma this was approximately `-1384.52` instead of `-1.38452`.

The browser now supplies `log(sigma) / 4`. A static operator-by-operator audit
found the remaining Full CD graph consistent with upstream: input/noise
scaling, `c_in`/`c_skip`/`c_out`, condition capture and injection order, stem
embedding and FiLM order, STFT adapter, time/frequency transformers, masks,
complex multiplication, final affine, and clamping.

### CD stem-embedding copies invalidated the mapping command encoder

The packaged weight buffer was created with `STORAGE | COPY_DST`, but
`createMappings` also uses it as the source of four `copyBufferToBuffer`
operations for the learned stem embeddings. WebGPU therefore invalidated the
entire CD mapping command encoder. Before explicit error scopes were added,
Chrome did not turn that validation failure into a rejected separation, so the
refiner continued with unusable FiLM vectors.

The package buffer now includes `COPY_SRC`. The unattended Full-mode check
reproduced the original validation error before this change and completes with
valid per-stem mappings after it. Validation, out-of-memory, and internal GPU
errors are now surfaced at the inference boundary instead of silently flowing
into waveform output.

### Arbitrary-length files were sent through one unbounded graph

DiCoSe is trained and evaluated on 485,100-sample (11-second) items. The old
browser runtime instead built one time-attention graph for an entire track.
For `trust_nobody.wav` (16-kHz mono, 127.168 seconds), conversion to 44.1 kHz
produces 5,608,109 samples and 12,717 STFT frames. At that shape:

- the deterministic `wide` activation alone requires about 2.26 GiB;
- time attention grows quadratically;
- specialized STFT-adapter convolution dispatches exceed the WebGPU grid;
- invalid or failed GPU work could flow into CD noise instead of producing a
  useful error.

Long Full input now uses fixed 485,100-sample model items, 50% overlap,
reflected outer context, the upstream 10% endpoint-inclusive linear fade,
normalized overlap-add, and exact crop/length restoration. This follows the
generic MSST whole-track policy without reproducing its batch-dependent
edge-window bug or its final chunk whose output is entirely cropped. The
overlap noise is keyed by padded-track coordinate so the final CD affine does
not crossfade independent noise fields at seams. WebGPU validation,
out-of-memory, and internal errors are scoped and surfaced.

Fast keeps the same fixed model item, fade, normalization, and exact output
geometry, but advances by 436,590 samples so adjacent chunks overlap only the
48,510-sample fade region. This is a deliberate long-track performance policy,
not an upstream whole-track equivalence claim. At the model-rate length of
`trust_nobody.wav`, Full plans 25 chunks and Fast plans 13. The 25/13, or
approximately 1.92×, reduction is a chunk-count projection; listening quality
and end-to-end wall time are not inferred from it.

Pure numeric tests cover both long-track plans, reflection, adaptive final
padding, positive coverage at every output sample, identity reconstruction at
boundary lengths, asymmetric stereo, and Full overlap-noise continuity.

The long-file wrappers are explicit browser policies, not a claim that upstream
published a whole-track stochastic oracle. Full's chunk, reflection, fade,
normalization, and crop geometry follow the generic MSST helper. The Full path
uses one coordinate-stable noise field across overlaps to avoid crossfading
independent CD noise at seams; upstream's sampler draws a new contiguous field
per invocation. Fast retains the common fixed-item and overlap-add machinery
but deliberately changes the step and reflected border described above.

### Non-44.1-kHz resampling and output geometry differed from the loader

Upstream uses torchaudio's default Hann-windowed sinc resampler when source and
target rates differ. The browser used two-point linear interpolation, endpoint
hold, and a rounded output length. Torchaudio uses a centered zero-padded
polyphase filter and a ceiling output length. This is material for both the
included 22.05-kHz fixture and the reported 16-kHz long file. The production
path now uses the torchaudio 2.0.2 geometry and coefficients, with rate-pair,
edge-impulse, identity, and output-length numeric oracles.

The file API also used to return the internal 44.1-kHz timeline instead of the
source file's timeline. Upstream restores the original sample rate and exact
frame count after inference. The browser now retains that input geometry,
resamples all stems back after the model, and trims or zero-pads the single
rounding frame when needed. For `trust_nobody.wav`, output is therefore exactly
2,034,688 frames at 16 kHz rather than 5,608,109 frames at 44.1 kHz. This
mismatch could retain out-of-band model residue and wasted transfer/storage,
but could not change duration or turn valid PCM into broadband noise.
Upstream's generic CLI uses librosa `kaiser_best` for this reverse conversion;
the browser reuses its validated Hann-sinc resampler, so native geometry is
matched but reverse-resampler samples are not claimed bit-exact.

### Over-range Fast exports were clipped

The demo always encoded PCM16 and therefore hard-clipped deterministic/Fast
samples outside `[-1, 1]`. Upstream selects float WAV when a stem peak exceeds
one. The demo now uses the same peak rule: in-range stems remain PCM16, while
over-range stems use IEEE-float WAV and preserve their samples. Full already
clamps final output by design, matching the released sampler, so this was not
the reported Full-mode noise mechanism.

### A fresh manifest could be paired with stale packed weights

The manifest was fetched with `no-store`, while the fixed-name 623 MB weight
blob used `force-cache`. Only byte length was checked. Several packing layouts
have the same total length, so an updated manifest could address a stale cached
blob with incompatible layouts and produce arbitrary tensors without a load
error.

The weight URL is now content-addressed with the manifest's declared SHA-256 as
its cache key. Manifest parsing rejects malformed weight digests, and the
stream still enforces the exact declared byte count. A package revision can no
longer reuse another revision's cached blob.

## Audited matches

The following paths were independently compared with upstream and did not
contain a semantic mismatch after the model-math and mapping fixes:

- centered periodic-Hann STFT/ISTFT geometry and exact requested length;
- stereo spectrum packing and `[left real, left imag, right real, right imag]`
  ordering;
- 62-band split/scatter layout and stem ordering;
- RMSNorm, RoPE frequencies and positions, gated attention, residual order,
  feed-forward GLU, mask estimation, zero-DC handling, and complex masks;
- deterministic-to-CD condition tensors and every condition-add location;
- CD sampler coefficients and one-step schedule;
- checkpoint tensor selection, names, shapes, and package hashes;
- peak-selected PCM16/IEEE-float WAV interleaving and headers.

Expected numerical differences remain from binary16 storage, WebGPU FFT
rounding, and the browser's deterministic RNG. Those are not semantic graph
changes.

## Full/CD dynamic parity

The Full oracle uses exactly the first 485,100 samples of the included WAV
after the production browser Hann-sinc conversion, not a separately recreated
input. Its planar input SHA-256 is
`2f47b43d1129549916fa2cd9a70ca0dae4b650f7e1833fe811a550d5283ff3f2`.
The official implementation and browser share fixed seed `0xd1c05e`.

Final refined waveform agreement is:

| Stem | NRMSE | SNR | Cosine |
| --- | ---: | ---: | ---: |
| drums | 0.001114 | 59.06 dB | 0.99999938 |
| bass | 0.000858 | 61.33 dB | 0.99999963 |
| other | 0.000766 | 62.32 dB | 0.99999971 |
| vocals | 0.021789 | 33.24 dB | 0.99991330 |

The vocal reference is nearly silent (RMS `0.0000914`); its absolute RMSE is
`0.00000199`. Before the final consistency affine, raw CD-model NRMSE is
0.00052–0.01187 with minimum cosine 0.999932. The gate additionally compares
17 internal CD seams, including time embedding, mapping input/output, first
FiLM vectors, both condition adapters, early and late transformer states,
final normalization, and masks. Every stem/call is checked against a
stage-specific envelope rather than an aggregate signal-energy heuristic.

Waveforms were generated by upstream PyTorch float32. Internal stages were
generated with the released float16 weights under PyTorch autocast so the
comparison isolates graph/layout errors from expected precision differences.
Independent oracle-generation runs were byte-identical.
An additional reconstruction loaded both checkpoint files independently,
recomputed the condition adapter without browser intermediates, and matched
the fixture's condition stages at 0.00043–0.00050 NRMSE.

## What the two demo modes mean

- **Full** is the released deterministic graph followed by the released
  one-step CD graph. Its long-track wrapper retains the upstream 50% overlap
  policy.
- **Fast** is the released deterministic checkpoint returned directly. It is
  a valid upstream model output per item, but omits the learned CD refinement.
  On inputs above 12 seconds it also uses 10% rather than 50% overlap, so its
  whole-track result is an explicit approximation rather than an upstream
  reference-equivalence claim.
The retired Balanced experiment used a stride-2 CD temporal trunk for which
there is no released checkpoint or upstream graph. Its measurements remain in
OPT-0025 as historical evidence, but the mode was removed rather than retained
as a dormant public switch. The old observation that it sounded less broken on
one short file did not validate the then-incorrect CD graph.

The retired Extra Fast experiment changed the full temporal grid to hop 882.
It met the 30-second timing target, but listening found its degradation too
large relative to Fast. OPT-0027 retains the measurements as historical
evidence; the public mode, runtime geometry, shader support, and probe-only
stride-2 primitives were removed completely.

## Gates

- `pnpm test` covers DSP primitives, model/package contracts, chunking, and
  overlap-add behavior.
- `pnpm test:reference-quality` compares Fast/deterministic waveforms and
  intermediate tensors with the official PyTorch model and fails on envelope
  violations.
- `pnpm test:refined-reference-quality` compares deterministic, raw CD, and
  final Full waveforms plus 17 internal CD seams with a frozen-noise official
  PyTorch execution. It enforces both aggregate and per-stem/per-call limits.
- `pnpm test:output-mode-quality` checks same-worker Full/Fast isolation and
  verifies their deterministic diagnostics agree exactly.
