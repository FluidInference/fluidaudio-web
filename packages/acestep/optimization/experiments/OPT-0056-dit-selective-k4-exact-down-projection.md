# OPT-0056 — DiT selective K4 with exact MLP down projection

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate FP16 K4 partials retained in eight of nine repeated dense
  operations per layer

## First-principles basis

OPT-0037 failed its final-latent maximum-error gate even though NRMSE, SNR, and
Pearson passed. Treating every dense projection as equally risky would discard
the useful part of OPT-0032 without testing the structurally exceptional
operation.

Eight repeated dense operations reduce K1024: four self/cross query/output
projections, two self key/value projections, and the MLP gate/up projections.
Only `mlp-down-projection` reduces K6144. It therefore injects `1,536` bounded
K4 partials into each FP32 running output instead of `256`, then immediately
feeds the gated residual that carries state to the next layer. It is the
highest-leverage error-amplification suspect.

Down projection is about `26.1%` of scheduled repeated-dense MACs. If it stays
exact while the other `73.9%` retains OPT-0032's measured `1.405x` primitive
gain, the ideal dense-family speedup remains about `1.27x`, or roughly nine
seconds versus the exact `42.85 s` dense projection. This is large enough to
test; reverting all three FFN contractions would retain too little upside.

## Frozen diagnostic direction

- Keep the authenticated revision-8 K4 package and add an exact-increasing-K,
  FP32 multiply/add owner that directly consumes its physical K4 layout. This
  owner exists to isolate arithmetic without generating another 3 GB package;
  it must reproduce the revision-7 FP16-weight/FP32-accumulator result at the
  primitive and actual-layer boundaries.
- Add a distinct diagnostic selector: route exactly the 24
  `mlp-down-projection` weights/operations through that exact owner, and route
  the other 192 repeated dense operations through OPT-0032 K4. Cross-cache and
  shared BF16 projections remain unchanged.
- Do not change the sampler, DCW, attention, normalization, residuals,
  activation precision, package bytes, graph order, or production default.
- Report the actual kernel owner for all 216 repeated dense routes and fail
  closed on any label/shape mismatch.

## Gates

1. Exhaustively prove K4-layout logical indexing and raw FP16 weight identity.
   On all four production shapes and actual down-projection weights, require
   the exact-layout owner to match revision-7 output U32 words exactly.
2. Capture the output latent after every one of eight sampler evaluations for
   the exact control, the known all-K4 arm, and the exact-down hybrid. This
   localizes when errors amplify; the all-K4 OPT-0037 receipt may be reused
   only for its final aggregate, not substituted for missing step taps.
3. Against the exact control, the hybrid must pass the unchanged OPT-0037
   final thresholds: NRMSE at most `0.02`, SNR at least `34 dB`, Pearson at
   least `0.999`, maximum absolute error at most `0.25`, no non-finite values,
   deterministic repeats, and clean lifecycle/cancellation. Do not waive the
   maximum-error condition because the all-K4 aggregate metrics were close.
4. If correctness passes, use a short balanced layer/evaluation-slice screen
   and require at least `1.15x` realized repeated-dense speedup over the exact
   control before package-layout integration. The diagnostic exact K4-layout
   owner's own speed is not the proposed production layout; a later
   replace-not-duplicate package may retain native exact layout for the 24
   down weights under a new integration identity.

If exact-down alone fails the final-latent envelope, stop this selector. Any
broader exact subset, layer-dependent precision, error-feedback scheme, or
altered K-partial length requires a new ID. No waveform, listening, production,
or product-speed claim is authorized until this numerical gate passes.

## Closeout — final-latent gate failed

The authenticated three-arm trajectory result is
[`result.json`](../results/OPT-0056/result.json), SHA-256
`9d84fcb6daa6c18a702f34ac645e47d3611691eb45e13c5ab6b9308238bbdf96`.
The selective exact-down arm was deterministic and improved the all-K4 final
maximum absolute error from `0.9955760241` to `0.6395987272`, while NRMSE
`0.0078057355`, SNR `42.1517 dB`, and Pearson `0.9999693484` passed. Its
maximum error still exceeded the unchanged `0.25` cap, so the literal
correctness gate failed and timing was correctly skipped. All arms stayed
finite and sequential with clean disposal. Stop this selector as negative and
abandoned; no revision-8 package, diagnostic owner, or route is integrated.
