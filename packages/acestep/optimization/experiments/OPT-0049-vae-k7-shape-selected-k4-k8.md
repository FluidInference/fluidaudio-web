# OPT-0049 — VAE K7 shape-selected K4/K8 partials

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: approximate, per-operation bounded FP16 reduction selection

## First-principles basis

OPT-0041 rejected one universal K8 owner because C512 was `0.9730x` versus
K4, but it established a coherent channel-dependent result: K8 won C1024,
C256, and the dominant C128 tier by `1.1042x`, `1.1304x`, and `1.4000x`.
Keeping K4 only for C512 and selecting K8 for those three winning tiers gives
an evidence-derived C300-weighted score of `3,510.899929 ms` versus
`4,279.349978 ms` for general K4 (`1.218875520x`). Applied only as planning
arithmetic to OPT-0024's long K7 projection, that is about `6.28 s` additional
potential saving; it is not a long measurement.

This tests shape selection, not another reduction length. The candidate uses
the unchanged native O-K-I package, unchanged OPT-0024 K4 owner for C512, and
the unchanged OPT-0041 K8 owner for C1024/C256/C128. All unbiased K7
operations remain on the exact owner. No runtime default changes in this
experiment.

## Frozen gate

Use the authenticated revision-6 C512 fixture and the same batch64 topology on
both arms. The control routes every eligible biased K7 operation through K4;
the candidate routes strictly by the frozen channel map above. Before timing,
require complete deterministic outputs, exact owner/count reconciliation, no
non-finite values, clean canaries/lifecycle, and the complete OPT-0024 C512
waveform envelope against the same exact accepted oracle. Time one balanced
AB/BA screen after the ordinary single level-0 thermal check. Require both
directions to improve homogeneous K7 wall, median K7 speedup at least `1.10x`,
and no decoder-wall regression. A pass authorizes only the existing OPT-0044
trajectory/product-quality escalation; it does not itself change production.

If the candidate fails, retain general K4. Do not retune the channel map after
the result. Revisit only if a different K7 physical layout materially changes
the per-tier K4/K8 behavior.

## Closeout — superseded by row reuse

The frozen selector gate was not executed, so the OPT-0041-derived projection
remains inconclusive planning evidence. OPT-0051/0057 subsequently established
a materially stronger shape-selected row-reuse direction while retaining the
same bounded K4 arithmetic, superseding this K4/K8 selector. No OPT-0049
runtime route or production default was integrated.
