# OPT-0041 — VAE K7 bounded FP16 K8/K16 partials

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate bounded K8/K16 FP16 partial reduction with FP32 running state

## Hypothesis

OPT-0024 already proved that native FP16 dot4 partials widened into FP32 make
the dominant K7 family materially faster and remain inside the C512 waveform
envelope. Its direct, barrier-free kernel still updates each of 32 FP32
accumulators once per Cin4 group. Combining two or four consecutive dot4
partials in FP16 before one FP32 update can shorten that dependency chain by
2-4x without the shared-memory/barrier failure mode of OPT-0017 or the
unbounded FP16 accumulator rejected by OPT-0009.

## Frozen mechanism and gate

Keep OPT-0024's native O-K-I weights, WG128 fixed32 ownership, eight rows by
four output channels per lane, increasing kernel/Cin block order, FP32 bias and
running state, and explicit FP16 output store. Add only bounded K8 and K16
local-partial variants, handling a final short group without reading outside
Cin. Compare OPT-0024/K8/K16 over the frozen primitive tiers and adversarial
inputs. Require finite deterministic complete outputs and the OPT-0024
numerical envelope. A candidate must improve the weighted primitive wall by at
least `1.15x` over OPT-0024 before a complete C512 waveform gate; production
routing, long projection, and listening remain unauthorized until then.

## Result

All four production tiers and four adversarial fixtures passed the frozen
numerical and lifecycle gates. Each candidate was executed twice over 16
probes, with `2,428,928` raw-U16 comparisons per candidate, zero non-finite
outputs, zero finite-to-zero or signed-zero changes, complete qNaN overwrite,
intact canaries, deterministic reruns, and no live resources after cleanup.
Aggregate K8/K16 maximum absolute error versus K4 was one FP16 step
(`0.000030517578125`); all per-probe NRMSE, SNR, Pearson, and relative-error
limits passed.

The one reportable timing launch followed `34.582 s` idle and exactly one
thermal-level-0 observation. Six balanced permutations per tier produced the
following medians:

| tier | production weight | K4 (ms) | K8 (ms) | K16 (ms) | K8/K4 | K16/K4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1024 | 282 | 2.6500 | 2.4000 | 5.2000 | 1.1042x | 0.5096x |
| C512 | 423 | 1.8000 | 1.8500 | 2.4500 | 0.9730x | 0.7347x |
| C256 | 423 | 1.3000 | 1.1500 | 2.3500 | 1.1304x | 0.5532x |
| C128 | 1,269 | 1.7500 | 1.2500 | 1.4000 | 1.4000x | 1.2500x |

The exact production-weighted scores were `4,279.349978 ms` for K4,
`3,532.049960 ms` for K8 (`1.211576854x`), and `5,273.399997 ms` for K16
(`0.811497323x`). K8 therefore cleared the aggregate threshold but lost the
C512 tier; K16 lost three tiers. Neither satisfied the frozen all-tier-win
rule, so the declared decision is `negative-stop-before-c512`: retain K4 as
the general owner, do not run C512, and do not authorize production routing or
listening.

The heterogeneous result is still useful mechanism evidence, not authority to
reinterpret this gate: a separately registered per-shape K4/K8 selector could
use K8 only on C1024/C256/C128, but must receive its own measurement and
quality escalation. One earlier page preparation was rejected before timed
dispatch because the preflight interval reached `67.325 s`, outside the
frozen 30–35-second launch window; it is not a performance sample.

Compact receipt: [`../results/OPT-0041/result.json`](../results/OPT-0041/result.json),
SHA-256 `8c638a562e6a00ac9c49e26fb51b5ed6caf81c08d1ba6c8c1fd2e372a0e5da74`.
