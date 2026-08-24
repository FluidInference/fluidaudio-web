# OPT-0025 — CD-only stride-2 temporal trunk

## First-principles target

OPT-0022 established that temporal row reduction has enough arithmetic leverage
to move the full runtime materially, but reducing both networks damaged the
deterministic separator before refinement began. This narrower arm preserves
the deterministic separator and all of its condition adapters at the released
full resolution. Only the four consistency-distilled transformer trunks use
the 595 anchors at original frame positions `0, 2, …, 1188`.

The CD band split and mask estimator remain at all 1,189 frames. Each full
deterministic condition tensor is sampled directly while it is added to the
low-resolution CD destination, avoiding persistent low-resolution condition
copies. Original-position RoPE is retained. The full mask feature is restored
with the same residual-delta bridge as OPT-0022:

```text
cd_band_full + interpolate(cd_trunk_low - sampled_cd_band_full)
```

This leaves the deterministic graph and its diagnostics byte-path equivalent
to full mode while reducing arithmetic in the four CD evaluations that
dominate refined inference.

## Validation

The raw WebGPU probe covered even, odd, and clamped-tail anchor addition with
zero mismatches across 42 f16 words. Fourteen fixture words distinguish the
correct uploaded-f16 add from a naive full-precision reference, proving the
rounding check is active. The existing decimation, residual bridge, and
original-position RoPE probes also remained exact. Type checking, all 8 unit
tests, the complete WebGPU probe, and diff checks passed before the paired
waveform run.

## Paired supplied-WAV result

Both arms used blockwise Flash, the same decoded PCM, and seed `0xd1c05e` in
the isolated-Chrome quality harness. The candidate changed only the four CD
workspaces; deterministic diagnostics matched the full control exactly.

| Stage | Full refined | CD stride 2 | Full / candidate |
| --- | ---: | ---: | ---: |
| deterministic | 4,947.3 ms | 4,819.1 ms | 1.03× |
| four refinements | 16,340.6 ms | 7,103.6 ms | 2.30× |
| ISTFT | 1,311.6 ms | 1,471.8 ms | 0.89× |
| **total** | **23,214.6 ms** | **14,081.3 ms** | **1.6486×** |

The candidate saved **9,133.3 ms** end to end. Unlike the all-network arm,
global waveform drift stayed small:

| Stem | NRMSE | SNR | cosine | worst-window NRMSE | worst-window cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| drums | 0.0230327 | 32.7531 dB | 0.9997358 | 0.23294 | 0.97274 |
| bass | 0.0222445 | 33.0555 dB | 0.9997531 | 0.54917 | 0.97686 |
| other | 0.0353758 | 29.0259 dB | 0.9993754 | 0.09258 | 0.99572 |
| vocals | 0.0180505 | 34.8702 dB | 0.9998604 | 0.03713 | 0.999635 |

Peak-localized errors were also bounded on this fixture: maximum absolute
error/reference peak was `0.0396217/0.07330` for drums,
`0.0208697/0.05809` for bass, `0.0217871/0.06172` for other, and
`0.00042005/0.03228` for vocals. The bass worst-window NRMSE of 0.54917 is the
clearest remaining warning that low global error is not a listening or
ground-truth quality result.

## Sustained benchmark

The selectable release harness also ran one warmup and three measured
`refined` + `cd-stride2` passes in isolated Chrome 151. End-to-end samples were
17,534.9, 17,908.1, and 18,617.0 ms, for a **17,908.1-ms median** (range
17,534.9–18,617.0 ms). Median stages were 6,494.4 ms deterministic, 9,562.7 ms
refinement, 1,262.8 ms ISTFT, and 17,898.3 ms model total.

This panel immediately followed the longer full-resolution sustained panel;
its nominally unchanged deterministic stage was 23% slower than the full
panel's median and continued rising, exposing substantial device thermal
state. Therefore 17.91 s is a conservative hot-device absolute result, while
the adjacent-arm 14.08 s result above remains the controlled speed comparison.

## Disposition

Retain `cd-stride2` as an explicit experimental balanced mode. It recovers
most of the temporal-reduction speedup while avoiding the large waveform drift
caused by decimating the deterministic separator. Full refined inference stays
the default until licensed ground-truth stems, per-stem SDR/SI-SDR and
transient gates, and blind listening establish that the local errors are
acceptable.
