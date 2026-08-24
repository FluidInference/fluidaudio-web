# OPT-0022 — stride-2 full transformer trunk

## First-principles target

The supplied WAV produces 1,189 frames × 62 bands = 73,718 transformer
rows. Dense projections account for about 24.45 TFLOP and full time attention
for another 7.18 TFLOP, so raising contraction utilization alone cannot make
the exact 32.11-TFLOP graph substantially sub-realtime. This arm instead
tested reducing the temporal row count of every deterministic and CD
transformer block.

The experimental arm selected 595 original-frame anchors at positions
`0, 2, …, 1188`. Band splitting and mask decoding remain at all 1,189 frames.
After the low-resolution trunk, the full feature is reconstructed as:

```text
band_full + interpolate(trunk_low - sampled_band_full)
```

This bridge is the identity if the trunk is the identity and retains local
odd-frame band features. Time-axis RoPE uses original positions rather than
compressed positions; frequency attention remains unchanged.

The accounted arithmetic falls from about 32.11 to 14.57 TFLOP, a 54.6%
reduction.

## Primitive evidence

The browser WebGPU probe passed with:

- 0/42 temporal-anchor f16 mismatches across even and odd endpoint layouts;
- zero residual-bridge identity, constant-delta, and odd-tail mismatches;
- 0/18,432 fused-QKV mismatches against selected original-position rows;
- zero mapped fused/non-fused attention mismatches;
- 5,846 output words changed versus the deliberately wrong compressed-RoPE
  control, proving the position remap is active.

Type checking, all 8 unit tests, and the complete WebGPU probe passed before a
full model run.

## Paired supplied-WAV result

Both arms used blockwise Flash, the same decoded PCM, and seed `0xd1c05e` in a
fresh isolated Chrome 151 process.

| Stage | Full | Stride 2 | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 5,458.0 ms | 2,276.6 ms | 2.40× |
| four refinements | 16,316.7 ms | 7,037.1 ms | 2.32× |
| total | 23,845.9 ms | 11,388.2 ms | 2.09× |

The 12,457.7-ms saving validates the arithmetic model. Quality does not:

| Stem | NRMSE | SNR | cosine | worst-window NRMSE |
| --- | ---: | ---: | ---: | ---: |
| drums | 0.2164 | 13.29 dB | 0.97634 | 0.9959 |
| bass | 0.4771 | 6.43 dB | 0.88150 | 0.9999 |
| other | 0.4115 | 7.71 dB | 0.92521 | 1.1770 |
| vocals | 0.1564 | 16.11 dB | 0.98820 | 0.9053 |

## Disposition

The all-network stride-2 arm is rejected and its public/runtime selection was
pruned after recording this evidence. It proves that temporal row reduction
has the required performance leverage, but untrained resampling of the
deterministic separator changes the waveform far too much. OPT-0025 preserves
the deterministic network exactly and applies reduced resolution only to the
small consistency correction.
