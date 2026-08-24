# OPT-0026 — Fast 10% long-track overlap

## First-principles target

OPT-0024 removes the four consistency-distilled evaluations from Fast, but the
long-track wrapper still inherited Full's 50% overlap. That schedule evaluates
almost every source coordinate twice even though Fast runs only the released
deterministic separator. Reducing duplicate fixed-size calls has more leverage
than another small kernel change and does not alter the neural graph within a
model item.

Full remains the released deterministic plus CD graph with the generic MSST
50% whole-track overlap policy. Fast keeps the 485,100-sample model item and
the existing 48,510-sample endpoint-inclusive linear fade, but advances by
436,590 samples. Adjacent Fast chunks therefore overlap exactly the fade region:

| Geometry | Full | Fast |
| --- | ---: | ---: |
| Model item | 485,100 samples | 485,100 samples |
| Step | 242,550 samples / 550 STFT hops | 436,590 samples / 990 STFT hops |
| Overlap and reflected border | 242,550 samples / 50% | 48,510 samples / 10% |
| Linear fade | 48,510 samples / 10% | 48,510 samples / 10% |

Both schedules remain aligned to the 441-sample STFT hop. In Fast, the two
linear window ramps span the complete overlap and normalized overlap-add keeps
every output coordinate covered.

## Long-track call-count projection

After input resampling, `trust_nobody.wav` has 5,608,109 model-rate samples.
The exact schedule geometry is:

| Schedule | Chunks | Evaluated model samples | Evaluated / source |
| --- | ---: | ---: | ---: |
| Full 50% overlap | 25 | 12,127,500 | 2.1625× |
| Fast 10% overlap | 13 | 6,306,300 | 1.1245× |

Because every item has the same size, the 25/13 ratio projects a **1.92×
reduction in chunk calls** for this input. This is deliberately not reported as
a 1.92× wall-time result: browser scheduling, DSP, allocation, and sustained
thermal behavior do not scale exactly with call count.

## End-to-end measurement

A fresh isolated Chrome 151 process ran `trust_nobody.wav` in Fast with one
warmup followed by three measured passes through the same worker and model
package. End-to-end browser samples were:

| Sample | Wall time |
| ---: | ---: |
| 1 | 71.57 s |
| 2 | 113.05 s |
| 3 | 79.61 s |

The sustained median was **79.61 s** (range **71.57–113.05 s**). The pass at
the median wall time reported 69.51 s deterministic model compute, 7.49 s
ISTFT, and 1.94 s preparation. A separate fresh-Chrome, no-warmup sample took
60.85 s end to end. The spread is large enough that the sustained panel, not
the isolated best sample, is the decision metric.

This misses the 30-second goal. At the sustained median, the deterministic
graph alone consumes 69.51 seconds, so further overlap reduction cannot close
the gap: the schedule is already only 13 fixed items for 127.17 seconds of
source. The next material step must reduce or accelerate deterministic-graph
and DSP work while preserving the checkpoint's useful quality.

## Correctness and quality boundary

The per-item deterministic checkpoint, STFT geometry, mask reconstruction,
and output restoration are unchanged. Inputs at or below the existing
single-pass threshold are also unchanged. Full retains its 50% schedule and
reference gates.

Fast's long-track context and crossfade differ from the generic upstream MSST
policy. Numeric plan and identity-overlap tests can establish exact length,
positive coverage, and absence of arithmetic gaps, but they cannot establish
perceptual seam quality. No listening-quality claim follows from the call-count
projection; long-track listening remains the acceptance gate for that tradeoff.

## Disposition

Retain the 10% schedule only for explicit Fast requests. Keep Full as the
default 50%-overlap path. Do not promote the Fast schedule as upstream-
equivalent, and do not describe it as meeting the 30-second long-track target.
