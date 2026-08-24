# OPT-0018 — cooperative Flash score exponentials

## Hypothesis

The Q64×K16 Flash owner used all 256 lanes for score construction, then only
64 lanes for softmax while each active lane evaluated 16 exponentials. Giving
four lanes to each query could expose four times more score-exp concurrency.

The benchmark arm kept the current owner's ascending block maximum and final
ascending probability sum. It published the next maximum, let all 256 lanes
compute four identical scalar exponentials apiece, and returned the stored
probabilities to the original owner for state update. Workgroup storage stayed
at 25,344 bytes. Reordering the phases required one additional barrier per K16
block, from four to five.

## Correctness

The narrow Chrome 151 probe compared the current Flash output and cooperative
arm across all 56,832 f16 words and found zero raw-word mismatches. This
confirmed identical target-browser score, softmax, and output association.

## Production-shape timing

Two fresh-profile panels used two warmups and seven timestamped samples per
arm and geometry. Compilation, upload, submission, and readback were excluded.

| Run | Geometry | Current median | Cooperative median | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | time 62×1,189 | 193.396736 ms | 189.988864 ms | 1.0179× |
| 1 | frequency 1,189×62 | 11.534336 ms | 11.534336 ms | 1.0000× |
| 2 | time 62×1,189 | 193.069056 ms | 190.251008 ms | 1.0148× |
| 2 | frequency 1,189×62 | 11.534336 ms | 11.468800 ms | 1.0057× |

Across the production 40 time and 40 frequency calls, those medians project
only 115–136 ms of saving. The extra barrier almost completely cancels the
benefit of parallel score exponentials, proving that the apparent 75% idle
phase was not a multi-second standalone bottleneck.

## Disposition

Negative. The benchmark arm was pruned and production Flash remained
unchanged. Any later attention work must alter a larger dataflow or arithmetic
term rather than repeat this ownership-only softmax split.
