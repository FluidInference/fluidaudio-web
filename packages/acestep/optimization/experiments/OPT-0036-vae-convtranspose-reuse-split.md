# OPT-0036 — VAE ConvTranspose reuse-axis split

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: exact tap-then-Cin FP32 accumulation

## Hypothesis

OPT-0026 owns four phase rows by four adjacent output channels per lane: 16
FP32 accumulators. OPT-0029 doubled both dimensions at once, leaving 64 live
accumulators/lane; despite fourfold larger output tiles, it regressed to
`0.9473x`. The result does not distinguish useful reuse from register-pressure
damage. Two 32-accumulator variants can isolate the axes:

- row-reuse: eight phase rows by four channels/lane, a `32 x 128` workgroup
  output tile, halving weight-load instructions per output; and
- channel-reuse: four phase rows by eight channels/lane, a `16 x 256` tile,
  halving input broadcast/workgroup overhead per output.

Both directly consume the revision-6 `[phase,tap,Cin,Cout]` FP16 layout and
retain exact increasing tap then Cin FP32 arithmetic and explicit FP16 store.

## Gate

Add isolated benchmark owners for exactly those two geometries; do not alter
OPT-0026 or production routing. Compile all five exact C512 production shapes,
compare every raw output U16 against OPT-0026, require complete deterministic
writes and clean teardown, then time one balanced A/B/C sequence after the
ordinary 30-second nominal check. Promote only a variant that is no slower on
every block and improves the summed five-shape median by at least `1.15x`.
Otherwise retain the result and stop. No converter/package, long-window,
waveform, or product claim is authorized by this screen.

## Result

One stock-Chrome/M3 timing launch followed a 30.068-second idle and a single
nominal thermal-level-0 observation. Both candidates compiled, completed, and
were raw-U16 exact against OPT-0026 over all five production C512 shapes:
`565,248,000` candidate comparisons plus `282,624,000` deterministic-rerun
comparisons had zero mismatches or unwritten values. Cleanup destroyed all 25
buffers and the device; the browser reported no warning or error.

The summed five-shape medians were `534.0 ms` for OPT-0026, `373.0 ms` for
row reuse (`1.431635x`), and `372.5 ms` for channel reuse (`1.433557x`). Both
candidates were faster on every shape and passed the frozen `1.15x` gate.
Channel reuse was the fastest single arm, but the shape response is decisive:
channel reuse won blocks 0-2 while row reuse won blocks 3-4. Selecting those
already measured exact owners per operation has a planning-only summed-median
score of `326.75 ms` (`1.634277x` versus OPT-0026); that hybrid was not timed
as a production decoder here and requires its own integration gate.

Receipt: [`../results/OPT-0036/result.json`](../results/OPT-0036/result.json),
SHA-256 `7d51fa2a9a2ac4feddd6645e9c2cbfa9e9ebd0e11d07e6b103cb424cbd9951e1`.
External thermal receipt:
[`../results/OPT-0036/thermal.json`](../results/OPT-0036/thermal.json), SHA-256
`2c2a1c3b8528d5ffcf10178425227d4d3e7e84baf5346ecb801349a8c00f6d92`.
