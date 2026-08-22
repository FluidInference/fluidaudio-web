# OPT-0031 — DiT dense WG512 cooperative panels

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: exact FP16 operands with increasing-K FP32 accumulation

## Hypothesis

The M3 adapter reports stock WebGPU limits of 1,024 compute invocations and
32 KiB workgroup storage, but OPT-0019–0021 requested only WG256/16 KiB. A
WG512 `M128 x N128 x K32` tile can cover twice the rows and twice the K depth
of OPT-0019, reducing weighted production panel iterations and barriers by
4x, while preserving the selected OPT-0009 arithmetic exactly.

## Frozen mechanism

- WG512 / 16 fixed-32 subgroups; output tile `M128 x N128`.
- Each subgroup owns eight rows by 128 columns; each lane owns eight rows by
  four adjacent columns (32 FP32 accumulators).
- Shared FP16 panels: padded A `[128,33]` and B `[32,132]`, 16,896 bytes total.
- Consume the existing converter-native N256/K32 FP16 weights by half-tile.
- For each K32 panel, load, barrier, execute K `0..31` with FP32 multiply/add
  in the same order, then barrier. No reassociation or precision change.
- Request only stock WebGPU limits; no browser flags, WebNN, MPS, or native API.

This is materially different from OPT-0019's WG256 `M64 x N128 x K16`
mechanism: it halves workgroups and halves panel iterations per workgroup,
quartering total panel/barrier events. It does not repeat OPT-0020's
horizontal FP32 dot or OPT-0021's layout-only change.

## Gate

Compare the four exact M2250 production shapes against current OPT-0009 with
complete raw-U32 exactness, finite/complete writes, balanced timing, and the
ordinary one-shot 30-second nominal check. Continue only if every shape is no
slower and the weighted `4/2/2/1` score is at least `1.25x`; otherwise retain
the negative result. No package, selector, trajectory, or product claim is
authorized by this primitive screen.

## Result

Stock Chrome 151 on the fixed32 Apple M3 compiled and executed WG512 with
16,896 bytes of workgroup storage. Before timing, four executions per shape
covered `101,376,000` raw-U32 comparisons with zero mismatches. Current and
candidate reruns were deterministic, every qNaN prefill was overwritten, all
outputs were finite, tail rows were written, and adjacent canaries remained
intact.

After the one 30-second nominal gate, four rotated AB/BA samples per arm gave:

| Shape | Multiplicity | Current median | WG512 median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| H→H | 4 | 11.9500 ms | 13.1500 ms | 0.90875x |
| H→1024 | 2 | 6.3500 ms | 7.0000 ms | 0.90714x |
| H→6144 | 2 | 35.1000 ms | 38.3500 ms | 0.91525x |
| 6144→H | 1 | 36.1000 ms | 38.1500 ms | 0.94626x |
| **Weighted 4/2/2/1** | | **166.8000 ms** | **181.4500 ms** | **0.91926x** |

The candidate lost every stratum and missed the `1.25x` gate. Wider advertised
limits are usable, but this WG512/shared-panel geometry lowers occupancy or
adds enough synchronization cost to outweigh its reduced panel traffic. Stop
without package, selector, trajectory, or product escalation. Revisit only if
a browser/compiler/GPU change invalidates this result or a materially different
matrix-instruction capability becomes available in stock WebGPU.

The canonical receipt is
[`optimization/results/OPT-0031/result.json`](../results/OPT-0031/result.json),
SHA-256 `399508ad1f1235969e45f58cdfccdf447ba8df6ce386e96fc1d513d067e3b42e`.
Cleanup destroyed all 20 buffers and the device. Two earlier button attempts
were rejected by the thermal form before timed GPU dispatch (missing fields,
then a Vite reload during concurrent edits); they are setup failures, not
timing samples. The accepted page used a 30-second launch-latency allowance so
browser-control overhead could not invalidate an otherwise immediate click.
