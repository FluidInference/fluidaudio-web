# OPT-0033 — DiT query8 K/V key blocking

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: exact relative to the current query8 arithmetic and key order

## Hypothesis

OPT-0030 proved that widening query8 to WG512/WG1024 loses more occupancy and
synchronization than it saves in K/V loads. The current winning WG256 query8
still stages one 128-wide K/V row and executes two workgroup barriers for every
key: about 4,500 barriers per workgroup at M2250. Stage a small block of keys,
then let each subgroup execute the same ascending-key online-softmax updates
without modifying shared storage until the block is complete.

## Frozen screen

Compare current row-staged query8 with two otherwise identical WG256 arms:

- key-block 8: shared K/V `[8,128]` each, 8 KiB total;
- key-block 16: shared K/V `[16,128]` each, 16 KiB total.

The 256 lanes cooperatively load the complete block, barrier once, every
subgroup processes valid keys in the original ascending order using the same
FP32 QK expression, `subgroupAdd`, online max/denominator updates, and weighted
value updates, then barrier once before overwriting the block. Tail keys are
guarded without adding arithmetic updates. Query ownership, GQA sharing,
workgroup count, output layout, and all storage types remain query8. This cuts
barriers from `2 * 2250` to `2 * ceil(2250/block)` without widening the group
or changing K/V scalar load count.

This is not OPT-0030: no additional query streams or threads are added. It is
the missing key-axis blocking component of FlashAttention-style execution,
not merely online softmax.

## Gate

Require bit-exact equality to current query8 across all 4,608,000 M2250 F32
outputs, complete finite writes, stock Chrome/WebGPU, clean lifecycle, and one
balanced nominal timing screen. Continue only if the best blocked arm is at
least `1.35x` faster than query8. No production or product claim follows from
the primitive screen.

## Result

Stock Chrome/M3 compiled and ran both WG256 candidates. Query8, key-block8,
and key-block16 produced the same SHA-256
`2882495eccd1f1971e998b957dfdd12ee517a2019a6ddccd9d76e38bb81c9892`.
Each candidate compared all `4,608,000` outputs against query8: `9,216,000`
F32 comparisons total, zero bit mismatches, zero non-finite values, complete
writes, and intact canaries.

After the 30-second nominal gate, three samples per arm produced medians:

| Arm | Barriers/workgroup | Median | Speedup vs query8 |
| --- | ---: | ---: | ---: |
| query8 | 4,500 | 108.5 ms | 1.000x |
| key-block8 | 564 | 125.0 ms | 0.868x |
| key-block16 | 282 | 119.7 ms | 0.90643x |

The best blocked arm missed the `1.35x` threshold and was slower than control.
Therefore per-key barriers are not the decisive query8 bottleneck on this
compiler/GPU; increased shared-memory footprint, block indexing, or occupancy
cost outweighs the nominal barrier reduction. Stop without integration. Do not
retry unchanged key blocking unless the browser compiler or GPU changes.

The canonical receipt is
[`optimization/results/OPT-0033/result.json`](../results/OPT-0033/result.json),
SHA-256 `77ce04b318e0714fdc7577a49a12c0a8f275e397f4283318da97c644f7c129f0`.
All six buffers and the device were destroyed. One earlier click was rejected
for missing thermal form fields before timed GPU dispatch and is not a timing
sample; the accepted gate records a 30.062-second nominal interval.
