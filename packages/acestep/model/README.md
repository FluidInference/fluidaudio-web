# Reproducing the ACE-Step browser model packages

This directory is the sole model-preparation workspace for
`ace-step-1.5.wgsl`. `convert.py` downloads the immutable upstream artifacts,
checks their byte identities and complete safetensor inventories, selects the
v1 text-to-music graph, writes bounded runtime shards, creates a deterministic
manifest, verifies the result independently, and atomically installs it. No
notebook, manually edited weight, or unrecorded shell step is part of the
package recipe.

The structure deliberately follows `../parakeet.wgsl/model`, pinned at
Parakeet commit `7ee112738262a6f5a0efd2f150748a4087432fbb`. ACE-Step has a
larger staged graph, so its source contracts and phase-oriented shard plan are
specific to this project.

Generated weights are not committed. The ordinary test suite uses synthetic
safetensor and package fixtures and does not download model data.

## Requirements and disk budget

- [`uv`](https://docs.astral.sh/uv/) and CPython 3.13
- approximately 35–40 GB free while keeping the verified source cache, the
  three-package production tuple, and one transactional staging directory
- a stable network connection for the first source download

The environment requires Python `>=3.13,<3.14`. Its direct dependencies are
exact pins (`numpy==2.4.6` and `requests==2.32.5`) and `uv.lock` captures the
complete resolution. Always use `uv run --frozen`; a dependency or transform
change can alter package bytes and requires a new converter revision and new
canonical manifest identities.

## One-command generation

Run from the repository root:

```bash
# Exact current browser-production tuple (recommended for a fresh clone).
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile production

# Correctness-first BF16-bit-preserving package (the default).
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile reference

# Raw FP16 candidate package.
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile fp16

# Produce both revision-4 development profiles from the verified cache.
# This is not the optimized production tuple.
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile all

# OPT-0037 revision-8 mixed DiT layer package. This atomically replaces only
# the authenticated revision-7 package after complete staging verification.
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile fp16-dit-dense-experimental --offline
```

`--profile production` downloads and authenticates the pinned upstream files,
then deterministically installs exactly the three packages selected by the
demo:

| Component | Directory | Converter revision | Manifest SHA-256 |
| --- | --- | ---: | --- |
| reference/shared | `model/files-reference` | 4 | `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6` |
| production DiT | `model/files-fp16-dit-rev7-oracle` | 7 | `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f` |
| production VAE | `model/files-fp16-vae-revision7-experimental` | 7 | `36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7` |

Each staged manifest must match its committed production SHA-256 before the
package can be installed. To regenerate just one optimized production package,
use `--profile production-dit-rev7` or
`--profile production-vae-rev7`. These component selectors retain
`--output-dir` support; the complete `production` selector does not because its
three output paths are part of the browser setup contract.

The development defaults are `model/cache/`, `model/files-reference/`, and
`model/files-fp16/`. `--cache-dir` relocates downloads. `--output-dir`
relocates one package and is intentionally incompatible with multi-package
profiles.
After a complete first download, `--offline` proves that generation uses only
the pinned cache. To audit cached source bytes and headers without generating
a package:

```bash
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --offline --verify-cache-only
```

Downloads stream in bounded chunks. An interrupted partial remains resumable;
a complete bad-length or bad-hash download is removed. A cached file is never
trusted solely because its filename exists. Conversion reads large tensor
payloads through read-only memory maps and bounded regions, not one complete
Python model object.

Each target is built in a hidden sibling directory. The converter verifies its
manifest, every file hash, tensor bounds, and alignment before using atomic
renames. It refuses symlink targets and refuses to overwrite a nonempty
directory that is not already a recognized package. If installation fails
after moving an existing valid package aside, it restores the prior package.

## Immutable source contract

The model snapshots are permanently fixed to:

```text
ACE-Step/Ace-Step1.5
19671f406d603126926c1b7e2adc169acbcade22

ACE-Step/acestep-5Hz-lm-0.6B
148d8ea0225bdab342ee1ae3a354275ccd60ca80
```

Reference behavior is fixed to upstream source commit:

```text
https://github.com/ace-step/ACE-Step-1.5.git
6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
```

The large files are:

| Source | Bytes | SHA-256 | Tensors | Inventory SHA-256 |
| --- | ---: | --- | ---: | --- |
| ACE Turbo | 4,787,825,604 | `3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7` | 677 | `0befc4e612073b1bd27bee7d98fa119aa96a6a05c9ebd6c1952e4796ab60e33d` |
| Qwen embedding | 1,191,586,416 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` | 310 | `7b21319da9ddf0ee1bef76ac0b6b8b5afa3ad92d4fffe520416e3904563f3a76` |
| Oobleck VAE | 337,431,388 | `da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0` | 365 | `71b0ccfc5236b9aba8593b5ccece3367714b603f1464a444447e01cc6f467c4b` |
| Silence latent | 3,841,215 | `a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b` | — | — |
| 0.6B planner | 1,325,804,024 | `5d92a60806e2e88c04de58ddc6dde93f2bc8f1336162b3ad5853886c9bcc6b82` | 310 | `447349101df0840017aa014a63e2989246caa81430beb9ecb8724feabff5dc96` |

That is **1,662 source tensors**. The contract is derived from and checks the
four complete headers. Header
length and header SHA-256 are also fixed in `source_contract.py`, so a rename,
shape change, dtype change, reordered/changed offset, or metadata/header change
is rejected before conversion.

The same contract pins every required configuration and tokenizer asset by
length and SHA-256. Both text and planner sets include `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, `merges.txt`, added and special tokens,
and the exact chat template. These are needed to preserve upstream prompt
formatting, planner Chain-of-Thought metadata generation, multilingual token
behavior, and constrained semantic decoding. The upstream model cards and
license/notice files are copied into every package.

## Current package recipe

Every source tensor receives exactly one disposition in
`conversion-plan.json`: included, consumed by a named transform, or excluded
with a committed reason.

- All 310 planner tensors and all 310 text-embedding tensors are retained.
- The ACE condition encoder, semantic detokenizer, decoder-side
  `tokenizer.quantizer.project_out`, and complete 24-layer Turbo decoder are
  retained. In pinned upstream, direct text-to-music computes a tokenizer and
  detokenizer result from the silence latent and then discards it through the
  all-false `is_covers` selection. The scoped browser graph removes that dead
  computation; planner-provided codes still use `project_out` and the complete
  detokenizer.
- Exactly 30 encode-only ACE tensors are excluded: all tensors below
  `tokenizer.audio_acoustic_proj`, `tokenizer.attention_pooler`, and
  `tokenizer.quantizer.project_in`. They total 210,035,724 source bytes /
  105,017,862 BF16 parameters. `quantizer.project_out` and every
  `detokenizer.*` tensor remain packaged. `null_condition_emb` is also excluded;
  Turbo's ordinary no-CFG inference path does not use the training-time dropped
  condition.
- The VAE encoder is excluded; all decoder tensors are retained.
- VAE weight normalization is fused offline and all decoder weights remain
  FP32 in both profiles. The versioned fusion recipe uses an explicit adjacent-
  pair FP32 norm-reduction tree and FP32 `g * v / (norm + 1e-9)` operations;
  it does not delegate package-defining reductions to platform BLAS. Fused
  Conv1d tensors are transposed from source `[out,in,kernel]` to runtime
  `[out,kernel,in]`, fused ConvTranspose1d tensors from source
  `[in,out,kernel]` to the same runtime order, and Snake alpha/beta tensors are
  flattened from `[1,C,1]` to `[C]`. Their manifest layouts and transformations
  carry explicit `v1` identifiers.
- The `reference` profile preserves every other BF16 payload bit and exposes
  it as aligned packed BF16 pairs for shader-side FP32 decode.
- The `fp16` profile deterministically converts those eligible BF16 values to
  IEEE FP16. FP32-sensitive runtime arithmetic remains a shader/runtime policy,
  not a weight-package claim.

Output is grouped by runtime phase and transformer/VAE block. Files are capped
at 120 MiB, below the portable 128 MiB storage-binding limit. Individual large
embeddings and VAE matrices are split only after conversion, along runtime axis
zero, and all tensor starts are 256-byte aligned. ConvTranspose1d shards cover
output-channel ranges rather than the source tensor's input-channel axis. The
manifest records runtime logical and storage shapes, layout, phase/lifetime,
source tensor, and transformation for every output span. The
approximately 1–4 MiB transport chunks used later by the browser are separate
from these logical GPU-binding shards.

Converter revision 4 stores the exact 271 rank-two DiT GEMM weights in the
operation-native scalar order `[N/128,K/32,32,128]`. The packed-BF16 reference
and raw-FP16 profiles share that physical layout; only their scalar dtype and
transformation differ. Each eligible matrix is one complete logical tensor,
so the portable and fixed-32 subgroup DiT kernels consume the same bytes
without runtime repacking, package expansion, or duplicate GPU residency.
Other transformer tensors retain their established layouts, alongside the
operation-native VAE layouts required by the decoder. Future layout or
precision changes increment the converter revision and remain reproducible
here. Lossy compression is deferred and will extend the same pipeline rather
than create another one.

Converter revision 8 keeps the focused 24-layer mixed DiT package and replaces
only its 216 repeated dense FP16 matrices with the OPT-0032 physical order
`[N/128,K/4,output4,lane32,K4]`. The remaining 240 layer tensors retain their
revision-7 reference-BF16 support layouts. Each K4 matrix remains one complete
logical tensor and occupies exactly the same number of bytes as its logical
FP16 matrix; the package never carries the old and new dense copies together.
The default target is
`model/files-fp16-dit-layer-mixed-experimental`. Only the committed, fully
re-hashed predecessor identity may occupy that target during an atomic
replacement; no older layout is accepted as a revision-8 package fallback.

Production intentionally retains the separately selectable converter-revision-7
DiT layout `[N/256,K/32,K-in-tile,N-in-tile]`. Revision 8 failed OPT-0037's
trajectory gate and is not a production substitute. The production selector
therefore restores the exact revision-7 bytes under their own output directory
instead of weakening the rev8 gate or silently changing runtime math.

## Tests

Run the focused no-download suite:

```bash
uv run --frozen --project model --python 3.13 \
  python3 -m unittest discover -s model/tests -v
```

It checks source revisions and tokenizer contracts, exact encode-only tensor
accounting, strict safetensor parsing, complete tensor disposition logic, BF16
conversion, element-by-element VAE layout mappings, runtime-axis row splitting,
alignment, deterministic manifests, hash rejection, future-format rejection,
and safe transactional-target behavior. After both revision-4 packages have
been generated, run the opt-in complete source/package hash audit with:

```bash
ACE_STEP_VERIFY_FULL_PACKAGE=1 \
  uv run --frozen --project model --python 3.13 \
  python3 -m unittest discover -s model/tests -v
```

After generating the production tuple, audit all sources, package files, and
the three exact manifest identities with:

```bash
ACE_STEP_VERIFY_PRODUCTION_PACKAGES=1 \
  uv run --frozen --project model --python 3.13 \
  python3 -m unittest model.tests.test_full_package -v
```

## Canonical package identities

Both profiles have been generated twice from the verified offline source cache
with identical manifest hashes, and the installed copies passed the opt-in full
audit. The machine-readable record is
[`canonical-packages.json`](./canonical-packages.json).

| Profile | Manifest SHA-256 | Total package bytes |
| --- | --- | ---: |
| packed-BF16 reference | `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6` | 7,500,802,986 |
| raw FP16 candidate | `c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b` | 7,500,757,041 |

The exact browser-production tuple is recorded separately under
`productionPackages` in `canonical-packages.json`. `replacementPackages`
contains only authenticated package identities that may already occupy an
experimental target during an atomic upgrade; those entries are not the
production output identities.

Each manifest accounts for 1,662 source tensors, 1,412 logical package
tensors including the extracted silence latent, 1,420 records after row
sharding, 163 listed files, and 138 weight shards. Package byte totals include
the manifest, immutable tokenizer/config/license assets, shard alignment, and
tensor payloads. Generated package directories remain ignored and are not
committed.

## Licenses

ACE-Step source and the pinned ACE model cards declare MIT licensing. Qwen3
components are Apache-2.0; applicable attribution, NOTICE, patent, and license
terms must be preserved. `manifest.json` carries structured license metadata,
and packages include the pinned ACE model cards plus the committed ACE license,
complete Apache License 2.0 text, and Qwen notice. Consult upstream terms before
redistributing generated model packages; generated weights remain separate
from this source repository.
