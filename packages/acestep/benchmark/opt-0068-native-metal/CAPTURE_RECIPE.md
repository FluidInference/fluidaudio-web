# OPT-0068 actual-M2250 fixture capture recipe

No usable activation bytes were present when the Phase-1 harness was written.
OPT-0067 retained exact attention-output hashes and comparison counters, not
the FP32 inputs and outputs of the nine repeated-layer dense primitives.  This
recipe is therefore a required future step.  Never replace it with zeros,
pseudorandom data, a 12-second shape, or an activation reconstructed by the
native candidate.

There is deliberately no runnable OPT-0068 capture URL yet.  The existing
OPT-0067 page cannot emit these tensors, so running it again does not clear the
blocker.  Before using the commands below, a separately reviewed, isolated
`test/browser/opt-0068-native-dense-fixture.{html,ts}` plus worker/contract must
implement this recipe from a pushed checkpoint.  It may add an explicit
benchmark-only observation seam, but must not alter runtime selection or the
ordinary production path.

Once those reviewed files exist, serve them only from the repository root:

```sh
pnpm exec vite --host 127.0.0.1
```

The capture page's frozen route is
`http://127.0.0.1:5173/test/browser/opt-0068-native-dense-fixture.html`.
It must reuse `parseOpt0018RunIdentity`, so the URL must supply `coreCommit`,
`harnessCommit`, `machineModel=Mac15%2C12`, `osVersion=26.5.2`,
`osBuild=25F84`, the URL-encoded full Chrome version, `gpuCoreCount=10`, and
`memoryBytes=17179869184`.  Until that route exists and is reviewed, do not run
the native `inspect`, `measure`, or `sustained` commands with substitute data.

## Frozen capture

Start from a pushed OPT-0068 checkpoint.  Add a benchmark-only capture owner;
do not select it from the demo or alter the production profile.  It must fail
closed unless all of these are exact:

- main manifest
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`;
- revision-7 DiT manifest
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`;
- `ace-turbo-v1-correctness`, direct 180 seconds, planner disabled, shift-3
  Euler with eight evaluations, Haar DCW `double`, strengths `0.05/0.02`;
- batch 1, M2250, C98, evaluation 0, layer 0; and
- the pinned request/token/seed identities already enforced by OPT-0018 and
  OPT-0067.

The exact canonical request is 366 UTF-8 bytes with SHA-256
`031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f`:

```json
{"generationProfile":"ace-turbo-v1-correctness","prompt":"Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.","lyrics":"","instrumental":true,"durationSeconds":180,"seed":"0000000000c0ffee","planner":{"mode":"disabled"},"metadata":{"bpm":104,"keyScale":"D minor","timeSignature":"4"}}
```

Use the ordinary exact dense owner
`opt-0009-fp16-fp32-dense-v1`.  The capture target is precisely the graph
label prefix `ace-dit-eval-0-layer-0`.  At the construction seam in
`src/webgpu/ace-dit.ts`, the six input bindings are `s.selfModulated`,
`s.selfMergedAttention`, `s.crossNormalized`, `s.crossMergedAttention`,
`s.mlpModulated`, and `s.gatedActivation`; the nine corresponding output
bindings are `s.selfQueryFlat`, `s.selfKeyFlat`, `s.selfValueFlat`,
`s.selfProjectedAttention`, `s.crossQueryFlat`, `s.crossProjectedAttention`,
`s.gate`, `s.up`, and `s.projectedMlp`.  Capture only this evaluation/layer,
and assert every expected label exactly once.

Capture these six producer outputs immediately before their first listed dense
consumer.  Preserve the GPU buffer's FP32 little-endian bytes; native operand
rounding happens later in the harness.

| activation ID | shape | dense consumers |
| --- | ---: | --- |
| `self-modulated` | `2250x2048` | self query/key/value |
| `self-merged-attention` | `2250x2048` | self output |
| `cross-normalized` | `2250x2048` | cross query |
| `cross-merged-attention` | `2250x2048` | cross output |
| `mlp-modulated` | `2250x2048` | MLP gate/up |
| `mlp-gated-activation` | `2250x6144` | MLP down |

Capture the nine dense outputs immediately after the accepted OPT-0009
FP16-input/FP32-accumulate primitive: `self-query`, `self-key`, `self-value`,
`self-output`, `cross-query`, `cross-output`, `mlp-gate`, `mlp-up`, and
`mlp-down`.  These are the accepted-WebGPU comparator files.

Also write the ordinary evaluation-0 result as
`evaluation-0-result.f32le`.  It is exactly 1,152,000 bytes / 288,000 FP32
elements and must hash to
`d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286`,
the accepted OPT-0067 identity.  A mismatch rejects the entire capture even if
all 15 dense files look plausible.

Use one reusable readback allocation, no larger than the largest 55,296,000
byte tensor.  Drain before mapping, stream each mapped range directly to a
transactional file, unmap, and reuse the allocation.  Do not retain a second
copy in JavaScript.  Capture is untimed and may add explicit drains; retain the
ordinary evaluation-0 result hash and require it to match the identical-request
OPT-0067 oracle so the observation path cannot silently perturb math.

For every file record shape, exact byte length, SHA-256, finite/nonzero counts,
minimum/maximum, and bounded diagnostic slices.  Write
`ace-opt-0068-m2250-native-fixture-v1` JSON conforming to
`fixture.schema.json`, including the capture commit and a SHA-256 over every
source file controlling the seam.  Raw tensor files remain ignored.

The capture owner must expose a user-selected output directory and stream each
mapped range transactionally to that directory; it must not collect the files
inside a page array, Blob bundle, or worker message.  Its receipt must list the
canonical request hash, all 16 file names/lengths/hashes, exact label counts,
evaluation-result hash, final live buffer/map counts, capture commit, and the
aggregate source authority used as `--capture-source-sha256` below.  The
aggregate must cover the page, worker, contract, and every runtime source file
that controls the added observation seam.

Name the files `activation-<activation ID>.f32le` and
`output-<output ID>.f32le`, then assemble the manifest without a manual binary
or JSON editing step.  The generator independently scans every FP32 word,
rejects non-finite or all-zero files, records exact finite/nonzero counts and
range, records eight-word head/tail bit slices, and authenticates the frozen
evaluation result:

```sh
uv run --python 3.13 \
  python benchmark/opt-0068-native-metal/make_fixture_manifest.py \
  --capture-directory benchmark/artifacts/OPT-0068/actual-m2250 \
  --capture-commit OPT0068_CAPTURE_COMMIT \
  --capture-source-sha256 OPT0068_CAPTURE_SOURCE_SHA256 \
  --output benchmark/artifacts/OPT-0068/actual-m2250/fixture.json
```

## Independent verification

From the repository root, before any native device is created:

```sh
swift run --package-path benchmark/opt-0068-native-metal ace-opt-0068-mps \
  --mode inspect \
  --package-dir model/files-fp16-dit-rev7-oracle \
  --fixture-manifest benchmark/artifacts/OPT-0068/actual-m2250/fixture.json
```

The inspector hashes the revision-7 manifest, both touched layer-0 shards, all
16 captured files, recomputes their FP32 evidence, validates the exact request,
package, capture, and evaluation identities, and proves the
4/2/2/1 operation mix.  A missing file or mismatched byte rejects the fixture.
