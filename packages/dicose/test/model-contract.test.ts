import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { parseModelManifest, type DiCoSeTensorManifest } from "../src/model/manifest.js";
import { DICOSE_BANDS } from "../src/runtime/bs-roformer.js";

const ROOT = resolve(import.meta.dirname, "..");
const PACKAGE_MANIFEST = resolve(ROOT, "public/model/manifest.json");
const REFERENCE_CONTRACT = resolve(ROOT, "test/fixtures/deterministic-reference.json");

interface DeterministicReferenceContract {
  readonly schema: string;
  readonly fixture: Readonly<{
    readonly sha256: string;
    readonly preprocessing: Readonly<{
      readonly frames: number;
      readonly stft: Readonly<{
        readonly frames: number;
        readonly nFft: number;
        readonly hopLength: number;
      }>;
    }>;
  }>;
  readonly reference: Readonly<{
    readonly stems: Readonly<Record<string, Readonly<{ readonly rms: number; readonly peak: number }>>>;
  }>;
}

const contract = JSON.parse(readFileSync(REFERENCE_CONTRACT, "utf8")) as DeterministicReferenceContract;

describe("DiCoSe deterministic reference contract", () => {
  it("pins the supplied WAV preprocessing and non-zero expected stem scales", () => {
    expect(contract.schema).toBe("dicose-deterministic-reference-v1");
    expect(contract.fixture.sha256).toBe("9e487f3a84b974b11b47442d0fd99512ab4826130d04351e8c9625d84e107bb7");
    expect(contract.fixture.preprocessing.frames).toBe(524_288);
    expect(contract.fixture.preprocessing.stft).toMatchObject({ frames: 1_189, nFft: 2_048, hopLength: 441 });
    for (const name of ["drums", "bass", "other"] as const) {
      expect(contract.reference.stems[name]?.rms).toBeGreaterThan(0.04);
      expect(contract.reference.stems[name]?.peak).toBeGreaterThan(0.3);
    }
    expect(contract.reference.stems.vocals?.rms).toBeLessThan(0.001);
  });
});

describe("DiCoSe WebGPU tensor contract", () => {
  const packageTest = existsSync(PACKAGE_MANIFEST) ? it : it.skip;

  packageTest("covers every tensor family consumed by the raw WGSL graph", () => {
    const manifest = parseModelManifest(
      JSON.parse(readFileSync(PACKAGE_MANIFEST, "utf8")),
    );
    expect(manifest.tensors).toHaveLength(2_857);
    expect(manifest.config.freqsPerBands).toEqual(DICOSE_BANDS);
    expect(DICOSE_BANDS).toHaveLength(62);
    expect(DICOSE_BANDS.reduce((total, band) => total + band, 0)).toBe(1_025);

    const tensors = new Map(manifest.tensors.map((tensor) => [tensor.name, tensor]));
    assertSeparatorTensors(tensors, "det", false);
    assertSeparatorTensors(tensors, "cd", true);
    assertCdConditioningTensors(tensors);
  });
});

function assertSeparatorTensors(
  tensors: ReadonlyMap<string, DiCoSeTensorManifest>,
  prefix: "det" | "cd",
  hasMapping: boolean,
): void {
  for (let band = 0; band < DICOSE_BANDS.length; band += 1) {
    const width = DICOSE_BANDS[band]! * 4;
    const base = `${prefix}.band_split.to_features.${band}`;
    assertTensor(tensors, `${base}.0.gamma`, [width], "row-major");
    assertTensor(tensors, `${base}.1.weight`, [width, 384], "linear-in-out");
    assertTensor(tensors, `${base}.1.bias`, [384], "row-major");
  }

  for (let layer = 0; layer < 8; layer += 1) {
    for (const axis of [0, 1] as const) {
      const base = `${prefix}.layers.${layer}.${axis}.layers.0`;
      assertTensor(tensors, `${base}.0.norm.gamma`, [384], "row-major");
      assertTensor(tensors, `${base}.0.to_qkv.weight`, [384, 1_536], "linear-in-out");
      assertTensor(tensors, `${base}.0.to_gates.weight`, [384, 8], "linear-in-out");
      assertTensor(tensors, `${base}.0.to_gates.bias`, [8], "row-major");
      assertTensor(tensors, `${base}.0.to_out.0.weight`, [512, 384], "linear-in-out");
      assertTensor(tensors, `${base}.1.net.0.gamma`, [384], "row-major");
      assertTensor(tensors, `${base}.1.net.1.weight`, [384, 1_536], "linear-in-out");
      assertTensor(tensors, `${base}.1.net.1.bias`, [1_536], "row-major");
      assertTensor(tensors, `${base}.1.net.4.weight`, [1_536, 384], "linear-in-out");
      assertTensor(tensors, `${base}.1.net.4.bias`, [384], "row-major");
      if (hasMapping) {
        const scaleShift = `${base}.0.to_scale_shift.to_scale_shift.1`;
        assertTensor(tensors, `${scaleShift}.weight`, [1_536, 768], "linear-in-out");
        assertTensor(tensors, `${scaleShift}.bias`, [768], "row-major");
        const ffScaleShift = `${base}.1.to_scale_shift.to_scale_shift.1`;
        assertTensor(tensors, `${ffScaleShift}.weight`, [1_536, 768], "linear-in-out");
        assertTensor(tensors, `${ffScaleShift}.bias`, [768], "row-major");
      }
    }
  }

  assertTensor(tensors, `${prefix}.final_norm.gamma`, [384], "row-major");
  for (let stem = 0; stem < 4; stem += 1) {
    for (let band = 0; band < DICOSE_BANDS.length; band += 1) {
      const width = DICOSE_BANDS[band]! * 4;
      const base = `${prefix}.mask_estimators.${stem}.to_freqs.${band}.0`;
      assertTensor(tensors, `${base}.0.weight`, [384, 768], "linear-in-out");
      assertTensor(tensors, `${base}.0.bias`, [768], "row-major");
      assertTensor(tensors, `${base}.2.weight`, [768, width * 2], "linear-in-out");
      assertTensor(tensors, `${base}.2.bias`, [width * 2], "row-major");
    }
  }
}

function assertCdConditioningTensors(tensors: ReadonlyMap<string, DiCoSeTensorManifest>): void {
  assertTensor(tensors, "cd.stem_embedding.weight", [4, 1_536], "row-major");
  assertTensor(tensors, "cd.to_time.0.1.weight", [384, 1_536], "linear-in-out");
  assertTensor(tensors, "cd.to_time.0.1.bias", [1_536], "row-major");
  assertTensor(tensors, "cd.to_mapping.0.weight", [1_536, 1_536], "linear-in-out");
  assertTensor(tensors, "cd.to_mapping.0.bias", [1_536], "row-major");
  assertTensor(tensors, "cd.to_mapping.2.weight", [1_536, 1_536], "linear-in-out");
  assertTensor(tensors, "cd.to_mapping.2.bias", [1_536], "row-major");

  assertTensor(tensors, "cd.stft_feature_adapter.1.weight", [128, 4, 3, 3], "conv-oihw");
  assertTensor(tensors, "cd.stft_feature_adapter.1.bias", [128], "row-major");
  assertTensor(tensors, "cd.stft_feature_adapter.3.weight", [128, 128, 1, 1], "conv-oihw");
  assertTensor(tensors, "cd.stft_feature_adapter.5.weight", [128, 128, 1, 1], "conv-oihw");
  assertTensor(tensors, "cd.stft_feature_adapter.7.weight", [4, 128, 3, 3], "conv-oihw");
  assertTensor(tensors, "cd.band_split_feature_adapter.0.weight", [384, 384], "linear-in-out");
  assertTensor(tensors, "cd.band_split_feature_adapter.2.weight", [384, 384], "linear-in-out");
  for (let adapter = 0; adapter < 16; adapter += 1) {
    assertTensor(tensors, `cd.transformer_feature_adapters.${adapter}.0.weight`, [384, 384], "linear-in-out");
    assertTensor(tensors, `cd.transformer_feature_adapters.${adapter}.2.weight`, [384, 384], "linear-in-out");
  }
}

function assertTensor(
  tensors: ReadonlyMap<string, DiCoSeTensorManifest>,
  name: string,
  shape: readonly number[],
  layout: DiCoSeTensorManifest["layout"],
): void {
  const tensor = tensors.get(name);
  expect(tensor, `missing tensor ${name}`).toBeDefined();
  expect(tensor?.shape, `shape for ${name}`).toEqual(shape);
  const expectedLayout = layout === "linear-in-out" ? packedLinearLayout(shape) : layout;
  expect(tensor?.layout, `layout for ${name}`).toBe(expectedLayout);
}

function packedLinearLayout(shape: readonly number[]): DiCoSeTensorManifest["layout"] {
  const [inner, columns] = shape;
  if (shape.length !== 2 || inner === undefined || columns === undefined || inner % 32 !== 0) {
    return "linear-in-out";
  }
  if (columns % 256 === 0) return "linear-tile-n256-k32";
  if (columns % 128 === 0) return "linear-tile-n128-k32";
  return "linear-in-out";
}
