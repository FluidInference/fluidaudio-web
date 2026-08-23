import { describe, expect, it } from "vitest";

import { aceRuntimePackageFiles } from "../src/model/acquire.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  type AcePackageFileRecord,
  type AcePackageManifest,
} from "../src/model/manifest.js";
import {
  createAceMainAcquisitionManifest,
  createAceOpt0011VaeAcquisitionManifest,
} from "../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES,
  isAceReferenceDitLayerWeightFile,
} from "../src/webgpu/dit-fp16-package.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../src/webgpu/vae-fp16-package.js";

describe("production dual-manifest acquisition", () => {
  it("excludes reference VAE-only and replaced DiT-layer shards", () => {
    const manifest = syntheticReferenceManifest();
    const acquisition = createAceMainAcquisitionManifest(manifest);
    const selected = new Set(
      aceRuntimePackageFiles(acquisition).map((file) => file.name),
    );
    const phasesByShard = new Map<string, Set<string>>();
    for (const tensor of Object.values(manifest.tensors)) {
      const phases = phasesByShard.get(tensor.shard) ?? new Set<string>();
      phases.add(tensor.phase);
      phasesByShard.set(tensor.shard, phases);
    }
    const vaeOnly = [...phasesByShard]
      .filter(([, phases]) => phases.size === 1 && phases.has("vae"))
      .map(([name]) => name);

    expect(vaeOnly.length).toBeGreaterThan(0);
    expect(vaeOnly.every((name) => !selected.has(name))).toBe(true);
    expect(
      Object.values(manifest.tensors)
        .filter((tensor) => tensor.phase !== "vae")
        .every((tensor) =>
          isAceReferenceDitLayerWeightFile(tensor.shard) ||
          selected.has(tensor.shard)
        ),
    ).toBe(true);
    expect(
      ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES.every((name) => !selected.has(name)),
    ).toBe(true);
    expect(acquisition.tensors).toBe(manifest.tensors);
  });

  it("re-resolves exactly seven original revision-5 VAE records", () => {
    const manifest = syntheticVaeManifest();
    const acquisition = createAceOpt0011VaeAcquisitionManifest(manifest);

    expect(acquisition.files.map((file) => file.name)).toEqual(
      ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
    );
    expect(acquisition.files).toHaveLength(7);
    expect(
      acquisition.files.reduce((sum, file) => sum + file.byteLength, 0),
    ).toBe(168_791_552);
    for (const file of acquisition.files) {
      expect(file).toBe(
        manifest.files.find((candidate) => candidate.name === file.name),
      );
    }
    expect(acquisition.tensors).toBe(manifest.tensors);
  });
});

function syntheticReferenceManifest(): AcePackageManifest {
  const layerFiles = ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES.map(
    (name, index) => file(
      name,
      index % 2 === 0 ? 121_668_608 : 4_198_400,
    ),
  );
  return {
    profile: "reference",
    files: [
      ...layerFiles,
      file("weights/vae-only.bin", 64),
      file("weights/shared.bin", 128),
      file("weights/dit.bin", 256),
      file("assets/qwen/tokenizer.json", 32, "upstream-asset"),
    ],
    tensors: {
      ...Object.fromEntries(layerFiles.map((record, index) => [
        `ace.decoder.layers.${Math.floor(index / 2)}.fixture-${index}`,
        tensor("dit", record.name),
      ])),
      "vae.only": tensor("vae", "weights/vae-only.bin"),
      "vae.shared": tensor("vae", "weights/shared.bin"),
      "dit.shared": tensor("dit", "weights/shared.bin"),
      "dit.only": tensor("dit", "weights/dit.bin"),
    },
  } as unknown as AcePackageManifest;
}

function syntheticVaeManifest(): AcePackageManifest {
  const byteLengths = [
    117_469_184,
    16_795_648,
    25_189_376,
    5_254_656,
    1_316_608,
    922_880,
    1_843_200,
  ] as const;
  const files = ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.map((name, index) =>
    file(name, byteLengths[index]!)
  );
  return {
    profile: "fp16-vae-experimental",
    provenance: {
      converterRevision: ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    },
    files,
    tensors: Object.fromEntries(
      files.map((record, index) => [
        `vae.fixture-${index}`,
        tensor("vae", record.name),
      ]),
    ),
  } as unknown as AcePackageManifest;
}

function file(
  name: string,
  byteLength: number,
  kind: AcePackageFileRecord["kind"] = "weights",
): AcePackageFileRecord {
  return Object.freeze({
    name,
    byteLength,
    sha256: "a".repeat(64),
    kind,
  });
}

function tensor(phase: string, shard: string) {
  return { phase, shard };
}
