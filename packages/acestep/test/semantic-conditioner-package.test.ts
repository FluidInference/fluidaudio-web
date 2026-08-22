import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import type { AceGpuLogicalTensor } from "../src/model/gpu-tensors.js";
import type { AcePackageManifest } from "../src/model/manifest.js";
import {
  ACE_DIRECT_CONDITIONER_LOGICAL_TENSOR_NAMES,
  ACE_SEMANTIC_LOGICAL_TENSOR_NAMES,
  ACE_UNUSED_TIMBRE_SPECIAL_TOKEN,
  aceConditionerExpectedLogicalShape,
  resolveAceDirectConditionerPackage,
  resolveAceSemanticPackageWeights,
  validateAceConditionerPackageInventory,
  type AceConditionerTensorResolver,
} from "../src/webgpu/semantic-conditioner-package.js";

const LOCAL_MANIFEST_URLS = Object.freeze({
  reference: new URL("../model/files-reference/manifest.json", import.meta.url),
  fp16: new URL("../model/files-fp16/manifest.json", import.meta.url),
});
const HAS_LOCAL_CANONICAL_MANIFESTS = Object.values(LOCAL_MANIFEST_URLS)
  .every((url) => existsSync(url));

describe("ACE semantic/direct conditioner package graph", () => {
  it("resolves every learned weight, silence constant, and audited dead parameter", () => {
    const resolver = syntheticResolver("raw-fp16");
    const semantic = resolveAceSemanticPackageWeights(resolver, "raw-fp16");
    const direct = resolveAceDirectConditionerPackage(resolver, "raw-fp16");
    expect(semantic.layers).toHaveLength(2);
    expect(direct.weights.lyricLayers).toHaveLength(8);
    expect(direct.weights.timbreLayers).toHaveLength(4);
    expect(direct.silenceSource.size).toBe(3_840_000);
    expect(direct.auditedUnusedTimbreSpecialToken).toEqual(
      resolver.binding(ACE_UNUSED_TIMBRE_SPECIAL_TOKEN),
    );
  });

  it("rejects changed logical shapes before graph assembly", () => {
    const resolver = syntheticResolver("raw-fp16", {
      logicalShapeOverride: new Map([
        [ACE_SEMANTIC_LOGICAL_TENSOR_NAMES[0]!, [1]],
      ]),
    });
    expect(() => resolveAceSemanticPackageWeights(resolver, "raw-fp16")).toThrow(
      /has shape/,
    );
  });

  it("rejects a package profile mismatch instead of decoding wrong storage", () => {
    expect(() => resolveAceSemanticPackageWeights(
      syntheticResolver("reference-bf16"),
      "raw-fp16",
    )).toThrow(/violates its graph contract/);
  });

  it("fails closed on unknown logical names", () => {
    expect(() => aceConditionerExpectedLogicalShape("ace.hidden.weight"))
      .toThrow(/Unknown ACE conditioner tensor/);
  });
});

describe.skipIf(!HAS_LOCAL_CANONICAL_MANIFESTS)(
  "ACE generated semantic/direct conditioner packages",
  () => {
    it.each(["reference", "fp16"] as const)(
      "authenticates the complete %s phase inventories",
      (profile) => {
        const manifest = loadManifest(profile);
        expect(() => validateAceConditionerPackageInventory(manifest)).not.toThrow();
        const semantic = Object.values(manifest.tensors)
          .filter((tensor) => tensor.phase === "semantic");
        const conditioner = Object.values(manifest.tensors)
          .filter((tensor) => tensor.phase === "conditioner");
        expect(semantic).toHaveLength(30);
        expect(conditioner).toHaveLength(140);
        expect(new Set(semantic.map((tensor) => tensor.logicalTensor))).toEqual(
          new Set(ACE_SEMANTIC_LOGICAL_TENSOR_NAMES),
        );
        expect(new Set(conditioner.map((tensor) => tensor.logicalTensor))).toEqual(
          new Set(ACE_DIRECT_CONDITIONER_LOGICAL_TENSOR_NAMES),
        );
      },
    );

    it("rejects a hidden semantic tensor even when required names remain", () => {
      const hidden = structuredClone(loadManifest("fp16")) as MutableManifest;
      hidden.tensors["ace.detokenizer.hidden.weight"] = {
        ...hidden.tensors[ACE_SEMANTIC_LOGICAL_TENSOR_NAMES[0]!]!,
        logicalTensor: "ace.detokenizer.hidden.weight",
      };
      expect(() => validateAceConditionerPackageInventory(hidden)).toThrow(
        /semantic manifest inventory differs/,
      );
    });
  },
);

function loadManifest(profile: keyof typeof LOCAL_MANIFEST_URLS): AcePackageManifest {
  return JSON.parse(readFileSync(LOCAL_MANIFEST_URLS[profile], "utf8")) as
    AcePackageManifest;
}

function syntheticResolver(
  profile: "reference-bf16" | "raw-fp16",
  options: Readonly<{
    logicalShapeOverride?: ReadonlyMap<string, readonly number[]>;
  }> = {},
): AceConditionerTensorResolver {
  const dtype = profile === "reference-bf16" ? "uint32-bf16-pairs" : "float16";
  const layout = profile === "reference-bf16"
    ? "source-row-major-bf16-pairs-lsb-u32"
    : "source-row-major";
  const buffers = new Map<string, GPUBuffer>();
  const binding = (name: string): GPUBufferBinding => {
    const shape = aceConditionerExpectedLogicalShape(name);
    const elements = shape.reduce((product, value) => product * value, 1);
    const byteLength = name === "constants.silence_latent"
      ? elements * 4
      : elements * 2;
    let buffer = buffers.get(name);
    if (buffer === undefined) {
      buffer = { size: byteLength } as GPUBuffer;
      buffers.set(name, buffer);
    }
    return { buffer, offset: 0, size: byteLength };
  };
  return {
    binding,
    logicalTensor(name): AceGpuLogicalTensor {
      const expectedShape = aceConditionerExpectedLogicalShape(name);
      const logicalShape = options.logicalShapeOverride?.get(name) ?? expectedShape;
      const isConstant = name === "constants.silence_latent";
      const phase = isConstant
        ? "constants"
        : name.startsWith("ace.detokenizer.") || name.startsWith("ace.tokenizer.")
          ? "semantic"
          : "conditioner";
      const tensor = {
        logicalTensor: name,
        logicalShape,
        phase,
        dtype: isConstant ? "float32" : dtype,
        layout: isConstant ? "contiguous-nct-f32" : layout,
        partAxis: 0,
        partStart: 0,
        partEnd: logicalShape[0]!,
      } as AcePackageManifest["tensors"][string];
      return {
        logicalTensor: name,
        logicalShape,
        parts: [{ tensorName: name, tensor, binding: binding(name) }],
      };
    },
  };
}

type MutableManifest = Omit<AcePackageManifest, "tensors"> & {
  tensors: Record<string, AcePackageManifest["tensors"][string]>;
};
