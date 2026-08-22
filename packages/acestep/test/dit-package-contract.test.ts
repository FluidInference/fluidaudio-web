import { existsSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import type { AceGpuLogicalTensor } from "../src/model/gpu-tensors.js";
import {
  ACE_DIT_GEMM_FP16_TRANSFORMATION,
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_TILE_LAYOUT,
  ACE_DIT_DENSE_K4_FP16_LAYOUT,
  ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
  parseAcePackageManifest,
  type AcePackageTensorRecord,
} from "../src/model/manifest.js";
import type { AceModelProfileId } from "../src/webgpu/capabilities.js";

import {
  ACE_DIT_LAYER_COUNT,
  ACE_DIT_SHARED_TENSOR_NAMES,
  aceDitCanonicalTensorNames,
  aceDitExpectedLogicalShape,
  aceDitLayerTensorNames,
  createAceDitCanonicalTensorNameMap,
  isAceDitGemmWeightTensorName,
  isAceDitRepeatedDenseWeightTensorName,
  resolveAceDitPackageWeights,
} from "../src/webgpu/ace-dit-package.js";
import {
  ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  requireAceOpt0037DitK4PackageIdentity,
} from "../src/webgpu/dit-fp16-package.js";

const LOCAL_MANIFEST_URLS = Object.freeze({
  reference: new URL("../model/files-reference/manifest.json", import.meta.url),
  fp16: new URL("../model/files-fp16/manifest.json", import.meta.url),
});
const HAS_LOCAL_CANONICAL_MANIFESTS = Object.values(LOCAL_MANIFEST_URLS)
  .every((url) => existsSync(url));
const LOCAL_OPT_0037_MANIFEST_URL = new URL(
  "../model/files-fp16-dit-layer-mixed-experimental/manifest.json",
  import.meta.url,
);
const HAS_LOCAL_OPT_0037_MANIFEST = existsSync(LOCAL_OPT_0037_MANIFEST_URL);
const FAKE_BUFFER = { size: 4 } as GPUBuffer;

describe("ACE DiT canonical package bindings", () => {
  it("maps shared and repeated graph weights to exact canonical names", () => {
    expect(ACE_DIT_SHARED_TENSOR_NAMES.conditionProjection).toEqual({
      weight: "ace.decoder.condition_embedder.weight",
      bias: "ace.decoder.condition_embedder.bias",
    });
    expect(ACE_DIT_SHARED_TENSOR_NAMES.inputProjection.weight).toBe(
      "ace.decoder.proj_in.1.weight",
    );
    expect(ACE_DIT_SHARED_TENSOR_NAMES.timestep.projectionWeight).toBe(
      "ace.decoder.time_embed.time_proj.weight",
    );
    expect(ACE_DIT_SHARED_TENSOR_NAMES.relativeTimestep.linear1Bias).toBe(
      "ace.decoder.time_embed_r.linear_1.bias",
    );
    expect(ACE_DIT_SHARED_TENSOR_NAMES.output.projection).toBe(
      "ace.decoder.proj_out.1.weight",
    );
    expect(aceDitLayerTensorNames(23)).toMatchObject({
      scaleShiftTable: "ace.decoder.layers.23.scale_shift_table",
      selfAttentionNorm: "ace.decoder.layers.23.self_attn_norm.weight",
      selfValueProjection: "ace.decoder.layers.23.self_attn.v_proj.weight",
      crossKeyNorm: "ace.decoder.layers.23.cross_attn.k_norm.weight",
      downProjection: "ace.decoder.layers.23.mlp.down_proj.weight",
    });
    expect(() => aceDitLayerTensorNames(-1)).toThrow(/\[0, 23\]/);
    expect(() => aceDitLayerTensorNames(24)).toThrow(/\[0, 23\]/);
    expect(() => aceDitLayerTensorNames(1.5)).toThrow(/\[0, 23\]/);
  });

  it("emits exactly 20 shared and 24 by 19 layer tensors", () => {
    const map = createAceDitCanonicalTensorNameMap();
    expect(ACE_DIT_LAYER_COUNT).toBe(24);
    expect(map.layers).toHaveLength(24);
    for (const layer of map.layers) expect(Object.values(layer)).toHaveLength(19);
    const names = aceDitCanonicalTensorNames();
    expect(names).toHaveLength(20 + 24 * 19);
    expect(new Set(names).size).toBe(names.length);
  });

  it("pins exactly 271 unique rank-two GEMM weight names", () => {
    const canonicalNames = aceDitCanonicalTensorNames();
    const gemmNames = canonicalNames.filter(isAceDitGemmWeightTensorName);
    expect(gemmNames).toHaveLength(7 + 24 * 11);
    expect(new Set(gemmNames).size).toBe(gemmNames.length);
    expect(canonicalNames.filter(
      (name) => !isAceDitGemmWeightTensorName(name),
    )).toHaveLength(205);
    for (const name of gemmNames) {
      expect(aceDitExpectedLogicalShape(name)).toHaveLength(2);
    }
    expect(gemmNames).toContain("ace.decoder.condition_embedder.weight");
    expect(gemmNames).toContain("ace.decoder.time_embed.linear_1.weight");
    expect(gemmNames).toContain("ace.decoder.time_embed_r.time_proj.weight");
    expect(gemmNames).toContain("ace.decoder.layers.0.self_attn.k_proj.weight");
    expect(gemmNames).toContain("ace.decoder.layers.23.cross_attn.o_proj.weight");
    expect(gemmNames).toContain("ace.decoder.layers.23.mlp.down_proj.weight");

    expect(isAceDitGemmWeightTensorName(
      "ace.decoder.condition_embedder.bias",
    )).toBe(false);
    expect(isAceDitGemmWeightTensorName(
      "ace.decoder.proj_in.1.weight",
    )).toBe(false);
    expect(isAceDitGemmWeightTensorName(
      "ace.decoder.layers.0.self_attn.q_norm.weight",
    )).toBe(false);
    expect(isAceDitGemmWeightTensorName(
      "ace.decoder.layers.24.self_attn.q_proj.weight",
    )).toBe(false);
    expect(isAceDitGemmWeightTensorName(
      "ace.decoder.layers.0.self_attn.q_proj.bias",
    )).toBe(false);
  });

  it.each(["reference-bf16", "raw-fp16"] as const)(
    "resolves every %s name once with its exact tiled or row-major contract",
    (profile) => {
      const requested: string[] = [];
      const resolver = {
        logicalTensor(name: string) {
          return fakeLogicalTensor(name, profile);
        },
        binding(name: string) {
          requested.push(name);
          return { buffer: FAKE_BUFFER, offset: 0, size: 4 };
        },
      };
      const weights = resolveAceDitPackageWeights(resolver, profile);
      expect(weights.layers).toHaveLength(24);
      expect(weights.crossCaches).toHaveLength(24);
      expect(requested).toHaveLength(476);
      expect(new Set(requested)).toEqual(new Set(aceDitCanonicalTensorNames()));
      expect(weights.crossCaches[23]?.keyProjection).toEqual({
        buffer: FAKE_BUFFER,
        offset: 0,
        size: 4,
      });

      expect(() => resolveAceDitPackageWeights({
        ...resolver,
        logicalTensor(name) {
          const logical = resolver.logicalTensor(name);
          return name === ACE_DIT_SHARED_TENSOR_NAMES.inputProjection.weight
            ? { ...logical, logicalShape: [1] }
            : logical;
        },
      }, profile)).toThrow(/has shape/);
    },
  );

  it("routes exactly 216 rev8 K4 dense tensors and retains 48 cross-cache plus 192 BF16 layer tensors", () => {
    const referenceRequests: string[] = [];
    const mixedRequests: string[] = [];
    const referenceResolver = {
      logicalTensor(name: string) {
        return fakeLogicalTensor(name, "reference-bf16");
      },
      binding(name: string) {
        referenceRequests.push(name);
        return { buffer: FAKE_BUFFER, offset: 0, size: 4 };
      },
    };
    const mixedResolver = {
      logicalTensor(name: string) {
        return fakeOpt0037MixedLogicalTensor(name);
      },
      binding(name: string) {
        mixedRequests.push(name);
        return { buffer: FAKE_BUFFER, offset: 0, size: 4 };
      },
    };

    const weights = resolveAceDitPackageWeights(
      referenceResolver,
      "reference-bf16",
      mixedResolver,
      ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
    );
    expect(weights.layers).toHaveLength(24);
    expect(referenceRequests).toHaveLength(20);
    expect(mixedRequests).toHaveLength(24 * 19);
    const denseNames = mixedRequests.filter(
      isAceDitRepeatedDenseWeightTensorName,
    );
    const crossCacheNames = mixedRequests.filter((name) =>
      /\.cross_attn\.(?:k|v)_proj\.weight$/u.test(name)
    );
    const supportNames = mixedRequests.filter((name) =>
      !isAceDitRepeatedDenseWeightTensorName(name) &&
      !/\.cross_attn\.(?:k|v)_proj\.weight$/u.test(name)
    );
    expect(denseNames).toHaveLength(216);
    expect(crossCacheNames).toHaveLength(48);
    expect(supportNames).toHaveLength(192);
    expect(new Set(mixedRequests).size).toBe(456);
    for (const name of denseNames) {
      const logical = fakeOpt0037MixedLogicalTensor(name);
      const tensor = logical.parts[0]!.tensor;
      expect(tensor).toMatchObject({
        dtype: "float16",
        layout: ACE_DIT_DENSE_K4_FP16_LAYOUT,
        transformation: ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
        storageShape: [
          logical.logicalShape[0]! / 128,
          logical.logicalShape[1]! / 4,
          4,
          32,
          4,
        ],
      });
    }
    for (const name of [...crossCacheNames, ...supportNames]) {
      expect(fakeOpt0037MixedLogicalTensor(name).parts[0]!.tensor.dtype)
        .toBe("uint32-bf16-pairs");
    }

    expect(() => resolveAceDitPackageWeights(
      referenceResolver,
      "reference-bf16",
      mixedResolver,
    )).toThrow(/must be supplied together/);

    const target = denseNames[0]!;
    const malformed = fakeOpt0037MixedLogicalTensor(target);
    const malformedTensor = forgeLogicalTensor(malformed, {
      ...malformed.parts[0]!.tensor,
      storageShape: [...malformed.logicalShape],
    });
    expect(() => resolveAceDitPackageWeights(
      referenceResolver,
      "reference-bf16",
      {
        ...mixedResolver,
        logicalTensor(name: string) {
          return name === target
            ? malformedTensor
            : fakeOpt0037MixedLogicalTensor(name);
        },
      },
      ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
    )).toThrow(/package contract/);
  });

  it.each(["reference-bf16", "raw-fp16"] as const)(
    "rejects wrong %s layout, dtype, transformation, shape, and source",
    (profile) => {
      const target = ACE_DIT_SHARED_TENSOR_NAMES.conditionProjection.weight;
      const correct = fakeLogicalTensor(target, profile);
      const correctTensor = correct.parts[0]!.tensor;
      const mutations: readonly Partial<AcePackageTensorRecord>[] = [
        { layout: "source-row-major" },
        {
          dtype: profile === "reference-bf16"
            ? "float16"
            : "uint32-bf16-pairs",
        },
        {
          transformation: profile === "reference-bf16"
            ? "preserve-bf16-bits-pack-u32-pairs"
            : "bf16-to-ieee-fp16",
        },
        { logicalShape: [128, 32] },
        { storageShape: [1] },
        { source: "ace-turbo-weights:decoder.condition_embedder.bias" },
      ];
      for (const mutation of mutations) {
        const forged = forgeLogicalTensor(correct, {
          ...correctTensor,
          ...mutation,
        });
        expect(
          () => resolveAceDitPackageWeights(
            resolverReplacing(target, forged, profile),
            profile,
          ),
          JSON.stringify(mutation),
        ).toThrow(/package contract/);
      }
    },
  );

  it.each(["reference-bf16", "raw-fp16"] as const)(
    "rejects a tiled contract on non-GEMM %s tensors",
    (profile) => {
      const target = ACE_DIT_SHARED_TENSOR_NAMES.conditionProjection.bias;
      const correct = fakeLogicalTensor(target, profile);
      const forged = forgeLogicalTensor(correct, {
        ...correct.parts[0]!.tensor,
        layout: ACE_DIT_GEMM_TILE_LAYOUT,
        transformation: profile === "reference-bf16"
          ? ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
          : ACE_DIT_GEMM_FP16_TRANSFORMATION,
      });
      expect(() => resolveAceDitPackageWeights(
        resolverReplacing(target, forged, profile),
        profile,
      )).toThrow(/package contract/);
    },
  );
});

describe.skipIf(!HAS_LOCAL_CANONICAL_MANIFESTS)(
  "ACE DiT generated-package resolution",
  () => {
    it.each(["reference", "fp16"] as const)(
    "resolves every emitted name and no hidden DiT weight in the %s manifest",
    (profile) => {
      const manifest = JSON.parse(readFileSync(
        LOCAL_MANIFEST_URLS[profile],
        "utf8",
      )) as { readonly tensors: Readonly<Record<string, unknown>> };
      const manifestNames = Object.keys(manifest.tensors)
        .filter((name) => name.startsWith("ace.decoder."))
        .sort();
      const emittedNames = [...aceDitCanonicalTensorNames()].sort();
      for (const name of emittedNames) {
        expect(
          Object.hasOwn(manifest.tensors, name),
          `missing ${profile} tensor ${name}`,
        ).toBe(true);
      }
      expect(emittedNames).toEqual(manifestNames);
    },
    );
  },
);

describe.skipIf(!HAS_LOCAL_OPT_0037_MANIFEST)(
  "ACE OPT-0037 generated package identity",
  () => {
    it("independently authenticates the exact rev8 manifest and K4 inventory", () => {
      const bytes = readFileSync(LOCAL_OPT_0037_MANIFEST_URL);
      expect(bytes.byteLength).toBe(ACE_OPT_0037_DIT_K4_MANIFEST_BYTES);
      const sha256 = createHash("sha256").update(bytes).digest("hex");
      expect(sha256).toBe(ACE_OPT_0037_DIT_K4_MANIFEST_SHA256);
      const manifest = parseAcePackageManifest(
        JSON.parse(bytes.toString("utf8")),
        "fp16-dit-dense-experimental",
      );
      expect(Object.keys(manifest.tensors)).toHaveLength(457);
      expect(manifest.files).toHaveLength(73);
      expect(Object.values(manifest.tensors).filter((tensor) =>
        tensor.layout === ACE_DIT_DENSE_K4_FP16_LAYOUT &&
        tensor.transformation === ACE_DIT_DENSE_K4_FP16_TRANSFORMATION
      )).toHaveLength(216);
      const loaded = {
        manifest,
        manifestUrl: LOCAL_OPT_0037_MANIFEST_URL.href,
        manifestSha256: sha256,
        manifestByteLength: bytes.byteLength,
        manifestId: `ace-step-webgpu-v1:${manifest.profile}:${sha256}`,
      };
      expect(() => requireAceOpt0037DitK4PackageIdentity(loaded)).not.toThrow();
      expect(() => requireAceOpt0037DitK4PackageIdentity({
        ...loaded,
        manifestSha256: "0".repeat(64),
      })).toThrow(/authenticated identity validation/);
    });
  },
);

function fakeLogicalTensor(
  name: string,
  profile: AceModelProfileId,
): AceGpuLogicalTensor {
  const logicalShape = aceDitExpectedLogicalShape(name);
  const elements = logicalShape.reduce((product, extent) => product * extent, 1);
  const reference = profile === "reference-bf16";
  const tiled = isAceDitGemmWeightTensorName(name);
  const tensor: AcePackageTensorRecord = {
    shard: "weights/dit/test.bin",
    byteOffset: 0,
    byteLength: elements * 2,
    dtype: reference ? "uint32-bf16-pairs" : "float16",
    logicalShape,
    storageShape: reference ? [Math.ceil(elements / 2)] : logicalShape,
    layout: tiled
      ? ACE_DIT_GEMM_TILE_LAYOUT
      : reference
        ? "source-row-major-bf16-pairs-lsb-u32"
        : "source-row-major",
    phase: "dit",
    lifetime: "dit",
    source: `ace-turbo-weights:${name.slice("ace.".length)}`,
    transformation: tiled
      ? reference
        ? ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
        : ACE_DIT_GEMM_FP16_TRANSFORMATION
      : reference
        ? "preserve-bf16-bits-pack-u32-pairs"
        : "bf16-to-ieee-fp16",
    logicalTensor: name,
    partAxis: 0,
    partStart: 0,
    partEnd: logicalShape[0]!,
  };
  return forgeLogicalTensor({
    logicalTensor: name,
    logicalShape,
  }, tensor);
}

function fakeOpt0037MixedLogicalTensor(name: string): AceGpuLogicalTensor {
  const reference = fakeLogicalTensor(name, "reference-bf16");
  if (!isAceDitRepeatedDenseWeightTensorName(name)) return reference;
  const [outputs, inputs] = reference.logicalShape;
  return forgeLogicalTensor(reference, {
    ...reference.parts[0]!.tensor,
    dtype: "float16",
    storageShape: [outputs! / 128, inputs! / 4, 4, 32, 4],
    layout: ACE_DIT_DENSE_K4_FP16_LAYOUT,
    transformation: ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
  });
}

function forgeLogicalTensor(
  logical: Pick<AceGpuLogicalTensor, "logicalTensor" | "logicalShape">,
  tensor: AcePackageTensorRecord,
): AceGpuLogicalTensor {
  return {
    logicalTensor: logical.logicalTensor,
    logicalShape: logical.logicalShape,
    parts: [{
      tensorName: tensor.logicalTensor,
      tensor,
      binding: { buffer: FAKE_BUFFER, offset: 0, size: 4 },
    }],
  };
}

function resolverReplacing(
  target: string,
  replacement: AceGpuLogicalTensor,
  profile: AceModelProfileId,
) {
  return {
    logicalTensor(name: string) {
      return name === target ? replacement : fakeLogicalTensor(name, profile);
    },
    binding() {
      return { buffer: FAKE_BUFFER, offset: 0, size: 4 };
    },
  };
}
