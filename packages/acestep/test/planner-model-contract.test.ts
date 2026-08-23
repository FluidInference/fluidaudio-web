import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  parseAcePackageManifest,
  resolveAceLogicalTensor,
  type AcePackageManifest,
} from "../src/model/manifest.js";
import type {
  AcePlannerDecodeBatch,
  AcePlannerPrefillBatch,
} from "../src/runtime/planner.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  ACE_PLANNER_LAYER_TENSOR_NAMES,
  ACE_PLANNER_LOGICAL_TENSOR_NAMES,
  ACE_PLANNER_MODEL_CACHE_CONTRACT,
  ACE_PLANNER_MODEL_REFERENCE_PROVENANCE,
  ACE_PLANNER_PHYSICAL_TENSOR_NAMES,
  ACE_PLANNER_SHARED_TENSOR_NAMES,
  acePlannerLayerTensorNames,
  createAcePlannerModelControlData,
  planAcePlannerHeadSlices,
  planAcePlannerModel,
  resolveAcePlannerModelWeights,
  validateAcePlannerManifestInventory,
  validateAcePlannerModelBatch,
  type AcePlannerTensorResolver,
} from "../src/webgpu/planner-model.js";

const LOCAL_MANIFEST_URLS = Object.freeze({
  reference: new URL("../model/files-reference/manifest.json", import.meta.url),
  fp16: new URL("../model/files-fp16/manifest.json", import.meta.url),
});
const HAS_LOCAL_CANONICAL_MANIFESTS = Object.values(LOCAL_MANIFEST_URLS)
  .every((url) => existsSync(url));

describe("ACE production planner-model contract", () => {
  it("pins the cached Qwen3ForCausalLM behavior and exact package inventory", () => {
    expect(ACE_PLANNER_MODEL_REFERENCE_PROVENANCE).toEqual({
      aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      plannerSnapshotRevision: "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
      qwenImplementation: "transformers-4.57.6-Qwen3ForCausalLM",
      cacheLayout: "batch-kv-head-physical-token-head-dimension",
      logitsSelection: "last-physical-token-per-left-padded-row",
      tiedOutput: "model.embed_tokens.weight-is-lm-head-weight",
    });
    expect(ACE_PLANNER_MODEL_CACHE_CONTRACT.phaseCapacity).toBe(
      "exact-requested-capacity",
    );
    expect(ACE_PLANNER_SHARED_TENSOR_NAMES).toEqual({
      embedding: "planner.model.embed_tokens.weight",
      finalNorm: "planner.model.norm.weight",
    });
    expect(ACE_PLANNER_LAYER_TENSOR_NAMES).toHaveLength(28);
    expect(Object.values(acePlannerLayerTensorNames(27))).toEqual([
      "planner.model.layers.27.input_layernorm.weight",
      "planner.model.layers.27.self_attn.q_proj.weight",
      "planner.model.layers.27.self_attn.k_proj.weight",
      "planner.model.layers.27.self_attn.v_proj.weight",
      "planner.model.layers.27.self_attn.q_norm.weight",
      "planner.model.layers.27.self_attn.k_norm.weight",
      "planner.model.layers.27.self_attn.o_proj.weight",
      "planner.model.layers.27.post_attention_layernorm.weight",
      "planner.model.layers.27.mlp.gate_proj.weight",
      "planner.model.layers.27.mlp.up_proj.weight",
      "planner.model.layers.27.mlp.down_proj.weight",
    ]);
    expect(ACE_PLANNER_LOGICAL_TENSOR_NAMES).toHaveLength(310);
    expect(new Set(ACE_PLANNER_LOGICAL_TENSOR_NAMES).size).toBe(310);
    expect(ACE_PLANNER_PHYSICAL_TENSOR_NAMES).toHaveLength(314);
    expect(ACE_PLANNER_PHYSICAL_TENSOR_NAMES.slice(0, 5)).toEqual([
      "planner.model.embed_tokens.weight.rows-000000-049152",
      "planner.model.embed_tokens.weight.rows-049152-098304",
      "planner.model.embed_tokens.weight.rows-098304-147456",
      "planner.model.embed_tokens.weight.rows-147456-196608",
      "planner.model.embed_tokens.weight.rows-196608-217204",
    ]);
    expect(ACE_PLANNER_EMBEDDING_ROW_PARTS).toEqual([
      { firstRow: 0, rowCount: 49_152 },
      { firstRow: 49_152, rowCount: 49_152 },
      { firstRow: 98_304, rowCount: 49_152 },
      { firstRow: 147_456, rowCount: 49_152 },
      { firstRow: 196_608, rowCount: 20_596 },
    ]);
    expect(() => acePlannerLayerTensorNames(-1)).toThrow(/\[0, 27\]/);
    expect(() => acePlannerLayerTensorNames(28)).toThrow(/\[0, 27\]/);
  });

  it("plans exact phase-sized prefill memory without allocating 40960 slots", () => {
    const plan = planAcePlannerModel("reference-bf16", prefillBatch());
    expect(plan).toEqual({
      kind: "prefill",
      batch: 2,
      tokens: 4,
      cacheCapacity: 512,
      rows: 8,
      layerCount: 28,
      hiddenElements: 8_192,
      queryElements: 16_384,
      keyValueElements: 8_192,
      intermediateElements: 24_576,
      blockScratchElements: 262_144,
      transientActivationElements: 296_960,
      transientActivationBytes: 1_187_840,
      logitsElements: 434_408,
      logitsBytes: 1_737_632,
      kvCacheElementsPerLayer: 2_097_152,
      kvCacheBytesPerLayer: 8_388_608,
      kvCacheElements: 58_720_256,
      kvCacheBytes: 234_881_024,
      totalActivationElements: 59_451_624,
      totalActivationBytes: 237_806_496,
      cacheValidityBytes: 4_096,
      invocationControlBytes: 8_328,
      totalNonWeightBytes: 237_818_920,
    });
    const oneMoreSlot = planAcePlannerModel(
      "reference-bf16",
      { ...prefillBatch(), cacheCapacity: 513 },
    );
    expect(oneMoreSlot.kvCacheBytes - plan.kvCacheBytes).toBe(458_752);
    expect(oneMoreSlot.totalNonWeightBytes - plan.totalNonWeightBytes).toBe(
      458_760,
    );
  });

  it("intersects the exact semantic and EOS domains with tied shards", () => {
    expect(planAcePlannerHeadSlices({
      firstTokenId: 151_669,
      tokenCount: 64_000,
    })).toEqual([
      {
        shardIndex: 3,
        globalFirstRow: 151_669,
        localFirstRow: 4_213,
        rowCount: 44_939,
      },
      {
        shardIndex: 4,
        globalFirstRow: 196_608,
        localFirstRow: 0,
        rowCount: 19_061,
      },
    ]);
    expect(planAcePlannerHeadSlices({
      firstTokenId: 151_645,
      tokenCount: 1,
    })).toEqual([{
      shardIndex: 3,
      globalFirstRow: 151_645,
      localFirstRow: 4_189,
      rowCount: 1,
    }]);
    expect(() => planAcePlannerHeadSlices({
      firstTokenId: 217_203,
      tokenCount: 2,
    })).toThrow(/outside the vocabulary/);
  });

  it("halves only activation storage in the raw-FP16 profile", () => {
    const reference = planAcePlannerModel("reference-bf16", decodeBatch());
    const fp16 = planAcePlannerModel("raw-fp16", decodeBatch());
    expect(reference).toMatchObject({
      kind: "decode",
      batch: 2,
      tokens: 1,
      cacheCapacity: 512,
      transientActivationElements: 75_776,
      logitsElements: 434_408,
      kvCacheElements: 58_720_256,
      totalActivationElements: 59_230_440,
      totalActivationBytes: 236_921_760,
      cacheValidityBytes: 4_096,
      invocationControlBytes: 2_112,
      totalNonWeightBytes: 236_927_968,
    });
    expect(fp16.totalActivationBytes).toBe(reference.totalActivationBytes / 2);
    expect(fp16.cacheValidityBytes).toBe(reference.cacheValidityBytes);
    expect(fp16.invocationControlBytes).toBe(reference.invocationControlBytes);
    expect(fp16.totalNonWeightBytes).toBe(118_467_088);
  });

  it("preserves left-padded physical positions and gathers each last physical row", () => {
    const batch = prefillBatch();
    const controls = createAcePlannerModelControlData(batch);
    expect([...controls.queryPositions]).toEqual([
      0, 1, 2, 3,
      0, 1, 2, 3,
    ]);
    expect([...controls.sourceValidity]).toEqual([
      0, 0, 1, 1,
      0, 1, 1, 1,
    ]);
    expect([...controls.lastPhysicalRowIndices]).toEqual([3, 3]);
    expect(controls.clearCacheBeforeDispatch).toBe(true);
    expect([...controls.cosine.slice(0, 128)]).toEqual(
      new Array<number>(128).fill(1),
    );

    const decode = createAcePlannerModelControlData(decodeBatch());
    expect([...decode.queryPositions]).toEqual([4, 4]);
    expect([...decode.lastPhysicalRowIndices]).toEqual([0, 0]);
    expect(decode.clearCacheBeforeDispatch).toBe(false);
  });

  it("fails closed when a batch no longer represents the pinned physical cache", () => {
    const base = prefillBatch();
    expect(() => validateAcePlannerModelBatch({
      ...base,
      rotaryPositionIds: new Uint32Array([0, 1, 2, 2, 0, 1, 2, 3]),
    })).toThrow(/physical cache slot/);
    expect(() => validateAcePlannerModelBatch({
      ...base,
      keyValidity: new Uint32Array([1, 0, 1, 1, 0, 1, 1, 1]),
    })).toThrow(/left-padding validity differs/);
    const decode = decodeBatch();
    expect(() => validateAcePlannerModelBatch({
      ...decode,
      inputIds: new Uint32Array([42, 43]),
    })).toThrow(/same token/);
  });
});

describe.skipIf(!HAS_LOCAL_CANONICAL_MANIFESTS)(
  "ACE generated planner-package resolution",
  () => {
    it.each(["reference", "fp16"] as const)(
      "accepts and resolves the exact %s 310/314 inventory",
      (profile) => {
        const manifest = loadManifest(profile);
        expect(() => validateAcePlannerManifestInventory(manifest)).not.toThrow();
        const weights = resolveAcePlannerModelWeights(
          fakeResolver(manifest),
          profile === "reference" ? "reference-bf16" : "raw-fp16",
        );
        expect(weights.embedding.map(({ firstRow, rowCount }) => ({
          firstRow,
          rowCount,
        }))).toEqual(ACE_PLANNER_EMBEDDING_ROW_PARTS);
        expect(weights.layers).toHaveLength(28);
        expect(weights.finalNorm.size).toBe(2_048);
        // The model API intentionally exposes no independent LM-head weight.
        expect(Object.keys(weights).sort()).toEqual([
          "embedding",
          "finalNorm",
          "layers",
        ]);
      },
    );

    it("rejects a hidden planner tensor and an edited storage layout", () => {
      const hidden = rawManifest("reference");
      const source = hidden.tensors["planner.model.norm.weight"]!;
      hidden.tensors["planner.model.hidden.weight"] = {
        ...source,
        logicalTensor: "planner.model.hidden.weight",
      };
      expect(() => validateAcePlannerManifestInventory(
        hidden as unknown as AcePackageManifest,
      )).toThrow(/hidden=.*planner.model.hidden.weight/);

      const reshaped = rawManifest("fp16");
      reshaped.tensors["planner.model.layers.0.self_attn.q_proj.weight"]!
        .storageShape = [1024, 2048];
      expect(() => validateAcePlannerManifestInventory(
        reshaped as unknown as AcePackageManifest,
      )).toThrow(/storage shape changed/);
    });
  },
);

function prefillBatch(): AcePlannerPrefillBatch {
  const queryPositions = new Uint32Array([
    0, 1, 2, 3,
    0, 1, 2, 3,
  ]);
  const sourceValidity = new Uint32Array([
    0, 0, 1, 1,
    0, 1, 1, 1,
  ]);
  return {
    kind: "prefill",
    rows: 2,
    tokens: 4,
    cacheCapacity: 512,
    inputIds: new Uint32Array([
      151_643, 151_643, 10, 11,
      151_643, 20, 21, 22,
    ]),
    keyValidity: sourceValidity.slice(),
    rotaryPositionIds: queryPositions.slice(),
    causal: {
      rowStartPositions: new Uint32Array([0, 0]),
      validLengths: new Uint32Array([4, 4, 4, 4]),
      queryPositions,
      sourceValidity,
    },
    conditionalRow: 0,
    unconditionalRow: 1,
  };
}

function decodeBatch(): AcePlannerDecodeBatch {
  return {
    kind: "decode",
    rows: 2,
    tokens: 1,
    cacheCapacity: 512,
    cachedTokensBeforeAppend: 4,
    inputIds: new Uint32Array([42, 42]),
    rotaryPositionIds: new Uint32Array([4, 4]),
    causal: {
      rowStartPositions: new Uint32Array([4, 4]),
      validLengths: new Uint32Array([1, 5, 1, 5]),
      queryPositions: new Uint32Array([4, 4]),
      sourceValidity: new Uint32Array([1, 1]),
    },
    conditionalRow: 0,
    unconditionalRow: 1,
  };
}

function loadManifest(profile: keyof typeof LOCAL_MANIFEST_URLS): AcePackageManifest {
  return parseAcePackageManifest(rawManifest(profile), profile);
}

function rawManifest(profile: keyof typeof LOCAL_MANIFEST_URLS): {
  tensors: Record<string, Record<string, unknown>>;
} {
  return JSON.parse(readFileSync(LOCAL_MANIFEST_URLS[profile], "utf8")) as {
    tensors: Record<string, Record<string, unknown>>;
  };
}

function fakeResolver(manifest: AcePackageManifest): AcePlannerTensorResolver {
  const buffers = new Map<string, GPUBuffer>();
  for (const file of manifest.files) {
    buffers.set(file.name, { size: file.byteLength } as GPUBuffer);
  }
  return {
    logicalTensor(logicalTensor) {
      const resolved = resolveAceLogicalTensor(manifest, logicalTensor);
      return {
        logicalTensor,
        logicalShape: resolved.logicalShape,
        parts: resolved.parts.map(({ tensorName, tensor }) => ({
          tensorName,
          tensor,
          binding: {
            buffer: buffers.get(tensor.shard)!,
            offset: tensor.byteOffset,
            size: tensor.byteLength,
          },
        })),
      };
    },
    binding(logicalTensor) {
      const logical = this.logicalTensor(logicalTensor);
      if (logical.parts.length !== 1) {
        throw new Error("test resolver requires an unsharded logical tensor");
      }
      return logical.parts[0]!.binding;
    },
  };
}
