import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  parseAcePackageManifest,
  resolveAceLogicalTensor,
  type AcePackageManifest,
} from "../src/model/manifest.js";
import {
  ACE_TEXT_ENCODER_INPUT_CONTRACT,
  ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES,
  ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES,
  ACE_TEXT_ENCODER_PHYSICAL_TENSOR_NAMES,
  ACE_TEXT_ENCODER_REFERENCE_PROVENANCE,
  ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES,
  aceTextEncoderLayerTensorNames,
  createAceTextEncoderControlData,
  planAceTextEncoder,
  resolveAceTextEncoderWeights,
  validateAceTextEncoderManifestInventory,
  type AceTextEncoderTensorResolver,
} from "../src/webgpu/text-encoder.js";

const LOCAL_MANIFEST_URLS = Object.freeze({
  reference: new URL("../model/files-reference/manifest.json", import.meta.url),
  fp16: new URL("../model/files-fp16/manifest.json", import.meta.url),
});
const HAS_LOCAL_CANONICAL_MANIFESTS = Object.values(LOCAL_MANIFEST_URLS)
  .every((url) => existsSync(url));

describe("ACE production text-encoder contract", () => {
  it("pins the audited upstream call, dependency, and input semantics", () => {
    expect(ACE_TEXT_ENCODER_REFERENCE_PROVENANCE).toEqual({
      aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      modelSnapshotRevision: "19671f406d603126926c1b7e2adc169acbcade22",
      conditioningEmbedSourceSha256:
        "45e80702219496edcef5712fa8403583c61f0df128833b9657d0d129d035a3f2",
      conditioningTextSourceSha256:
        "0313ef5fc5ade1baa5a806f3748e5c853961768d057045e040fcb28a33a153b6",
      componentLoaderSourceSha256:
        "621fc7d24ee847835de2eb66f35451120cb92310cf5e112e4edbdf955535d6de",
      qwenConfigSha256:
        "bb23c1607cfe059a58d8f0196cf1cebb52082b1056b8e358a579da80a5759420",
      transformersVersion: "4.57.6",
      transformersWheelSha256:
        "4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550",
      transformersQwen3SourceSha256:
        "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
      transformersInputDecoratorSourceSha256:
        "94728825190dff491f2e0b93beb59c8246adb0b6049e3203c511405f43bb7e30",
    });
    expect(ACE_TEXT_ENCODER_INPUT_CONTRACT).toEqual({
      batch: 1,
      tokenizerPadding: "longest-single-item",
      textTruncationTokens: 256,
      lyricTruncationTokens: 2_048,
      textAttention: "causal-all-keys-valid",
      downstreamTextMask: "tokenizer-mask-bypasses-qwen-and-is-retained-for-conditioner",
      positionIds: "zero-based-physical-token-index",
      upstreamKeyValueCache: "created-by-config-default-but-output-dead",
      browserKeyValueCache: "elided-with-identical-last-hidden-state",
      output: "last-hidden-state-after-final-rmsnorm",
      lyricOutput: "embedding-table-lookup-only",
      downstreamLyricMask: "tokenizer-mask-bypasses-embedding-and-is-retained-for-conditioner",
    });
  });

  it("maps exactly 28 by 11 decoder weights plus embedding and final norm", () => {
    expect(ACE_TEXT_ENCODER_SHARED_TENSOR_NAMES).toEqual({
      embedding: "text.embed_tokens.weight",
      finalNorm: "text.norm.weight",
    });
    expect(ACE_TEXT_ENCODER_LAYER_TENSOR_NAMES).toHaveLength(28);
    expect(Object.values(aceTextEncoderLayerTensorNames(27))).toEqual([
      "text.layers.27.input_layernorm.weight",
      "text.layers.27.self_attn.q_proj.weight",
      "text.layers.27.self_attn.k_proj.weight",
      "text.layers.27.self_attn.v_proj.weight",
      "text.layers.27.self_attn.q_norm.weight",
      "text.layers.27.self_attn.k_norm.weight",
      "text.layers.27.self_attn.o_proj.weight",
      "text.layers.27.post_attention_layernorm.weight",
      "text.layers.27.mlp.gate_proj.weight",
      "text.layers.27.mlp.up_proj.weight",
      "text.layers.27.mlp.down_proj.weight",
    ]);
    expect(ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES).toHaveLength(310);
    expect(new Set(ACE_TEXT_ENCODER_LOGICAL_TENSOR_NAMES).size).toBe(310);
    expect(ACE_TEXT_ENCODER_PHYSICAL_TENSOR_NAMES).toHaveLength(313);
    expect(ACE_TEXT_ENCODER_PHYSICAL_TENSOR_NAMES.slice(0, 4)).toEqual([
      "text.embed_tokens.weight.rows-000000-049152",
      "text.embed_tokens.weight.rows-049152-098304",
      "text.embed_tokens.weight.rows-098304-147456",
      "text.embed_tokens.weight.rows-147456-151669",
    ]);
    expect(() => aceTextEncoderLayerTensorNames(-1)).toThrow(/\[0, 27\]/);
    expect(() => aceTextEncoderLayerTensorNames(28)).toThrow(/\[0, 27\]/);
  });

  it("plans the complete 256-token batch-one graph without a score matrix", () => {
    expect(planAceTextEncoder("reference-bf16", {
      batch: 1,
      tokens: 256,
    })).toEqual({
      batch: 1,
      tokens: 256,
      rows: 256,
      layerCount: 28,
      hiddenElements: 262_144,
      queryElements: 524_288,
      keyValueElements: 262_144,
      intermediateElements: 786_432,
      outputElements: 262_144,
      scratchActivationElements: 9_175_040,
      scratchActivationBytes: 36_700_160,
      residentActivationElements: 9_437_184,
      residentActivationBytes: 37_748_736,
    });
    expect(planAceTextEncoder("raw-fp16", {
      batch: 1,
      tokens: 256,
    })).toMatchObject({
      scratchActivationBytes: 18_350_080,
      residentActivationBytes: 18_874_368,
    });
    expect(() => planAceTextEncoder("reference-bf16", {
      batch: 1,
      tokens: 257,
    })).toThrow(/exceed 256/);
  });

  it("emits exact unmasked causal controls and Transformers physical positions", () => {
    const controls = createAceTextEncoderControlData(4);
    expect([...controls.validLengths]).toEqual([4, 4]);
    expect([...controls.queryPositions]).toEqual([0, 1, 2, 3]);
    expect([...controls.keyValidity]).toEqual([1, 1, 1, 1]);
    expect([...controls.cosine.slice(0, 128)]).toEqual(
      new Array<number>(128).fill(1),
    );
    expect([...controls.sine.slice(0, 128)]).toEqual(
      new Array<number>(128).fill(0),
    );
  });
});

describe.skipIf(!HAS_LOCAL_CANONICAL_MANIFESTS)(
  "ACE generated text-package resolution",
  () => {
    it.each(["reference", "fp16"] as const)(
      "accepts the exact %s physical/logical inventory and resolves all bindings",
      (profile) => {
        const manifest = loadManifest(profile);
        expect(() => validateAceTextEncoderManifestInventory(manifest)).not.toThrow();
        const resolver = fakeResolver(manifest);
        const weights = resolveAceTextEncoderWeights(
          resolver,
          profile === "reference" ? "reference-bf16" : "raw-fp16",
        );
        expect(weights.embedding.map(({ firstRow, rowCount }) => ({
          firstRow,
          rowCount,
        }))).toEqual([
          { firstRow: 0, rowCount: 49_152 },
          { firstRow: 49_152, rowCount: 49_152 },
          { firstRow: 98_304, rowCount: 49_152 },
          { firstRow: 147_456, rowCount: 4_213 },
        ]);
        expect(weights.layers).toHaveLength(28);
        expect(weights.finalNorm.size).toBe(2_048);
      },
    );

    it("rejects a hidden text tensor even when all required names remain", () => {
      const parsed = JSON.parse(readFileSync(
        LOCAL_MANIFEST_URLS.reference,
        "utf8",
      )) as { tensors: Record<string, Record<string, unknown>> };
      const source = parsed.tensors["text.norm.weight"]!;
      parsed.tensors["text.hidden.weight"] = {
        ...source,
        logicalTensor: "text.hidden.weight",
      };
      const manifest = parsed as unknown as AcePackageManifest;
      expect(() => validateAceTextEncoderManifestInventory(manifest)).toThrow(
        /hidden=.*text.hidden.weight/,
      );
    });
  },
);

function loadManifest(profile: keyof typeof LOCAL_MANIFEST_URLS): AcePackageManifest {
  return parseAcePackageManifest(
    JSON.parse(readFileSync(LOCAL_MANIFEST_URLS[profile], "utf8")),
    profile,
  );
}

function fakeResolver(manifest: AcePackageManifest): AceTextEncoderTensorResolver {
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
      if (logical.parts.length !== 1) throw new Error("test resolver requires one part");
      return logical.parts[0]!.binding;
    },
  };
}
