import { describe, expect, it } from "vitest";

import {
  AceCorrectnessKvCacheWriteKernel,
  aceCorrectnessKvCacheWriteWgsl,
  planAceKvCacheWrite,
} from "../src/webgpu/kernels/kv-cache.js";

describe("ACE planner KV-cache write contract", () => {
  it("uses the attention-native B/KVH/T/D layout with a bounded 2D dispatch", () => {
    expect(
      planAceKvCacheWrite({
        batch: 2,
        keyValueHeads: 8,
        appendTokens: 137,
        cacheCapacity: 512,
        headDimension: 128,
      }),
    ).toEqual({
      batch: 2,
      keyValueHeads: 8,
      appendTokens: 137,
      cacheCapacity: 512,
      headDimension: 128,
      sourceElements: 280_576,
      cacheElements: 1_048_576,
      sourceValidityElements: 274,
      cacheValidityElements: 1_024,
      workgroupsX: 4_384,
      workgroupsY: 1,
    });
  });

  it("emits all-or-nothing row range checks and fail-closed validity writes", () => {
    const source = aceCorrectnessKvCacheWriteWgsl("reference-bf16", {
      batch: 2,
      keyValueHeads: 2,
      appendTokens: 3,
      cacheCapacity: 8,
      headDimension: 4,
    });
    expect(source).toContain("let last_allowed_start = CACHE_CAPACITY - APPEND_TOKENS");
    expect(source).toContain("if (!row_range_is_valid) { return; }");
    expect(source).toContain("write_status[batch] = select(0u, 1u, row_range_is_valid)");
    expect(source).toContain("select(0u, 1u, source_mask == 1u)");
    expect(source).toContain("((batch * KV_HEADS + kv_head) * CACHE_CAPACITY");
  });

  it("emits raw-FP16 cache storage only for the raw profile", () => {
    const source = aceCorrectnessKvCacheWriteWgsl("raw-fp16", {
      batch: 1,
      keyValueHeads: 1,
      appendTokens: 1,
      cacheCapacity: 2,
      headDimension: 2,
    });
    expect(source).toContain("enable f16;");
    expect(source).toContain("source_key: array<f16>");
    expect(source).toContain("cache_key: array<f16>");
  });

  it.each([
    { batch: 0, keyValueHeads: 1, appendTokens: 1, cacheCapacity: 1, headDimension: 1 },
    { batch: 1, keyValueHeads: 1, appendTokens: 3, cacheCapacity: 2, headDimension: 1 },
    { batch: 1, keyValueHeads: 1, appendTokens: 1.5, cacheCapacity: 2, headDimension: 1 },
    { batch: 1, keyValueHeads: 1, appendTokens: 1, cacheCapacity: 2, headDimension: -1 },
  ])("rejects malformed cache geometry", (shape) => {
    expect(() => planAceKvCacheWrite(shape)).toThrow();
  });

  it("rejects undersized and overlapping logical bindings before compilation", async () => {
    const shape = {
      batch: 1,
      keyValueHeads: 1,
      appendTokens: 1,
      cacheCapacity: 2,
      headDimension: 1,
    };
    const kernel = AceCorrectnessKvCacheWriteKernel.create(
      fakeKvCacheDevice(),
      "reference-bf16",
    );
    try {
      const common = completeBindings(shape);
      await expect(
        kernel.createDispatch("undersized", shape, {
          ...common,
          sourceKey: fakeBinding(3),
        }),
      ).rejects.toThrow(/source key binding does not expose 4 bytes/);

      const shared = { size: 8 } as GPUBuffer;
      await expect(
        kernel.createDispatch("overlap", shape, {
          ...common,
          sourceKey: fakeBinding(4, shared),
          sourceValue: fakeBinding(4, shared),
        }),
      ).rejects.toThrow(/source key overlaps .*source value/);
    } finally {
      kernel.destroy();
    }
  });

  it("fails closed for an unknown model profile", () => {
    expect(() =>
      aceCorrectnessKvCacheWriteWgsl("future" as never, {
        batch: 1,
        keyValueHeads: 1,
        appendTokens: 1,
        cacheCapacity: 1,
        headDimension: 1,
      }),
    ).toThrow(/Unknown ACE KV-cache model profile/);
  });
});

function completeBindings(shape: {
  batch: number;
  keyValueHeads: number;
  appendTokens: number;
  cacheCapacity: number;
  headDimension: number;
}) {
  const sourceBytes =
    shape.batch * shape.keyValueHeads * shape.appendTokens * shape.headDimension * 4;
  const cacheBytes =
    shape.batch * shape.keyValueHeads * shape.cacheCapacity * shape.headDimension * 4;
  return {
    sourceKey: fakeBinding(sourceBytes),
    sourceValue: fakeBinding(sourceBytes),
    sourceValidity: fakeBinding(shape.batch * shape.appendTokens * 4),
    cacheKey: fakeBinding(cacheBytes),
    cacheValue: fakeBinding(cacheBytes),
    cacheValidity: fakeBinding(shape.batch * shape.cacheCapacity * 4),
    rowStartPositions: fakeBinding(shape.batch * 4),
    writeStatus: fakeBinding(shape.batch * 4),
  };
}

function fakeBinding(size: number, buffer: GPUBuffer = { size } as GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size };
}

function fakeKvCacheDevice(): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 64,
      maxComputeWorkgroupSizeX: 64,
    },
  } as unknown as GPUDevice;
}
