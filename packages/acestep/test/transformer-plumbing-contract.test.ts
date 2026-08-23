import { describe, expect, it } from "vitest";

import {
  aceCorrectnessEmbeddingWgsl,
  planAceEmbedding,
} from "../src/webgpu/kernels/embedding.js";
import {
  AceCorrectnessTensorCopyKernel,
  aceCorrectnessAxisCopyWgsl,
  aceCorrectnessGatherRowsWgsl,
  aceCorrectnessStablePackGatherWgsl,
  aceCorrectnessStablePackIndicesWgsl,
  planAceConcat,
  planAceCrop,
  planAceGatherRows,
  planAceRepeat,
  planAceRightPad,
  planAceStablePack,
} from "../src/webgpu/kernels/tensor-copy.js";
import {
  AceCorrectnessTransformerPlumbingKernel,
  aceCorrectnessElementwiseWgsl,
  aceCorrectnessHeadTransformWgsl,
  planAceHeadTransform,
  planAceTransformerTensor,
} from "../src/webgpu/kernels/transformer-plumbing.js";

describe("ACE transformer correctness plumbing", () => {
  it("plans shard-complete embeddings without assembling a vocabulary buffer", () => {
    expect(
      planAceEmbedding(
        { tokenCount: 4, width: 3, vocabularySize: 6 },
        [
          { firstRow: 0, rowCount: 2 },
          { firstRow: 2, rowCount: 4 },
        ],
      ),
    ).toEqual({
      tokenCount: 4,
      width: 3,
      vocabularySize: 6,
      outputElements: 12,
      workgroupsX: 1,
      workgroupsY: 1,
      shards: [
        { firstRow: 0, rowCount: 2 },
        { firstRow: 2, rowCount: 4 },
      ],
    });
  });

  it.each([
    [[{ firstRow: 1, rowCount: 5 }], "start"],
    [[{ firstRow: 0, rowCount: 2 }, { firstRow: 3, rowCount: 3 }], "start"],
    [[{ firstRow: 0, rowCount: 5 }], "cover"],
    [[{ firstRow: 0, rowCount: 7 }], "cover"],
    [[], "at least one"],
  ] as const)("rejects embedding shard gaps, overlaps, and incomplete coverage", (shards, message) => {
    expect(() =>
      planAceEmbedding(
        { tokenCount: 1, width: 4, vocabularySize: 6 },
        shards,
      ),
    ).toThrow(message);
  });

  it("decodes shard-local packed BF16 and has a true FP16 storage path", () => {
    const shape = { tokenCount: 3, width: 5, vocabularySize: 7 };
    const bf16 = aceCorrectnessEmbeddingWgsl(
      "reference-bf16",
      shape,
      { firstRow: 2, rowCount: 3 },
    );
    expect(bf16).toContain("const FIRST_ROW: u32 = 2u");
    expect(bf16).toContain("let local_element = (token_id - FIRST_ROW) * WIDTH");
    expect(bf16).toContain("weight: array<u32>");
    expect(bf16).toContain("bits16 << 16u");
    expect(bf16).toContain("token_id >= VOCABULARY_SIZE");

    const fp16 = aceCorrectnessEmbeddingWgsl(
      "raw-fp16",
      shape,
      { firstRow: 0, rowCount: 7 },
    );
    expect(fp16).toContain("enable f16;");
    expect(fp16).toContain("weight: array<f16>");
    expect(fp16).toContain("output: array<f16>");
  });

  it("plans physical head transforms and transformer broadcasts", () => {
    expect(
      planAceHeadTransform({
        batch: 2,
        tokens: 3,
        heads: 2,
        headDimension: 2,
      }),
    ).toEqual({
      batch: 2,
      tokens: 3,
      heads: 2,
      headDimension: 2,
      width: 4,
      elements: 24,
      workgroupsX: 1,
      workgroupsY: 1,
    });
    expect(planAceTransformerTensor({ batch: 2, tokens: 3, width: 4 })).toEqual({
      batch: 2,
      tokens: 3,
      width: 4,
      elements: 24,
      broadcastElements: 8,
      workgroupsX: 1,
      workgroupsY: 1,
    });
  });

  it("emits exact split-head and inverse merge-head indexing", () => {
    const shape = { batch: 2, tokens: 3, heads: 2, headDimension: 4 };
    const split = aceCorrectnessHeadTransformWgsl(
      "reference-bf16",
      "split-heads",
      shape,
    );
    expect(split).toContain("let token = (index / HEAD_DIMENSION) % TOKENS");
    expect(split).toContain("((batch * TOKENS + token) * HEADS + head)");
    const merge = aceCorrectnessHeadTransformWgsl(
      "raw-fp16",
      "merge-heads",
      shape,
    );
    expect(merge).toContain("enable f16;");
    expect(merge).toContain("let head = (index / HEAD_DIMENSION) % HEADS");
    expect(merge).toContain("((batch * HEADS + head) * TOKENS + token)");
  });

  it("keeps the reference elementwise graph FP32 and broadcasts by batch/feature", () => {
    const shape = { batch: 2, tokens: 3, width: 4 };
    const adaln = aceCorrectnessElementwiseWgsl("reference-bf16", "adaln", shape);
    expect(adaln).toContain("normalized: array<f32>");
    expect(adaln).toContain("let batch = index / (TOKENS * WIDTH)");
    expect(adaln).toContain(
      "normalized[index] * (1.0 + scale[broadcast_index]) + shift[broadcast_index]",
    );
    const gated = aceCorrectnessElementwiseWgsl(
      "reference-bf16",
      "gated-residual",
      shape,
    );
    expect(gated).toContain(
      "residual[index] + branch[index] * gate[broadcast_index]",
    );
  });

  it("states the raw-FP16 nonlinear and arithmetic rounding boundaries", () => {
    const shape = { batch: 1, tokens: 2, width: 3 };
    const silu = aceCorrectnessElementwiseWgsl("raw-fp16", "silu", shape);
    expect(silu).toContain("let value = f32(input[index])");
    expect(silu).toContain("output[index] = f16(value / (1.0 + exp(-value)))");
    const swiglu = aceCorrectnessElementwiseWgsl("raw-fp16", "swiglu", shape);
    expect(swiglu).toContain("let activated = f16(");
    expect(swiglu).toContain("output[index] = activated * up[index]");
    const adaln = aceCorrectnessElementwiseWgsl("raw-fp16", "adaln", shape);
    expect(adaln).toContain("normalized[index] * (f16(1.0) + scale[broadcast_index])");
  });

  it.each([
    "residual-add",
    "broadcast-add",
    "broadcast-multiply",
    "silu",
    "swiglu",
    "adaln",
    "gated-residual",
  ] as const)("emits a bounded %s shader", (operation) => {
    const source = aceCorrectnessElementwiseWgsl(
      "reference-bf16",
      operation,
      { batch: 1, tokens: 2, width: 3 },
    );
    expect(source).toContain("if (index >= ELEMENTS) { return; }");
    expect(source).toContain("@workgroup_size(256, 1, 1)");
  });

  it("plans stable generic row gather and axis helpers", () => {
    expect(
      planAceGatherRows({ outer: 2, sourceRows: 3, outputRows: 2, width: 2 }),
    ).toEqual({
      outer: 2,
      sourceRows: 3,
      outputRows: 2,
      width: 2,
      sourceElements: 12,
      indexElements: 4,
      outputElements: 8,
      workgroupsX: 1,
      workgroupsY: 1,
    });
    expect(
      planAceRightPad({ outer: 2, inputLength: 2, outputLength: 3, inner: 2 }),
    ).toMatchObject({ operation: "right-pad", inputElements: 8, outputElements: 12 });
    expect(
      planAceCrop({ outer: 1, inputLength: 5, offset: 1, outputLength: 3, inner: 2 }),
    ).toMatchObject({ operation: "crop", inputElements: 10, outputElements: 6 });
    expect(
      planAceRepeat({ outer: 1, inputLength: 2, repeats: 3, inner: 2 }),
    ).toMatchObject({ operation: "repeat", outputLength: 6, outputElements: 12 });
    expect(
      planAceConcat({ outer: 2, leftLength: 2, rightLength: 1, inner: 2 }),
    ).toMatchObject({
      operation: "concat",
      inputElements: 8,
      secondElements: 4,
      outputElements: 12,
    });
  });

  it("emits bounded gather, right-pad, crop, tiled repeat, and concat mappings", () => {
    const gather = aceCorrectnessGatherRowsWgsl("reference-bf16", {
      outer: 2,
      sourceRows: 3,
      outputRows: 2,
      width: 2,
    });
    expect(gather).toContain("source_row = indices[outer * OUTPUT_ROWS + output_row]");
    expect(gather).toContain("source_row >= SOURCE_ROWS");

    const pad = aceCorrectnessAxisCopyWgsl(
      "reference-bf16",
      planAceRightPad({ outer: 1, inputLength: 2, outputLength: 3, inner: 2 }),
    );
    expect(pad).toContain("if (output_axis < INPUT_LENGTH)");
    expect(pad).toContain("output[index] = 0.0");
    const crop = aceCorrectnessAxisCopyWgsl(
      "reference-bf16",
      planAceCrop({ outer: 1, inputLength: 4, offset: 1, outputLength: 2, inner: 2 }),
    );
    expect(crop).toContain("source_axis = output_axis + OFFSET");
    const repeat = aceCorrectnessAxisCopyWgsl(
      "raw-fp16",
      planAceRepeat({ outer: 1, inputLength: 2, repeats: 2, inner: 2 }),
    );
    expect(repeat).toContain("source_axis = output_axis % INPUT_LENGTH");
    expect(repeat).toContain("array<f16>");
    const concat = aceCorrectnessAxisCopyWgsl(
      "reference-bf16",
      planAceConcat({ outer: 1, leftLength: 2, rightLength: 1, inner: 2 }),
    );
    expect(concat).toContain("second_axis = output_axis - INPUT_LENGTH");
  });

  it("plans and emits ACE's stable valid-first sequence packing", () => {
    const shape = { batch: 2, leftLength: 2, rightLength: 1, width: 2 };
    expect(planAceStablePack(shape)).toEqual({
      ...shape,
      packedLength: 3,
      leftElements: 8,
      rightElements: 4,
      packedRows: 6,
      outputElements: 12,
      indexWorkgroupsX: 1,
      indexWorkgroupsY: 1,
      gatherWorkgroupsX: 1,
      gatherWorkgroupsY: 1,
    });
    const indices = aceCorrectnessStablePackIndicesWgsl(shape);
    expect(indices).toContain("let wants_valid = destination < valid_count");
    expect(indices).toContain("if (mask_at(batch, source) == wants_valid)");
    expect(indices).toContain("output_mask[index] = select(0u, 1u, wants_valid)");
    const gather = aceCorrectnessStablePackGatherWgsl("reference-bf16", shape);
    expect(gather).toContain("source < LEFT_LENGTH");
    expect(gather).toContain("right_row = source - LEFT_LENGTH");
  });

  it("surfaces stable-pack index and gather as dependent cooperative quanta", async () => {
    const kernel = AceCorrectnessTensorCopyKernel.create(
      fakeDevice(false),
      "reference-bf16",
    );
    try {
      const dispatch = await kernel.createStablePackDispatch(
        "stable",
        { batch: 1, leftLength: 1, rightLength: 1, width: 1 },
        {
          left: fakeBinding(4),
          right: fakeBinding(4),
          leftMask: fakeBinding(4),
          rightMask: fakeBinding(4),
          indicesScratch: fakeBinding(8),
          output: fakeBinding(8),
          outputMask: fakeBinding(8),
        },
      );
      expect(dispatch.cooperativeQuanta?.map(({ id }) => id)).toEqual([
        "stable-indices",
        "stable-gather",
      ]);
    } finally {
      kernel.destroy();
    }
  });

  it.each([
    () => planAceHeadTransform({ batch: 1, tokens: 0, heads: 2, headDimension: 4 }),
    () => planAceTransformerTensor({ batch: 1, tokens: 2, width: Number.NaN }),
    () => planAceGatherRows({ outer: 1, sourceRows: 2, outputRows: 0, width: 2 }),
    () => planAceRightPad({ outer: 1, inputLength: 3, outputLength: 2, inner: 1 }),
    () => planAceCrop({ outer: 1, inputLength: 3, offset: 2, outputLength: 2, inner: 1 }),
    () => planAceRepeat({ outer: 1, inputLength: 2, repeats: 0, inner: 1 }),
    () => planAceConcat({ outer: 1, leftLength: 2, rightLength: 0, inner: 1 }),
  ])("rejects malformed transformer movement geometry", (operation) => {
    expect(operation).toThrow();
  });

  it("rejects geometry that cannot be represented by WGSL u32 indices", () => {
    expect(() =>
      planAceEmbedding(
        { tokenCount: 1, width: 1, vocabularySize: 0x1_0000_0000 },
        [{ firstRow: 0, rowCount: 0x1_0000_0000 }],
      ),
    ).toThrow(/u32/);
    expect(() =>
      planAceTransformerTensor({ batch: 65_536, tokens: 65_536, width: 1 }),
    ).toThrow(/u32/);
  });

  it("fails closed for unknown JavaScript profiles and operations", () => {
    expect(() =>
      aceCorrectnessEmbeddingWgsl(
        "unknown" as never,
        { tokenCount: 1, width: 2, vocabularySize: 2 },
        { firstRow: 0, rowCount: 2 },
      ),
    ).toThrow(/Unknown ACE embedding model profile/);
    expect(() =>
      aceCorrectnessElementwiseWgsl(
        "reference-bf16",
        "unknown" as never,
        { batch: 1, tokens: 1, width: 1 },
      ),
    ).toThrow(/Unknown ACE elementwise operation/);
    expect(() =>
      aceCorrectnessHeadTransformWgsl(
        "reference-bf16",
        "unknown" as never,
        { batch: 1, tokens: 1, heads: 1, headDimension: 1 },
      ),
    ).toThrow(/Unknown ACE head transform/);
    expect(() =>
      aceCorrectnessAxisCopyWgsl("reference-bf16", {
        ...planAceRightPad({
          outer: 1,
          inputLength: 1,
          outputLength: 2,
          inner: 1,
        }),
        operation: "unknown" as never,
      }),
    ).toThrow(/Unknown ACE axis-copy operation/);
  });

  it("validates required feature support and refuses work after destroy", async () => {
    expect(() =>
      AceCorrectnessTransformerPlumbingKernel.create(
        fakeDevice(false),
        "raw-fp16",
      ),
    ).toThrow(/shader-f16/);
    const kernel = AceCorrectnessTransformerPlumbingKernel.create(
      fakeDevice(false),
      "reference-bf16",
    );
    kernel.destroy();
    await expect(
      kernel.createSiluDispatch(
        "dead",
        { batch: 1, tokens: 1, width: 1 },
        { input: fakeBinding(4), output: fakeBinding(4) },
      ),
    ).rejects.toThrow(/destroyed/);
  });

  it("validates exposed bytes and rejects overlapping output aliases", async () => {
    const kernel = AceCorrectnessTransformerPlumbingKernel.create(
      fakeDevice(false),
      "reference-bf16",
    );
    try {
      await expect(
        kernel.createSiluDispatch(
          "short",
          { batch: 1, tokens: 2, width: 2 },
          { input: fakeBinding(16), output: fakeBinding(12) },
        ),
      ).rejects.toThrow(/does not expose 16 bytes/);
      const shared = fakeBuffer(32);
      await expect(
        kernel.createSiluDispatch(
          "alias",
          { batch: 1, tokens: 2, width: 2 },
          {
            input: { buffer: shared, offset: 0, size: 16 },
            output: { buffer: shared, offset: 8, size: 16 },
          },
        ),
      ).rejects.toThrow(/must not overlap/);
    } finally {
      kernel.destroy();
    }
  });
});

function fakeDevice(shaderF16: boolean): GPUDevice {
  const pipeline = {
    getBindGroupLayout: () => ({}) as GPUBindGroupLayout,
  } as unknown as GPUComputePipeline;
  return {
    features: new Set(shaderF16 ? ["shader-f16"] : []),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
    createShaderModule: () => ({}) as GPUShaderModule,
    createComputePipelineAsync: async () => pipeline,
    createBindGroup: () => ({}) as GPUBindGroup,
  } as unknown as GPUDevice;
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}
