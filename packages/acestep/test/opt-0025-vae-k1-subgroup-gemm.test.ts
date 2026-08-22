import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
  aceOpt0025VaeK1SubgroupGemmRangeWgsl,
  aceOpt0025VaeK1SubgroupGemmWgsl,
  packAceOpt0025VaeK1WeightU16,
  planAceOpt0025VaeK1SubgroupGemm,
  planAceOpt0025VaeK1SubgroupGemmRange,
  planAceOpt0025VaeK1SubgroupGemmRangeDispatch,
} from "../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  buildOpt0025ShapeSpecs,
  buildOpt0025TimingOrders,
  summarizeOpt0025Timing,
} from "./browser/opt-0025-vae-k1-subgroup-gemm.js";

describe("OPT-0025 VAE K1 fixed32 subgroup GEMM", () => {
  it("pins the five exact C512-window K1 production shapes", () => {
    const specs = buildOpt0025ShapeSpecs();
    expect(specs.map(({ id, frames, channels, operationMultiplicity }) => ({
      id,
      frames,
      channels,
      operationMultiplicity,
    }))).toEqual([
      { id: "c1024", frames: 5_120, channels: 1_024, operationMultiplicity: 3 },
      { id: "c512", frames: 30_720, channels: 512, operationMultiplicity: 3 },
      { id: "c256", frames: 122_880, channels: 256, operationMultiplicity: 3 },
      { id: "c128-a", frames: 491_520, channels: 128, operationMultiplicity: 3 },
      { id: "c128-b", frames: 983_040, channels: 128, operationMultiplicity: 3 },
    ]);
    expect(specs.map(({ shape }) =>
      planAceOpt0025VaeK1SubgroupGemm(shape).workgroupCount
    )).toEqual([1_280, 3_840, 7_680, 15_360, 30_720]);
    expect(specs.every(({ shape }) => {
      const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
      return plan.packedWeightStorageShape[2] === 32 &&
        plan.packedWeightStorageShape[3] === 128;
    })).toBe(true);
  });

  it("packs native [Cout,1,Cin] bits into exact [Nt,Kt,32,128] tiles", () => {
    const inputChannels = 128;
    const outputChannels = 128;
    const native = new Uint16Array(inputChannels * outputChannels);
    for (let column = 0; column < outputChannels; column += 1) {
      for (let inner = 0; inner < inputChannels; inner += 1) {
        native[column * inputChannels + inner] = column * 32 + inner;
      }
    }
    const packed = packAceOpt0025VaeK1WeightU16(
      native,
      inputChannels,
      outputChannels,
    );
    for (let inner = 0; inner < inputChannels; inner += 1) {
      for (let column = 0; column < outputChannels; column += 1) {
        expect(packed[inner * 128 + column]).toBe(
          native[column * inputChannels + inner],
        );
      }
    }
  });

  it("emits fixed32 subgroups, increasing-Cin FP32 accumulation, and FP16 stores", () => {
    const source = aceOpt0025VaeK1SubgroupGemmWgsl(
      buildOpt0025ShapeSpecs()[1]!.shape,
    );
    expect(ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID).toContain("opt-0025");
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain("var acc0 = load_bias(column);");
    expect(source).toContain("inner_in_tile += 1u");
    expect(source).toContain("subgroupBroadcast(lane_a, 7u)");
    expect(source).toContain("acc7 = acc7 + vec4<f32>(a7) * b;");
    expect(source).toContain("output[output_base] = f16(acc0.x);");
  });

  it("rejects geometry outside the exact square production family", () => {
    const base = buildOpt0025ShapeSpecs()[0]!.shape;
    expect(() => planAceOpt0025VaeK1SubgroupGemm({
      ...base,
      outputChannels: 512,
    })).toThrow(/square/);
    expect(() => planAceOpt0025VaeK1SubgroupGemm({
      ...base,
      kernelSize: 7,
    })).toThrow(/K1/);
    expect(() => packAceOpt0025VaeK1WeightU16(
      new Uint16Array(1),
      128,
      128,
    )).toThrow(/length/);
  });

  it("maps cooperative complete-row ranges without changing K1 math", () => {
    const plan = planAceOpt0025VaeK1SubgroupGemm(
      buildOpt0025ShapeSpecs()[1]!.shape,
    );
    expect(planAceOpt0025VaeK1SubgroupGemmRange(plan, {
      base: 64 * plan.columns,
      count: 96 * plan.columns,
    })).toEqual({
      base: 64 * plan.columns,
      count: 96 * plan.columns,
      firstRow: 64,
      rowCount: 96,
      workgroupCount: 12,
    });
    expect(planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
      plan,
      planAceOpt0025VaeK1SubgroupGemmRange(plan, {
        base: 64 * plan.columns,
        count: 96 * plan.columns,
      }),
    )).toEqual({
      mapping: "flat-x",
      workgroupsX: 12,
      workgroupsY: 1,
      workgroupsZ: 1,
    });
    const source = aceOpt0025VaeK1SubgroupGemmRangeWgsl(plan.shape);
    expect(source).toContain("@binding(4) var<uniform> output_range");
    expect(source).toContain("first_row + row_tile * 32u");
    expect(source).toContain("lane_row < first_row + range_row_count");
    expect(() => planAceOpt0025VaeK1SubgroupGemmRange(plan, {
      base: 1,
      count: plan.columns,
    })).toThrow(/complete in-bounds rows/);
  });

  it("shards C2378-scale cooperative ranges over row-Y without changing C512", () => {
    const boundaryShape = {
      batch: 1,
      inputFrames: 65_535 * 32,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    } as const;
    const boundary = planAceOpt0025VaeK1SubgroupGemm(boundaryShape);
    const boundaryRange = planAceOpt0025VaeK1SubgroupGemmRange(boundary, {
      base: 0,
      count: boundary.outputElements,
    });
    expect(boundary.workgroupCount).toBe(65_535);
    expect(planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
      boundary,
      boundaryRange,
    )).toEqual({
      mapping: "flat-x",
      workgroupsX: 65_535,
      workgroupsY: 1,
      workgroupsZ: 1,
    });

    const c2378Scale = planAceOpt0025VaeK1SubgroupGemm({
      ...boundaryShape,
      inputFrames: 2_378 * 1_920,
    });
    expect(c2378Scale.workgroupCount).toBe(142_680);
    const boundedRange = planAceOpt0025VaeK1SubgroupGemmRange(
      c2378Scale,
      { base: 3 * 1_048_576, count: 1_048_576 },
    );
    expect(planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
      c2378Scale,
      boundedRange,
    )).toEqual({
      mapping: "column-x-row-y",
      workgroupsX: 1,
      workgroupsY: 256,
      workgroupsZ: 1,
    });
    const source = aceOpt0025VaeK1SubgroupGemmRangeWgsl(
      c2378Scale.shape,
    );
    expect(source).toContain("let row_tile = group.y;");
    expect(source).toContain("let column_tile = group.x;");
    expect(source).toContain(
      "first_row + row_tile * 32u +",
    );

    const onePastFlat = planAceOpt0025VaeK1SubgroupGemm({
      ...boundaryShape,
      inputFrames: 65_535 * 32 + 1,
    });
    expect(onePastFlat.workgroupCount).toBe(65_536);
    expect(() => planAceOpt0025VaeK1SubgroupGemmRange(
      onePastFlat,
      { base: 0, count: onePastFlat.outputElements },
    )).toThrow(/exceeds the 2D dispatch limits/);
  });

  it("uses balanced AB/BA rounds and reports true median-four throughput", () => {
    const orders = buildOpt0025TimingOrders();
    expect(orders).toHaveLength(20);
    for (let shapeIndex = 0; shapeIndex < 5; shapeIndex += 1) {
      const shapeOrders = orders.filter((entry) => entry.shapeIndex === shapeIndex);
      expect(shapeOrders.filter(({ order }) => order[0] === "current")).toHaveLength(2);
      expect(shapeOrders.filter(({ order }) => order[0] === "candidate")).toHaveLength(2);
    }
    const summary = summarizeOpt0025Timing(buildOpt0025ShapeSpecs().map(({ id }) => ({
      id,
      samples: {
        current: [12, 8, 10, 14],
        candidate: [4, 2, 3, 5],
      },
    })));
    expect(summary).toMatchObject({
      samplesPerArmPerShape: 4,
      c512DecoderWindow: {
        operationCount: 15,
        currentMilliseconds: 165,
        candidateMilliseconds: 52.5,
        speedup: 165 / 52.5,
      },
    });
  });

  it("keeps the primitive browser harness isolated from the decoder", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0025-vae-k1-subgroup-gemm.html",
      import.meta.url,
    ), "utf8");
    const harness = readFileSync(new URL(
      "./browser/opt-0025-vae-k1-subgroup-gemm.ts",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("opt-0025-vae-k1-subgroup-gemm.ts");
    expect(harness).toContain("fullRawFp16Identity: true");
    expect(harness).toContain("compilationAllocationUploadExcludedFromTiming: true");
    expect(harness).not.toContain("vae-fp16-decoder");
  });
});
