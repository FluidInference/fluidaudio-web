import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0032_DENSE_TILE_COLUMNS,
  ACE_OPT_0032_DENSE_TILE_INNER,
  ACE_OPT_0032_DENSE_TILE_ROWS,
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
  aceOpt0032DenseK4PartialsWgsl,
  aceOpt0032PackedWeightIndex,
  packAceOpt0032DenseWeightU16,
  planAceOpt0032DenseK4Partials,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0037_DENSE_K4_KERNEL_ID,
  ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT,
  AceOpt0037DenseK4ProductionKernel,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-production.js";

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0032-dit-dense-fp16-k4-partials.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0032-dit-dense-fp16-k4-partials.html",
  import.meta.url,
), "utf8");

describe("OPT-0032 dense FP16 K4 partials", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("plans fixed32 WG128 M32xN128 for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0032DenseK4Partials({
      rows: 2_250,
      inner,
      columns,
    });
    expect(plan).toMatchObject({
      tileRows: ACE_OPT_0032_DENSE_TILE_ROWS,
      tileColumns: ACE_OPT_0032_DENSE_TILE_COLUMNS,
      tileInner: ACE_OPT_0032_DENSE_TILE_INNER,
      workgroupSize: ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
      subgroupSize: ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
      rowTiles: 71,
      columnTiles: columns / 128,
      innerK4Groups: inner / 4,
      workgroupCount: 71 * columns / 128,
      outputRangeCount: 1,
    });
    expect(plan.packedWeightStorageShape).toEqual([
      columns / 128,
      inner / 4,
      4,
      32,
      4,
    ]);
  });

  it("packs B as [N/128,K/4,output4,lane32,K4] without loss", () => {
    const inner = 8;
    const columns = 128;
    const logical = Uint16Array.from(
      { length: inner * columns },
      (_, index) => index + 1,
    );
    const packed = packAceOpt0032DenseWeightU16(logical, inner, columns);
    const visited = new Set<number>();
    for (let k = 0; k < inner; k += 1) {
      for (let column = 0; column < columns; column += 1) {
        const physical = aceOpt0032PackedWeightIndex(
          k,
          column,
          inner,
          columns,
        );
        visited.add(physical);
        expect(packed[physical]).toBe(logical[k * columns + column]);
      }
    }
    expect(visited.size).toBe(logical.length);
    expect(aceOpt0032PackedWeightIndex(0, 0, inner, columns)).toBe(0);
    expect(aceOpt0032PackedWeightIndex(3, 0, inner, columns)).toBe(3);
    expect(aceOpt0032PackedWeightIndex(0, 4, inner, columns)).toBe(4);
    expect(aceOpt0032PackedWeightIndex(0, 1, inner, columns)).toBe(128);
    expect(aceOpt0032PackedWeightIndex(4, 0, inner, columns)).toBe(512);
  });

  it("emits four native FP16 K4 dots and only widens the partial vector", () => {
    const wgsl = aceOpt0032DenseK4PartialsWgsl({
      rows: 2_250,
      inner: 2_048,
      columns: 2_048,
    });
    expect(ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID).toMatch(
      /fixed32-wg128-m32-n128-v1$/,
    );
    expect(ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT).toBe(
      "ace-opt-0032-b-n128-k4-output4-lane32-k4-v1",
    );
    expect(wgsl).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(wgsl).toContain("weight: array<vec4<f16>>");
    expect(wgsl).toContain("var lane_a = vec4<f16>(0.0h)");
    expect(wgsl.match(/subgroupBroadcast\(lane_a, [0-7]u\)/g)).toHaveLength(8);
    expect(wgsl.match(/dot\(a0, b[0-3]\)/g)).toHaveLength(4);
    expect(wgsl).toContain("let partial0 = vec4<f16>(");
    expect(wgsl).toContain("acc0 = acc0 + vec4<f32>(partial0)");
    expect(wgsl).toContain("var acc0 = vec4<f32>(0.0)");
    expect(wgsl).not.toContain("var<workgroup>");
    expect(wgsl).not.toContain("workgroupBarrier");
    expect(wgsl).not.toMatch(/var acc\d+\s*=\s*vec4<f16>/);
    expect(KERNEL_SOURCE).not.toContain("OPT-0020");
  });

  it("rejects unregistered geometry and malformed benchmark packing", () => {
    expect(() => planAceOpt0032DenseK4Partials({
      rows: 2_250,
      inner: 1_024,
      columns: 2_048,
    })).toThrow(/non-production/);
    expect(() => packAceOpt0032DenseWeightU16(
      new Uint16Array(1),
      8,
      128,
    )).toThrow(/expected 1024/);
    expect(() => aceOpt0032PackedWeightIndex(8, 0, 8, 128)).toThrow(
      /out of bounds/,
    );
  });

  it("binds the measured owner behind the distinct production profile and fails closed after destroy", async () => {
    const device = {
      features: new Set(["shader-f16", "subgroups"]),
      limits: {
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
      },
      createShaderModule: () => ({
        getCompilationInfo: async () => ({ messages: [] }),
      }),
      createComputePipelineAsync: async () => ({
        getBindGroupLayout: () => ({}),
      }),
      createBindGroup: () => ({}),
    } as unknown as GPUDevice;
    const binding = (size: number): GPUBufferBinding => ({
      buffer: { size } as GPUBuffer,
      offset: 0,
      size,
    });
    const bindings = {
      activation: binding(2_048 * 4),
      weight: binding(2_048 * 2_048 * 2),
      output: binding(2_048 * 4),
    };
    const kernel = AceOpt0037DenseK4ProductionKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const dispatch = await kernel.createDispatch(
      "opt-0037-production",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    expect(ACE_OPT_0037_DENSE_K4_KERNEL_ID).toBe(
      ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
    );
    expect(dispatch.weightLayout).toBe(ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT);
    const plan = dispatch.plan as ReturnType<
      typeof planAceOpt0032DenseK4Partials
    >;
    expect(plan.packedWeightStorageShape).toEqual([
      16, 512, 4, 32, 4,
    ]);
    kernel.destroy();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "opt-0037-after-destroy",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    )).rejects.toThrow(/production K4 kernel was destroyed/);
  });

  it("gates READY on full outputs and adversarial numerics before balanced timing", () => {
    for (const shape of [
      'fullSpec("h-h", 2_048, 2_048, 4, 0)',
      'fullSpec("h-1024", 2_048, 1_024, 2, 1)',
      'fullSpec("h-6144", 2_048, 6_144, 2, 2)',
      'fullSpec("6144-h", 6_144, 2_048, 1, 3)',
    ]) expect(HARNESS_SOURCE).toContain(shape);
    for (const adversarial of [
      '"signed-zero"',
      '"cancellation"',
      '"range"',
      '"long-k"',
    ]) expect(HARNESS_SOURCE).toContain(adversarial);
    expect(HARNESS_SOURCE).toContain("FULL_OUTPUT_COUNT = 25_344_000");
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(HARNESS_SOURCE).toContain("prefixCanaryIntact");
    expect(HARNESS_SOURCE).toContain("candidateDeterministicRawU32: true");
    expect(HARNESS_SOURCE).toContain("classChangeCount");
    expect(HARNESS_SOURCE).toContain("candidateNonFiniteCount");
    expect(HARNESS_SOURCE).toContain("relativeRmsError");
    expect(HARNESS_SOURCE).toContain("nrmse");
    expect(HARNESS_SOURCE).toContain("snrDecibels");
    expect(HARNESS_SOURCE).toContain("pearsonCorrelation");
    expect(HARNESS_SOURCE).toContain("completedBeforeReady: true");
    expect(HARNESS_SOURCE).toContain(
      "READY — four full outputs and all adversarial screens passed",
    );
    expect(HARNESS_SOURCE).toContain("productionMultiplicity: 4 | 2 | 1");
    expect(HARNESS_SOURCE).toContain("requiredWeightedSpeedup");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0032_RESULT__ = receipt");
    expect(HARNESS_SOURCE).not.toContain("workgroupBarrier");
    expect(HARNESS_SOURCE).not.toContain("OPT-0020");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain("leave the machine nominal and idle for 30 seconds");
    expect(HARNESS_HTML).toContain(
      "opt-0032-dit-dense-fp16-k4-partials.ts",
    );
  });
});
