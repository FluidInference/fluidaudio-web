import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
  ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION,
  ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
  AceOpt0050DenseK4OutputVectorFmaKernel,
  aceOpt0050DenseK4OutputVectorFmaWgsl,
  aceOpt0050PackedWeightIndex,
  packAceOpt0050DenseWeightU16,
  planAceOpt0050DenseK4OutputVectorFma,
  unpackAceOpt0050DenseWeightU16,
} from
  "../src/webgpu/kernels/dit-dense-fp16-k4-output-vector-fma.js";
import {
  buildOpt0050Cases,
  parseOpt0050ThermalGate,
} from "./browser/opt-0050-dit-dense-k4-output-vector-fma.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const CORE_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-k4-output-vector-fma.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0050-dit-dense-k4-output-vector-fma.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0050-dit-dense-k4-output-vector-fma.html",
  import.meta.url,
), "utf8");

describe("OPT-0050 dense bounded K4 output-vector FMA", () => {
  it.each([
    [2_048, 2_048],
    [2_048, 1_024],
    [2_048, 6_144],
    [6_144, 2_048],
  ])("retains OPT-0032 M32xN128 ownership for K%i/N%i", (inner, columns) => {
    const plan = planAceOpt0050DenseK4OutputVectorFma({
      rows: 2_250,
      inner,
      columns,
    });
    expect(plan).toMatchObject({
      kernelId: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
      weightLayout: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
      reductionSemantics: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION,
      tileRows: 32,
      tileColumns: 128,
      tileInner: 4,
      workgroupSize: 128,
      subgroupSize: 32,
      rowTiles: 71,
      columnTiles: columns / 128,
      innerK4Groups: inner / 4,
    });
    expect(plan.packedWeightStorageShape).toEqual([
      columns / 128,
      inner / 4,
      4,
      32,
      4,
    ]);
  });

  it("exhaustively proves `[N/128,K/4,K4,lane32,output4]` identity", () => {
    const inner = 8;
    const columns = 128;
    const logical = Uint16Array.from(
      { length: inner * columns },
      (_, index) => (index * 4051 + 17) & 0xffff,
    );
    const packed = packAceOpt0050DenseWeightU16(logical, inner, columns);
    const visited = new Uint8Array(logical.length);
    for (let k = 0; k < inner; k += 1) {
      for (let column = 0; column < columns; column += 1) {
        const physical = aceOpt0050PackedWeightIndex(k, column, inner, columns);
        expect(visited[physical]).toBe(0);
        visited[physical] = 1;
        expect(packed[physical]).toBe(logical[k * columns + column]);
      }
    }
    expect(visited.every((entry) => entry === 1)).toBe(true);
    expect(unpackAceOpt0050DenseWeightU16(packed, inner, columns))
      .toEqual(logical);
    expect(aceOpt0050PackedWeightIndex(0, 0, inner, columns)).toBe(0);
    expect(aceOpt0050PackedWeightIndex(0, 1, inner, columns)).toBe(1);
    expect(aceOpt0050PackedWeightIndex(0, 4, inner, columns)).toBe(4);
    expect(aceOpt0050PackedWeightIndex(1, 0, inner, columns)).toBe(128);
    expect(aceOpt0050PackedWeightIndex(4, 0, inner, columns)).toBe(512);
  });

  it("emits four bounded FP16 output-vector FMAs then one FP32 widening", () => {
    const source = aceOpt0050DenseK4OutputVectorFmaWgsl({
      rows: 2_250,
      inner: 2_048,
      columns: 2_048,
    });
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain("weight: array<vec4<f16>>");
    expect(source.match(/var partial[0-7] = vec4<f16>\(0\.0h\)/g))
      .toHaveLength(8);
    expect(source.match(/partial[0-7] = fma\(vec4<f16>\(a[0-7]\.[xyzw]\), b[0-3], partial[0-7]\)/g))
      .toHaveLength(32);
    expect(source.match(/acc[0-7] = acc[0-7] \+ vec4<f32>\(partial[0-7]\)/g))
      .toHaveLength(8);
    expect(source).not.toContain("dot(");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/var acc\d+ = vec4<f16>/);
    expect(CORE_SOURCE).not.toContain("OPT-0020");
  });

  it("rejects malformed packs and non-production planner shapes", () => {
    expect(() => packAceOpt0050DenseWeightU16(
      new Uint16Array(1),
      8,
      128,
    )).toThrow(/expected 1024/);
    expect(() => aceOpt0050PackedWeightIndex(8, 0, 8, 128))
      .toThrow(/out of bounds/);
    expect(() => planAceOpt0050DenseK4OutputVectorFma({
      rows: 2_250,
      inner: 1_024,
      columns: 2_048,
    })).toThrow(/non-production/);
  });

  it("caches the isolated owner and dies closed", async () => {
    const device = fakeDevice();
    const owner = AceOpt0050DenseK4OutputVectorFmaKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const bindings = fakeBindings(1, 2_048, 2_048);
    const first = await owner.createDispatch(
      "first",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    const second = await owner.createDispatch(
      "second",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    );
    expect(first.weightLayout).toBe(
      ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
    );
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
    } as unknown as GPUComputePassEncoder;
    first.encode(pass);
    second.encodeRange(pass, 0);
    expect((pass.dispatchWorkgroups as ReturnType<typeof vi.fn>).mock.calls)
      .toEqual([[16, 1, 1], [16, 1, 1]]);
    expect(() => second.encodeRange(pass, 1)).toThrow(/range must be zero/);
    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      { rows: 1, inner: 2_048, columns: 2_048 },
      bindings,
    )).rejects.toThrow(/was destroyed/);
  });

  it("freezes the four OPT-0032 shapes and adversarial gate", () => {
    const cases = buildOpt0050Cases();
    expect(cases.full.map(({ id, productionMultiplicity }) => [
      id,
      productionMultiplicity,
    ])).toEqual([
      ["h-h", 4],
      ["h-1024", 2],
      ["h-6144", 2],
      ["6144-h", 1],
    ]);
    expect(cases.adversarial.map(({ fixtureKind }) => fixtureKind)).toEqual([
      "signed-zero",
      "cancellation",
      "range",
      "long-k",
    ]);
    expect(HARNESS_SOURCE).toContain("FULL_OUTPUT_COUNT = 25_344_000");
    expect(HARNESS_SOURCE).toContain("ADVERSARIAL_OUTPUT_COUNT = 17_408");
    expect(HARNESS_SOURCE).toContain("candidateDeterministicRawU32: true");
    expect(HARNESS_SOURCE).toContain("prefixCanaryIntact");
    expect(HARNESS_SOURCE).toContain("tailRowWritten");
    expect(HARNESS_SOURCE).toContain("relativeRmsError");
    expect(HARNESS_SOURCE).toContain("nrmse");
    expect(HARNESS_SOURCE).toContain("snrDecibels");
    expect(HARNESS_SOURCE).toContain("pearsonCorrelation");
    expect(HARNESS_SOURCE).toContain("everyProductionShapeNonSlower");
    expect(HARNESS_SOURCE).toContain("REQUIRED_WEIGHTED_SPEEDUP = 1.15");
    expect(HARNESS_SOURCE).toContain("packingOutsideTimedRegion: true");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0050_RESULT__ = receipt");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
  });

  it("accepts exactly one fresh level-0 observation after 30 seconds", () => {
    const valid = thermal(10_000, 40_050, 1, 0);
    expect(parseOpt0050ThermalGate(valid, 9_000, 40_100)).toMatchObject({
      waitDurationMilliseconds: 30_050,
      checkCount: 1,
      thermalLevel: 0,
      launchDelayMilliseconds: 50,
    });
    for (const invalid of [
      thermal(10_000, 39_999, 1, 0),
      thermal(10_000, 40_050, 2, 0),
      thermal(10_000, 40_050, 1, 1),
    ]) {
      expect(() => parseOpt0050ThermalGate(invalid, 9_000, 40_100))
        .toThrow(/exactly one level-0 notifyutil check/);
    }
  });
});

function fakeDevice(): GPUDevice & {
  createShaderModule: ReturnType<typeof vi.fn>;
  createComputePipelineAsync: ReturnType<typeof vi.fn>;
  createBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: async () => ({ messages: [] }),
    })),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: () => ({}),
    })),
    createBindGroup: vi.fn(() => ({})),
  } as unknown as ReturnType<typeof fakeDevice>;
}

function fakeBindings(rows: number, inner: number, columns: number) {
  const binding = (size: number): GPUBufferBinding => Object.freeze({
    buffer: { size } as GPUBuffer,
    offset: 0,
    size,
  });
  return Object.freeze({
    activation: binding(rows * inner * 4),
    weight: binding(inner * columns * 2),
    output: binding(rows * columns * 4),
  });
}

function thermal(
  wait: number,
  checked: number,
  count: number,
  level: number,
): URLSearchParams {
  return new URLSearchParams({
    thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
    waitStartedAtEpochMilliseconds: String(wait),
    checkedAtEpochMilliseconds: String(checked),
    checkCount: String(count),
    thermalLevel: String(level),
  });
}
