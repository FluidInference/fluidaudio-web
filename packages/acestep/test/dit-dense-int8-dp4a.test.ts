import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
  ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
  ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
  ACE_OPT_0058_GROUP_SIZES,
  ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE,
  AceOpt0058DenseInt8Dp4aKernel,
  aceOpt0058DenseInt8Dp4aWgsl,
  aceOpt0058DynamicQuantizerWgsl,
  aceOpt0058PackedWeightWordIndex,
  aceOpt0058WeightScaleIndex,
  packAceOpt0058SignedI8x4,
  planAceOpt0058DenseInt8,
  quantizeAndPackAceOpt0058DenseWeight,
  unpackAceOpt0058DenseWeightI8,
  unpackAceOpt0058SignedI8,
  type AceOpt0058GroupSize,
} from "../src/webgpu/kernels/dit-dense-int8-dp4a.js";
import {
  buildOpt0058Cases,
  parseOpt0058ThermalGate,
} from "./browser/opt-0058-dit-dense-int8-dp4a.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const CORE_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/dit-dense-int8-dp4a.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0058-dit-dense-int8-dp4a.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0058-dit-dense-int8-dp4a.html",
  import.meta.url,
), "utf8");

describe("OPT-0058 dynamic-int8 DP4a dense screen", () => {
  it.each(ACE_OPT_0058_GROUP_SIZES.flatMap((groupSize) => [
    [2_048, 2_048, groupSize],
    [2_048, 1_024, groupSize],
    [2_048, 6_144, groupSize],
    [6_144, 2_048, groupSize],
  ] as const))(
    "plans M32xN128 fixed32 ownership for K%i/N%i/G%i",
    (inner, columns, groupSize) => {
      const plan = planAceOpt0058DenseInt8(
        { rows: 2_250, inner, columns },
        groupSize,
      );
      expect(plan).toMatchObject({
        kernelId: ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
        weightLayout: ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
        scaleLayout: ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
        tileRows: 32,
        tileColumns: 128,
        tileInner: groupSize,
        workgroupSize: 128,
        subgroupSize: 32,
        groupSize,
        rowTiles: 71,
        columnTiles: columns / 128,
        innerGroups: inner / groupSize,
        packedKPerGroup: groupSize / 4,
        maximumIntegerPartialMagnitude: 127 * 127 * groupSize,
        packedActivationWords: 2_250 * inner / 4,
        packedWeightWords: inner * columns / 4,
      });
      expect(plan.packedWeightStorageShape).toEqual([
        columns / 128,
        inner / groupSize,
        groupSize / 4,
        4,
        32,
      ]);
      expect(plan.weightScaleStorageShape).toEqual([
        columns / 128,
        inner / groupSize,
        32,
        4,
      ]);
      expect(plan.packedActivationStorageShape).toEqual([
        2_250,
        inner / groupSize,
        groupSize / 4,
      ]);
      expect(plan.maximumIntegerPartialMagnitude).toBeLessThan(2 ** 31);
    },
  );

  it.each(ACE_OPT_0058_GROUP_SIZES)(
    "exhaustively proves signed-byte pack/index/inverse for G%i",
    (groupSize) => {
      const inner = groupSize * 2;
      const columns = 128;
      const logical = new Float32Array(inner * columns);
      const expected = new Int8Array(inner * columns);
      for (let column = 0; column < columns; column += 1) {
        for (let group = 0; group < 2; group += 1) {
          let maximum = 0;
          for (let offset = 0; offset < groupSize; offset += 1) {
            const k = group * groupSize + offset;
            const value = group === 0
              ? 0
              : ((k * 17 + column * 13) % 253 - 126) / 19;
            logical[k * columns + column] = value;
            maximum = Math.max(maximum, Math.abs(value));
          }
          const scale = maximum === 0 ? 0 : Math.fround(maximum / 127);
          for (let offset = 0; offset < groupSize; offset += 1) {
            const k = group * groupSize + offset;
            const normalized = scale === 0
              ? 0
              : Math.fround(logical[k * columns + column]! / scale);
            expected[k * columns + column] = Math.max(
              -127,
              Math.min(
                127,
                Math.sign(normalized) * Math.floor(Math.abs(normalized) + 0.5),
              ),
            );
          }
        }
      }
      const first = quantizeAndPackAceOpt0058DenseWeight(
        logical,
        inner,
        columns,
        groupSize,
      );
      const second = quantizeAndPackAceOpt0058DenseWeight(
        logical,
        inner,
        columns,
        groupSize,
      );
      expect(first.packed).toEqual(second.packed);
      expect(first.scales).toEqual(second.scales);
      expect(first.zeroGroupCount).toBe(columns);
      expect(unpackAceOpt0058DenseWeightI8(
        first.packed,
        inner,
        columns,
        groupSize,
      )).toEqual(expected);

      const visitedWords = new Uint8Array(first.packed.length);
      const visitedScales = new Uint8Array(first.scales.length);
      for (let column = 0; column < columns; column += 1) {
        for (let group = 0; group < 2; group += 1) {
          const scaleIndex = aceOpt0058WeightScaleIndex(
            column,
            group,
            inner,
            columns,
            groupSize,
          );
          expect(visitedScales[scaleIndex]).toBe(0);
          visitedScales[scaleIndex] = 1;
          for (let packedK = 0; packedK < groupSize / 4; packedK += 1) {
            const wordIndex = aceOpt0058PackedWeightWordIndex(
              group,
              packedK,
              column,
              inner,
              columns,
              groupSize,
            );
            expect(visitedWords[wordIndex]).toBe(0);
            visitedWords[wordIndex] = 1;
          }
        }
      }
      expect(visitedWords.every((value) => value === 1)).toBe(true);
      expect(visitedScales.every((value) => value === 1)).toBe(true);
    },
  );

  it("round-trips all symmetric signed bytes and rejects non-finite weights", () => {
    for (let x = -127; x <= 127; x += 1) {
      const word = packAceOpt0058SignedI8x4(x, -x, 0, 127);
      expect(unpackAceOpt0058SignedI8(word, 0)).toBe(x);
      expect(unpackAceOpt0058SignedI8(word, 1)).toBe(x === 0 ? 0 : -x);
      expect(unpackAceOpt0058SignedI8(word, 2)).toBe(0);
      expect(unpackAceOpt0058SignedI8(word, 3)).toBe(127);
    }
    expect(() => packAceOpt0058SignedI8x4(-128, 0, 0, 0)).toThrow(/\[-127,127\]/);
    const invalid = new Float32Array(32 * 128);
    invalid[19] = Number.NaN;
    expect(() => quantizeAndPackAceOpt0058DenseWeight(invalid, 32, 128, 32))
      .toThrow(/rejects non-finite/);
  });

  it.each(ACE_OPT_0058_GROUP_SIZES)(
    "emits a bounded exact-i32 G%i DP4a reduction and dynamic quantizer",
    (groupSize) => {
      const shape = { rows: 2_250, inner: 6_144, columns: 2_048 };
      const contraction = aceOpt0058DenseInt8Dp4aWgsl(shape, groupSize);
      const quantizer = aceOpt0058DynamicQuantizerWgsl(shape, groupSize);
      expect(contraction).toContain(
        `requires ${ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE};`,
      );
      expect(contraction).toContain("@compute @workgroup_size(128, 1, 1)");
      expect(contraction.match(/dot4I8Packed\(a[0-7], b[0-3]\)/g))
        .toHaveLength(32);
      expect(contraction.match(/var partial[0-7] = vec4<i32>\(0\)/g))
        .toHaveLength(8);
      expect(contraction.match(/acc[0-7] \+= vec4<f32>\(partial[0-7]\)/g))
        .toHaveLength(8);
      expect(contraction).toContain(`const PACKED_K_PER_GROUP = ${groupSize / 4}u;`);
      expect(contraction).toContain("atomicLoad(&status[0]) != 0u");
      expect(contraction).toContain("0x7fc05800u | (atomicLoad(&status[0]) & 0xffu)");
      expect(contraction).toContain("bitcast<vec4<f32>>(vec4<u32>(failure_bits))");
      expect(contraction).not.toContain("var<workgroup>");
      expect(contraction).not.toContain("workgroupBarrier");

      expect(quantizer).toContain("enable subgroups;");
      expect(quantizer).toContain("var<workgroup> subgroup_maximum: array<f32, 4>");
      expect(quantizer).toContain(`@compute @workgroup_size(${groupSize}, 1, 1)`);
      expect(quantizer).toContain("subgroupMax(abs(value))");
      expect(quantizer).toContain("atomicAdd(&status[0], 1u)");
      expect(quantizer).not.toContain("atomicAdd(&status[1]");
      expect(quantizer).toContain("round(value / scale)");
      expect(quantizer).toContain("clamp(i32(round(value / scale)), -127, 127)");
      expect(quantizer).toContain("workgroupBarrier");
    },
  );

  it("requires the actual language feature, compiles two owners, and dies closed", async () => {
    const device = fakeDevice();
    expect(() => AceOpt0058DenseInt8Dp4aKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      packed4x8IntegerDotProduct: false,
    })).toThrow(/packed_4x8_integer_dot_product/);
    const kernel = AceOpt0058DenseInt8Dp4aKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      packed4x8IntegerDotProduct: true,
    });
    const dispatch = await kernel.createDispatch(
      "owner",
      { rows: 1, inner: 2_048, columns: 2_048 },
      64,
      fakeBindings(1, 2_048, 2_048, 64),
    );
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const pass = fakePass();
    dispatch.encodeComplete(pass);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [32, 1, 1],
      [16, 1, 1],
    ]);
    kernel.destroy();
    kernel.destroy();
    expect(() => dispatch.encodePrequantized(pass)).toThrow(/destroyed/);
    await expect(kernel.createDispatch(
      "after",
      { rows: 1, inner: 2_048, columns: 2_048 },
      64,
      fakeBindings(1, 2_048, 2_048, 64),
    )).rejects.toThrow(/destroyed/);
  });

  it("freezes the conservative complete-pipeline browser gate", () => {
    const cases = buildOpt0058Cases();
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
      "zero-group",
      "cancellation",
      "range",
      "saturation",
      "long-k",
    ]);
    for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
      expect(HARNESS_SOURCE).toContain(`"g${groupSize}Prequantized"`);
      expect(HARNESS_SOURCE).toContain(`"g${groupSize}Complete"`);
    }
    for (const fixture of [
      '"signed-zero"',
      '"zero-group"',
      '"cancellation"',
      '"range"',
      '"saturation"',
      '"long-k"',
    ]) expect(HARNESS_SOURCE).toContain(fixture);
    expect(HARNESS_SOURCE).toContain("FULL_OUTPUT_COUNT = 25_344_000");
    expect(HARNESS_SOURCE).toContain("REQUIRED_COMPLETE_SPEEDUP = 1.50");
    expect(HARNESS_SOURCE).toContain("oneDynamicQuantizationPerGemm: true");
    expect(HARNESS_SOURCE).toContain("prequantizedCeilingIsNotPrimary: true");
    expect(HARNESS_SOURCE).toContain("wgslLanguageFeatures");
    expect(HARNESS_SOURCE).toContain("candidateDeterministicRawU32: true");
    expect(HARNESS_SOURCE).toContain("dynamicQuantizerDeterministicRawU32: true");
    expect(HARNESS_SOURCE).toContain("finiteToZeroCollapseCount");
    expect(HARNESS_SOURCE).toContain("activationSaturationCount");
    expect(HARNESS_SOURCE).toContain("weightSaturationCount");
    expect(HARNESS_SOURCE).toContain("unwrittenPackedByteCount");
    expect(HARNESS_SOURCE).toContain("gpuMilliseconds");
    expect(HARNESS_SOURCE).toContain("fencedWallMilliseconds");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0058_RESULT__ = receipt");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(CORE_SOURCE).not.toContain("OPT-0020");
  });

  it("accepts exactly one fresh level-0 observation after 30 seconds", () => {
    const valid = thermal(10_000, 40_050, 1, 0);
    expect(parseOpt0058ThermalGate(valid, 9_000, 40_100)).toMatchObject({
      waitDurationMilliseconds: 30_050,
      checkCount: 1,
      thermalLevel: 0,
      launchDelayMilliseconds: 50,
    });
    for (const invalid of [
      thermal(10_000, 39_999, 1, 0),
      thermal(10_000, 40_050, 2, 0),
      thermal(10_000, 40_050, 1, 1),
      thermal(10_000, 40_050, 1, 0, "wrong-command"),
    ]) {
      expect(() => parseOpt0058ThermalGate(invalid, 9_000, 40_100))
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
    features: new Set(["subgroups"]),
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

function fakeBindings(
  rows: number,
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
) {
  const plan = planAceOpt0058DenseInt8({ rows, inner, columns }, groupSize);
  const binding = (size: number): GPUBufferBinding => Object.freeze({
    buffer: { size } as GPUBuffer,
    offset: 0,
    size,
  });
  return Object.freeze({
    activation: binding(plan.activationElements * 4),
    packedActivation: binding(plan.packedActivationWords * 4),
    activationScale: binding(plan.activationScaleElements * 4),
    weight: binding(plan.packedWeightWords * 4),
    weightScale: binding(plan.weightScaleElements * 4),
    quantizationStatus: binding(4),
    output: binding(plan.outputElements * 4),
  });
}

function fakePass(): GPUComputePassEncoder & {
  dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as ReturnType<typeof fakePass>;
}

function thermal(
  wait: number,
  checked: number,
  count: number,
  level: number,
  command = "notifyutil -g com.apple.system.thermalpressurelevel",
): URLSearchParams {
  return new URLSearchParams({
    thermalCommand: command,
    waitStartedAtEpochMilliseconds: String(wait),
    checkedAtEpochMilliseconds: String(checked),
    checkCount: String(count),
    thermalLevel: String(level),
  });
}
