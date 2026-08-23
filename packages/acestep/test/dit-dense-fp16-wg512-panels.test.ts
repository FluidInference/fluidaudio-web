import { describe, expect, it } from "vitest";

import browserSource from
  "./browser/opt-0031-dit-dense-wg512-panels.ts?raw";
import browserHtml from
  "./browser/opt-0031-dit-dense-wg512-panels.html?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0031_DENSE_ACCUMULATORS_PER_THREAD,
  ACE_OPT_0031_DENSE_BARRIERS_PER_INNER_TILE,
  ACE_OPT_0031_DENSE_COLUMNS_PER_THREAD,
  ACE_OPT_0031_DENSE_INPUT_PANEL_STRIDE,
  ACE_OPT_0031_DENSE_ROWS_PER_SUBGROUP,
  ACE_OPT_0031_DENSE_ROWS_PER_THREAD,
  ACE_OPT_0031_DENSE_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0031_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0031_DENSE_TILE_COLUMNS,
  ACE_OPT_0031_DENSE_TILE_INNER,
  ACE_OPT_0031_DENSE_TILE_ROWS,
  ACE_OPT_0031_DENSE_WEIGHT_PANEL_STRIDE,
  ACE_OPT_0031_DENSE_WG512_KERNEL_SET_ID,
  ACE_OPT_0031_DENSE_WORKGROUP_SIZE,
  ACE_OPT_0031_DENSE_WORKGROUP_STORAGE_BYTES,
  AceOpt0031DenseWg512PanelsKernel,
  aceOpt0031DenseWg512PanelsWgsl,
  planAceOpt0031DenseWg512Panels,
} from "../src/webgpu/kernels/dit-dense-fp16-wg512-panels.js";
import {
  buildOpt0031ShapeSpecs,
  buildOpt0031TimingOrders,
  parseOpt0031ThermalGate,
  summarizeOpt0031Timing,
  type Opt0031TimingInput,
} from "./browser/opt-0031-dit-dense-wg512-panels.js";

const M2250 = 2_250;

describe("OPT-0031 DiT dense WG512 cooperative panels", () => {
  it("pins the exact geometry, storage, and four production plans", () => {
    expect(ACE_OPT_0031_DENSE_WG512_KERNEL_SET_ID).toBe(
      "opt-0031-m128-n128-k32-wg512-panels-fp16-fp32-v1",
    );
    expect({
      rows: ACE_OPT_0031_DENSE_TILE_ROWS,
      columns: ACE_OPT_0031_DENSE_TILE_COLUMNS,
      inner: ACE_OPT_0031_DENSE_TILE_INNER,
      workgroup: ACE_OPT_0031_DENSE_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0031_DENSE_SUBGROUP_SIZE,
      subgroups: ACE_OPT_0031_DENSE_SUBGROUPS_PER_WORKGROUP,
      rowsPerSubgroup: ACE_OPT_0031_DENSE_ROWS_PER_SUBGROUP,
      rowsPerThread: ACE_OPT_0031_DENSE_ROWS_PER_THREAD,
      columnsPerThread: ACE_OPT_0031_DENSE_COLUMNS_PER_THREAD,
      accumulatorsPerThread: ACE_OPT_0031_DENSE_ACCUMULATORS_PER_THREAD,
      inputStride: ACE_OPT_0031_DENSE_INPUT_PANEL_STRIDE,
      weightStride: ACE_OPT_0031_DENSE_WEIGHT_PANEL_STRIDE,
      barriersPerInnerTile: ACE_OPT_0031_DENSE_BARRIERS_PER_INNER_TILE,
      storageBytes: ACE_OPT_0031_DENSE_WORKGROUP_STORAGE_BYTES,
    }).toEqual({
      rows: 128,
      columns: 128,
      inner: 32,
      workgroup: 512,
      subgroup: 32,
      subgroups: 16,
      rowsPerSubgroup: 8,
      rowsPerThread: 8,
      columnsPerThread: 4,
      accumulatorsPerThread: 32,
      inputStride: 33,
      weightStride: 132,
      barriersPerInnerTile: 2,
      storageBytes: 16_896,
    });

    const cases = [
      {
        inner: 2_048,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 64,
        workgroups: 288,
        scheduled: 9_663_676_416,
        valid: 9_437_184_000,
        activationBytes: 294_912_000,
        weightBytes: 150_994_944,
        operandBytes: 445_906_944,
        barrierEvents: 36_864,
        bindingBytes: [18_432_000, 8_388_608, 18_432_000],
      },
      {
        inner: 2_048,
        columns: 1_024,
        columnTiles: 8,
        innerTiles: 64,
        workgroups: 144,
        scheduled: 4_831_838_208,
        valid: 4_718_592_000,
        activationBytes: 147_456_000,
        weightBytes: 75_497_472,
        operandBytes: 222_953_472,
        barrierEvents: 18_432,
        bindingBytes: [18_432_000, 4_194_304, 9_216_000],
      },
      {
        inner: 2_048,
        columns: 6_144,
        columnTiles: 48,
        innerTiles: 64,
        workgroups: 864,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 884_736_000,
        weightBytes: 452_984_832,
        operandBytes: 1_337_720_832,
        barrierEvents: 110_592,
        bindingBytes: [18_432_000, 25_165_824, 55_296_000],
      },
      {
        inner: 6_144,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 192,
        workgroups: 288,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 884_736_000,
        weightBytes: 452_984_832,
        operandBytes: 1_337_720_832,
        barrierEvents: 110_592,
        bindingBytes: [55_296_000, 25_165_824, 18_432_000],
      },
    ] as const;

    for (const expected of cases) {
      const plan = planAceOpt0031DenseWg512Panels({
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
      });
      expect(plan).toMatchObject({
        kernelSetId: ACE_OPT_0031_DENSE_WG512_KERNEL_SET_ID,
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
        workgroupsX: expected.columnTiles,
        workgroupsY: 18,
        rowTiles: 18,
        columnTiles: expected.columnTiles,
        innerTiles: expected.innerTiles,
        workgroupCount: expected.workgroups,
        scheduledRows: 2_304,
        scheduledMultiplyAdds: expected.scheduled,
        validMultiplyAdds: expected.valid,
        barriersPerWorkgroup: expected.innerTiles * 2,
        barrierEvents: expected.barrierEvents,
        estimatedGlobalActivationBytes: expected.activationBytes,
        estimatedGlobalWeightBytes: expected.weightBytes,
        estimatedGlobalOperandBytes: expected.operandBytes,
        outputRangeCount: 1,
      });
      expect(plan.packedWeightStorageShape).toEqual([
        expected.columns / 256,
        expected.inner / 32,
        32,
        256,
      ]);
      expect(plan.outputRanges).toEqual([{
        firstOutput: 0,
        outputCount: M2250 * expected.columns,
        firstWorkgroup: 0,
        workgroupCount: expected.workgroups,
        multiplyAdds: expected.scheduled,
      }]);
      expect([
        plan.activationElements * 4,
        plan.weightElements * 2,
        plan.outputElements * 4,
      ]).toEqual(expected.bindingBytes);
      expect(Object.isFrozen(plan)).toBe(true);
      expect(Object.isFrozen(plan.outputRanges)).toBe(true);
      expect(Object.isFrozen(plan.packedWeightStorageShape)).toBe(true);
    }

    const multiplicities = [4, 2, 2, 1] as const;
    const weighted = cases.map((entry, index) => ({
      plan: planAceOpt0031DenseWg512Panels({
        rows: M2250,
        inner: entry.inner,
        columns: entry.columns,
      }),
      multiplicity: multiplicities[index]!,
    }));
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.workgroupCount * multiplicity, 0)).toBe(3_456);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.scheduledMultiplyAdds * multiplicity, 0)).toBe(
      135_291_469_824,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.validMultiplyAdds * multiplicity, 0)).toBe(132_120_576_000);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalActivationBytes * multiplicity, 0)).toBe(
      4_128_768_000,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalWeightBytes * multiplicity, 0)).toBe(
      2_113_929_216,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalOperandBytes * multiplicity, 0)).toBe(
      6_242_697_216,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.barrierEvents * multiplicity, 0)).toBe(516_096);
  });

  it("assigns every M128/N128 output scalar to exactly one subgroup lane", () => {
    const owners = new Uint8Array(128 * 128);
    for (let subgroup = 0; subgroup < 16; subgroup += 1) {
      for (let lane = 0; lane < 32; lane += 1) {
        for (let ownedRow = 0; ownedRow < 8; ownedRow += 1) {
          for (let component = 0; component < 4; component += 1) {
            const row = subgroup * 8 + ownedRow;
            const column = lane * 4 + component;
            const index = row * 128 + column;
            owners[index] = owners[index]! + 1;
          }
        }
      }
    }
    expect([...owners].every((count) => count === 1)).toBe(true);
  });

  it("loads every padded-panel payload element and N256 half-tile exactly once", () => {
    const inputVectors = new Uint8Array(1_024);
    for (let localIndex = 0; localIndex < 512; localIndex += 1) {
      for (let panelVector = localIndex; panelVector < 1_024;
        panelVector += 512) {
        inputVectors[panelVector] = inputVectors[panelVector]! + 1;
      }
    }
    expect([...inputVectors].every((count) => count === 1)).toBe(true);

    const physicalRecords = new Set<number>();
    const candidateColumnTiles = 2_048 / 128;
    for (let groupX = 0; groupX < candidateColumnTiles; groupX += 1) {
      for (let localIndex = 0; localIndex < 512; localIndex += 1) {
        const k = Math.floor(localIndex / 16);
        const n8 = localIndex % 16;
        const n256 = Math.floor(groupX / 2);
        const half = groupX % 2;
        const record = ((n256 * 32 + k) * 32) + half * 16 + n8;
        expect(physicalRecords.has(record)).toBe(false);
        physicalRecords.add(record);
      }
    }
    expect(physicalRecords.size).toBe(32 * 2_048 / 8);
  });

  it("emits the frozen fixed32 WG512 shader with exact increasing-K arithmetic", () => {
    const source = aceOpt0031DenseWg512PanelsWgsl({
      rows: M2250,
      inner: 2_048,
      columns: 2_048,
    });
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(512, 1, 1)");
    expect(source).toContain("subgroup_size != 32u");
    expect(source).toContain("array<f16, 4224>");
    expect(source.match(/array<f16, 4224>/g)).toHaveLength(2);
    expect(source.match(/var acc\d = vec4<f32>\(0\.0\);/g)).toHaveLength(8);
    expect(source.match(/subgroupBroadcast\(lane_a, \d+u\)/g)).toHaveLength(8);
    expect(source.match(/acc\d = acc\d \+ vec4<f32>\(f32\(a\d\)\) \* b;/g))
      .toHaveLength(8);
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).toContain("inner_in_tile += 1u");
    expect(source).toContain("packed_n128_half * 16u + packed_n8");
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/atomic/i);
  });

  it("rejects capability, shape, and stock-limit drift", () => {
    const capable = fakeDevice();
    const kernel = AceOpt0031DenseWg512PanelsKernel.create(
      capable,
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    );
    kernel.destroy();
    expect(() => AceOpt0031DenseWg512PanelsKernel.create(
      fakeDevice({ features: ["shader-f16"] }),
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    )).toThrow(/subgroups/);
    expect(() => AceOpt0031DenseWg512PanelsKernel.create(
      fakeDevice({ maxComputeInvocationsPerWorkgroup: 256 }),
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    )).toThrow(/512x1/);
    expect(() => AceOpt0031DenseWg512PanelsKernel.create(
      fakeDevice({ maxComputeWorkgroupStorageSize: 16_895 }),
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    )).toThrow(/16896/);
    expect(() => AceOpt0031DenseWg512PanelsKernel.create(
      capable,
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32/);
    for (const shape of [
      { rows: 2_249, inner: 2_048, columns: 2_048 },
      { rows: 2_250, inner: 4_096, columns: 2_048 },
      { rows: 2_250, inner: 2_048, columns: 4_096 },
    ]) {
      expect(() => planAceOpt0031DenseWg512Panels(shape)).toThrow();
    }
  });

  it("pins the browser correctness, thermal, balanced-order, and score gate", () => {
    expect(buildOpt0031ShapeSpecs().map(({ id, productionMultiplicity }) =>
      [id, productionMultiplicity])).toEqual([
      ["h-h", 4],
      ["h-1024", 2],
      ["h-6144", 2],
      ["6144-h", 1],
    ]);
    const orders = buildOpt0031TimingOrders();
    expect(orders).toHaveLength(16);
    for (let round = 0; round < 4; round += 1) {
      const entries = orders.filter((entry) => entry.roundIndex === round);
      expect(entries.map(({ shapeIndex }) => shapeIndex)).toEqual(
        [0, 1, 2, 3].map((index) => (index + round) % 4),
      );
      expect(entries.every(({ order }) =>
        order.join("/") === (round % 2 === 0
          ? "current/candidate"
          : "candidate/current"))).toBe(true);
    }

    const passing = timingInputs(40, 20);
    const pass = summarizeOpt0031Timing(passing);
    expect(pass["everyShapeNoSlower"]).toBe(true);
    expect(pass["passed"]).toBe(true);
    expect((pass["weighted"] as Record<string, unknown>)["speedup"]).toBe(2);
    const failing = timingInputs(40, 31);
    failing[3] = Object.freeze({
      id: "6144-h",
      samples: Object.freeze({
        current: Object.freeze([40, 40, 40, 40]),
        candidate: Object.freeze([41, 41, 41, 41]),
      }),
    });
    expect(summarizeOpt0031Timing(failing)["passed"]).toBe(false);

    const preparedAt = 1_000_000;
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: String(preparedAt),
      thermalCompletedAtEpochMilliseconds: String(preparedAt + 30_000),
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1000",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0031ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toMatchObject({ durationMilliseconds: 30_000, observationCount: 31 });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0031ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toThrow(/thermal gate/);

    expect(browserSource).toContain("__ACE_OPT0031_RESULT__");
    expect(browserSource).toContain("executeCorrectnessDispatch");
    expect(browserSource).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(browserSource).toContain("buildOpt0031TimingOrders()");
    expect(browserSource).toContain("timedGpuWorkBeforeButton: false");
    expect(browserHtml).toContain("id=\"thermal-gate\"");
    expect(browserHtml).toContain("id=\"run\"");
    expect(browserHtml).toContain("raw U32");
    expect(ACE_DIT_DENSE_FP16_TILE_LAYOUT).toBe(
      "dit-gemm-n256-k32-tile-major-v1",
    );
  });
});

function fakeDevice(overrides: Readonly<{
  features?: readonly GPUFeatureName[];
  maxComputeInvocationsPerWorkgroup?: number;
  maxComputeWorkgroupSizeX?: number;
  maxComputeWorkgroupStorageSize?: number;
}> = {}): GPUDevice {
  return {
    features: new Set(overrides.features ?? ["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup:
        overrides.maxComputeInvocationsPerWorkgroup ?? 512,
      maxComputeWorkgroupSizeX: overrides.maxComputeWorkgroupSizeX ?? 512,
      maxComputeWorkgroupStorageSize:
        overrides.maxComputeWorkgroupStorageSize ?? 16_896,
    },
  } as unknown as GPUDevice;
}

function timingInputs(
  current: number,
  candidate: number,
): Opt0031TimingInput[] {
  return buildOpt0031ShapeSpecs().map(({ id }) => Object.freeze({
    id,
    samples: Object.freeze({
      current: Object.freeze([current, current, current, current]),
      candidate: Object.freeze([
        candidate,
        candidate,
        candidate,
        candidate,
      ]),
    }),
  }));
}
