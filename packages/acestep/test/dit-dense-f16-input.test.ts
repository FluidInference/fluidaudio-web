import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0081_DENSE_F16_INPUT_ACCUMULATORS_PER_LANE,
  ACE_OPT_0081_DENSE_F16_INPUT_ACTIVATION_ELEMENT_BYTES,
  ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE,
  ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID,
  ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP,
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE,
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS,
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER,
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS,
  ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE,
  AceOpt0081DenseF16InputKernel,
  aceOpt0081DenseF16InputWgsl,
  planAceOpt0081DenseF16Input,
} from "../src/webgpu/kernels/dit-dense-f16-input.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

const M2250 = 2_250;
const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

describe("OPT-0081 DiT scalar typed-F16 dense input", () => {
  it("pins arm B to OPT-0009 geometry and exact four-shape accounting", () => {
    expect(ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID).toBe(
      "opt-0081-m32-n256-k32-wg128-scalar-f16-input-fp32-output-v1",
    );
    expect({
      rows: ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS,
      columns: ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS,
      inner: ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER,
      workgroup: ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE,
      subgroups: ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP,
      rowsPerSubgroup: ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP,
      columnsPerLane: ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE,
      accumulatorsPerLane: ACE_OPT_0081_DENSE_F16_INPUT_ACCUMULATORS_PER_LANE,
      activationElementBytes:
        ACE_OPT_0081_DENSE_F16_INPUT_ACTIVATION_ELEMENT_BYTES,
    }).toEqual({
      rows: 32,
      columns: 256,
      inner: 32,
      workgroup: 128,
      subgroup: 32,
      subgroups: 4,
      rowsPerSubgroup: 8,
      columnsPerLane: 8,
      accumulatorsPerLane: 64,
      activationElementBytes: 2,
    });
    expect([
      ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS,
      ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS,
      ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER,
      ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE,
      ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE,
    ]).toEqual([
      ACE_OPT_0009_DENSE_TILE_ROWS,
      ACE_OPT_0009_DENSE_TILE_COLUMNS,
      ACE_OPT_0009_DENSE_TILE_INNER,
      ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
      ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
    ]);

    const cases = [
      {
        inner: 2_048,
        columns: 2_048,
        columnTiles: 8,
        innerTiles: 64,
        workgroups: 568,
        scheduled: 9_529_458_688,
        valid: 9_437_184_000,
        activationBytes: 74_448_896,
        weightBytes: 2_382_364_672,
        operandBytes: 2_456_813_568,
        recordsPerWorkgroup: 262_144,
        bindingBytes: [9_216_000, 8_388_608, 18_432_000],
      },
      {
        inner: 2_048,
        columns: 1_024,
        columnTiles: 4,
        innerTiles: 64,
        workgroups: 284,
        scheduled: 4_764_729_344,
        valid: 4_718_592_000,
        activationBytes: 37_224_448,
        weightBytes: 1_191_182_336,
        operandBytes: 1_228_406_784,
        recordsPerWorkgroup: 262_144,
        bindingBytes: [9_216_000, 4_194_304, 9_216_000],
      },
      {
        inner: 2_048,
        columns: 6_144,
        columnTiles: 24,
        innerTiles: 64,
        workgroups: 1_704,
        scheduled: 28_588_376_064,
        valid: 28_311_552_000,
        activationBytes: 223_346_688,
        weightBytes: 7_147_094_016,
        operandBytes: 7_370_440_704,
        recordsPerWorkgroup: 262_144,
        bindingBytes: [9_216_000, 25_165_824, 55_296_000],
      },
      {
        inner: 6_144,
        columns: 2_048,
        columnTiles: 8,
        innerTiles: 192,
        workgroups: 568,
        scheduled: 28_588_376_064,
        valid: 28_311_552_000,
        activationBytes: 223_346_688,
        weightBytes: 7_147_094_016,
        operandBytes: 7_370_440_704,
        recordsPerWorkgroup: 786_432,
        bindingBytes: [27_648_000, 25_165_824, 18_432_000],
      },
    ] as const;

    for (const expected of cases) {
      const candidate = planAceOpt0081DenseF16Input(
        shape(expected.inner, expected.columns),
      );
      const control = planAceOpt0009DenseGemm(
        shape(expected.inner, expected.columns),
      );
      expect(candidate).toMatchObject({
        kernelSetId: ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID,
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
        workgroupsX: expected.columnTiles,
        workgroupsY: 71,
        rowTiles: 71,
        columnTiles: expected.columnTiles,
        innerTiles: expected.innerTiles,
        workgroupCount: expected.workgroups,
        scheduledRows: 2_272,
        scheduledMultiplyAdds: expected.scheduled,
        validMultiplyAdds: expected.valid,
        packedRecordLoadsPerWorkgroup: expected.recordsPerWorkgroup,
        estimatedGlobalActivationBytes: expected.activationBytes,
        estimatedGlobalWeightBytes: expected.weightBytes,
        estimatedGlobalOperandBytes: expected.operandBytes,
        outputRangeCount: 1,
      });
      expect(candidate.packedWeightStorageShape).toEqual(
        control.packedWeightStorageShape,
      );
      expect(candidate.workgroupCount).toBe(control.workgroupCount);
      expect(candidate.outputRanges).toEqual(control.outputRanges);
      expect(candidate.outputRanges).toEqual([{
        firstOutput: 0,
        outputCount: M2250 * expected.columns,
        firstWorkgroup: 0,
        workgroupCount: expected.workgroups,
        multiplyAdds: expected.scheduled,
      }]);
      expect([
        candidate.activationElements * 2,
        candidate.weightElements * 2,
        candidate.outputElements * 4,
      ]).toEqual(expected.bindingBytes);
      expect(Object.isFrozen(candidate)).toBe(true);
      expect(Object.isFrozen(candidate.outputRanges)).toBe(true);
      expect(Object.isFrozen(candidate.outputRanges[0])).toBe(true);
      expect(Object.isFrozen(candidate.packedWeightStorageShape)).toBe(true);
    }

    const multiplicities = [4, 2, 2, 1] as const;
    const weighted = cases.map((entry, index) => ({
      plan: planAceOpt0081DenseF16Input(shape(entry.inner, entry.columns)),
      multiplicity: multiplicities[index]!,
    }));
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.workgroupCount * multiplicity, 0)).toBe(6_816);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.scheduledMultiplyAdds * multiplicity, 0)).toBe(
      133_412_421_632,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.validMultiplyAdds * multiplicity, 0)).toBe(132_120_576_000);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalActivationBytes * multiplicity, 0)).toBe(
      1_042_284_544,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalWeightBytes * multiplicity, 0)).toBe(
      33_353_105_408,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalOperandBytes * multiplicity, 0)).toBe(
      34_395_389_952,
    );
  });

  it("assigns every M32/N256 output exactly once and guards the M2250 tail", () => {
    const owners = new Uint8Array(
      ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS *
        ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS,
    );
    for (let subgroup = 0;
      subgroup < ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP;
      subgroup += 1) {
      for (let lane = 0;
        lane < ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE;
        lane += 1) {
        for (let ownedRow = 0;
          ownedRow < ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP;
          ownedRow += 1) {
          for (let component = 0;
            component < ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE;
            component += 1) {
            const row = subgroup *
              ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP + ownedRow;
            const column = lane *
              ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE + component;
            const index = row * ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS +
              column;
            owners[index] = owners[index]! + 1;
          }
        }
      }
    }
    expect(owners).toHaveLength(8_192);
    expect(owners.every((count) => count === 1)).toBe(true);

    const tailRows: number[] = [];
    for (let subgroup = 0; subgroup < 4; subgroup += 1) {
      for (let ownedRow = 0; ownedRow < 8; ownedRow += 1) {
        const row = 70 * 32 + subgroup * 8 + ownedRow;
        if (row < M2250) tailRows.push(row);
      }
    }
    expect(tailRows).toEqual(
      Array.from({ length: 10 }, (_, index) => 2_240 + index),
    );
  });

  it("retains the native packed-weight index and four subgroup requests", () => {
    for (const { inner, columns } of [
      shape(2_048, 2_048),
      shape(2_048, 1_024),
      shape(2_048, 6_144),
      shape(6_144, 2_048),
    ]) {
      const innerTiles = inner / 32;
      const recordCount = inner * columns / 8;
      const readCounts = new Uint8Array(recordCount);
      let firstMismatch: readonly [number, number] | undefined;
      for (let groupX = 0; groupX < columns / 256; groupX += 1) {
        for (let innerTile = 0; innerTile < innerTiles; innerTile += 1) {
          const tileBase = (groupX * innerTiles + innerTile) * 1_024;
          for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
            for (let lane = 0; lane < 32; lane += 1) {
              const candidate = tileBase + innerInTile * 32 + lane;
              const k = innerTile * 32 + innerInTile;
              const n = groupX * 256 + lane * 8;
              const current = existingPackedRecord(inner, k, n);
              if (candidate !== current && firstMismatch === undefined) {
                firstMismatch = [candidate, current];
              }
              for (let subgroup = 0; subgroup < 4; subgroup += 1) {
                readCounts[candidate] = readCounts[candidate]! + 1;
              }
            }
          }
        }
      }
      expect(firstMismatch).toBeUndefined();
      expect(readCounts.every((count) => count === 4)).toBe(true);
    }
  });

  it("emits scalar native-F16 loads and strict increasing-K FP32 arithmetic", () => {
    const exactShape = shape(2_048, 2_048);
    const historical = aceOpt0009DenseGemmWgsl(exactShape);
    const expected = replaceOnce(
      replaceOnce(
        `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID}\n` +
          "// reference-owner: OPT-0009-exact-body\n" +
          "// activation-storage: scalar-array-f16-producer-boundary\n" +
          "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product\n" +
          historical,
        "@group(0) @binding(0) var<storage, read> activation: array<f32>;",
        "@group(0) @binding(0) var<storage, read> activation: array<f16>;",
      ),
      "lane_a = f16(activation[lane_row * INNER + inner]);",
      "lane_a = activation[lane_row * INNER + inner];",
    );
    const source = aceOpt0081DenseF16InputWgsl(exactShape);
    expect(source).toBe(expected);
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID}`,
    );
    expect(source).toContain(
      "// activation-storage: scalar-array-f16-producer-boundary",
    );
    expect(source).toContain(
      "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product",
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain(
      "var<storage, read> activation: array<f16>;",
    );
    expect(source).not.toMatch(/activation:\s*array<vec/);
    expect(source).toContain("lane_a = activation[lane_row * INNER + inner]");
    expect(source).not.toContain("lane_a = f16(activation[");
    expect(source.match(/var acc[0-7]_[01] = vec4<f32>\(0\.0\);/g))
      .toHaveLength(16);
    expect(source.match(/subgroupBroadcast\(lane_a, [0-7]u\)/g))
      .toHaveLength(8);
    expect(source.match(/acc[0-7]_[01] = acc[0-7]_[01] \+/g))
      .toHaveLength(16);
    expect(source).toContain("let b0 = unpack_f16x4(packed_b.x, packed_b.y)");
    expect(source).toContain("vec4<f32>(f32(a0)) * vec4<f32>(b0)");
    expect(source).toContain(
      "var<storage, read_write> output: array<vec4<f32>>",
    );
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/atomic/i);
  });

  it("caches pipelines and resource bind groups and exposes one range", async () => {
    const device = fakeDevice();
    const owner = AceOpt0081DenseF16InputKernel.create(device, FIXED32);
    const firstShape = shape(2_048, 2_048);
    const resources = bindingsFor(firstShape);
    const first = await owner.createDispatch("first", firstShape, resources);
    const second = await owner.createDispatch("second", firstShape, resources);
    const changed = await owner.createDispatch(
      "changed",
      firstShape,
      bindingsFor(firstShape),
    );
    const otherShape = shape(2_048, 1_024);
    const other = await owner.createDispatch(
      "other",
      otherShape,
      bindingsFor(otherShape),
    );

    expect(first).toMatchObject({
      kernelSetId: ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      rangeCount: 1,
      plan: { workgroupsX: 8, workgroupsY: 71 },
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(second.plan).toEqual(first.plan);
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroupLayout).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(3);
    const firstLayout = device.createBindGroupLayout.mock.calls[0]![0];
    expect(firstLayout.entries.map((entry: GPUBindGroupLayoutEntry) =>
      entry.buffer?.minBindingSize
    )).toEqual([9_216_000, 8_388_608, 18_432_000]);

    const pass = fakePass();
    first.encodeRange(pass, 0);
    second.encode(pass);
    changed.encode(pass);
    other.encode(pass);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [8, 71, 1],
      [8, 71, 1],
      [8, 71, 1],
      [4, 71, 1],
    ]);
    expect(() => first.encodeRange(pass, 1)).toThrow(/range must be zero/);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch("late", firstShape, resources))
      .rejects.toThrow(/was destroyed/);
  });

  it("fails closed on capabilities, shapes, bias, and device limits", async () => {
    expect(() => AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ features: ["subgroups"] }),
      FIXED32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ features: ["shader-f16"] }),
      FIXED32,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0081DenseF16InputKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    expect(() => AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      FIXED32,
    )).toThrow(/128x1/);
    expect(() => AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 127 }),
      FIXED32,
    )).toThrow(/128x1/);

    for (const rejected of [
      { rows: 2_249, inner: 2_048, columns: 2_048 },
      { rows: M2250, inner: 1_024, columns: 2_048 },
      { rows: M2250, inner: 2_048, columns: 4_096 },
      { rows: Number.NaN, inner: 2_048, columns: 2_048 },
    ]) expect(() => planAceOpt0081DenseF16Input(rejected)).toThrow();

    const wideShape = shape(2_048, 6_144);
    const dispatchLimited = AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumDispatch: 23 }),
      FIXED32,
    );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);
    const dispatchYLimited = AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumDispatch: 70 }),
      FIXED32,
    );
    const narrowShape = shape(2_048, 1_024);
    await expect(dispatchYLimited.createDispatch(
      "dispatch-y-limit",
      narrowShape,
      bindingsFor(narrowShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);
    const bufferLimited = AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
      FIXED32,
    );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const allocationLimited = AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ maximumBuffer: 1 }),
      FIXED32,
    );
    await expect(allocationLimited.createDispatch(
      "allocation-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const badAlignment = AceOpt0081DenseF16InputKernel.create(
      fakeDevice({ storageAlignment: 3 }),
      FIXED32,
    );
    await expect(badAlignment.createDispatch(
      "bad-alignment",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/invalid alignment/);

    const biasedDevice = fakeDevice();
    const biased = AceOpt0081DenseF16InputKernel.create(
      biasedDevice,
      FIXED32,
    );
    const exactShape = shape(2_048, 2_048);
    await expect(biased.createDispatch("biased", exactShape, {
      ...bindingsFor(exactShape),
      bias: fakeBinding(8_192),
    })).rejects.toThrow(/rejects bias/);
    expect(biasedDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("requires exact 2-byte activation bindings and rejects overlap", async () => {
    const exactShape = shape(2_048, 2_048);
    const plan = planAceOpt0081DenseF16Input(exactShape);
    const activationBytes = plan.activationElements * 2;

    const shortDevice = fakeDevice();
    const short = AceOpt0081DenseF16InputKernel.create(shortDevice, FIXED32);
    await expect(short.createDispatch("short", exactShape, {
      ...bindingsFor(exactShape),
      activation: fakeBinding(activationBytes - 4),
    })).rejects.toThrow(new RegExp(`aligned ${activationBytes}-byte binding`));
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned = AceOpt0081DenseF16InputKernel.create(
      misalignedDevice,
      FIXED32,
    );
    await expect(misaligned.createDispatch("misaligned", exactShape, {
      ...bindingsFor(exactShape),
      activation: {
        buffer: fakeBuffer(activationBytes + 256),
        offset: 4,
        size: activationBytes,
      },
    })).rejects.toThrow(/does not expose an aligned/);
    expect(misalignedDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const alias = AceOpt0081DenseF16InputKernel.create(aliasDevice, FIXED32);
    const shared = fakeBuffer(Math.max(
      activationBytes,
      plan.outputElements * 4,
    ));
    await expect(alias.createDispatch("alias", exactShape, {
      ...bindingsFor(exactShape),
      activation: { buffer: shared, offset: 0, size: activationBytes },
      output: {
        buffer: shared,
        offset: 0,
        size: plan.outputElements * 4,
      },
    })).rejects.toThrow(/output must not overlap an input binding/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("evicts failed compilation and rejects destruction during compilation", async () => {
    const exactShape = shape(2_048, 1_024);
    const bindings = bindingsFor(exactShape);
    const failedDevice = fakeDevice({ compilationError: "typed input rejected" });
    const failed = AceOpt0081DenseF16InputKernel.create(failedDevice, FIXED32);
    await expect(failed.createDispatch("first-failure", exactShape, bindings))
      .rejects.toThrow(/typed input rejected/);
    await Promise.resolve();
    await expect(failed.createDispatch("retry", exactShape, bindings))
      .rejects.toThrow(/typed input rejected/);
    expect(failedDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(failedDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(failedDevice.createBindGroup).not.toHaveBeenCalled();

    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const pendingDevice = fakeDevice({ pipeline });
    const pendingOwner = AceOpt0081DenseF16InputKernel.create(
      pendingDevice,
      FIXED32,
    );
    const pending = pendingOwner.createDispatch("pending", exactShape, bindings);
    await vi.waitFor(() => {
      expect(pendingDevice.createComputePipelineAsync).toHaveBeenCalledOnce();
    });
    pendingOwner.destroy();
    resolvePipeline({ label: "late" } as GPUComputePipeline);
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(pendingDevice.createBindGroup).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function shape(inner: number, columns: number): AceGemmShape {
  return { rows: M2250, inner, columns };
}

function existingPackedRecord(inner: number, k: number, n: number): number {
  return ((Math.floor(n / 256) * (inner / 32) + Math.floor(k / 32)) * 32 +
    k % 32) * 32 + Math.floor((n % 256) / 8);
}

function replaceOnce(
  source: string,
  expected: string,
  replacement: string,
): string {
  expect(source.split(expected)).toHaveLength(2);
  return source.replace(expected, replacement);
}

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroupLayout: ReturnType<typeof vi.fn>;
  readonly createPipelineLayout: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly features?: readonly string[];
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly storageAlignment?: number;
  readonly compilationError?: string;
  readonly pipeline?: Promise<GPUComputePipeline>;
} = {}): FakeDevice {
  return {
    features: new Set(options.features ?? ["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      minStorageBufferOffsetAlignment: options.storageAlignment ?? 256,
    },
    createShaderModule: vi.fn(() => ({
      label: "module",
      getCompilationInfo: vi.fn(async () => ({
        messages: options.compilationError === undefined
          ? []
          : [{
              type: "error",
              lineNum: 1,
              linePos: 1,
              message: options.compilationError,
            }],
      })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(() =>
      options.pipeline ?? Promise.resolve({ label: "pipeline" })
    ),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
  } as unknown as FakeDevice;
}

function bindingsFor(shape: AceGemmShape): AceGemmBufferBindings {
  const plan = planAceOpt0081DenseF16Input(shape);
  return {
    activation: fakeBinding(plan.activationElements * 2),
    weight: fakeBinding(plan.weightElements * 2),
    output: fakeBinding(plan.outputElements * 4),
  };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakePass(): GPUComputePassEncoder & {
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
  };
}
