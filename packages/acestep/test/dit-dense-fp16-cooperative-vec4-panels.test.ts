import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0021_DENSE_ACCUMULATORS_PER_THREAD,
  ACE_OPT_0021_DENSE_BARRIERS_PER_INNER_TILE,
  ACE_OPT_0021_DENSE_COLUMNS_PER_THREAD,
  ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
  ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE,
  ACE_OPT_0021_DENSE_ROWS_PER_THREAD,
  ACE_OPT_0021_DENSE_TILE_COLUMNS,
  ACE_OPT_0021_DENSE_TILE_INNER,
  ACE_OPT_0021_DENSE_TILE_ROWS,
  ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE,
  ACE_OPT_0021_DENSE_WORKGROUP_SIZE,
  ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X,
  ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y,
  ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
  AceOpt0021DenseCooperativeVec4PanelsKernel,
  aceOpt0021DenseCooperativeVec4PanelsWgsl,
  planAceOpt0021DenseCooperativeVec4Panels,
} from "../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

const M2250 = 2_250;

describe("OPT-0021 DiT dense cooperative vec4 panels", () => {
  it("pins the exact owner, resources, and four production plans", () => {
    expect(ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID).toBe(
      "opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1",
    );
    expect({
      rows: ACE_OPT_0021_DENSE_TILE_ROWS,
      columns: ACE_OPT_0021_DENSE_TILE_COLUMNS,
      inner: ACE_OPT_0021_DENSE_TILE_INNER,
      workgroupX: ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X,
      workgroupY: ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y,
      workgroup: ACE_OPT_0021_DENSE_WORKGROUP_SIZE,
      rowsPerThread: ACE_OPT_0021_DENSE_ROWS_PER_THREAD,
      columnsPerThread: ACE_OPT_0021_DENSE_COLUMNS_PER_THREAD,
      accumulatorsPerThread: ACE_OPT_0021_DENSE_ACCUMULATORS_PER_THREAD,
      inputStride: ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE,
      weightStride: ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE,
      barriersPerInnerTile: ACE_OPT_0021_DENSE_BARRIERS_PER_INNER_TILE,
      storageBytes: ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
    }).toEqual({
      rows: 64,
      columns: 128,
      inner: 16,
      workgroupX: 16,
      workgroupY: 16,
      workgroup: 256,
      rowsPerThread: 4,
      columnsPerThread: 8,
      accumulatorsPerThread: 32,
      inputStride: 17,
      weightStride: 33,
      barriersPerInnerTile: 2,
      storageBytes: 6_400,
    });

    const cases = [
      {
        inner: 2_048,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 128,
        workgroups: 576,
        scheduled: 9_663_676_416,
        valid: 9_437_184_000,
        activationBytes: 294_912_000,
        weightBytes: 301_989_888,
        operandBytes: 596_901_888,
        barrierEvents: 147_456,
        bindingBytes: [18_432_000, 8_388_608, 18_432_000],
      },
      {
        inner: 2_048,
        columns: 1_024,
        columnTiles: 8,
        innerTiles: 128,
        workgroups: 288,
        scheduled: 4_831_838_208,
        valid: 4_718_592_000,
        activationBytes: 147_456_000,
        weightBytes: 150_994_944,
        operandBytes: 298_450_944,
        barrierEvents: 73_728,
        bindingBytes: [18_432_000, 4_194_304, 9_216_000],
      },
      {
        inner: 2_048,
        columns: 6_144,
        columnTiles: 48,
        innerTiles: 128,
        workgroups: 1_728,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 884_736_000,
        weightBytes: 905_969_664,
        operandBytes: 1_790_705_664,
        barrierEvents: 442_368,
        bindingBytes: [18_432_000, 25_165_824, 55_296_000],
      },
      {
        inner: 6_144,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 384,
        workgroups: 576,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 884_736_000,
        weightBytes: 905_969_664,
        operandBytes: 1_790_705_664,
        barrierEvents: 442_368,
        bindingBytes: [55_296_000, 25_165_824, 18_432_000],
      },
    ] as const;

    for (const expected of cases) {
      const plan = planAceOpt0021DenseCooperativeVec4Panels({
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
      });
      expect(plan).toMatchObject({
        kernelSetId: ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
        workgroupsX: expected.columnTiles,
        workgroupsY: 36,
        rowTiles: 36,
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
        inputPanelElements: 272,
        weightPanelElements: 528,
        workgroupStorageBytes: 6_400,
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
      expect(Object.isFrozen(plan.outputRanges[0])).toBe(true);
      expect(Object.isFrozen(plan.packedWeightStorageShape)).toBe(true);
    }

    const multiplicities = [4, 2, 2, 1] as const;
    const weighted = cases.map((entry, index) => ({
      plan: planAceOpt0021DenseCooperativeVec4Panels({
        rows: M2250,
        inner: entry.inner,
        columns: entry.columns,
      }),
      multiplicity: multiplicities[index]!,
    }));
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.workgroupCount * multiplicity,
      0,
    )).toBe(6_912);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.scheduledMultiplyAdds * multiplicity,
      0,
    )).toBe(135_291_469_824);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.validMultiplyAdds * multiplicity,
      0,
    )).toBe(132_120_576_000);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.estimatedGlobalOperandBytes * multiplicity,
      0,
    )).toBe(8_356_626_432);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.estimatedGlobalActivationBytes * multiplicity,
      0,
    )).toBe(4_128_768_000);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.estimatedGlobalWeightBytes * multiplicity,
      0,
    )).toBe(4_227_858_432);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.barrierEvents * multiplicity,
      0,
    )).toBe(2_064_384);
    expect(weighted.reduce(
      (total, { plan, multiplicity }) =>
        total + plan.workgroupCount * plan.innerTiles * multiplicity,
      0,
    )).toBe(1_032_192);
  });

  it("assigns each M64/N128 output to one 4x8 thread owner", () => {
    const owners = new Uint8Array(
      ACE_OPT_0021_DENSE_TILE_ROWS * ACE_OPT_0021_DENSE_TILE_COLUMNS,
    );
    for (let localY = 0; localY < 16; localY += 1) {
      for (let localX = 0; localX < 16; localX += 1) {
        for (let row = 0; row < 4; row += 1) {
          for (let column = 0; column < 8; column += 1) {
            const tileRow = localY * 4 + row;
            const tileColumn = localX * 8 + column;
            const index = tileRow * ACE_OPT_0021_DENSE_TILE_COLUMNS +
              tileColumn;
            owners[index] = owners[index]! + 1;
          }
        }
      }
    }
    expect(owners).toHaveLength(8_192);
    expect([...owners].every((count) => count === 1)).toBe(true);
  });

  it("maps every N128/K16 quadrant to the existing N256/K32 packed bytes", () => {
    const inner = 64;
    const columns = 512;
    const records = inner * columns / 8;
    const owners = new Uint8Array(records);
    const pinned = new Map([
      ["0,0,0,0", 0],
      ["0,0,15,15", 495],
      ["1,0,0,0", 16],
      ["0,1,0,0", 512],
      ["1,1,15,15", 1_023],
      ["0,2,0,0", 1_024],
      ["2,0,0,0", 2_048],
      ["3,3,15,15", 4_095],
    ]);
    for (let columnTile = 0; columnTile < columns / 128; columnTile += 1) {
      for (let innerTile = 0; innerTile < inner / 16; innerTile += 1) {
        for (let localY = 0; localY < 16; localY += 1) {
          for (let localX = 0; localX < 16; localX += 1) {
            const record = candidatePackedRecord(
              inner,
              columnTile,
              innerTile,
              localY,
              localX,
            );
            expect(record).toBeLessThan(records);
            owners[record] = owners[record]! + 1;
            const expected = pinned.get(
              `${columnTile},${innerTile},${localY},${localX}`,
            );
            if (expected !== undefined) expect(record).toBe(expected);
            for (let component = 0; component < 8; component += 1) {
              const k = innerTile * 16 + localY;
              const n = columnTile * 128 + localX * 8 + component;
              const scalar = existingPackedScalar(inner, k, n);
              expect(record * 16 + component * 2).toBe(scalar * 2);
            }
          }
        }
      }
    }
    expect([...owners].every((count) => count === 1)).toBe(true);
  });

  it("pins padded K-major vec4 ownership and the retained B byte stride", () => {
    const inputOwners = new Uint8Array(16 * 17);
    const inputScalarOwners = new Uint8Array(16 * 64);
    const weightOwners = new Uint8Array(16 * 33);
    for (let localY = 0; localY < 16; localY += 1) {
      for (let localX = 0; localX < 16; localX += 1) {
        const inputVector = localX * 17 + localY;
        inputOwners[inputVector] = inputOwners[inputVector]! + 1;
        for (let component = 0; component < 4; component += 1) {
          const scalar = localX * 64 + localY * 4 + component;
          inputScalarOwners[scalar] = inputScalarOwners[scalar]! + 1;
        }

        const lowWeightVector = localY * 33 + localX * 2;
        const highWeightVector = lowWeightVector + 1;
        weightOwners[lowWeightVector] = weightOwners[lowWeightVector]! + 1;
        weightOwners[highWeightVector] = weightOwners[highWeightVector]! + 1;
        expect(lowWeightVector * 8).toBe(
          (localY * 132 + localX * 8) * 2,
        );
        expect(highWeightVector * 8).toBe(
          (localY * 132 + localX * 8 + 4) * 2,
        );
      }
    }
    expect([...inputScalarOwners].every((count) => count === 1)).toBe(true);
    for (let inner = 0; inner < 16; inner += 1) {
      expect([...inputOwners.slice(inner * 17, inner * 17 + 16)])
        .toEqual(Array(16).fill(1));
      expect(inputOwners[inner * 17 + 16]).toBe(0);
      expect([...weightOwners.slice(inner * 33, inner * 33 + 32)])
        .toEqual(Array(32).fill(1));
      expect(weightOwners[inner * 33 + 32]).toBe(0);
    }
  });

  it("generates vector panels and sixteen ordered scalar-K FP32 updates", () => {
    const source = aceOpt0021DenseCooperativeVec4PanelsWgsl(
      shape(2_048, 2_048),
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID}`,
    );
    expect(source).toContain(
      "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product",
    );
    expect(source).toContain("enable f16;");
    expect(source).not.toContain("enable subgroups;");
    expect(source).not.toMatch(/\bsubgroup/);
    expect(source).toMatch(/@workgroup_size\(\s*16,\s*16,\s*1\s*\)/);
    expect(source.match(/var<workgroup> input_panel:/g)).toHaveLength(1);
    expect(source.match(/var<workgroup> weight_panel:/g)).toHaveLength(1);
    expect(source).toContain("array<vec4<f16>, 272>");
    expect(source).toContain("array<vec4<f16>, 528>");
    expect(source).not.toMatch(/var<workgroup> \w+: array<f16,/);
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).toContain(
      "for (var inner_tile = 0u; inner_tile < 128u; inner_tile += 1u)",
    );
    expect(source).not.toContain("inner_in_tile");
    expect(source).not.toContain("local_invocation_index");
    expect(source.match(/\/\/ lexical-k4-base: (?:0|4|8|12)/g))
      .toEqual([
        "// lexical-k4-base: 0",
        "// lexical-k4-base: 4",
        "// lexical-k4-base: 8",
        "// lexical-k4-base: 12",
      ]);
    expect(source.match(/var acc[0-3]_[01] = vec4<f32>\(0\.0\);/g))
      .toHaveLength(8);
    expect(source.match(/acc[0-3]_[01] = acc[0-3]_[01] \+/g))
      .toHaveLength(128);
    expect(source.match(/let a(?:[0-9]|1[0-5]) = vec4<f32>\(/g))
      .toHaveLength(16);
    expect(source.match(/let b(?:[0-9]|1[0-5])_[01] = vec4<f32>\(/g))
      .toHaveLength(32);
    for (let row = 0; row < 4; row += 1) {
      const component = ["x", "y", "z", "w"][row]!;
      for (let half = 0; half < 2; half += 1) {
        const updates = [...source.matchAll(new RegExp(
          `acc${row}_${half} = acc${row}_${half} \\+\\s*` +
            `vec4<f32>\\(a(\\d+)\\.${component}\\) \\* b\\1_${half};`,
          "g",
        ))].map((match) => Number(match[1]));
        expect(updates).toEqual(Array.from({ length: 16 }, (_, k) => k));
      }
    }
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/\batomic/);
    expect(source).not.toContain("return;");
    expect(source).toContain("packed_n256_tile = group.x / 2u");
    expect(source).toContain("packed_n128_half = group.x % 2u");
    expect(source).toContain("packed_k32_tile = packed_inner / 32u");
    expect(source).toContain("packed_k_in_tile = packed_inner % 32u");
    expect(source).toContain("array<vec4<u32>>");
    expect(source).toContain("var a_value = vec4<f16>(0.0h)");
    expect(source).toContain(
      "local_id.x * INPUT_PANEL_STRIDE + local_id.y",
    );
    expect(source).toContain(
      "local_id.y * WEIGHT_PANEL_STRIDE + local_id.x * 2u",
    );
    expect(source.match(/if \(global_a_row(?: \+ [123]u)? < ROWS\)/g))
      .toHaveLength(4);
    expect(source).toContain("weight_panel[weight_panel_base] = packed_low;");
    expect(source).toContain(
      "weight_panel[weight_panel_base + 1u] = packed_high;",
    );
    expect(source).toContain("if (row < ROWS && column_base + 7u < COLUMNS)");
    for (const barrier of indicesOf(source, "workgroupBarrier();")) {
      expect(braceDepthAt(source, barrier)).toBe(2);
    }
  });

  it("caches shape pipelines and resource bind groups and encodes one range", async () => {
    const device = fakeDevice();
    const owner = AceOpt0021DenseCooperativeVec4PanelsKernel.create(device);
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
      kernelSetId: ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      rangeCount: 1,
      plan: { workgroupsX: 16, workgroupsY: 36 },
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
    )).toEqual([18_432_000, 8_388_608, 18_432_000]);

    const pass = fakePass();
    first.encodeRange(pass, 0);
    second.encode(pass);
    changed.encode(pass);
    other.encode(pass);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [16, 36, 1],
      [16, 36, 1],
      [16, 36, 1],
      [8, 36, 1],
    ]);
    expect(() => first.encodeRange(pass, 1)).toThrow(/range must be zero/);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch("late", firstShape, resources))
      .rejects.toThrow(/was destroyed/);
  });

  it("fails closed on capabilities, shapes, bias, and device limits", async () => {
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ shaderF16: false }),
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumInvocations: 255 }),
    )).toThrow(/requires WG256/);
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 15 }),
    )).toThrow(/requires WG256/);
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumWorkgroupSizeY: 15 }),
    )).toThrow(/requires WG256/);
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumWorkgroupStorage: 6_399 }),
    )).toThrow(/requires 6400 workgroup-storage bytes/);
    expect(() => AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 16, maximumWorkgroupSizeY: 16 }),
    )).not.toThrow();

    for (const rejected of [
      { rows: 2_249, inner: 2_048, columns: 2_048 },
      { rows: M2250, inner: 1_024, columns: 2_048 },
      { rows: M2250, inner: 2_048, columns: 4_096 },
      { rows: Number.NaN, inner: 2_048, columns: 2_048 },
    ]) {
      expect(() => planAceOpt0021DenseCooperativeVec4Panels(rejected)).toThrow();
    }

    const exactShape = shape(2_048, 6_144);
    const dispatchLimited = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumDispatch: 47 }),
    );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the device dispatch dimension/);
    const dispatchYLimited = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumDispatch: 35 }),
    );
    const narrowShape = shape(2_048, 1_024);
    await expect(dispatchYLimited.createDispatch(
      "dispatch-y-limit",
      narrowShape,
      bindingsFor(narrowShape),
    )).rejects.toThrow(/exceeds the device dispatch dimension/);
    const bufferLimited = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
    );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const allocationLimited = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ maximumBuffer: 1 }),
    );
    await expect(allocationLimited.createDispatch(
      "allocation-limit",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const badAlignment = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      fakeDevice({ storageAlignment: 3 }),
    );
    await expect(badAlignment.createDispatch(
      "bad-alignment",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/invalid storage alignment/);

    const biasedDevice = fakeDevice();
    const biased = AceOpt0021DenseCooperativeVec4PanelsKernel.create(biasedDevice);
    await expect(biased.createDispatch("biased", shape(2_048, 2_048), {
      ...bindingsFor(shape(2_048, 2_048)),
      bias: fakeBinding(8_192),
    })).rejects.toThrow(/reject bias/);
    expect(biasedDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("rejects short, misaligned, and aliased bindings before compilation", async () => {
    const exactShape = shape(2_048, 2_048);
    const plan = planAceOpt0021DenseCooperativeVec4Panels(exactShape);

    const shortDevice = fakeDevice();
    const short = AceOpt0021DenseCooperativeVec4PanelsKernel.create(shortDevice);
    await expect(short.createDispatch("short", exactShape, {
      ...bindingsFor(exactShape),
      activation: fakeBinding(plan.activationElements * 4 - 4),
    })).rejects.toThrow(/does not expose an aligned/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      misalignedDevice,
    );
    await expect(misaligned.createDispatch("misaligned", exactShape, {
      ...bindingsFor(exactShape),
      activation: {
        buffer: fakeBuffer(plan.activationElements * 4 + 256),
        offset: 4,
        size: plan.activationElements * 4,
      },
    })).rejects.toThrow(/does not expose an aligned/);
    expect(misalignedDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const alias = AceOpt0021DenseCooperativeVec4PanelsKernel.create(aliasDevice);
    const shared = fakeBuffer(Math.max(
      plan.activationElements * 4,
      plan.outputElements * 4,
    ));
    await expect(alias.createDispatch("alias", exactShape, {
      ...bindingsFor(exactShape),
      activation: {
        buffer: shared,
        offset: 0,
        size: plan.activationElements * 4,
      },
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
    const failedDevice = fakeDevice({ compilationError: "panels rejected" });
    const failed = AceOpt0021DenseCooperativeVec4PanelsKernel.create(failedDevice);
    await expect(failed.createDispatch("first-failure", exactShape, bindings))
      .rejects.toThrow(/panels rejected/);
    await Promise.resolve();
    await expect(failed.createDispatch("retry", exactShape, bindings))
      .rejects.toThrow(/panels rejected/);
    expect(failedDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(failedDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(failedDevice.createBindGroup).not.toHaveBeenCalled();

    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const pendingDevice = fakeDevice({ pipeline });
    const pendingOwner = AceOpt0021DenseCooperativeVec4PanelsKernel.create(
      pendingDevice,
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

function existingPackedScalar(inner: number, k: number, n: number): number {
  return (((Math.floor(n / 256) * (inner / 32) + Math.floor(k / 32)) * 32 +
    k % 32) * 256 + n % 256);
}

function candidatePackedRecord(
  inner: number,
  columnTile: number,
  innerTile: number,
  localY: number,
  localX: number,
): number {
  const packedInner = innerTile * 16 + localY;
  return ((Math.floor(columnTile / 2) * (inner / 32) +
    Math.floor(packedInner / 32)) * 32 + packedInner % 32) * 32 +
    columnTile % 2 * 16 + localX;
}

function indicesOf(source: string, needle: string): readonly number[] {
  const found: number[] = [];
  for (let index = source.indexOf(needle); index >= 0;) {
    found.push(index);
    index = source.indexOf(needle, index + needle.length);
  }
  return found;
}

function braceDepthAt(source: string, index: number): number {
  let depth = 0;
  for (const character of source.slice(0, index)) {
    if (character === "{") depth += 1;
    if (character === "}") depth -= 1;
  }
  return depth;
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
  readonly shaderF16?: boolean;
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupSizeY?: number;
  readonly maximumWorkgroupStorage?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly storageAlignment?: number;
  readonly compilationError?: string;
  readonly pipeline?: Promise<GPUComputePipeline>;
} = {}): FakeDevice {
  return {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 16,
      maxComputeWorkgroupSizeY: options.maximumWorkgroupSizeY ?? 16,
      maxComputeWorkgroupStorageSize:
        options.maximumWorkgroupStorage ?? 16_384,
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
  const plan = planAceOpt0021DenseCooperativeVec4Panels(shape);
  return {
    activation: fakeBinding(plan.activationElements * 4),
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
