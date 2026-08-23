import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0079_DENSE_ACCUMULATORS_PER_LANE,
  ACE_OPT_0079_DENSE_BARRIERS_PER_INNER_TILE,
  ACE_OPT_0079_DENSE_COLUMNS_PER_LANE,
  ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID,
  ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE,
  ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_PACKED_RECORD,
  ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE,
  ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_LANE,
  ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP,
  ACE_OPT_0079_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0079_DENSE_TILE_COLUMNS,
  ACE_OPT_0079_DENSE_TILE_INNER,
  ACE_OPT_0079_DENSE_TILE_ROWS,
  ACE_OPT_0079_DENSE_WORKGROUP_SIZE,
  ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES,
  AceOpt0079DenseDecodedHalfTileKernel,
  aceOpt0079DenseDecodedHalfTileWgsl,
  planAceOpt0079DenseDecodedHalfTile,
} from "../src/webgpu/kernels/dit-dense-fp16-decoded-half-tile.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

const M2250 = 2_250;
const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

describe("OPT-0079 DiT dense decoded half-tile multicast", () => {
  it("pins the exact WG256 owner, 8-KiB panel, and production accounting", () => {
    expect(ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID).toBe(
      "opt-0079-m64-n128-k32-wg256-decoded-half-tile-fp16-fp32-v1",
    );
    expect({
      rows: ACE_OPT_0079_DENSE_TILE_ROWS,
      columns: ACE_OPT_0079_DENSE_TILE_COLUMNS,
      inner: ACE_OPT_0079_DENSE_TILE_INNER,
      workgroup: ACE_OPT_0079_DENSE_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0079_DENSE_SUBGROUP_SIZE,
      subgroups: ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP,
      rowsPerSubgroup: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP,
      columnsPerLane: ACE_OPT_0079_DENSE_COLUMNS_PER_LANE,
      accumulatorsPerLane: ACE_OPT_0079_DENSE_ACCUMULATORS_PER_LANE,
      recordsPerTile: ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE,
      recordsPerLane: ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_LANE,
      vectorsPerRecord: ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_PACKED_RECORD,
      vectorsPerTile: ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE,
      barriersPerTile: ACE_OPT_0079_DENSE_BARRIERS_PER_INNER_TILE,
      storageBytes: ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES,
    }).toEqual({
      rows: 64,
      columns: 128,
      inner: 32,
      workgroup: 256,
      subgroup: 32,
      subgroups: 8,
      rowsPerSubgroup: 8,
      columnsPerLane: 4,
      accumulatorsPerLane: 32,
      recordsPerTile: 512,
      recordsPerLane: 2,
      vectorsPerRecord: 2,
      vectorsPerTile: 1_024,
      barriersPerTile: 2,
      storageBytes: 8_192,
    });

    const cases = [
      {
        inner: 2_048,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 64,
        workgroups: 576,
        scheduled: 9_663_676_416,
        valid: 9_437_184_000,
        activationBytes: 301_989_888,
        weightBytes: 301_989_888,
        operandBytes: 603_979_776,
        barrierEvents: 73_728,
        recordsPerWorkgroup: 32_768,
        vectorsPerWorkgroup: 65_536,
        bindingBytes: [18_432_000, 8_388_608, 18_432_000],
      },
      {
        inner: 2_048,
        columns: 1_024,
        columnTiles: 8,
        innerTiles: 64,
        workgroups: 288,
        scheduled: 4_831_838_208,
        valid: 4_718_592_000,
        activationBytes: 150_994_944,
        weightBytes: 150_994_944,
        operandBytes: 301_989_888,
        barrierEvents: 36_864,
        recordsPerWorkgroup: 32_768,
        vectorsPerWorkgroup: 65_536,
        bindingBytes: [18_432_000, 4_194_304, 9_216_000],
      },
      {
        inner: 2_048,
        columns: 6_144,
        columnTiles: 48,
        innerTiles: 64,
        workgroups: 1_728,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 905_969_664,
        weightBytes: 905_969_664,
        operandBytes: 1_811_939_328,
        barrierEvents: 221_184,
        recordsPerWorkgroup: 32_768,
        vectorsPerWorkgroup: 65_536,
        bindingBytes: [18_432_000, 25_165_824, 55_296_000],
      },
      {
        inner: 6_144,
        columns: 2_048,
        columnTiles: 16,
        innerTiles: 192,
        workgroups: 576,
        scheduled: 28_991_029_248,
        valid: 28_311_552_000,
        activationBytes: 905_969_664,
        weightBytes: 905_969_664,
        operandBytes: 1_811_939_328,
        barrierEvents: 221_184,
        recordsPerWorkgroup: 98_304,
        vectorsPerWorkgroup: 196_608,
        bindingBytes: [55_296_000, 25_165_824, 18_432_000],
      },
    ] as const;

    for (const expected of cases) {
      const plan = planAceOpt0079DenseDecodedHalfTile(
        shape(expected.inner, expected.columns),
      );
      expect(plan).toMatchObject({
        kernelSetId: ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID,
        rows: M2250,
        inner: expected.inner,
        columns: expected.columns,
        workgroupsX: expected.columnTiles,
        workgroupsY: 36,
        rowTiles: 36,
        columnTiles: expected.columnTiles,
        physicalColumnTiles: expected.columns / 256,
        innerTiles: expected.innerTiles,
        workgroupCount: expected.workgroups,
        scheduledRows: 2_304,
        scheduledMultiplyAdds: expected.scheduled,
        validMultiplyAdds: expected.valid,
        barriersPerWorkgroup: expected.innerTiles * 2,
        barrierEvents: expected.barrierEvents,
        packedRecordLoadsPerWorkgroup: expected.recordsPerWorkgroup,
        decodedVectorWritesPerWorkgroup: expected.vectorsPerWorkgroup,
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
      expect(Object.isFrozen(plan.outputRanges[0])).toBe(true);
      expect(Object.isFrozen(plan.packedWeightStorageShape)).toBe(true);
    }

    const multiplicities = [4, 2, 2, 1] as const;
    const weighted = cases.map((entry, index) => ({
      plan: planAceOpt0079DenseDecodedHalfTile(
        shape(entry.inner, entry.columns),
      ),
      multiplicity: multiplicities[index]!,
    }));
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.workgroupCount * multiplicity, 0)).toBe(6_912);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.scheduledMultiplyAdds * multiplicity, 0)).toBe(
      135_291_469_824,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.validMultiplyAdds * multiplicity, 0)).toBe(132_120_576_000);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalActivationBytes * multiplicity, 0)).toBe(
      4_227_858_432,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalWeightBytes * multiplicity, 0)).toBe(
      4_227_858_432,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalOperandBytes * multiplicity, 0)).toBe(
      8_455_716_864,
    );
  });

  it("assigns every M64/N128 output exactly once and guards the M2250 tail", () => {
    const owners = new Uint8Array(
      ACE_OPT_0079_DENSE_TILE_ROWS * ACE_OPT_0079_DENSE_TILE_COLUMNS,
    );
    for (let subgroup = 0;
      subgroup < ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP;
      subgroup += 1) {
      for (let lane = 0; lane < ACE_OPT_0079_DENSE_SUBGROUP_SIZE; lane += 1) {
        for (let ownedRow = 0;
          ownedRow < ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP;
          ownedRow += 1) {
          for (let component = 0;
            component < ACE_OPT_0079_DENSE_COLUMNS_PER_LANE;
            component += 1) {
            const row = subgroup * ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP +
              ownedRow;
            const column = lane * ACE_OPT_0079_DENSE_COLUMNS_PER_LANE +
              component;
            const index = row * ACE_OPT_0079_DENSE_TILE_COLUMNS + column;
            owners[index] = owners[index]! + 1;
          }
        }
      }
    }
    expect(owners).toHaveLength(8_192);
    expect([...owners].every((count) => count === 1)).toBe(true);

    const tailRows: number[] = [];
    for (let subgroup = 0; subgroup < 8; subgroup += 1) {
      for (let ownedRow = 0; ownedRow < 8; ownedRow += 1) {
        const row = 35 * 64 + subgroup * 8 + ownedRow;
        if (row < M2250) tailRows.push(row);
      }
    }
    expect(tailRows).toEqual(
      Array.from({ length: 10 }, (_, index) => 2_240 + index),
    );
  });

  it("owns all 512 records and all 1,024 decoded destinations exactly once", () => {
    const recordOwners = new Uint8Array(512);
    const decodedOwners = new Uint8Array(1_024);
    for (let localIndex = 0; localIndex < 256; localIndex += 1) {
      for (const logicalRecord of [localIndex, localIndex + 256]) {
        recordOwners[logicalRecord] = recordOwners[logicalRecord]! + 1;
        decodedOwners[logicalRecord * 2] =
          decodedOwners[logicalRecord * 2]! + 1;
        decodedOwners[logicalRecord * 2 + 1] =
          decodedOwners[logicalRecord * 2 + 1]! + 1;
      }
    }
    expect([...recordOwners].every((count) => count === 1)).toBe(true);
    expect([...decodedOwners].every((count) => count === 1)).toBe(true);

    const readCounts = new Uint8Array(1_024);
    for (let subgroup = 0; subgroup < 8; subgroup += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          const index = innerInTile * 32 + lane;
          readCounts[index] = readCounts[index]! + 1;
        }
      }
    }
    expect([...readCounts].every((count) => count === 8)).toBe(true);
  });

  it("maps both N128 halves to every converter-native N256/K32 record", () => {
    for (const { inner, columns } of [
      shape(2_048, 2_048),
      shape(2_048, 1_024),
      shape(2_048, 6_144),
      shape(6_144, 2_048),
    ]) {
      const innerTiles = inner / 32;
      const physicalOwners = new Uint8Array(inner * columns / 8);
      let firstMismatch: readonly [number, number] | undefined;
      for (let groupX = 0; groupX < columns / 128; groupX += 1) {
        const physicalColumnTile = Math.floor(groupX / 2);
        const halfRecordOffset = (groupX % 2) * 16;
        for (let innerTile = 0; innerTile < innerTiles; innerTile += 1) {
          const physicalTileBase =
            (physicalColumnTile * innerTiles + innerTile) * 1_024;
          for (let logicalRecord = 0; logicalRecord < 512; logicalRecord += 1) {
            const physical = physicalTileBase +
              Math.floor(logicalRecord / 16) * 32 + halfRecordOffset +
              logicalRecord % 16;
            const k = innerTile * 32 + Math.floor(logicalRecord / 16);
            const n = groupX * 128 + (logicalRecord % 16) * 8;
            const current = existingPackedRecord(inner, k, n);
            if (physical !== current && firstMismatch === undefined) {
              firstMismatch = [physical, current];
            }
            physicalOwners[physical] = physicalOwners[physical]! + 1;
          }
        }
      }
      expect(firstMismatch).toBeUndefined();
      expect(physicalOwners.every((count) => count === 1)).toBe(true);
    }
  });

  it("emits typed FP16 staging and strict increasing-K FP32 arithmetic", () => {
    const source = aceOpt0079DenseDecodedHalfTileWgsl(shape(2_048, 2_048));
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID}`,
    );
    expect(source).toContain(
      "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product",
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(256, 1, 1)");
    expect(source.match(/var<workgroup> weight_panel:/g)).toHaveLength(1);
    expect(source).toContain("array<vec4<f16>, 1024>");
    expect(source).toContain("let physical_column_tile = group.x / 2u;");
    expect(source).toContain("let half_record_offset = (group.x % 2u)");
    expect(source).toContain(
      "(physical_column_tile * INNER_TILES + inner_tile) * PHYSICAL_RECORDS_PER_TILE;",
    );
    for (const record of [0, 1]) {
      expect(source).toContain(
        `(logical_record${record} / HALF_RECORDS_PER_K_ROW) * PHYSICAL_RECORDS_PER_K_ROW +`,
      );
      expect(source).toContain(
        `half_record_offset + logical_record${record} % HALF_RECORDS_PER_K_ROW;`,
      );
    }
    expect(source.match(/let packed[01] = weight\[physical_record[01]\]/g))
      .toHaveLength(2);
    expect(source.match(/weight_panel\[logical_record[01] \* 2u/g))
      .toHaveLength(4);
    expect(source.match(/decode_f16x4\(packed[01]\./g)).toHaveLength(4);
    expect(source.match(/var acc[0-7] = vec4<f32>\(0\.0\);/g))
      .toHaveLength(8);
    expect(source.match(/subgroupBroadcast\(lane_a, [0-7]u\)/g))
      .toHaveLength(8);
    expect(source.match(/acc[0-7] = acc[0-7] \+/g)).toHaveLength(8);
    expect(source).toContain("lane_a = f16(activation[");
    expect(source).toContain(
      "acc0 = acc0 + vec4<f32>(f32(a0)) * vec4<f32>(b);",
    );
    expect(source).toContain(
      "for (var inner_in_tile = 0u; inner_in_tile < TILE_INNER; inner_in_tile += 1u)",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    const barriers = indicesOf(source, "workgroupBarrier();");
    expect(source.indexOf("weight_panel[logical_record1 * 2u + 1u]"))
      .toBeLessThan(barriers[0]!);
    expect(barriers[0]!).toBeLessThan(source.indexOf("for (var inner_in_tile"));
    expect(source.indexOf("for (var inner_in_tile")).toBeLessThan(barriers[1]!);
    for (const barrier of barriers) expect(braceDepthAt(source, barrier)).toBe(2);
    expect(source).not.toContain("return;");
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/atomic/i);
  });

  it("caches pipelines and bindings, dispatches one range, and destroys", async () => {
    const device = fakeDevice();
    const owner = AceOpt0079DenseDecodedHalfTileKernel.create(device, FIXED32);
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
      kernelSetId: ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID,
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
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ features: ["subgroups"] }),
      FIXED32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ features: ["shader-f16"] }),
      FIXED32,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumInvocations: 255 }),
      FIXED32,
    )).toThrow(/256x1/);
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 255 }),
      FIXED32,
    )).toThrow(/256x1/);
    expect(() => AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumWorkgroupStorage: 8_191 }),
      FIXED32,
    )).toThrow(/8192 workgroup-storage bytes/);

    for (const rejected of [
      { rows: 2_249, inner: 2_048, columns: 2_048 },
      { rows: M2250, inner: 1_024, columns: 2_048 },
      { rows: M2250, inner: 2_048, columns: 4_096 },
      { rows: Number.NaN, inner: 2_048, columns: 2_048 },
    ]) {
      expect(() => planAceOpt0079DenseDecodedHalfTile(rejected)).toThrow();
    }

    const wideShape = shape(2_048, 6_144);
    const dispatchLimited = AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumDispatch: 47 }),
      FIXED32,
    );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);
    const dispatchYLimited = AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumDispatch: 35 }),
      FIXED32,
    );
    const narrowShape = shape(2_048, 1_024);
    await expect(dispatchYLimited.createDispatch(
      "dispatch-y-limit",
      narrowShape,
      bindingsFor(narrowShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);
    const bufferLimited = AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
      FIXED32,
    );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const allocationLimited = AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ maximumBuffer: 1 }),
      FIXED32,
    );
    await expect(allocationLimited.createDispatch(
      "allocation-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const badAlignment = AceOpt0079DenseDecodedHalfTileKernel.create(
      fakeDevice({ storageAlignment: 3 }),
      FIXED32,
    );
    await expect(badAlignment.createDispatch(
      "bad-alignment",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/invalid alignment/);

    const biasedDevice = fakeDevice();
    const biased = AceOpt0079DenseDecodedHalfTileKernel.create(
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

  it("rejects bad bindings, evicts failed compilation, and rejects late publish", async () => {
    const exactShape = shape(2_048, 2_048);
    const plan = planAceOpt0079DenseDecodedHalfTile(exactShape);
    const shortDevice = fakeDevice();
    const short = AceOpt0079DenseDecodedHalfTileKernel.create(
      shortDevice,
      FIXED32,
    );
    await expect(short.createDispatch("short", exactShape, {
      ...bindingsFor(exactShape),
      activation: fakeBinding(plan.activationElements * 4 - 4),
    })).rejects.toThrow(/does not expose an aligned/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned = AceOpt0079DenseDecodedHalfTileKernel.create(
      misalignedDevice,
      FIXED32,
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
    const alias = AceOpt0079DenseDecodedHalfTileKernel.create(
      aliasDevice,
      FIXED32,
    );
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

    const failedDevice = fakeDevice({ compilationError: "decoded rejected" });
    const failed = AceOpt0079DenseDecodedHalfTileKernel.create(
      failedDevice,
      FIXED32,
    );
    const bindings = bindingsFor(exactShape);
    await expect(failed.createDispatch("first-failure", exactShape, bindings))
      .rejects.toThrow(/decoded rejected/);
    await Promise.resolve();
    await expect(failed.createDispatch("retry", exactShape, bindings))
      .rejects.toThrow(/decoded rejected/);
    expect(failedDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(failedDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(failedDevice.createBindGroup).not.toHaveBeenCalled();

    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const pendingDevice = fakeDevice({ pipeline });
    const pendingOwner = AceOpt0079DenseDecodedHalfTileKernel.create(
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
  readonly features?: readonly string[];
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupStorage?: number;
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
  const plan = planAceOpt0079DenseDecodedHalfTile(shape);
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
