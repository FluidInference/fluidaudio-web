import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_GEMM_WEIGHT_LAYOUT } from
  "../src/webgpu/kernels/gemm.js";
import {
  ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_WORKGROUP_BYTES,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_BYTES,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_STRIDE,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_WORDS,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE,
  AceOpt0083PlannerDirectLowRowBf16GemvKernel,
  AceOpt0083PlannerLowRowBf16GemvKernel,
  aceOpt0083PlannerDirectLowRowBf16GemvWgsl,
  aceOpt0083PlannerLowRowBf16GemvWgsl,
  planAceOpt0083PlannerDirectLowRowBf16Gemv,
  planAceOpt0083PlannerLowRowBf16Gemv,
} from "../src/webgpu/kernels/planner-low-row-bf16-gemv.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

describe("OPT-0083 planner low-row packed-BF16 GEMV", () => {
  it("pins candidate C geometry and all production planner shapes", () => {
    expect(ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID).toBe(
      "opt-0083-planner-m1-m2-n128-k64-wg128-source-row-major-bf16-fp32-v1",
    );
    expect({
      columns: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
      inner: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
      workgroup: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE,
      panelStride:
        ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_STRIDE,
      panelWords: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_WORDS,
      panelBytes: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_BYTES,
      maximumStorage:
        ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_WORKGROUP_BYTES,
    }).toEqual({
      columns: 128,
      inner: 64,
      workgroup: 128,
      panelStride: 129,
      panelWords: 4_128,
      panelBytes: 16_512,
      maximumStorage: 17_024,
    });

    const layerShapes = [
      { inner: 1_024, columns: 2_048 },
      { inner: 1_024, columns: 1_024 },
      { inner: 2_048, columns: 1_024 },
      { inner: 1_024, columns: 3_072 },
      { inner: 3_072, columns: 1_024 },
    ] as const;
    for (const rows of [1, 2] as const) {
      for (const dense of layerShapes) {
        const plan = planAceOpt0083PlannerLowRowBf16Gemv({ rows, ...dense });
        expect(plan).toMatchObject({
          kernelId: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
          rows,
          inner: dense.inner,
          columns: dense.columns,
          workgroupsX: dense.columns / 128,
          workgroupsY: 1,
          columnTiles: dense.columns / 128,
          innerTiles: dense.inner / 64,
          workgroupCount: dense.columns / 128,
          scheduledColumns: dense.columns,
          scheduledInner: dense.inner,
          scheduledMultiplyAdds: rows * dense.inner * dense.columns,
          validMultiplyAdds: rows * dense.inner * dense.columns,
          activationBytes: rows * dense.inner * 4,
          weightBytes: dense.inner * dense.columns * 2,
          outputBytes: rows * dense.columns * 4,
          workgroupStorageBytes: rows === 1 ? 16_768 : 17_024,
          barriersPerWorkgroup: dense.inner / 32,
          outputRangeCount: 1,
        });
        expect(plan.packedWeightStorageShape).toEqual([
          dense.columns,
          dense.inner / 2,
        ]);
        expect(plan.outputRanges).toEqual([{
          firstOutput: 0,
          outputCount: rows * dense.columns,
          firstWorkgroup: 0,
          workgroupCount: dense.columns / 128,
          multiplyAdds: rows * dense.inner * dense.columns,
        }]);
        expect(Object.isFrozen(plan)).toBe(true);
        expect(Object.isFrozen(plan.packedWeightStorageShape)).toBe(true);
        expect(Object.isFrozen(plan.outputRanges)).toBe(true);
        expect(Object.isFrozen(plan.outputRanges[0])).toBe(true);
      }
    }

    const fullHead = planAceOpt0083PlannerLowRowBf16Gemv({
      rows: 2,
      inner: 1_024,
      columns: 217_204,
    });
    expect(fullHead).toMatchObject({
      columnTiles: 1_697,
      scheduledColumns: 217_216,
      scheduledMultiplyAdds: 444_858_368,
      validMultiplyAdds: 444_833_792,
      weightBytes: 444_833_792,
    });
    expect(fullHead.outputRanges[0]).toMatchObject({
      outputCount: 434_408,
      workgroupCount: 1_697,
    });

    const compactTail = planAceOpt0083PlannerLowRowBf16Gemv({
      rows: 2,
      inner: 1_024,
      columns: 44_939,
    });
    expect(compactTail).toMatchObject({
      columnTiles: 352,
      scheduledColumns: 45_056,
      scheduledMultiplyAdds: 92_274_688,
      validMultiplyAdds: 92_035_072,
      weightBytes: 92_035_072,
    });
    const eosTail = planAceOpt0083PlannerLowRowBf16Gemv({
      rows: 1,
      inner: 1_024,
      columns: 1,
    });
    expect(eosTail).toMatchObject({
      columnTiles: 1,
      scheduledColumns: 128,
      scheduledMultiplyAdds: 131_072,
      validMultiplyAdds: 1_024,
      weightBytes: 2_048,
      outputBytes: 4,
    });
    const innerTail = planAceOpt0083PlannerLowRowBf16Gemv({
      rows: 1,
      inner: 1_026,
      columns: 129,
    });
    expect(innerTail).toMatchObject({
      columnTiles: 2,
      innerTiles: 17,
      scheduledColumns: 256,
      scheduledInner: 1_088,
      scheduledMultiplyAdds: 278_528,
      validMultiplyAdds: 132_354,
    });
  });

  it("pins distinct direct Arm B plans without workgroup staging", () => {
    expect(ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID).toBe(
      "opt-0083-planner-direct-m1-m2-n128-wg128-source-row-major-bf16-fp32-v1",
    );
    expect(ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID).not.toBe(
      ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
    );
    for (const rows of [1, 2] as const) {
      const plan = planAceOpt0083PlannerDirectLowRowBf16Gemv({
        rows,
        inner: 1_026,
        columns: 129,
      });
      expect(plan).toMatchObject({
        kernelId:
          ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
        rows,
        inner: 1_026,
        columns: 129,
        workgroupsX: 2,
        workgroupsY: 1,
        tileRows: 2,
        tileColumns: 128,
        tileInner: 2,
        workgroupSize: 128,
        columnTiles: 2,
        innerPairs: 513,
        workgroupCount: 2,
        scheduledColumns: 256,
        scheduledMultiplyAdds: rows * 262_656,
        validMultiplyAdds: rows * 132_354,
        activationBytes: rows * 4_104,
        weightBytes: 264_708,
        outputBytes: rows * 516,
        workgroupStorageBytes: 0,
        barriersPerWorkgroup: 0,
      });
      expect(plan.packedWeightStorageShape).toEqual([129, 513]);
      expect(plan.outputRanges).toEqual([{
        firstOutput: 0,
        outputCount: rows * 129,
        firstWorkgroup: 0,
        workgroupCount: 2,
        multiplyAdds: rows * 262_656,
      }]);
      expect(Object.isFrozen(plan)).toBe(true);
      expect(Object.isFrozen(plan.outputRanges)).toBe(true);
    }

    const fullHead = planAceOpt0083PlannerDirectLowRowBf16Gemv({
      rows: 2,
      inner: 1_024,
      columns: 217_204,
    });
    expect(fullHead).toMatchObject({
      workgroupsX: 1_697,
      scheduledMultiplyAdds: 444_858_368,
      validMultiplyAdds: 444_833_792,
      weightBytes: 444_833_792,
      workgroupStorageBytes: 0,
    });
  });

  it("maps every valid tail record exactly once into the padded transpose", () => {
    const columns = 129;
    const inner = 66;
    const columnTiles = Math.ceil(columns / 128);
    const innerTiles = Math.ceil(inner / 64);
    const sourceRecords = new Uint8Array(columns * inner / 2);

    for (let group = 0; group < columnTiles; group += 1) {
      for (let innerTile = 0; innerTile < innerTiles; innerTile += 1) {
        const sharedOwners = new Uint8Array(
          ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_WORDS,
        );
        for (let lane = 0; lane < 128; lane += 1) {
          for (let item = lane; item < 128 * 32; item += 128) {
            const localColumn = Math.floor(item / 32);
            const localPair = item % 32;
            const shared = localPair * 129 + localColumn;
            sharedOwners[shared] = sharedOwners[shared]! + 1;
            const sourceColumn = group * 128 + localColumn;
            const sourcePair = innerTile * 32 + localPair;
            if (sourceColumn < columns && sourcePair < inner / 2) {
              const source = sourceColumn * (inner / 2) + sourcePair;
              sourceRecords[source] = sourceRecords[source]! + 1;
            }
          }
        }
        expect([...sharedOwners].filter((count) => count === 1)).toHaveLength(
          4_096,
        );
        expect([...sharedOwners].filter((count) => count === 0)).toHaveLength(
          32,
        );
        for (let pair = 0; pair < 32; pair += 1) {
          expect(sharedOwners[pair * 129 + 128]).toBe(0);
        }
      }
    }
    expect([...sourceRecords].every((count) => count === 1)).toBe(true);

    for (const rows of [1, 2]) {
      const owners = new Uint8Array(rows * columns);
      for (let group = 0; group < columnTiles; group += 1) {
        for (let lane = 0; lane < 128; lane += 1) {
          const column = group * 128 + lane;
          if (column >= columns) continue;
          for (let row = 0; row < rows; row += 1) {
            const output = row * columns + column;
            owners[output] = owners[output]! + 1;
          }
        }
      }
      expect([...owners].every((count) => count === 1)).toBe(true);
    }
  });

  it("emits the bank-padded source-row-major panel and strict K ordering", () => {
    const m1 = aceOpt0083PlannerLowRowBf16GemvWgsl({
      rows: 1,
      inner: 1_026,
      columns: 129,
    });
    const m2 = aceOpt0083PlannerLowRowBf16GemvWgsl({
      rows: 2,
      inner: 1_024,
      columns: 44_939,
    });
    for (const source of [m1, m2]) {
      expect(source).toContain(
        `// kernel-id: ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID}`,
      );
      expect(source).toContain(
        "// weight-layout: source-row-major-packed-bf16-[N,K/2]",
      );
      expect(source).toContain(
        "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product",
      );
      expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
      expect(source).toContain(
        "var<workgroup> weight_panel: array<u32, 32 * 129>",
      );
      expect(source).toContain(
        "weight[source_column * INNER_PAIRS + source_pair]",
      );
      expect(source).toContain(
        "weight_panel[local_pair * WEIGHT_PANEL_STRIDE + local_column] = packed",
      );
      expect(source).toContain("pair & 0xffffu");
      expect(source).toContain("pair & 0xffff0000u");
      expect(source).toContain("source_column < COLUMNS");
      expect(source).toContain("source_pair < INNER_PAIRS");
      expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(source).not.toMatch(/\bdot\s*\(/);
      expect(source).not.toMatch(/\bfma\s*\(/);
      expect(source).not.toMatch(/subgroup/i);
      expect(source).not.toMatch(/atomic/i);
      expect(source).not.toContain("enable f16");
      expect(source).not.toContain("return;");
    }
    expect(m1).toContain("activation_panel: array<f32, 64>");
    expect(m1).not.toContain("var sum1");
    expect(m2).toContain("activation_panel: array<f32, 128>");
    expect(m2).toContain("var sum1 = 0.0");

    const low = m2.indexOf(
      "sum0 = sum0 + activation_panel[local_inner] * weight_low",
    );
    const high = m2.indexOf(
      "sum0 = sum0 + activation_panel[local_inner + 1u] * weight_high",
    );
    expect(low).toBeGreaterThan(0);
    expect(high).toBeGreaterThan(low);
  });

  it("emits Arm B direct packed-pair reads with strict low-then-high K order", () => {
    const m1 = aceOpt0083PlannerDirectLowRowBf16GemvWgsl({
      rows: 1,
      inner: 1_026,
      columns: 129,
    });
    const m2 = aceOpt0083PlannerDirectLowRowBf16GemvWgsl({
      rows: 2,
      inner: 1_024,
      columns: 2_048,
    });
    for (const source of [m1, m2]) {
      expect(source).toContain(
        `// kernel-id: ${ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID}`,
      );
      expect(source).toContain(
        "// weight-layout: source-row-major-packed-bf16-[N,K/2]",
      );
      expect(source).toContain(
        "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product",
      );
      expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
      expect(source).toContain("let weight_base = column * INNER_PAIRS");
      expect(source).toContain("weight[weight_base + pair_index]");
      expect(source).toContain("pair_index * 2u");
      expect(source).not.toContain("var<workgroup>");
      expect(source).not.toContain("workgroupBarrier");
      expect(source).not.toMatch(/\bdot\s*\(/);
      expect(source).not.toMatch(/\bfma\s*\(/);
      expect(source).not.toMatch(/subgroup/i);
      expect(source).not.toMatch(/atomic/i);
      expect(source).not.toContain("enable f16");
    }
    expect(m1).not.toContain("var sum1");
    expect(m2).toContain("var sum1 = 0.0");
    const low = m2.indexOf(
      "sum0 = sum0 + activation[inner] * weight_low",
    );
    const high = m2.indexOf(
      "sum0 = sum0 + activation[inner + 1u] * weight_high",
    );
    expect(low).toBeGreaterThan(0);
    expect(high).toBeGreaterThan(low);
  });

  it("runs Arm B as a distinct cached owner with zero shared-memory requirement", async () => {
    const device = fakeDevice({ maximumWorkgroupStorage: 1 });
    const owner = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      device,
      "reference-bf16",
    );
    const firstShape = shape(2, 1_024, 2_048);
    const first = await owner.createDispatch(
      "direct-first",
      firstShape,
      directBindingsFor(firstShape),
    );
    const second = await owner.createDispatch(
      "direct-second",
      firstShape,
      directBindingsFor(firstShape),
    );
    const tailShape = shape(1, 1_026, 129);
    const tail = await owner.createDispatch(
      "direct-tail",
      tailShape,
      directBindingsFor(tailShape),
    );
    expect(first).toMatchObject({
      kernelId: ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
      weightLayout: "source-row-major",
      rangeCount: 1,
      plan: {
        workgroupsX: 16,
        workgroupsY: 1,
        workgroupStorageBytes: 0,
      },
    });
    expect(second.plan).toEqual(first.plan);
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(3);
    expect(device.createBindGroupLayout.mock.calls[0]![0].entries.map(
      (entry: GPUBindGroupLayoutEntry) => entry.buffer?.minBindingSize,
    )).toEqual([8_192, 4_194_304, 16_384]);

    const pass = fakePass();
    first.encodeRange(pass, 0);
    second.encode(pass);
    tail.encode(pass);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [16, 1, 1],
      [16, 1, 1],
      [2, 1, 1],
    ]);
    expect(() => first.encodeRange(pass, 1)).toThrow(/range must be zero/);
    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch(
      "direct-late",
      firstShape,
      directBindingsFor(firstShape),
    )).rejects.toThrow(/was destroyed/);
  });

  it("fails Arm B closed on profile, layout, shape, bias, limits, and bindings", async () => {
    expect(() => AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      fakeDevice(),
      "raw-fp16",
    )).toThrow(/requires reference-bf16/);
    expect(() => AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      fakeDevice(),
      "reference-bf16",
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    )).toThrow(/requires source-row-major/);
    expect(() => AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      "reference-bf16",
    )).toThrow(/128x1/);
    expect(() => planAceOpt0083PlannerDirectLowRowBf16Gemv({
      rows: 3,
      inner: 1_024,
      columns: 1_024,
    })).toThrow(/M1 or M2/);
    expect(() => planAceOpt0083PlannerDirectLowRowBf16Gemv({
      rows: 1,
      inner: 1_023,
      columns: 1_024,
    })).toThrow(/even K/);

    const exactShape = shape(2, 1_024, 2_048);
    const biasedDevice = fakeDevice();
    const biased = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      biasedDevice,
      "reference-bf16",
    );
    await expect(biased.createDispatch("direct-biased", exactShape, {
      ...directBindingsFor(exactShape),
      bias: fakeBinding(4_096),
    })).rejects.toThrow(/rejects bias/);
    expect(biasedDevice.createShaderModule).not.toHaveBeenCalled();

    const limited = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      fakeDevice({ maximumDispatch: 15 }),
      "reference-bf16",
    );
    await expect(limited.createDispatch(
      "direct-dispatch-limit",
      exactShape,
      directBindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);

    const plan = planAceOpt0083PlannerDirectLowRowBf16Gemv(exactShape);
    const shortDevice = fakeDevice();
    const short = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      shortDevice,
      "reference-bf16",
    );
    await expect(short.createDispatch("direct-short", exactShape, {
      ...directBindingsFor(exactShape),
      weight: fakeBinding(plan.weightBytes - 4),
    })).rejects.toThrow(/does not expose an aligned/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const alias = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      aliasDevice,
      "reference-bf16",
    );
    const shared = fakeBuffer(Math.max(plan.activationBytes, plan.outputBytes));
    await expect(alias.createDispatch("direct-alias", exactShape, {
      ...directBindingsFor(exactShape),
      activation: { buffer: shared, offset: 0, size: plan.activationBytes },
      output: { buffer: shared, offset: 0, size: plan.outputBytes },
    })).rejects.toThrow(/output must not overlap an input binding/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("caches shape pipelines, exposes one range, and rejects late use", async () => {
    const device = fakeDevice();
    const owner = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      device,
      "reference-bf16",
    );
    const firstShape = shape(2, 1_024, 2_048);
    const first = await owner.createDispatch(
      "first",
      firstShape,
      bindingsFor(firstShape),
    );
    const second = await owner.createDispatch(
      "second",
      firstShape,
      bindingsFor(firstShape),
    );
    const tailShape = shape(1, 1_026, 129);
    const tail = await owner.createDispatch(
      "tail",
      tailShape,
      bindingsFor(tailShape),
    );
    expect(first).toMatchObject({
      kernelId: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
      weightLayout: "source-row-major",
      rangeCount: 1,
      plan: { workgroupsX: 16, workgroupsY: 1 },
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(second.plan).toEqual(first.plan);
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(3);
    expect(device.createBindGroupLayout.mock.calls[0]![0].entries.map(
      (entry: GPUBindGroupLayoutEntry) => entry.buffer?.minBindingSize,
    )).toEqual([8_192, 4_194_304, 16_384]);

    const pass = fakePass();
    first.encodeRange(pass, 0);
    second.encode(pass);
    tail.encode(pass);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [16, 1, 1],
      [16, 1, 1],
      [2, 1, 1],
    ]);
    expect(() => first.encodeRange(pass, 1)).toThrow(/range must be zero/);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch("late", firstShape, bindingsFor(firstShape)))
      .rejects.toThrow(/was destroyed/);
  });

  it("fails closed on profile, layout, shapes, bias, and device limits", async () => {
    expect(() => AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice(),
      "raw-fp16",
    )).toThrow(/requires reference-bf16/);
    expect(() => AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice(),
      "reference-bf16",
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    )).toThrow(/requires source-row-major/);
    expect(() => AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      "reference-bf16",
    )).toThrow(/128x1/);
    expect(() => AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 127 }),
      "reference-bf16",
    )).toThrow(/128x1/);
    expect(() => AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice({ maximumWorkgroupStorage: 17_023 }),
      "reference-bf16",
    )).toThrow(/17024 workgroup-storage bytes/);

    for (const rejected of [
      { rows: 0, inner: 1_024, columns: 1_024 },
      { rows: 3, inner: 1_024, columns: 1_024 },
      { rows: 1, inner: 1_023, columns: 1_024 },
      { rows: 1, inner: Number.NaN, columns: 1_024 },
      { rows: 2, inner: 1_000_000, columns: 2_000 },
    ]) {
      expect(() => planAceOpt0083PlannerLowRowBf16Gemv(rejected)).toThrow();
    }

    const exactShape = shape(2, 1_024, 2_048);
    const biasedDevice = fakeDevice();
    const biased = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      biasedDevice,
      "reference-bf16",
    );
    await expect(biased.createDispatch("biased", exactShape, {
      ...bindingsFor(exactShape),
      bias: fakeBinding(4_096),
    })).rejects.toThrow(/rejects bias/);
    expect(biasedDevice.createShaderModule).not.toHaveBeenCalled();

    const limited = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice({ maximumDispatch: 15 }),
      "reference-bf16",
    );
    await expect(limited.createDispatch(
      "dispatch-limit",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);

    const bufferLimited = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
      "reference-bf16",
    );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      exactShape,
      bindingsFor(exactShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
  });

  it("rejects short, misaligned, and aliased bindings before compilation", async () => {
    const exactShape = shape(2, 1_024, 2_048);
    const plan = planAceOpt0083PlannerLowRowBf16Gemv(exactShape);

    const shortDevice = fakeDevice();
    const short = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      shortDevice,
      "reference-bf16",
    );
    await expect(short.createDispatch("short", exactShape, {
      ...bindingsFor(exactShape),
      weight: fakeBinding(plan.weightBytes - 4),
    })).rejects.toThrow(/does not expose an aligned/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      misalignedDevice,
      "reference-bf16",
    );
    await expect(misaligned.createDispatch("misaligned", exactShape, {
      ...bindingsFor(exactShape),
      activation: {
        buffer: fakeBuffer(plan.activationBytes + 256),
        offset: 4,
        size: plan.activationBytes,
      },
    })).rejects.toThrow(/does not expose an aligned/);
    expect(misalignedDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const alias = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      aliasDevice,
      "reference-bf16",
    );
    const shared = fakeBuffer(Math.max(plan.activationBytes, plan.outputBytes));
    await expect(alias.createDispatch("alias", exactShape, {
      ...bindingsFor(exactShape),
      activation: { buffer: shared, offset: 0, size: plan.activationBytes },
      output: { buffer: shared, offset: 0, size: plan.outputBytes },
    })).rejects.toThrow(/output must not overlap an input binding/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroupLayout: ReturnType<typeof vi.fn>;
  readonly createPipelineLayout: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupStorage?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly storageAlignment?: number;
  readonly compilationError?: string;
} = {}): FakeDevice {
  return {
    features: new Set(),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupStorageSize:
        options.maximumWorkgroupStorage ?? 32_768,
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
    createComputePipelineAsync: vi.fn(async () => ({ label: "pipeline" })),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
  } as unknown as FakeDevice;
}

function shape(rows: 1 | 2, inner: number, columns: number): AceGemmShape {
  return { rows, inner, columns };
}

function bindingsFor(shape: AceGemmShape): AceGemmBufferBindings {
  const plan = planAceOpt0083PlannerLowRowBf16Gemv(shape);
  return {
    activation: fakeBinding(plan.activationBytes),
    weight: fakeBinding(plan.weightBytes),
    output: fakeBinding(plan.outputBytes),
  };
}

function directBindingsFor(shape: AceGemmShape): AceGemmBufferBindings {
  const plan = planAceOpt0083PlannerDirectLowRowBf16Gemv(shape);
  return {
    activation: fakeBinding(plan.activationBytes),
    weight: fakeBinding(plan.weightBytes),
    output: fakeBinding(plan.outputBytes),
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
