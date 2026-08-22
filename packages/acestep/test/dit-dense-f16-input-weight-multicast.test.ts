import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../src/model/manifest.js";
import {
  ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID,
  aceOpt0078DenseWeightMulticastWgsl,
  planAceOpt0078DenseWeightMulticast,
} from "../src/webgpu/kernels/dit-dense-fp16-weight-multicast.js";
import {
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ACCUMULATORS_PER_LANE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_BARRIERS_PER_INNER_TILE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_COLUMNS_PER_LANE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_INNER_TILE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_LANE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ROWS_PER_SUBGROUP,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUP_SIZE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_COLUMNS,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_INNER,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_ROWS,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_SIZE,
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_STORAGE_BYTES,
  AceOpt0081DenseF16InputWeightMulticastKernel,
  aceOpt0081DenseF16InputWeightMulticastWgsl,
  planAceOpt0081DenseF16InputWeightMulticast,
} from "../src/webgpu/kernels/dit-dense-f16-input-weight-multicast.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";

const M2250 = 2_250;
const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

describe("OPT-0081 arm C typed-F16 dense weight multicast", () => {
  it("pins OPT-0078 ownership with two-byte activations on all exact shapes", () => {
    expect(ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID).toBe(
      "opt-0081-m32-n256-k32-wg256-weight-multicast-typed-f16-input-fp32-output-v1",
    );
    expect({
      rows: ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_ROWS,
      columns: ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_COLUMNS,
      inner: ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_INNER,
      workgroup: ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUP_SIZE,
      subgroups:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUPS_PER_WORKGROUP,
      rowsPerSubgroup:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ROWS_PER_SUBGROUP,
      columnsPerLane:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_COLUMNS_PER_LANE,
      accumulators:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ACCUMULATORS_PER_LANE,
      recordsPerTile:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_INNER_TILE,
      recordsPerLane:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_LANE,
      barriers:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_BARRIERS_PER_INNER_TILE,
      storage:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_STORAGE_BYTES,
    }).toEqual({
      rows: 32,
      columns: 256,
      inner: 32,
      workgroup: 256,
      subgroup: 32,
      subgroups: 8,
      rowsPerSubgroup: 4,
      columnsPerLane: 8,
      accumulators: 32,
      recordsPerTile: 1_024,
      recordsPerLane: 4,
      barriers: 2,
      storage: 16_384,
    });

    const cases = [
      {
        inner: 2_048,
        columns: 2_048,
        columnTiles: 8,
        innerTiles: 64,
        workgroups: 568,
        scheduled: 9_529_458_688,
        valid: 9_437_184_000,
        activationBytes: 9_216_000,
        weightBytes: 8_388_608,
        outputBytes: 18_432_000,
        activationRequests: 74_448_896,
        weightRequests: 595_591_168,
        operandRequests: 670_040_064,
      },
      {
        inner: 2_048,
        columns: 1_024,
        columnTiles: 4,
        innerTiles: 64,
        workgroups: 284,
        scheduled: 4_764_729_344,
        valid: 4_718_592_000,
        activationBytes: 9_216_000,
        weightBytes: 4_194_304,
        outputBytes: 9_216_000,
        activationRequests: 37_224_448,
        weightRequests: 297_795_584,
        operandRequests: 335_020_032,
      },
      {
        inner: 2_048,
        columns: 6_144,
        columnTiles: 24,
        innerTiles: 64,
        workgroups: 1_704,
        scheduled: 28_588_376_064,
        valid: 28_311_552_000,
        activationBytes: 9_216_000,
        weightBytes: 25_165_824,
        outputBytes: 55_296_000,
        activationRequests: 223_346_688,
        weightRequests: 1_786_773_504,
        operandRequests: 2_010_120_192,
      },
      {
        inner: 6_144,
        columns: 2_048,
        columnTiles: 8,
        innerTiles: 192,
        workgroups: 568,
        scheduled: 28_588_376_064,
        valid: 28_311_552_000,
        activationBytes: 27_648_000,
        weightBytes: 25_165_824,
        outputBytes: 18_432_000,
        activationRequests: 223_346_688,
        weightRequests: 1_786_773_504,
        operandRequests: 2_010_120_192,
      },
    ] as const;

    for (const expected of cases) {
      const shapeValue = shape(expected.inner, expected.columns);
      const reference = planAceOpt0078DenseWeightMulticast(shapeValue);
      const candidate = planAceOpt0081DenseF16InputWeightMulticast(shapeValue);
      expect(candidate).toMatchObject({
        kernelSetId:
          ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID,
        referenceKernelSetId:
          ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID,
        activationStorage: "scalar-f16",
        activationElementBytes: 2,
        weightElementBytes: 2,
        outputElementBytes: 4,
        activationBytes: expected.activationBytes,
        weightBytes: expected.weightBytes,
        outputBytes: expected.outputBytes,
        workgroupsX: expected.columnTiles,
        workgroupsY: 71,
        rowTiles: 71,
        columnTiles: expected.columnTiles,
        innerTiles: expected.innerTiles,
        workgroupCount: expected.workgroups,
        scheduledRows: 2_272,
        scheduledMultiplyAdds: expected.scheduled,
        validMultiplyAdds: expected.valid,
        estimatedGlobalActivationBytes: expected.activationRequests,
        estimatedGlobalWeightBytes: expected.weightRequests,
        estimatedGlobalOperandBytes: expected.operandRequests,
        outputRangeCount: 1,
      });
      expect(candidate.outputRanges).toEqual(reference.outputRanges);
      expect(candidate.packedWeightStorageShape).toEqual(
        reference.packedWeightStorageShape,
      );
      expect(candidate.estimatedGlobalActivationBytes * 2).toBe(
        reference.estimatedGlobalActivationBytes,
      );
      expect(Object.isFrozen(candidate)).toBe(true);
      expect(Object.isFrozen(candidate.outputRanges)).toBe(true);
      expect(Object.isFrozen(candidate.packedWeightStorageShape)).toBe(true);
    }

    const multiplicities = [4, 2, 2, 1] as const;
    const weighted = cases.map((entry, index) => ({
      plan: planAceOpt0081DenseF16InputWeightMulticast(
        shape(entry.inner, entry.columns),
      ),
      multiplicity: multiplicities[index]!,
    }));
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.workgroupCount * multiplicity, 0)).toBe(6_816);
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.scheduledMultiplyAdds * multiplicity, 0)).toBe(
      133_412_421_632,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalActivationBytes * multiplicity, 0)).toBe(
      1_042_284_544,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalWeightBytes * multiplicity, 0)).toBe(
      8_338_276_352,
    );
    expect(weighted.reduce((sum, { plan, multiplicity }) =>
      sum + plan.estimatedGlobalOperandBytes * multiplicity, 0)).toBe(
      9_380_560_896,
    );
  });

  it("derives only the typed input and identity from frozen OPT-0078 WGSL", () => {
    const exactShape = shape(2_048, 2_048);
    const historical = aceOpt0078DenseWeightMulticastWgsl(exactShape);
    const expected = replaceOnce(
      replaceOnce(
        replaceOnce(
          historical,
          `// kernel-id: ${ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
          `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
        ),
        "@group(0) @binding(0) var<storage, read> activation: array<f32>;",
        "@group(0) @binding(0) var<storage, read> activation: array<f16>;",
      ),
      "lane_a = f16(activation[lane_row * INNER + inner]);",
      "lane_a = activation[lane_row * INNER + inner];",
    );
    const source = aceOpt0081DenseF16InputWeightMulticastWgsl(exactShape);
    expect(source).toBe(expected);
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
    );
    expect(source).not.toContain(
      `// kernel-id: ${ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
    );
    expect(source).toContain("activation: array<f16>");
    expect(source).toContain(
      "lane_a = activation[lane_row * INNER + inner]",
    );
    expect(source).not.toContain("f16(activation[");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).toContain("array<vec4<u32>, 1024>");
    expect(source.match(/var acc[0-3]_[01] = vec4<f32>\(0\.0\);/g))
      .toHaveLength(8);
    expect(source).toContain("vec4<f32>(f32(a0)) * vec4<f32>(b0)");
    expect(source).toContain(
      "var<storage, read_write> output: array<vec4<f32>>",
    );
    expect(source).not.toMatch(/\bdot\s*\(/);
    expect(source).not.toMatch(/\bfma\s*\(/);
  });

  it("caches shape pipelines and physical-resource bind groups", async () => {
    const device = fakeDevice();
    const owner = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      device,
      FIXED32,
    );
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
      kernelSetId:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      rangeCount: 1,
      plan: {
        activationBytes: 9_216_000,
        workgroupsX: 8,
        workgroupsY: 71,
      },
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
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ features: ["subgroups"] }),
      FIXED32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ features: ["shader-f16"] }),
      FIXED32,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ maximumInvocations: 255 }),
      FIXED32,
    )).toThrow(/256x1/);
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 255 }),
      FIXED32,
    )).toThrow(/256x1/);
    expect(() => AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ maximumWorkgroupStorage: 16_383 }),
      FIXED32,
    )).toThrow(/16384 workgroup-storage bytes/);

    for (const rejected of [
      { rows: 2_249, inner: 2_048, columns: 2_048 },
      { rows: M2250, inner: 1_024, columns: 2_048 },
      { rows: M2250, inner: 2_048, columns: 4_096 },
      { rows: Number.NaN, inner: 2_048, columns: 2_048 },
    ]) {
      expect(() => planAceOpt0081DenseF16InputWeightMulticast(rejected))
        .toThrow();
    }

    const wideShape = shape(2_048, 6_144);
    const dispatchLimited = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ maximumDispatch: 23 }),
      FIXED32,
    );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the dispatch dimension/);
    const bufferLimited = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
      FIXED32,
    );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const allocationLimited =
      AceOpt0081DenseF16InputWeightMulticastKernel.create(
        fakeDevice({ maximumBuffer: 1 }),
        FIXED32,
      );
    await expect(allocationLimited.createDispatch(
      "allocation-limit",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/exceeds the device buffer limits/);
    const badAlignment = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      fakeDevice({ storageAlignment: 3 }),
      FIXED32,
    );
    await expect(badAlignment.createDispatch(
      "bad-alignment",
      wideShape,
      bindingsFor(wideShape),
    )).rejects.toThrow(/invalid alignment/);

    const exactShape = shape(2_048, 2_048);
    const biasedDevice = fakeDevice();
    const biased = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      biasedDevice,
      FIXED32,
    );
    await expect(biased.createDispatch("biased", exactShape, {
      ...bindingsFor(exactShape),
      bias: fakeBinding(8_192),
    })).rejects.toThrow(/rejects bias/);
    expect(biasedDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("rejects short, misaligned, and aliased bindings before compilation", async () => {
    const exactShape = shape(2_048, 2_048);
    const plan = planAceOpt0081DenseF16InputWeightMulticast(exactShape);

    const shortDevice = fakeDevice();
    const short = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      shortDevice,
      FIXED32,
    );
    await expect(short.createDispatch("short", exactShape, {
      ...bindingsFor(exactShape),
      activation: fakeBinding(plan.activationBytes - 4),
    })).rejects.toThrow(/does not expose an aligned/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      misalignedDevice,
      FIXED32,
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
    const alias = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      aliasDevice,
      FIXED32,
    );
    const shared = fakeBuffer(Math.max(plan.activationBytes, plan.outputBytes));
    await expect(alias.createDispatch("alias", exactShape, {
      ...bindingsFor(exactShape),
      activation: { buffer: shared, offset: 0, size: plan.activationBytes },
      output: { buffer: shared, offset: 0, size: plan.outputBytes },
    })).rejects.toThrow(/output must not overlap an input binding/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("evicts failed compilation and rejects destruction during compilation", async () => {
    const exactShape = shape(2_048, 1_024);
    const bindings = bindingsFor(exactShape);
    const failedDevice = fakeDevice({ compilationError: "typed f16 rejected" });
    const failed = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      failedDevice,
      FIXED32,
    );
    await expect(failed.createDispatch("first-failure", exactShape, bindings))
      .rejects.toThrow(/typed f16 rejected/);
    await Promise.resolve();
    await expect(failed.createDispatch("retry", exactShape, bindings))
      .rejects.toThrow(/typed f16 rejected/);
    expect(failedDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(failedDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(failedDevice.createBindGroup).not.toHaveBeenCalled();

    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const pendingDevice = fakeDevice({ pipeline });
    const pendingOwner = AceOpt0081DenseF16InputWeightMulticastKernel.create(
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

function replaceOnce(source: string, expected: string, replacement: string): string {
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

function bindingsFor(shapeValue: AceGemmShape): AceGemmBufferBindings {
  const plan = planAceOpt0081DenseF16InputWeightMulticast(shapeValue);
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
