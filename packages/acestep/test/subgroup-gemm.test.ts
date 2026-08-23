import { describe, expect, it, vi } from "vitest";

import {
  ACE_SUBGROUP_GEMM_TILE_COLUMNS,
  ACE_SUBGROUP_GEMM_TILE_INNER,
  ACE_SUBGROUP_GEMM_TILE_ROWS,
  ACE_SUBGROUP_GEMM_WORKGROUP_BYTES,
  ACE_SUBGROUP_GEMM_WORKGROUP_SIZE,
  ACE_SUBGROUP_SIZE,
  AceSubgroupGemmKernel,
  aceSubgroupGemmWgsl,
  planAceSubgroupGemm,
} from "../src/webgpu/kernels/subgroup-gemm.js";
import {
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
} from "../src/webgpu/kernels/gemm.js";

describe("production packed-BF16 subgroup GEMM", () => {
  it("plans the exact fixed geometry and bounded principal-shape ranges", () => {
    const plan = planAceSubgroupGemm({
      rows: 2_250,
      inner: 2_048,
      columns: 6_144,
    });
    expect(plan).toMatchObject({
      tileRows: 32,
      tileColumns: 128,
      tileInner: 32,
      workgroupSize: 128,
      subgroupSize: 32,
      rowTiles: 71,
      columnTiles: 48,
      innerTiles: 64,
      workgroupCount: 3_408,
      activationElements: 4_608_000,
      weightElements: 12_582_912,
      packedWeightWords: 6_291_456,
      outputElements: 13_824_000,
      outputRangeCount: 14,
      packedWeightStorageShape: [48, 64, 32, 128],
    });
    expect(plan.outputRanges.every(
      ({ outputCount }) => outputCount <= ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
    )).toBe(true);
    expect(plan.outputRanges.every(
      ({ multiplyAdds }) =>
        multiplyAdds <= ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
    )).toBe(true);
    expect(plan.outputRanges.map(({ firstWorkgroup }) => firstWorkgroup)).toEqual(
      [0, 256, 512, 768, 1_024, 1_280, 1_536, 1_792, 2_048, 2_304,
        2_560, 2_816, 3_072, 3_328],
    );
    expect(plan.outputRanges.reduce(
      (sum, range) => sum + range.outputCount,
      0,
    )).toBe(plan.outputElements);
    expect(plan.outputRanges.at(-1)).toEqual({
      firstOutput: 13_631_488,
      outputCount: 192_512,
      firstWorkgroup: 3_328,
      workgroupCount: 80,
      multiplyAdds: 671_088_640,
    });
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.outputRanges)).toBe(true);
    expect(Object.isFrozen(plan.packedWeightStorageShape)).toBe(true);
    expect(ACE_SUBGROUP_GEMM_WORKGROUP_BYTES).toBe(0);
  });

  it("counts only active output rows while budgeting padded tail work", () => {
    const plan = planAceSubgroupGemm({
      rows: 33,
      inner: 32,
      columns: 256,
    });
    expect(plan).toMatchObject({
      rowTiles: 2,
      columnTiles: 2,
      workgroupCount: 4,
      outputElements: 8_448,
      outputRangeCount: 1,
    });
    expect(plan.outputRanges).toEqual([{
      firstOutput: 0,
      outputCount: 8_448,
      firstWorkgroup: 0,
      workgroupCount: 4,
      multiplyAdds: 524_288,
    }]);
  });

  it.each([
    { rows: 0, inner: 32, columns: 128, reason: /positive safe integer/ },
    { rows: 1, inner: 31, columns: 128, reason: /K divisible by 32/ },
    { rows: 1, inner: 32, columns: 127, reason: /N divisible by 128/ },
    { rows: 1.5, inner: 32, columns: 128, reason: /positive safe integer/ },
    {
      rows: 1,
      inner: 32 * 16_385,
      columns: 128,
      reason: /one bounded subgroup output tile/,
    },
  ])("rejects unsupported $rows x $inner x $columns geometry", (shape) => {
    expect(() => planAceSubgroupGemm(shape)).toThrow(shape.reason);
  });

  it("emits direct packed BF16, fixed-32 broadcasts, and exact K-order adds", () => {
    const source = aceSubgroupGemmWgsl(
      { rows: 2_250, inner: 2_048, columns: 6_144 },
      false,
    );
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain("if (subgroup_size != 32u)");
    expect(source).toContain("(column_tile * INNER_TILES + inner_tile)");
    expect(source).toContain("inner_in_tile * 64u");
    expect(source).toContain("subgroup_lane * 2u");
    expect(source).toContain("acc0 = acc0 + vec4<f32>(a0) * b;");
    expect(source).toContain("acc7 = acc7 + vec4<f32>(a7) * b;");
    expect(source.match(/subgroupBroadcast\(lane_a, [0-7]u\)/g)).toHaveLength(8);
    expect(source.indexOf("var inner_tile = 0u")).toBeLessThan(
      source.indexOf("var inner_in_tile = 0u"),
    );
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/subgroup(Add|Mul|Min|Max)/);
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("@binding(3) var<storage, read> bias");
  });

  it("pins adversarial fixtures that distinguish rounding and K reassociation", () => {
    const activation = [
      Math.fround(-68.06752014160156),
      Math.fround(12.192401885986328),
    ] as const;
    const weight = [1, 5.5625] as const;
    let separatelyRounded = 0;
    let contractedExpression = 0;
    for (let index = 0; index < activation.length; index += 1) {
      separatelyRounded = Math.fround(
        separatelyRounded + Math.fround(activation[index]! * weight[index]!),
      );
      // This is the WGSL source expression. Chrome/Metal may contract it.
      contractedExpression = Math.fround(
        contractedExpression + activation[index]! * weight[index]!,
      );
    }
    expect(float32Bits(separatelyRounded)).toBe(0xbe7d_3800);
    expect(float32Bits(contractedExpression)).toBe(0xbe7d_3830);
    expect(contractedExpression).not.toBe(separatelyRounded);

    const cancellation = [16_777_216, 1, -16_777_216, 0.5] as const;
    let sourceOrder = 0;
    for (const value of cancellation) {
      sourceOrder = Math.fround(sourceOrder + Math.fround(value));
    }
    const reassociated = Math.fround(
      Math.fround(cancellation[0] + cancellation[2]) +
        Math.fround(cancellation[1] + cancellation[3]),
    );
    expect(sourceOrder).toBe(0.5);
    expect(reassociated).toBe(1.5);
  });

  it("adds logical packed-BF16 bias only after the full contraction", () => {
    const source = aceSubgroupGemmWgsl(
      { rows: 33, inner: 64, columns: 128 },
      true,
    );
    expect(source).toContain(
      "@group(0) @binding(3) var<storage, read> bias: array<u32>;",
    );
    expect(source).toContain("@group(0) @binding(4) var<uniform>");
    expect(source).toContain("fn load_bias_vec4(first_scalar: u32)");
    expect(source).toContain("value = value + load_bias_vec4(column);");
    expect(source.lastIndexOf("value = value + load_bias_vec4(column);")).toBeGreaterThan(
      source.lastIndexOf("inner_in_tile += 1u"),
    );
  });

  it("fails closed without a reported fixed-32 subgroup capability", () => {
    const noFeature = fakeDevice({ features: [] });
    expect(() =>
      AceSubgroupGemmKernel.create(noFeature, {
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
      })
    ).toThrow(/fixed 32-lane subgroups/);

    const variable = fakeDevice();
    expect(() =>
      AceSubgroupGemmKernel.create(variable, {
        subgroupMinSize: 4,
        subgroupMaxSize: 64,
      })
    ).toThrow(/fixed 32-lane subgroups/);
    expect(() =>
      AceSubgroupGemmKernel.create(variable, {})
    ).toThrow(/fixed 32-lane subgroups/);

    const tooSmall = fakeDevice({ maximumWorkgroupSize: 64 });
    expect(() =>
      AceSubgroupGemmKernel.create(tooSmall, {
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
      })
    ).toThrow(/requires WG128/);
  });

  it("compiles, binds, and encodes every bounded range", async () => {
    const device = fakeDevice();
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 1_024, inner: 32_768, columns: 128 };
    const dispatch = await kernel.createDispatch("candidate", shape, {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    });
    expect(dispatch.rangeCount).toBe(2);
    expect(dispatch.plan.outputRanges.map(({ firstWorkgroup }) => firstWorkgroup))
      .toEqual([0, 16]);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const parameterBuffer = device.createdBuffers[0]!;
    expect(new Uint32Array(parameterBuffer.mapped, 0, 1)[0]).toBe(0);
    expect(new Uint32Array(parameterBuffer.mapped, 256, 1)[0]).toBe(16);

    const pass = fakePass();
    dispatch.encodeRange(pass, 1);
    expect(pass.dispatchWorkgroups).toHaveBeenLastCalledWith(16, 1, 1);
    expect(() => dispatch.encodeRange(pass, 2)).toThrow(/outside \[0, 2\)/);
    dispatch.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(3);

    kernel.destroy();
    await Promise.resolve();
    expect(parameterBuffer.destroy).toHaveBeenCalledOnce();
    await expect(kernel.createDispatch("after-destroy", shape, {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    })).rejects.toThrow(/was destroyed/);
  });

  it("uses distinct bias/no-bias layouts without changing dispatch geometry", async () => {
    const device = fakeDevice({
      pipelineResults: [
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
      scopeResults: [null, null, null, null, null, null],
    });
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 33, inner: 32, columns: 128 };
    const common = {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    };
    const noBias = await kernel.createDispatch("no-bias", shape, common);
    const bias = await kernel.createDispatch("bias", shape, {
      ...common,
      bias: fakeBinding(shape.columns * 2),
    });
    expect(bias.plan.outputRanges).toEqual(noBias.plan.outputRanges);
    const calls = device.createBindGroup.mock.calls;
    const noBiasDescriptor = calls[0]?.[0] as GPUBindGroupDescriptor;
    const biasDescriptor = calls[1]?.[0] as GPUBindGroupDescriptor;
    expect(Array.from(noBiasDescriptor.entries, ({ binding }) => binding)).toEqual([
      0, 1, 2, 3,
    ]);
    expect(Array.from(biasDescriptor.entries, ({ binding }) => binding)).toEqual([
      0, 1, 2, 3, 4,
    ]);
    kernel.destroy();
    await Promise.resolve();
    expect(device.createdBuffers.every(
      (buffer) => vi.mocked(buffer.destroy).mock.calls.length === 1,
    )).toBe(true);
  });

  it("evicts failed compilation and can retry the same specialization", async () => {
    const compilationFailure = new Error("synthetic subgroup compile failure");
    const device = fakeDevice({
      pipelineResults: [
        Promise.reject(compilationFailure),
        Promise.resolve(fakePipeline()),
      ],
    });
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 32, inner: 32, columns: 128 };
    const bindings = {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    };
    await expect(kernel.createDispatch("first", shape, bindings)).rejects.toBe(
      compilationFailure,
    );
    await expect(kernel.createDispatch("retry", shape, bindings)).resolves.toMatchObject({
      label: "retry",
      rangeCount: 1,
    });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createdBuffers).toHaveLength(1);
  });

  it("destroys a scoped range buffer after allocation failure and retries", async () => {
    const allocationFailure = {
      message: "synthetic range-buffer validation failure",
    } as GPUError;
    const device = fakeDevice({
      pipelineResults: [
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
      scopeResults: [
        allocationFailure,
        null,
        null,
        null,
        null,
        null,
      ],
    });
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 32, inner: 32, columns: 128 };
    const bindings = {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    };
    await expect(kernel.createDispatch("allocation-fails", shape, bindings))
      .rejects.toThrow(/validation failure/);
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
    await expect(kernel.createDispatch("allocation-retry", shape, bindings))
      .resolves.toMatchObject({ label: "allocation-retry", rangeCount: 1 });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.pushErrorScope).toHaveBeenCalledTimes(6);
    expect(device.popErrorScope).toHaveBeenCalledTimes(6);
    expect(device.createdBuffers).toHaveLength(2);
  });

  it("destroys a pipeline completed after kernel destruction", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelineResults: [pending] });
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 32, inner: 32, columns: 128 };
    const dispatchPromise = kernel.createDispatch("pending", shape, {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    });
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(dispatchPromise).rejects.toThrow(/destroyed while compiling/);
    await Promise.resolve();
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
  });

  it("rejects undersized and overlapping bindings before compilation", async () => {
    const device = fakeDevice();
    const kernel = AceSubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    const shape = { rows: 32, inner: 32, columns: 128 };
    await expect(kernel.createDispatch("short", shape, {
      activation: fakeBinding(shape.rows * shape.inner * 4 - 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    })).rejects.toThrow(/activation binding/);

    const shared = fakeBuffer(shape.rows * shape.inner * 4 + 65_536);
    await expect(kernel.createDispatch("aliased", shape, {
      activation: {
        buffer: shared,
        offset: 0,
        size: shape.rows * shape.inner * 4,
      },
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: {
        buffer: shared,
        offset: 1_024,
        size: shape.rows * shape.columns * 4,
      },
    })).rejects.toThrow(/must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_SUBGROUP_SIZE).toBe(32);
expect(ACE_SUBGROUP_GEMM_TILE_ROWS).toBe(32);
expect(ACE_SUBGROUP_GEMM_TILE_COLUMNS).toBe(128);
expect(ACE_SUBGROUP_GEMM_TILE_INNER).toBe(32);
expect(ACE_SUBGROUP_GEMM_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

function float32Bits(value: number): number {
  const scalar = new Float32Array([value]);
  return new Uint32Array(scalar.buffer)[0]!;
}

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
  readonly pushErrorScope: ReturnType<typeof vi.fn>;
  readonly popErrorScope: ReturnType<typeof vi.fn>;
  readonly createdBuffers: ReturnType<typeof fakeMappedBuffer>[];
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly features?: readonly string[];
  readonly maximumWorkgroupSize?: number;
  readonly pipelineResults?: readonly Promise<GPUComputePipeline>[];
  readonly scopeResults?: readonly (GPUError | null)[];
} = {}): FakeDevice {
  const createdBuffers: ReturnType<typeof fakeMappedBuffer>[] = [];
  const pipelineResults = [...(options.pipelineResults ?? [
    Promise.resolve(fakePipeline()),
  ])];
  const scopeResults = [...(options.scopeResults ?? [null, null, null])];
  return {
    features: new Set(options.features ?? ["subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumWorkgroupSize ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSize ?? 256,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => scopeResults.shift() ?? null),
    createShaderModule: vi.fn(() => ({ label: "module" })),
    createComputePipelineAsync: vi.fn(() => {
      const result = pipelineResults.shift();
      if (result === undefined) {
        throw new Error("fake device exhausted pipeline results");
      }
      return result;
    }),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = fakeMappedBuffer(Number(descriptor.size));
      createdBuffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createdBuffers,
  } as unknown as FakeDevice;
}

function fakePipeline(): GPUComputePipeline {
  return {
    getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
  } as unknown as GPUComputePipeline;
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

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeMappedBuffer(size: number) {
  const mapped = new ArrayBuffer(size);
  return {
    size,
    mapped,
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer & {
    readonly mapped: ArrayBuffer;
    readonly destroy: ReturnType<typeof vi.fn>;
  };
}
