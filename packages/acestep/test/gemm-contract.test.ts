import { describe, expect, it, vi } from "vitest";

import {
  ACE_DIT_GEMM_WEIGHT_LAYOUT,
  ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  ACE_TILED_GEMM_WORKGROUP_BYTES,
  AceCorrectnessGemmKernel,
  aceCompositeCooperativeQuanta,
  aceCompositeCooperativeSequence,
  aceCorrectnessGemmWgsl,
  aceGemmWeightScalarIndex,
  aceScalarGemmOracleWgsl,
  planAceGemm,
  planAceTiledGemm,
  type AceCooperativeGemmPlan,
} from "../src/webgpu/kernels/gemm.js";

describe("ACE correctness GEMM contract", () => {
  it("plans the principal three-minute DiT shapes in two dimensions", () => {
    expect(planAceGemm({ rows: 2_250, inner: 2_048, columns: 6_144 })).toEqual({
      rows: 2_250,
      inner: 2_048,
      columns: 6_144,
      workgroupsX: 768,
      workgroupsY: 282,
      activationElements: 4_608_000,
      weightElements: 12_582_912,
      outputElements: 13_824_000,
    });
    expect(planAceGemm({ rows: 2_250, inner: 6_144, columns: 2_048 })).toMatchObject({
      workgroupsX: 256,
      workgroupsY: 282,
      weightElements: 12_582_912,
    });
  });

  it("plans conservative source-layout tiles and bounded output ranges", () => {
    const plan = planAceTiledGemm({
      rows: 2_250,
      inner: 2_048,
      columns: 6_144,
    });
    expect(plan).toMatchObject({
      tileRows: 16,
      tileColumns: 128,
      tileInner: 16,
      workgroupSize: 128,
      outputRangeCount: 14,
    });
    expect(ACE_TILED_GEMM_WORKGROUP_BYTES).toBe(9_216);
    expect(plan.outputRanges.every(
      ({ outputCount }) => outputCount <= ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
    )).toBe(true);
    expect(plan.outputRanges.every(
      ({ multiplyAdds }) =>
        multiplyAdds <= ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
    )).toBe(true);
    expect(plan.outputRanges.map(({ firstOutput }) => firstOutput)).toEqual(
      [...plan.outputRanges]
        .map(({ firstOutput }) => firstOutput)
        .sort((left, right) => left - right),
    );
    expect(plan.outputRanges.reduce(
      (total, { outputCount }) => total + outputCount,
      0,
    )).toBe(plan.outputElements);
  });

  it.each([
    { rows: 0, inner: 2, columns: 2 },
    { rows: 1, inner: -1, columns: 2 },
    { rows: 1.5, inner: 2, columns: 2 },
    { rows: 1, inner: 2, columns: Number.NaN },
  ])("rejects malformed shape $rows x $inner x $columns", (shape) => {
    expect(() => planAceGemm(shape)).toThrow();
  });

  it("emits packed-BF16 source-order tiled contraction with no FMA", () => {
    const source = aceCorrectnessGemmWgsl(
      "reference-bf16",
      { rows: 3, inner: 5, columns: 7 },
      false,
    );
    expect(source).toContain("const ROWS: u32 = 3u");
    expect(source).toContain("const INNER: u32 = 5u");
    expect(source).toContain("const COLUMNS: u32 = 7u");
    expect(source).toContain("pair & 0xffffu");
    expect(source).toContain("bits16 << 16u");
    expect(source).not.toContain("var<storage, read> bias");
    expect(source).not.toContain("load_bias");
    expect(source).toContain("ACE GEMM weight layout: source-row-major");
    expect(source).toContain("return column * INNER + inner");
    expect(source).toContain(
      "load_weight(weight_scalar_index(source_column, source_inner))",
    );
    expect(source).not.toContain("column / 128u");
    expect(source).toContain("var acc0 = vec4<f32>(0.0)");
    expect(source).toContain("acc0 = acc0 +");
    expect(source).not.toContain("fma(");
    expect(source).toContain("workgroupBarrier()");
  });

  it("emits a shader-f16 storage and arithmetic variant", () => {
    const source = aceCorrectnessGemmWgsl(
      "raw-fp16",
      { rows: 3, inner: 5, columns: 7 },
      true,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("array<f16>");
    expect(source).toContain("var acc0 = vec4<f16>(0.0h)");
    expect(source).toContain("acc0 = acc0 +");
    expect(source).toContain("@binding(3) var<storage, read> bias");
    expect(source).toContain("sum = sum + bias[output_column]");
    expect(source).toContain("return column * INNER + inner");
    expect(source).not.toContain("fma(");
  });

  it("maps both physical layouts exactly across N128 and K32 boundaries", () => {
    const shape = { inner: 64, columns: 256 };
    const coordinates = [
      [0, 0, 0],
      [1, 0, 1],
      [127, 0, 127],
      [0, 1, 128],
      [127, 31, 4_095],
      [0, 32, 4_096],
      [128, 0, 8_192],
      [129, 33, 12_417],
      [255, 63, 16_383],
    ] as const;
    for (const [column, inner, physical] of coordinates) {
      expect(aceGemmWeightScalarIndex(
        ACE_DIT_GEMM_WEIGHT_LAYOUT,
        shape,
        column,
        inner,
      )).toBe(physical);
      expect(aceGemmWeightScalarIndex(
        "source-row-major",
        shape,
        column,
        inner,
      )).toBe(column * shape.inner + inner);
    }
    const everyPhysicalScalar = new Set<number>();
    for (let column = 0; column < shape.columns; column += 1) {
      for (let inner = 0; inner < shape.inner; inner += 1) {
        everyPhysicalScalar.add(aceGemmWeightScalarIndex(
          ACE_DIT_GEMM_WEIGHT_LAYOUT,
          shape,
          column,
          inner,
        ));
      }
    }
    expect(everyPhysicalScalar.size).toBe(shape.columns * shape.inner);
    expect(Math.min(...everyPhysicalScalar)).toBe(0);
    expect(Math.max(...everyPhysicalScalar)).toBe(16_383);
  });

  it.each(["reference-bf16", "raw-fp16"] as const)(
    "emits direct %s tile-major scalar loads while keeping bias logical",
    (profile) => {
      const source = aceCorrectnessGemmWgsl(
        profile,
        { rows: 3, inner: 64, columns: 256 },
        true,
        ACE_DIT_GEMM_WEIGHT_LAYOUT,
      );
      expect(source).toContain(
        `ACE GEMM weight layout: ${ACE_DIT_GEMM_WEIGHT_LAYOUT}`,
      );
      expect(source).toContain(
        "((column / 128u) * (INNER / 32u) + (inner / 32u)) * 32u",
      );
      expect(source).toContain("(inner % 32u)) * 128u + (column % 128u)");
      expect(source).toContain(
        "weight_scalar_index(source_column, source_inner)",
      );
      if (profile === "reference-bf16") {
        expect(source).toContain("load_bias(output_column)");
      } else {
        expect(source).toContain("bias[output_column]");
      }
    },
  );

  it("fails closed on tile-major shapes outside exact N128/K32 geometry", async () => {
    expect(() => aceCorrectnessGemmWgsl(
      "reference-bf16",
      { rows: 1, inner: 32, columns: 127 },
      false,
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    )).toThrow(/N divisible by 128/);
    expect(() => aceCorrectnessGemmWgsl(
      "reference-bf16",
      { rows: 1, inner: 31, columns: 128 },
      false,
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    )).toThrow(/K divisible by 32/);

    const device = fakeGemmDevice();
    const kernel = AceCorrectnessGemmKernel.create(
      device,
      "reference-bf16",
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    );
    try {
      await expect(kernel.createDispatch(
        "bad-tile-shape",
        { rows: 1, inner: 32, columns: 127 },
        {
          activation: fakeBinding(128),
          weight: fakeBinding(128 * 127 * 2),
          output: fakeBinding(127 * 4),
        },
      )).rejects.toThrow(/N divisible by 128/);
      expect(device.createShaderModule).not.toHaveBeenCalled();
    } finally {
      kernel.destroy();
    }
  });

  it("pins layout in dispatch, shader, bind-group labels, and cache isolation", async () => {
    const device = fakeGemmDevice();
    const shape = { rows: 1, inner: 32, columns: 128 };
    const bindings = {
      activation: fakeBinding(shape.rows * shape.inner * 4),
      weight: fakeBinding(shape.columns * shape.inner * 2),
      output: fakeBinding(shape.rows * shape.columns * 4),
    };
    const rowMajor = AceCorrectnessGemmKernel.create(
      device,
      "reference-bf16",
    );
    const tileMajor = AceCorrectnessGemmKernel.create(
      device,
      "reference-bf16",
      ACE_DIT_GEMM_WEIGHT_LAYOUT,
    );
    try {
      const first = await rowMajor.createDispatch("row", shape, bindings);
      const second = await tileMajor.createDispatch("tile", shape, bindings);
      const cached = await tileMajor.createDispatch("tile-cached", shape, bindings);
      expect(first.weightLayout).toBe("source-row-major");
      expect(second.weightLayout).toBe(ACE_DIT_GEMM_WEIGHT_LAYOUT);
      expect(cached.weightLayout).toBe(ACE_DIT_GEMM_WEIGHT_LAYOUT);
      expect(device.createShaderModule).toHaveBeenCalledTimes(2);
      const shaderLabels = device.createShaderModule.mock.calls.map(
        ([descriptor]) => (descriptor as GPUShaderModuleDescriptor).label,
      );
      expect(shaderLabels).toEqual([
        expect.stringContaining("source-row-major"),
        expect.stringContaining(ACE_DIT_GEMM_WEIGHT_LAYOUT),
      ]);
      const bindLabels = device.createBindGroup.mock.calls.map(
        ([descriptor]) => (descriptor as GPUBindGroupDescriptor).label,
      );
      expect(bindLabels).toEqual([
        expect.stringContaining("source-row-major"),
        expect.stringContaining(ACE_DIT_GEMM_WEIGHT_LAYOUT),
        expect.stringContaining(ACE_DIT_GEMM_WEIGHT_LAYOUT),
      ]);
    } finally {
      rowMajor.destroy();
      tileMajor.destroy();
    }
  });

  it("retains the pre-tiling scalar shader strictly as a differential oracle", () => {
    const reference = aceScalarGemmOracleWgsl(
      "reference-bf16",
      { rows: 3, inner: 5, columns: 7 },
      true,
    );
    const fp16 = aceScalarGemmOracleWgsl(
      "raw-fp16",
      { rows: 3, inner: 5, columns: 7 },
      false,
    );
    expect(reference).toContain("weight_base = column * INNER");
    expect(reference).toContain("sum = sum + activation[");
    expect(fp16).toContain("var sum = 0.0h");
    expect(reference).not.toContain("workgroupBarrier");
    expect(fp16).not.toContain("workgroupBarrier");
  });

  it.each(["activation", "weight", "bias"] as const)(
    "rejects exact logical output overlap with %s before compilation",
    async (aliasedInput) => {
      const device = fakeGemmDevice();
      const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
      const shared = fakeBuffer(256);
      const bindings = {
        activation: fakeBinding(64),
        weight: fakeBinding(64),
        output: { buffer: shared, offset: 64, size: 64 },
        bias: fakeBinding(16),
      } satisfies Record<string, GPUBufferBinding>;
      bindings[aliasedInput] = {
        buffer: shared,
        offset: 96,
        size: 64,
      };
      try {
        await expect(kernel.createDispatch(
          `alias-${aliasedInput}`,
          { rows: 4, inner: 4, columns: 4 },
          bindings,
        )).rejects.toThrow(/output must not overlap/);
        expect(device.createShaderModule).not.toHaveBeenCalled();
      } finally {
        kernel.destroy();
      }
    },
  );

  it("allows unused exposed bytes to overlap outside exact logical ranges", async () => {
    const device = fakeGemmDevice();
    const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
    const shared = fakeBuffer(256);
    try {
      const dispatch = await kernel.createDispatch(
        "logical-range-only",
        { rows: 1, inner: 1, columns: 1 },
        {
          activation: { buffer: shared, offset: 0, size: 128 },
          weight: fakeBinding(4),
          output: { buffer: shared, offset: 64, size: 128 },
        },
      );
      expect(dispatch.rangeCount).toBe(1);
    } finally {
      kernel.destroy();
    }
  });

  it("destroys scoped range parameters when destruction races compilation", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipelinePromise = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const rangeParameters = fakeMappedBuffer(256);
    const device = fakeGemmDevice({ pipelinePromise, rangeParameters });
    const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
    const pending = kernel.createDispatch(
      "destroy-during-compile",
      { rows: 1, inner: 1, columns: 1 },
      {
        activation: fakeBinding(4),
        weight: fakeBinding(4),
        output: fakeBinding(4),
      },
    );
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(pending).rejects.toThrow(/destroyed while compiling/);
    await Promise.resolve();
    expect(rangeParameters.destroy).toHaveBeenCalledOnce();
    expect(device.createBindGroup).not.toHaveBeenCalled();
  });

  it("destroys scoped range parameters when bind-group construction fails", async () => {
    const rangeParameters = fakeMappedBuffer(256);
    const device = fakeGemmDevice({ rangeParameters });
    device.createBindGroup.mockImplementation(() => {
      throw new Error("bind group failed");
    });
    const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
    try {
      await expect(kernel.createDispatch(
        "bind-group-failure",
        { rows: 1, inner: 1, columns: 1 },
        {
          activation: fakeBinding(4),
          weight: fakeBinding(4),
          output: fakeBinding(4),
        },
      )).rejects.toThrow(/bind group failed/);
    } finally {
      kernel.destroy();
    }
    await Promise.resolve();
    expect(rangeParameters.destroy).toHaveBeenCalledOnce();
  });

  it("expands ranged GEMMs inside composites instead of calling encode", () => {
    const events: string[] = [];
    const ranged = {
      label: "ranged",
      plan: {
        outputRanges: [
          { multiplyAdds: 1_000_000_000 },
          { multiplyAdds: 1_000_000_000 },
          { multiplyAdds: 1_000_000_000 },
        ],
      } as unknown as AceCooperativeGemmPlan,
      rangeCount: 3,
      encodeRange(_pass: GPUComputePassEncoder, range: number) {
        events.push(`range:${range}`);
      },
      encode() {
        throw new Error("composite collapsed ranged GEMM");
      },
    };
    const scalar = {
      label: "scalar",
      encode() {
        events.push("scalar");
      },
    };
    const quanta = aceCompositeCooperativeQuanta([ranged, scalar]);
    expect(quanta.map(({ id }) => id)).toEqual([
      "ranged-range-0..ranged-range-1",
      "ranged-range-2..scalar",
    ]);
    expect(quanta.map(({ primitiveCount }) => primitiveCount)).toEqual([2, 2]);
    for (const quantum of quanta) {
      quantum.encode({} as GPUComputePassEncoder);
    }
    expect(events).toEqual(["range:0", "range:1", "range:2", "scalar"]);
  });

  it("describes compact composite membership without encoding or changing packing", () => {
    const events: string[] = [];
    const ranged = {
      label: "ranged",
      plan: {
        outputRanges: [
          { multiplyAdds: 1_000_000_000 },
          { multiplyAdds: 1_000_000_000 },
          { multiplyAdds: 1_000_000_000 },
        ],
      } as unknown as AceCooperativeGemmPlan,
      rangeCount: 3,
      encodeRange(_pass: GPUComputePassEncoder, range: number) {
        events.push(`range:${range}`);
      },
    };
    const scalar = {
      label: "scalar",
      encode() {
        events.push("scalar");
      },
    };
    const sequence = aceCompositeCooperativeSequence([ranged, scalar]);
    expect(sequence.quantumCount).toBe(2);
    expect(sequence.describeQuantum(0)).toEqual({
      id: "ranged-range-0..ranged-range-1",
      primitiveCount: 2,
      scheduledMultiplyAdds: 2_000_000_000,
      members: [
        {
          id: "ranged-range-0",
          label: "ranged",
          rangeIndex: 0,
          primitiveCount: 1,
          scheduledMultiplyAdds: 1_000_000_000,
        },
        {
          id: "ranged-range-1",
          label: "ranged",
          rangeIndex: 1,
          primitiveCount: 1,
          scheduledMultiplyAdds: 1_000_000_000,
        },
      ],
    });
    expect(sequence.describeQuantum(1)).toEqual({
      id: "ranged-range-2..scalar",
      primitiveCount: 2,
      scheduledMultiplyAdds: 1_000_000_000,
      members: [
        {
          id: "ranged-range-2",
          label: "ranged",
          rangeIndex: 2,
          primitiveCount: 1,
          scheduledMultiplyAdds: 1_000_000_000,
        },
        {
          id: "scalar",
          label: "scalar",
          rangeIndex: null,
          primitiveCount: 1,
          scheduledMultiplyAdds: 0,
        },
      ],
    });
    expect(events).toEqual([]);
    expect(Object.isFrozen(sequence.describeQuantum(0))).toBe(true);
    expect(Object.isFrozen(sequence.describeQuantum(0).members)).toBe(true);
    sequence.encodeQuantum({} as GPUComputePassEncoder, 0);
    sequence.encodeQuantum({} as GPUComputePassEncoder, 1);
    expect(events).toEqual(["range:0", "range:1", "range:2", "scalar"]);
    expect(() => sequence.describeQuantum(2)).toThrow(/outside \[0, 2\)/);
  });

  it("fails closed for an unknown JavaScript profile", () => {
    expect(() =>
      aceCorrectnessGemmWgsl(
        "future-profile" as never,
        { rows: 1, inner: 1, columns: 1 },
        false,
      ),
    ).toThrow(/Unknown ACE GEMM model profile/);
  });
});

function fakeGemmDevice(options: {
  readonly pipelinePromise?: Promise<GPUComputePipeline>;
  readonly rangeParameters?: ReturnType<typeof fakeMappedBuffer>;
} = {}): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  const scopeResults: (GPUError | null)[] = [null, null, null];
  const rangeParameters = options.rangeParameters ?? fakeMappedBuffer(256);
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 32_768,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => scopeResults.shift() ?? null),
    createShaderModule: vi.fn(() => ({ label: "module" })),
    createComputePipelineAsync: vi.fn(
      () => options.pipelinePromise ?? Promise.resolve(fakePipeline()),
    ),
    createBuffer: vi.fn(() => rangeParameters),
    createBindGroup: vi.fn(() => ({ label: "bindings" })),
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
  };
}

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

function fakePipeline(): GPUComputePipeline {
  return {
    getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
  } as unknown as GPUComputePipeline;
}

function fakeBinding(
  size: number,
  buffer: GPUBuffer = fakeBuffer(size),
): GPUBufferBinding {
  return { buffer, offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeMappedBuffer(size: number) {
  const mapped = new ArrayBuffer(size);
  return {
    size,
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer & {
    readonly destroy: ReturnType<typeof vi.fn>;
  };
}
