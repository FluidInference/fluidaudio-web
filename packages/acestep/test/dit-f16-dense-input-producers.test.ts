import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
  AceOpt0081F16DenseInputProducerKernel,
  aceOpt0081AdaLnF16OutputWgsl,
  aceOpt0081CrossRmsNormF16OutputWgsl,
  aceOpt0081MergeHeadsF16OutputWgsl,
  aceOpt0081SwiGluF16OutputWgsl,
  type AceOpt0081AdaLnBindings,
  type AceOpt0081CrossRmsNormBindings,
  type AceOpt0081MergeHeadsBindings,
  type AceOpt0081SwiGluBindings,
} from "../src/webgpu/kernels/dit-f16-dense-input-producers.js";
import {
  aceCorrectnessRmsNormWgsl,
} from "../src/webgpu/kernels/rmsnorm.js";
import {
  aceCorrectnessElementwiseWgsl,
  aceCorrectnessHeadTransformWgsl,
} from "../src/webgpu/kernels/transformer-plumbing.js";

const HIDDEN_SHAPE = { batch: 1, tokens: 2_250, width: 2_048 } as const;
const MERGED_HEAD_SHAPE = {
  batch: 1,
  tokens: 2_250,
  heads: 16,
  headDimension: 128,
} as const;
const CROSS_RMSNORM_SHAPE = {
  rows: 2_250,
  width: 2_048,
  epsilon: 1e-6,
} as const;
const INTERMEDIATE_SHAPE = {
  batch: 1,
  tokens: 2_250,
  width: 6_144,
} as const;

const HIDDEN_ELEMENTS = 4_608_000;
const INTERMEDIATE_ELEMENTS = 13_824_000;

describe("OPT-0081 scalar-F16 dense-input producers", () => {
  it("pins the closed six-role inventory and exact M2250 storage delta", async () => {
    expect(ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE).toBe(
      "opt-0081-six-dense-input-f16-storage-v1",
    );
    expect(ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID).toBe(
      "opt-0081-six-f32-producer-f16-storage-v1",
    );
    expect(ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES).toEqual([
      "selfModulated",
      "selfMergedAttention",
      "crossNormalized",
      "crossMergedAttention",
      "mlpModulated",
      "gatedActivation",
    ]);
    expect(Object.isFrozen(ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES)).toBe(true);

    const fake = createFakeDevice();
    const kernel = AceOpt0081F16DenseInputProducerKernel.create(fake.device);
    try {
      const dispatches = [
        await kernel.createSelfModulatedDispatch(
          "self-modulated",
          "selfModulated",
          "adaln",
          HIDDEN_SHAPE,
          adaLnBindings(),
        ),
        await kernel.createSelfMergedAttentionDispatch(
          "self-merged",
          "selfMergedAttention",
          "merge-heads",
          MERGED_HEAD_SHAPE,
          mergeHeadsBindings(),
        ),
        await kernel.createCrossNormalizedDispatch(
          "cross-normalized",
          "crossNormalized",
          "cross-rmsnorm",
          CROSS_RMSNORM_SHAPE,
          crossRmsNormBindings(),
        ),
        await kernel.createCrossMergedAttentionDispatch(
          "cross-merged",
          "crossMergedAttention",
          "merge-heads",
          MERGED_HEAD_SHAPE,
          mergeHeadsBindings(),
        ),
        await kernel.createMlpModulatedDispatch(
          "mlp-modulated",
          "mlpModulated",
          "adaln",
          HIDDEN_SHAPE,
          adaLnBindings(),
        ),
        await kernel.createGatedActivationDispatch(
          "gated-activation",
          "gatedActivation",
          "swiglu",
          INTERMEDIATE_SHAPE,
          swiGluBindings(),
        ),
      ];

      expect(dispatches.map(({ role }) => role)).toEqual(
        ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
      );
      expect(dispatches.map(({ plan }) => plan.elements)).toEqual([
        HIDDEN_ELEMENTS,
        HIDDEN_ELEMENTS,
        HIDDEN_ELEMENTS,
        HIDDEN_ELEMENTS,
        HIDDEN_ELEMENTS,
        INTERMEDIATE_ELEMENTS,
      ]);
      const totalElements = dispatches.reduce(
        (sum, dispatch) => sum + dispatch.plan.elements,
        0,
      );
      expect(totalElements).toBe(36_864_000);
      expect(totalElements * Float32Array.BYTES_PER_ELEMENT).toBe(147_456_000);
      expect(totalElements * Uint16Array.BYTES_PER_ELEMENT).toBe(73_728_000);
      expect(dispatches.every(({ inputStorage }) => inputStorage === "f32"))
        .toBe(true);
      expect(dispatches.every(({ outputStorage }) => outputStorage === "f16"))
        .toBe(true);
      expect(dispatches.every(({ arithmetic }) => arithmetic === "f32"))
        .toBe(true);
      expect(dispatches.every(Object.isFrozen)).toBe(true);

      // The two AdaLN roles and two merge roles are deliberately one owner
      // each; RMSNorm and SwiGLU bring the exact cache total to four.
      expect(fake.pipelineCalls).toBe(4);
      expect(fake.shaderModules).toHaveLength(4);
      expect(fake.bindGroups.map(({ entries }) => Array.from(entries).length)).toEqual([
        4,
        2,
        3,
        2,
        4,
        3,
      ]);

      const encoded: unknown[] = [];
      dispatches[0]!.encode({
        setPipeline: (pipeline: GPUComputePipeline) =>
          encoded.push(["pipeline", pipeline]),
        setBindGroup: (index: number, bindGroup: GPUBindGroup) =>
          encoded.push(["bind-group", index, bindGroup]),
        dispatchWorkgroups: (x: number, y: number, z: number) =>
          encoded.push(["dispatch", x, y, z]),
      } as unknown as GPUComputePassEncoder);
      expect(encoded[2]).toEqual(["dispatch", 18_000, 1, 1]);
    } finally {
      kernel.destroy();
    }
  });

  it("keeps every producer FP32 until exactly one terminal F16 cast", () => {
    const adaln = aceOpt0081AdaLnF16OutputWgsl(HIDDEN_SHAPE);
    expect(adaln).toBe(expectedF16FinalStore(
      aceCorrectnessElementwiseWgsl(
        "reference-bf16",
        "adaln",
        HIDDEN_SHAPE,
      ),
      "output[index] = normalized[index] * (1.0 + scale[broadcast_index]) + shift[broadcast_index];",
    ));
    expectSingleFinalF16Store(adaln);
    expect(adaln).toContain("normalized: array<f32>");
    expect(adaln).toContain("scale: array<f32>");
    expect(adaln).toContain("shift: array<f32>");
    expect(adaln).toContain(
      "normalized[index] * (1.0 + scale[broadcast_index]) + shift[broadcast_index]",
    );
    expect(adaln).not.toContain("f16(1.0)");

    const merge = aceOpt0081MergeHeadsF16OutputWgsl(MERGED_HEAD_SHAPE);
    expect(merge).toBe(expectedF16FinalStore(
      aceCorrectnessHeadTransformWgsl(
        "reference-bf16",
        "merge-heads",
        MERGED_HEAD_SHAPE,
      ),
      "output[index] = input[source_index];",
    ));
    expectSingleFinalF16Store(merge);
    expect(merge).toContain("input: array<f32>");
    expect(merge).toContain(
      "((batch * HEADS + head) * TOKENS + token) * HEAD_DIMENSION + dimension",
    );
    expect(merge).toContain("output[index] = f16(input[source_index]);");

    const rmsnorm = aceOpt0081CrossRmsNormF16OutputWgsl(
      CROSS_RMSNORM_SHAPE,
    );
    expect(rmsnorm).toBe(expectedF16FinalStore(
      aceCorrectnessRmsNormWgsl(
        "reference-bf16",
        CROSS_RMSNORM_SHAPE,
      ),
      "output[index] = value * inverse_rms * load_weight(column);",
    ));
    expectSingleFinalF16Store(rmsnorm);
    expect(rmsnorm).toContain("input: array<f32>");
    expect(rmsnorm).toContain("weight: array<u32>");
    expect(rmsnorm).toContain("partial_squares: array<f32, 256>");
    expect(rmsnorm).toContain(
      "let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u)",
    );
    expect(rmsnorm).toContain("return bitcast<f32>(bits16 << 16u);");
    expect(rmsnorm).toContain(
      "output[index] = f16(value * inverse_rms * load_weight(column));",
    );
    expect(rmsnorm).not.toContain("f16(value * inverse_rms) *");

    const swiglu = aceOpt0081SwiGluF16OutputWgsl(INTERMEDIATE_SHAPE);
    expect(swiglu).toBe(expectedF16FinalStore(
      aceCorrectnessElementwiseWgsl(
        "reference-bf16",
        "swiglu",
        INTERMEDIATE_SHAPE,
      ),
      "output[index] = (value / (1.0 + exp(-value))) * up[index];",
    ));
    expectSingleFinalF16Store(swiglu);
    expect(swiglu).toContain("gate: array<f32>");
    expect(swiglu).toContain("up: array<f32>");
    expect(swiglu).toContain(
      "(value / (1.0 + exp(-value))) * up[index]",
    );
    expect(swiglu).not.toContain("let activated = f16");
  });

  it("fails closed for any role/operation substitution at a named boundary", async () => {
    const kernel = AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice().device,
    );
    try {
      const substitutions = [
        () => kernel.createSelfModulatedDispatch(
          "bad-self-modulated",
          "mlpModulated" as never,
          "adaln",
          HIDDEN_SHAPE,
          adaLnBindings(),
        ),
        () => kernel.createSelfMergedAttentionDispatch(
          "bad-self-merged",
          "crossMergedAttention" as never,
          "merge-heads",
          MERGED_HEAD_SHAPE,
          mergeHeadsBindings(),
        ),
        () => kernel.createCrossNormalizedDispatch(
          "bad-cross-normalized",
          "gatedActivation" as never,
          "swiglu" as never,
          CROSS_RMSNORM_SHAPE,
          crossRmsNormBindings(),
        ),
        () => kernel.createCrossMergedAttentionDispatch(
          "bad-cross-merged",
          "selfMergedAttention" as never,
          "merge-heads",
          MERGED_HEAD_SHAPE,
          mergeHeadsBindings(),
        ),
        () => kernel.createMlpModulatedDispatch(
          "bad-mlp-modulated",
          "selfModulated" as never,
          "adaln",
          HIDDEN_SHAPE,
          adaLnBindings(),
        ),
        () => kernel.createGatedActivationDispatch(
          "bad-gated",
          "crossNormalized" as never,
          "cross-rmsnorm" as never,
          INTERMEDIATE_SHAPE,
          swiGluBindings(),
        ),
      ];
      for (const substitution of substitutions) {
        await expect(substitution()).rejects.toThrow(/rejects producer role\/operation/);
      }
    } finally {
      kernel.destroy();
    }
  });

  it("rejects geometry outside the six ACE producer boundaries", async () => {
    expect(() => aceOpt0081AdaLnF16OutputWgsl({
      batch: 2,
      tokens: 2_250,
      width: 2_048,
    })).toThrow(/batch 1, tokens 2250, and width 2048/);
    expect(() => aceOpt0081AdaLnF16OutputWgsl({
      batch: 1,
      tokens: 1,
      width: 2_048,
    })).toThrow(/tokens 2250/);
    expect(() => aceOpt0081MergeHeadsF16OutputWgsl({
      batch: 1,
      tokens: 2_250,
      heads: 8,
      headDimension: 128,
    })).toThrow(/16 heads of dimension 128/);
    expect(() => aceOpt0081CrossRmsNormF16OutputWgsl({
      rows: 2_250,
      width: 2_048,
      epsilon: 1e-5,
    })).toThrow(/epsilon 1e-6/);
    expect(() => aceOpt0081SwiGluF16OutputWgsl({
      batch: 1,
      tokens: 2_250,
      width: 2_048,
    })).toThrow(/width 6144/);

    const kernel = AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice().device,
    );
    try {
      await expect(kernel.createCrossNormalizedDispatch(
        "wrong-width",
        "crossNormalized",
        "cross-rmsnorm",
        { rows: 2_250, width: 1_024, epsilon: 1e-6 },
        crossRmsNormBindings(),
      )).rejects.toThrow(/width 2048/);
    } finally {
      kernel.destroy();
    }
  });

  it("requires shader-f16, WG256, RMS shared memory, and binding limits", () => {
    expect(() => AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice({ shaderF16: false }).device,
    )).toThrow(/shader-f16/);
    for (const limits of [
      { maxComputeInvocationsPerWorkgroup: 255 },
      { maxComputeWorkgroupSizeX: 255 },
      { maxComputeWorkgroupStorageSize: 1_023 },
    ]) {
      expect(() => AceOpt0081F16DenseInputProducerKernel.create(
        createFakeDevice({ limits }).device,
      )).toThrow(/WG256 and 1024 bytes/);
    }
    for (const limits of [
      { minStorageBufferOffsetAlignment: 0 },
      { maxStorageBufferBindingSize: 0 },
      { maxBufferSize: 0 },
    ]) {
      expect(() => AceOpt0081F16DenseInputProducerKernel.create(
        createFakeDevice({ limits }).device,
      )).toThrow(/storage-buffer binding limits/);
    }
  });

  it("requires exact aligned binding ranges within buffer and device limits", async () => {
    const requiredInput = HIDDEN_ELEMENTS * 4;
    const requiredBroadcast = 2_048 * 4;
    const requiredOutput = HIDDEN_ELEMENTS * 2;
    const kernel = AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice().device,
    );
    try {
      for (const output of [
        fakeBinding(requiredOutput - 2),
        fakeBinding(requiredOutput + 2),
        {
          buffer: fakeBuffer(requiredOutput + 256),
          offset: 4,
          size: requiredOutput,
        },
        {
          buffer: fakeBuffer(requiredOutput - 1),
          offset: 0,
          size: requiredOutput,
        },
      ]) {
        await expect(kernel.createSelfModulatedDispatch(
          "bad-output-binding",
          "selfModulated",
          "adaln",
          HIDDEN_SHAPE,
          {
            normalized: fakeBinding(requiredInput),
            scale: fakeBinding(requiredBroadcast),
            shift: fakeBinding(requiredBroadcast),
            output,
          },
        )).rejects.toThrow(/must expose exactly 9216000 aligned bytes/);
      }

      const shared = fakeBuffer(requiredInput + requiredOutput);
      await expect(kernel.createSelfModulatedDispatch(
        "overlap",
        "selfModulated",
        "adaln",
        HIDDEN_SHAPE,
        {
          normalized: { buffer: shared, offset: 0, size: requiredInput },
          scale: fakeBinding(requiredBroadcast),
          shift: fakeBinding(requiredBroadcast),
          output: {
            buffer: shared,
            offset: requiredOutput,
            size: requiredOutput,
          },
        },
      )).rejects.toThrow(/must not overlap/);

      const implicitExact = await kernel.createSelfModulatedDispatch(
        "implicit-exact",
        "selfModulated",
        "adaln",
        HIDDEN_SHAPE,
        {
          normalized: { buffer: fakeBuffer(requiredInput) },
          scale: { buffer: fakeBuffer(requiredBroadcast) },
          shift: { buffer: fakeBuffer(requiredBroadcast) },
          output: { buffer: fakeBuffer(requiredOutput) },
        },
      );
      expect(implicitExact.plan.elements).toBe(HIDDEN_ELEMENTS);

      await expect(kernel.createCrossNormalizedDispatch(
        "short-packed-weight",
        "crossNormalized",
        "cross-rmsnorm",
        CROSS_RMSNORM_SHAPE,
        {
          ...crossRmsNormBindings(),
          weight: fakeBinding(4_092),
        },
      )).rejects.toThrow(/weight binding must expose exactly 4096 aligned bytes/);
    } finally {
      kernel.destroy();
    }

    const limited = AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice({
        limits: { maxStorageBufferBindingSize: requiredInput - 1 },
      }).device,
    );
    try {
      await expect(limited.createSelfModulatedDispatch(
        "over-limit",
        "selfModulated",
        "adaln",
        HIDDEN_SHAPE,
        adaLnBindings(),
      )).rejects.toThrow(/must expose exactly 18432000 aligned bytes/);
    } finally {
      limited.destroy();
    }
  });

  it("evicts a rejected pipeline and reuses a later successful compilation", async () => {
    const fake = createFakeDevice({
      compile: (call, pipeline) => call === 1
        ? Promise.reject(new Error("synthetic compilation failure"))
        : Promise.resolve(pipeline),
    });
    const kernel = AceOpt0081F16DenseInputProducerKernel.create(fake.device);
    try {
      await expect(kernel.createSelfModulatedDispatch(
        "first",
        "selfModulated",
        "adaln",
        HIDDEN_SHAPE,
        adaLnBindings(),
      )).rejects.toThrow(/synthetic compilation failure/);
      await expect(kernel.createSelfModulatedDispatch(
        "second",
        "selfModulated",
        "adaln",
        HIDDEN_SHAPE,
        adaLnBindings(),
      )).resolves.toMatchObject({ role: "selfModulated" });
      expect(fake.pipelineCalls).toBe(2);
    } finally {
      kernel.destroy();
    }
  });

  it("has idempotent destruction and rejects before and during compilation", async () => {
    const readyKernel = AceOpt0081F16DenseInputProducerKernel.create(
      createFakeDevice().device,
    );
    const ready = await readyKernel.createSelfModulatedDispatch(
      "ready",
      "selfModulated",
      "adaln",
      HIDDEN_SHAPE,
      adaLnBindings(),
    );
    readyKernel.destroy();
    readyKernel.destroy();
    expect(() => ready.encode({} as GPUComputePassEncoder)).toThrow(
      /destroyed before encoding/,
    );

    let finishCompilation: (() => void) | undefined;
    const fake = createFakeDevice({
      compile: (_call, pipeline) => new Promise((resolve) => {
        finishCompilation = () => resolve(pipeline);
      }),
    });
    const kernel = AceOpt0081F16DenseInputProducerKernel.create(fake.device);
    const pending = kernel.createSelfModulatedDispatch(
      "pending",
      "selfModulated",
      "adaln",
      HIDDEN_SHAPE,
      adaLnBindings(),
    );
    expect(finishCompilation).toBeTypeOf("function");
    kernel.destroy();
    kernel.destroy();
    finishCompilation!();
    await expect(pending).rejects.toThrow(/destroyed while compiling/);
    expect(fake.bindGroups).toHaveLength(0);
    await expect(kernel.createSelfModulatedDispatch(
      "dead",
      "selfModulated",
      "adaln",
      HIDDEN_SHAPE,
      adaLnBindings(),
    )).rejects.toThrow(/destroyed/);
  });
});

function expectedF16FinalStore(
  referenceSource: string,
  referenceAssignment: string,
): string {
  const outputDeclaration =
    "var<storage, read_write> output: array<f32>;";
  const assignmentPrefix = "output[index] = ";
  const rightHandSide = referenceAssignment.slice(
    assignmentPrefix.length,
    -1,
  );
  return "enable f16;\n" + referenceSource
    .replace(
      outputDeclaration,
      "var<storage, read_write> output: array<f16>;",
    )
    .replace(
      referenceAssignment,
      `${assignmentPrefix}f16(${rightHandSide});`,
    );
}

function expectSingleFinalF16Store(source: string): void {
  expect(source).toContain("enable f16;");
  expect(source).toContain("output: array<f16>");
  expect(source.match(/\bf16\s*\(/g) ?? []).toHaveLength(1);
  expect(source.match(/output\[index\]\s*=/g) ?? []).toHaveLength(1);
}

function adaLnBindings(tokens = 2_250): AceOpt0081AdaLnBindings {
  const elements = tokens * 2_048;
  return {
    normalized: fakeBinding(elements * 4),
    scale: fakeBinding(2_048 * 4),
    shift: fakeBinding(2_048 * 4),
    output: fakeBinding(elements * 2),
  };
}

function mergeHeadsBindings(tokens = 2_250): AceOpt0081MergeHeadsBindings {
  const elements = tokens * 2_048;
  return {
    input: fakeBinding(elements * 4),
    output: fakeBinding(elements * 2),
  };
}

function crossRmsNormBindings(rows = 2_250): AceOpt0081CrossRmsNormBindings {
  const elements = rows * 2_048;
  return {
    input: fakeBinding(elements * 4),
    weight: fakeBinding(Math.ceil(2_048 / 2) * 4),
    output: fakeBinding(elements * 2),
  };
}

function swiGluBindings(tokens = 2_250): AceOpt0081SwiGluBindings {
  const elements = tokens * 6_144;
  return {
    gate: fakeBinding(elements * 4),
    up: fakeBinding(elements * 4),
    output: fakeBinding(elements * 2),
  };
}

interface FakeDeviceOptions {
  readonly shaderF16?: boolean;
  readonly limits?: Readonly<Record<string, number>>;
  readonly compile?: (
    call: number,
    pipeline: GPUComputePipeline,
  ) => Promise<GPUComputePipeline>;
}

interface FakeDevice {
  readonly device: GPUDevice;
  readonly shaderModules: GPUShaderModuleDescriptor[];
  readonly bindGroups: GPUBindGroupDescriptor[];
  pipelineCalls: number;
}

function createFakeDevice(options: FakeDeviceOptions = {}): FakeDevice {
  const state = {
    shaderModules: [] as GPUShaderModuleDescriptor[],
    bindGroups: [] as GPUBindGroupDescriptor[],
    pipelineCalls: 0,
  };
  const pipeline = {
    getBindGroupLayout: () => ({}) as GPUBindGroupLayout,
  } as unknown as GPUComputePipeline;
  const device = {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 1_024,
      minStorageBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      ...options.limits,
    },
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => {
      state.shaderModules.push(descriptor);
      return {} as GPUShaderModule;
    },
    createComputePipelineAsync: () => {
      state.pipelineCalls += 1;
      return options.compile?.(state.pipelineCalls, pipeline) ??
        Promise.resolve(pipeline);
    },
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => {
      state.bindGroups.push(descriptor);
      return {} as GPUBindGroup;
    },
  } as unknown as GPUDevice;
  return Object.assign(state, { device });
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}
