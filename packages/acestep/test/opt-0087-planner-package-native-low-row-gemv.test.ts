import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it, vi, type Mock } from "vitest";

import {
  ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
} from "../src/webgpu/kernels/planner-low-row-bf16-gemv.js";
import {
  planAceTiledGemm,
  type AceGemmBufferBindings,
  type AceGemmDispatch,
  type AceGemmKernel,
  type AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  AceCorrectnessPlannerModelRuntime,
  type AcePlannerModelBindings,
  type AcePlannerModelDispatch,
} from "../src/webgpu/planner-model.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  type AceQwen3BlockScratch,
  type AceQwen3BlockWeights,
} from "../src/webgpu/qwen3.js";
import {
  AceOpt0087PlannerDenseOwner,
  selectAceOpt0087PlannerDenseArm,
  type AceOpt0087PlannerDenseInvocation,
} from "../src/webgpu/planner-dense-owner.js";

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });
vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

describe("OPT-0087 planner package-native low-row GEMV", () => {
  it("selects B only for reference-BF16 planner M1/M2 single-token decode", () => {
    const shape = { rows: 1, inner: 1_024, columns: 2_048 } as const;
    const direct = (
      overrides: Partial<AceOpt0087PlannerDenseInvocation> = {},
      candidateAvailable = true,
      candidateShape: AceGemmShape = shape,
      hasBias = false,
      profile: "reference-bf16" | "raw-fp16" = "reference-bf16",
    ) => selectAceOpt0087PlannerDenseArm(profile, candidateAvailable, {
      owner: "planner",
      kind: "decode",
      batch: 1,
      tokens: 1,
      requestedArm: "direct-b",
      ...overrides,
    }, candidateShape, hasBias);

    expect(direct()).toEqual({ arm: "direct-b", reason: "direct-selected" });
    expect(direct({ batch: 2 }, true, { ...shape, rows: 2 })).toEqual({
      arm: "direct-b",
      reason: "direct-selected",
    });
    expect(direct({ requestedArm: "generic-a" })).toEqual({
      arm: "generic-a",
      reason: "control-requested",
    });
    expect(direct({ owner: "non-planner" })).toMatchObject({
      arm: "generic-a",
      reason: "non-planner-owner",
    });
    expect(direct({}, true, shape, false, "raw-fp16")).toMatchObject({
      arm: "generic-a",
      reason: "non-reference-profile",
    });
    expect(direct({}, false)).toMatchObject({
      arm: "generic-a",
      reason: "candidate-owner-unavailable",
    });
    expect(direct({ kind: "prefill" })).toMatchObject({
      arm: "generic-a",
      reason: "not-single-token-decode",
    });
    // The explicitly frozen one-batch-row, multi-token prefill trap.
    expect(direct(
      { kind: "prefill", batch: 1, tokens: 2 },
      true,
      { ...shape, rows: 2 },
    )).toMatchObject({
      arm: "generic-a",
      reason: "not-single-token-decode",
    });
    expect(direct({ batch: 3 }, true, { ...shape, rows: 3 })).toMatchObject({
      arm: "generic-a",
      reason: "unsupported-physical-rows",
    });
    expect(direct({ batch: 2 })).toMatchObject({
      arm: "generic-a",
      reason: "physical-row-mismatch",
    });
    expect(direct({}, true, shape, true)).toMatchObject({
      arm: "generic-a",
      reason: "bias-present",
    });
    expect(direct({}, true, { ...shape, inner: 1_023 })).toMatchObject({
      arm: "generic-a",
      reason: "direct-shape-rejected",
    });
  });

  it("owns both pipeline arms exactly once and cleans partial construction", async () => {
    const generic = fakeKernel("generic-a");
    const direct = fakeKernel("direct-b");
    const owner = AceOpt0087PlannerDenseOwner.createFromFactoriesForOpt0087(
      "reference-bf16",
      { generic: () => generic, direct: () => direct },
    );
    const bindings = fakeDenseBindings();
    const result = await owner.createDispatchForOpt0087({
      label: "layer-0-query",
      role: "query-projection",
      invocation: {
        owner: "planner",
        kind: "decode",
        batch: 1,
        tokens: 1,
        requestedArm: "direct-b",
      },
      shape: { rows: 1, inner: 1_024, columns: 2_048 },
      bindings,
    });
    expect(result.selection).toMatchObject({ selectedArm: "direct-b" });
    expect(direct.calls).toEqual([expect.objectContaining({ bindings })]);
    expect(generic.calls).toEqual([]);
    owner.destroy();
    owner.destroy();
    expect(generic.destroy).toHaveBeenCalledOnce();
    expect(direct.destroy).toHaveBeenCalledOnce();
    expect((bindings.weight.buffer as FakeGraphBuffer).destroy)
      .not.toHaveBeenCalled();

    const partialGeneric = fakeKernel("partial-generic");
    const failure = new Error("candidate construction failed");
    expect(() => AceOpt0087PlannerDenseOwner.createFromFactoriesForOpt0087(
      "reference-bf16",
      {
        generic: () => partialGeneric,
        direct: () => {
          throw failure;
        },
      },
    )).toThrow(failure);
    expect(partialGeneric.destroy).toHaveBeenCalledOnce();

    const rawGeneric = fakeKernel("raw-generic");
    const rawDirectFactory = vi.fn(() => fakeKernel("must-not-exist"));
    const raw = AceOpt0087PlannerDenseOwner.createFromFactoriesForOpt0087(
      "raw-fp16",
      { generic: () => rawGeneric, direct: rawDirectFactory },
    );
    expect(rawDirectFactory).not.toHaveBeenCalled();
    raw.destroy();
    expect(rawGeneric.destroy).toHaveBeenCalledOnce();
  });

  for (const batch of [1, 2] as const) {
    it(`routes all real M${batch} layer/head bindings through B without changing 33-quanta topology`, async () => {
      const device = fakePlannerDevice();
      const fixture = plannerBindings();
      const runtime = AceCorrectnessPlannerModelRuntime.createForOpt0087(
        device,
        "reference-bf16",
      );
      const shape = {
        kind: "decode" as const,
        batch,
        tokens: 1,
        cacheCapacity: 64,
      };
      const armA = await runtime.createQwen3PlannerDispatchForOpt0087(
        `opt0087-m${batch}-a`,
        "generic-a",
        ACE_PLANNER_QWEN3_CONFIG,
        shape,
        fixture.bindings,
      );
      const armB = await runtime.createQwen3PlannerDispatchForOpt0087(
        `opt0087-m${batch}-b`,
        "direct-b",
        ACE_PLANNER_QWEN3_CONFIG,
        shape,
        fixture.bindings,
      );

      expect(modelTopology(armA)).toEqual(modelTopology(armB));
      expect(armA.quanta).toHaveLength(33);
      expect(armA.quanta.filter(({ kind }) => kind === "layer")).toHaveLength(28);
      expect(armA.quanta.filter(({ kind }) => kind === "tied-lm-head")
        .map(({ primitiveCount }) => primitiveCount)).toEqual([2, 3]);
      expect(armB.quanta.filter(({ kind }) => kind === "tied-lm-head")
        .map(({ primitiveCount }) => primitiveCount)).toEqual([2, 3]);
      expect(armA.opt0087HeadQuantumSliceFirstRows).toEqual([
        [0, 49_152],
        [98_304, 147_456, 196_608],
      ]);
      expect(armB.opt0087HeadQuantumSliceFirstRows)
        .toEqual(armA.opt0087HeadQuantumSliceFirstRows);
      expect(armA.primitiveCount).toBe(armB.primitiveCount);

      const aSelections = armA.opt0087DenseSelections!;
      const bSelections = armB.opt0087DenseSelections!;
      expect(aSelections).toHaveLength(28 * 7 + 5);
      expect(bSelections).toHaveLength(28 * 7 + 5);
      expect(new Set(aSelections.map(({ selectedArm }) => selectedArm)))
        .toEqual(new Set(["generic-a"]));
      expect(new Set(bSelections.map(({ selectedArm }) => selectedArm)))
        .toEqual(new Set(["direct-b"]));

      const expectedLayerRoles = [
        "query-projection",
        "key-projection",
        "value-projection",
        "attention-output-projection",
        "gate-projection",
        "up-projection",
        "down-projection",
      ];
      for (let layer = 0; layer < 28; layer += 1) {
        const selections = bSelections.filter(({ label }) =>
          label.includes(`-layer-${layer}-`)
        );
        expect(selections.map(({ role }) => role)).toEqual(expectedLayerRoles);
        const weights = fixture.bindings.weights.layers[layer]!;
        expect(selections.map(({ weightBinding }) => weightBinding)).toEqual([
          weights.queryProjection,
          weights.keyProjection,
          weights.valueProjection,
          weights.outputProjection,
          weights.gateProjection,
          weights.upProjection,
          weights.downProjection,
        ]);
      }

      const aHeads = aSelections.filter(({ role }) => role === "tied-lm-head");
      const bHeads = bSelections.filter(({ role }) => role === "tied-lm-head");
      expect(aHeads.map(({ shape: dense }) => dense.columns)).toEqual(
        ACE_PLANNER_EMBEDDING_ROW_PARTS.map(({ rowCount }) => rowCount),
      );
      expect(bHeads.at(-1)!.shape.columns).toBe(20_596);
      expect(bHeads.at(-1)!.shape.columns % 128).toBe(116);
      for (let index = 0; index < bHeads.length; index += 1) {
        const owner = fixture.bindings.weights.embedding[index]!.weight;
        const aWeight = aHeads[index]!.weightBinding;
        const bWeight = bHeads[index]!.weightBinding;
        expect(aWeight.buffer).toBe(owner.buffer);
        expect(bWeight.buffer).toBe(owner.buffer);
        expect({ offset: aWeight.offset, size: aWeight.size }).toEqual({
          offset: bWeight.offset,
          size: bWeight.size,
        });
      }
      for (let index = 0; index < aSelections.length; index += 1) {
        expect(aSelections[index]!.role).toBe(bSelections[index]!.role);
        expect(aSelections[index]!.shape).toEqual(bSelections[index]!.shape);
        expect(aSelections[index]!.weightBinding.buffer)
          .toBe(bSelections[index]!.weightBinding.buffer);
      }

      runtime.destroy();
      runtime.destroy();
      await Promise.resolve();
      for (const weight of fixture.weightBuffers) {
        expect(weight.destroy).not.toHaveBeenCalled();
      }
      for (const owned of device.ownedBuffers) {
        expect(owned.destroy).toHaveBeenCalledOnce();
        // The only runtime-owned buffers are tiny generic range uniforms;
        // neither arm stages or duplicates an authenticated model weight.
        expect(owned.size).toBe(256);
      }
    });
  }

  it("fails a one-batch-row multi-token prefill closed to generic in the real composer", async () => {
    const device = fakePlannerDevice();
    const fixture = plannerBindings();
    const runtime = AceCorrectnessPlannerModelRuntime.createForOpt0087(
      device,
      "reference-bf16",
    );
    const dispatch = await runtime.createQwen3PlannerDispatchForOpt0087(
      "opt0087-prefill-fail-closed",
      "direct-b",
      ACE_PLANNER_QWEN3_CONFIG,
      { kind: "prefill", batch: 1, tokens: 2, cacheCapacity: 64 },
      fixture.bindings,
    );
    expect(dispatch.opt0087DenseSelections).toHaveLength(28 * 7 + 5);
    expect(new Set(dispatch.opt0087DenseSelections!.map(({ selectedArm }) =>
      selectedArm
    ))).toEqual(new Set(["generic-a"]));
    expect(new Set(dispatch.opt0087DenseSelections!.map(({ reason }) => reason)))
      .toEqual(new Set(["not-single-token-decode"]));

    const foreignConfig = {
      ...ACE_PLANNER_QWEN3_CONFIG,
      id: "non-planner-qwen-owner",
    };
    const nonPlanner = await runtime.createQwen3PlannerDispatchForOpt0087(
      "opt0087-non-planner-fail-closed",
      "direct-b",
      foreignConfig,
      { kind: "decode", batch: 1, tokens: 1, cacheCapacity: 64 },
      fixture.bindings,
    );
    expect(new Set(nonPlanner.opt0087DenseSelections!.map(({ selectedArm }) =>
      selectedArm
    ))).toEqual(new Set(["generic-a"]));
    expect(new Set(nonPlanner.opt0087DenseSelections!.map(({ reason }) => reason)))
      .toEqual(new Set(["non-planner-owner"]));
    runtime.destroy();
  });

  it("keeps ordinary production generic and marks every experiment surface internal", async () => {
    const device = fakePlannerDevice();
    const fixture = plannerBindings();
    const runtime = AceCorrectnessPlannerModelRuntime.create(
      device,
      "reference-bf16",
    );
    const dispatch = await runtime.createQwen3PlannerDispatch(
      "ordinary-production",
      ACE_PLANNER_QWEN3_CONFIG,
      { kind: "decode", batch: 1, tokens: 1, cacheCapacity: 64 },
      fixture.bindings,
    );
    expect(dispatch.opt0087DenseSelections).toBeUndefined();
    expect(device.shaderSources.some((source) =>
      source.includes(ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID)
    )).toBe(false);
    runtime.destroy();

    const build = JSON.parse(readFileSync(
      resolve(process.cwd(), "tsconfig.build.json"),
      "utf8",
    )) as { compilerOptions?: { stripInternal?: boolean } };
    expect(build.compilerOptions?.stripInternal).toBe(true);
    for (const file of [
      "src/webgpu/planner-dense-owner.ts",
      "src/webgpu/qwen3.ts",
      "src/webgpu/planner-model.ts",
      "src/webgpu/planner-executor.ts",
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), "utf8");
      const lines = source.split("\n");
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index]!;
        const namedDeclaration = /^export\s+(?:type|interface|class|function)\s+\w*Opt0087/.test(line);
        const namedClassMember = /^\s+(?:static\s+|async\s+)?\w*Opt0087\s*\(/.test(line);
        if (!namedDeclaration && !namedClassMember) continue;
        expect(lines.slice(Math.max(0, index - 6), index).join("\n"))
          .toContain("@internal");
      }
    }
    const plannerModel = readFileSync(
      resolve(process.cwd(), "src/webgpu/planner-model.ts"),
      "utf8",
    );
    expect(plannerModel).toMatch(
      /@internal Exact clone-free selector evidence[^]*opt0087DenseSelections\?/
    );
    expect(plannerModel).toMatch(
      /@internal Exact physical head slices[^]*opt0087HeadQuantumSliceFirstRows\?/
    );
  });
});

function modelTopology(dispatch: AcePlannerModelDispatch) {
  return dispatch.quanta.map(({ id, logicalId, kind, layer, primitiveCount }) => ({
    // A/B labels intentionally differ, so compare the stable suffix topology.
    id: id.replace(/^opt0087-m[12]-[ab]-/, ""),
    logicalId: logicalId?.replace(/^opt0087-m[12]-[ab]-/, ""),
    kind,
    layer,
    primitiveCount,
  }));
}

interface FakeKernel extends AceGemmKernel {
  readonly calls: Array<{
    readonly label: string;
    readonly shape: AceGemmShape;
    readonly bindings: AceGemmBufferBindings;
  }>;
  readonly destroy: Mock<() => void>;
}

function fakeKernel(id: string): FakeKernel {
  const calls: FakeKernel["calls"] = [];
  const destroy = vi.fn<() => void>();
  return {
    calls,
    destroy,
    async createDispatch(label, shape, bindings): Promise<AceGemmDispatch> {
      calls.push({ label, shape, bindings });
      const plan = planAceTiledGemm(shape);
      return {
        label: `${id}-${label}`,
        weightLayout: "source-row-major",
        plan,
        rangeCount: plan.outputRangeCount,
        encodeRange: vi.fn(),
        encode: vi.fn(),
      };
    },
  };
}

type FakeGraphBuffer = GPUBuffer & Readonly<{
  readonly destroy: Mock<() => undefined>;
}>;

function fakeDenseBindings(): AceGemmBufferBindings {
  return {
    activation: graphBinding(),
    weight: graphBinding(),
    output: graphBinding(),
  };
}

function graphBinding(size = 1_073_741_824): GPUBufferBinding {
  return { buffer: graphBuffer(size), offset: 0, size };
}

function graphBuffer(size = 1_073_741_824): FakeGraphBuffer {
  return {
    size,
    destroy: vi.fn<() => undefined>(() => undefined),
  } as unknown as FakeGraphBuffer;
}

function plannerBindings(): Readonly<{
  readonly bindings: AcePlannerModelBindings;
  readonly weightBuffers: readonly FakeGraphBuffer[];
}> {
  const weightBuffers: FakeGraphBuffer[] = [];
  const weight = (): GPUBufferBinding => {
    const binding = graphBinding();
    weightBuffers.push(binding.buffer as FakeGraphBuffer);
    return binding;
  };
  const activation = (): GPUBufferBinding => graphBinding();
  const layer = (): AceQwen3BlockWeights => ({
    inputLayerNorm: weight(),
    queryProjection: weight(),
    keyProjection: weight(),
    valueProjection: weight(),
    queryNorm: weight(),
    keyNorm: weight(),
    outputProjection: weight(),
    postAttentionLayerNorm: weight(),
    gateProjection: weight(),
    upProjection: weight(),
    downProjection: weight(),
  });
  const block: AceQwen3BlockScratch = {
    normalizedInput: activation(),
    queryFlat: activation(),
    keyFlat: activation(),
    valueFlat: activation(),
    queryHeads: activation(),
    keyHeads: activation(),
    valueHeads: activation(),
    normalizedQueryHeads: activation(),
    normalizedKeyHeads: activation(),
    rotatedQueryHeads: activation(),
    rotatedKeyHeads: activation(),
    attentionHeads: activation(),
    mergedAttention: activation(),
    projectedAttention: activation(),
    afterAttention: activation(),
    normalizedAfterAttention: activation(),
    gate: activation(),
    up: activation(),
    gatedActivation: activation(),
    projectedMlp: activation(),
  };
  const embedding = ACE_PLANNER_EMBEDDING_ROW_PARTS.map((part) => ({
    ...part,
    weight: weight(),
  }));
  const bindings: AcePlannerModelBindings = {
    tokenIds: activation(),
    weights: {
      embedding,
      layers: Array.from({ length: 28 }, layer),
      finalNorm: weight(),
    },
    controls: {
      validLengths: activation(),
      queryPositions: activation(),
      sourceValidity: activation(),
      rowStartPositions: activation(),
      cosine: activation(),
      sine: activation(),
      lastPhysicalRowIndices: activation(),
      cacheValidity: activation(),
      writeStatus: activation(),
    },
    cache: {
      layers: Array.from({ length: 28 }, () => ({
        key: activation(),
        value: activation(),
      })),
    },
    scratch: {
      embedded: activation(),
      block,
      layerOutputs: [activation(), activation()],
      normalizedSequence: activation(),
      lastHiddenRows: activation(),
    },
    logits: ACE_PLANNER_EMBEDDING_ROW_PARTS.map(() => activation()),
  };
  return Object.freeze({ bindings, weightBuffers: Object.freeze(weightBuffers) });
}

interface FakePlannerDevice extends GPUDevice {
  readonly shaderSources: string[];
  readonly ownedBuffers: Array<FakeGraphBuffer & {
    readonly getMappedRange: ReturnType<typeof vi.fn>;
    readonly unmap: ReturnType<typeof vi.fn>;
  }>;
}

function fakePlannerDevice(): FakePlannerDevice {
  const shaderSources: string[] = [];
  const ownedBuffers: FakePlannerDevice["ownedBuffers"] = [];
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 32_768,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createShaderModule: vi.fn((descriptor: GPUShaderModuleDescriptor) => {
      shaderSources.push(descriptor.code);
      return {
        label: descriptor.label,
        getCompilationInfo: vi.fn(async () => ({ messages: [] })),
      };
    }),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({ label: "auto-layout" })),
    })),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const mapped = new ArrayBuffer(Number(descriptor.size));
      const buffer = {
        size: Number(descriptor.size),
        getMappedRange: vi.fn(() => mapped),
        unmap: vi.fn(),
        destroy: vi.fn<() => undefined>(() => undefined),
      } as unknown as FakePlannerDevice["ownedBuffers"][number];
      ownedBuffers.push(buffer);
      return buffer;
    }),
    shaderSources,
    ownedBuffers,
  } as unknown as FakePlannerDevice;
}
