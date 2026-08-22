import { describe, expect, it, vi } from "vitest";

import pipelineSource from "../src/runtime/webgpu-pipeline.ts?raw";
import aceDitSource from "../src/webgpu/ace-dit.ts?raw";
import backendSource from "../src/webgpu/dit-backend.ts?raw";
import graphSource from "../src/webgpu/dit-graph.ts?raw";

import { ACE_DIRECT_DCW_CONFIGURATION } from "../src/api.js";
import {
  ACE_TURBO_DIT_CONFIG,
  AceCorrectnessDitRuntime,
  aceDitLayerScratchBytes,
  planAceDitEvaluation,
  requireAceDitDenseInputStorageProfile,
  type AceDitLayerBindings,
  type AceDitLayerScratch,
} from "../src/webgpu/ace-dit.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import {
  AceDitGpuBackend,
  planAceDitBackendArena,
  planAceDitGpuBackendMemory,
} from "../src/webgpu/dit-backend.js";
import {
  AceDitGraphOwner,
  planAceDitGraphMemory,
} from "../src/webgpu/dit-graph.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
} from "../src/webgpu/dit-sampler-profile.js";
import {
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
} from "../src/webgpu/kernels/dit-f16-dense-input-producers.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });
vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

const PROFILE = ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE;
const SHAPE = Object.freeze({
  batch: 1,
  latentFrames: 4_500,
  conditionTokens: 98,
});
const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });
const GEMM_CONFIGURATION = Object.freeze({
  backend: "subgroups" as const,
  capability: FIXED32,
});
const DENSE_CONFIGURATION = Object.freeze({
  backend: "opt-0009-fp16-fp32" as const,
  capability: FIXED32,
});
const ATTENTION_CONFIGURATION = Object.freeze({
  backend: "opt-0070-fixed32-quad-query32-full-self-production" as const,
  runtimeProfileId:
    ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  capability: FIXED32,
  expectedQueryTokens: 2_250,
  expectedConditionTokens: 98,
});

const ROLE_SLOTS = Object.freeze({
  selfModulated: 61,
  selfMergedAttention: 69,
  crossNormalized: 72,
  crossMergedAttention: 77,
  mlpModulated: 81,
  gatedActivation: 84,
} as const);

const EXPECTED_LIFETIMES = Object.freeze([
  { firstQuantum: 27, lastQuantum: 50 },
  { firstQuantum: 55, lastQuantum: 78 },
  { firstQuantum: 83, lastQuantum: 106 },
  { firstQuantum: 111, lastQuantum: 134 },
  { firstQuantum: 139, lastQuantum: 162 },
  { firstQuantum: 167, lastQuantum: 190 },
  { firstQuantum: 195, lastQuantum: 218 },
  { firstQuantum: 223, lastQuantum: 246 },
]);

describe("OPT-0081 diagnostic graph storage profile", () => {
  it("selectively shrinks exactly six established arena roles", () => {
    const reference = planAceDitBackendArena("reference-bf16", SHAPE);
    const candidate = planAceDitBackendArena(
      "reference-bf16",
      SHAPE,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      PROFILE,
    );

    expect(candidate.denseInputStorageProfile).toBe(PROFILE);
    expect(reference).not.toHaveProperty("denseInputStorageProfile");
    expect(reference.roles).toHaveLength(123);
    expect(candidate.roles).toHaveLength(123);
    expect(reference.slots).toHaveLength(98);
    expect(candidate.slots).toHaveLength(98);
    expect(reference.allocatedArenaBytes).toBe(674_815_488);
    expect(candidate.allocatedArenaBytes).toBe(601_087_488);
    expect(reference.allocatedArenaBytes - candidate.allocatedArenaBytes)
      .toBe(73_728_000);
    expect(candidate.logicalRoleBytes).toBe(644_446_260);

    const selectedKeys = new Set(ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES.map(
      (role) => `layerScratch.${role}`,
    ));
    let referenceSelectedBytes = 0;
    let candidateSelectedBytes = 0;
    for (const referenceRole of reference.roles) {
      const candidateRole = candidate.roles.find(
        ({ key }) => key === referenceRole.key,
      );
      expect(candidateRole, referenceRole.key).toBeDefined();
      expect(candidateRole!.lifetimes).toEqual(referenceRole.lifetimes);
      if (!selectedKeys.has(referenceRole.key)) {
        expect(candidateRole!.byteLength, referenceRole.key)
          .toBe(referenceRole.byteLength);
        expect(candidate.roleToSlot[referenceRole.key], referenceRole.key)
          .toBe(reference.roleToSlot[referenceRole.key]);
        continue;
      }
      referenceSelectedBytes += referenceRole.byteLength;
      candidateSelectedBytes += candidateRole!.byteLength;
      expect(candidateRole!.byteLength).toBe(referenceRole.byteLength / 2);
      expect(candidateRole!.lifetimes).toEqual(EXPECTED_LIFETIMES);
    }
    expect(referenceSelectedBytes).toBe(147_456_000);
    expect(candidateSelectedBytes).toBe(73_728_000);

    for (const role of ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES) {
      const key = `layerScratch.${role}`;
      const expectedBytes = role === "gatedActivation"
        ? 27_648_000
        : 9_216_000;
      expect(candidate.roleToSlot[key]).toBe(ROLE_SLOTS[role]);
      expect(candidate.roles.find(({ key: candidateKey }) =>
        candidateKey === key
      )).toMatchObject({ key, byteLength: expectedBytes });
      expect(candidate.slots[ROLE_SLOTS[role]]!.byteLength).toBe(expectedBytes);
    }
    expect(candidate.slots[61]!.roles).toEqual([
      "timestepScratch.timestepProjection",
      "layerScratch.selfModulated",
    ]);
  });

  it("reconciles graph, arena, and backend accounting without a cast slot", () => {
    const referenceGraph = planAceDitGraphMemory("reference-bf16", SHAPE);
    const candidateGraph = planAceDitGraphMemory(
      "reference-bf16",
      SHAPE,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      PROFILE,
    );
    const candidateBackend = planAceDitGpuBackendMemory(
      "reference-bf16",
      SHAPE,
      1_024,
      "mixed-opt-0009",
      1,
      false,
      false,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      false,
      PROFILE,
    );

    expect(referenceGraph.layerScratchBytes).toBe(608_305_152);
    expect(candidateGraph).toMatchObject({
      denseInputStorageProfile: PROFILE,
      activationElementBytes: 4,
      layerScratchBytes: 534_577_152,
      minimumGraphBytesExcludingWeights: 597_678_132,
      unaliasedGraphBytesExcludingWeights: 644_446_260,
      largestRequiredBindingBytes: 55_296_000,
    });
    expect(candidateBackend).toMatchObject({
      modelProfile: "reference-bf16",
      denseInputStorageProfile: PROFILE,
      graph: candidateGraph,
      arena: {
        denseInputStorageProfile: PROFILE,
        allocatedArenaBytes: 601_087_488,
        logicalRoleBytes: 644_446_260,
      },
    });
    expect(candidateBackend.arena.logicalRoleBytes).toBe(
      candidateBackend.graph.unaliasedGraphBytesExcludingWeights,
    );
    expect(candidateBackend.arena.roles.some(({ key }) =>
      key.toLowerCase().includes("cast")
    )).toBe(false);
    expect(candidateBackend.graph.largestRequiredBindingBytes).toBe(
      referenceGraph.largestRequiredBindingBytes,
    );
    expect(candidateBackend.commandBufferCount).toBe(
      planAceDitGpuBackendMemory(
        "reference-bf16",
        SHAPE,
        1_024,
        "mixed-opt-0009",
      ).commandBufferCount,
    );
  });

  it("fails closed outside the literal profile, model, and M2250/C98 shape", () => {
    expect(requireAceDitDenseInputStorageProfile()).toBeUndefined();
    expect(requireAceDitDenseInputStorageProfile(PROFILE)).toBe(PROFILE);
    expect(() => requireAceDitDenseInputStorageProfile(
      "future-storage-profile" as never,
    )).toThrow(/not authenticated/);
    expect(() => planAceDitGraphMemory(
      "raw-fp16",
      SHAPE,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      PROFILE,
    )).toThrow(/reference-bf16 M2250\/C98/);
    for (const shape of [
      { ...SHAPE, batch: 2 },
      { ...SHAPE, latentFrames: 4_498 },
      { ...SHAPE, conditionTokens: 97 },
    ]) expect(() => planAceDitBackendArena(
      "reference-bf16",
      shape,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      PROFILE,
    )).toThrow(/reference-bf16 M2250\/C98/);

    expect(aceDitLayerScratchBytes(
      "reference-bf16",
      "selfModulated",
      4_608_000,
      PROFILE,
    )).toBe(9_216_000);
    expect(aceDitLayerScratchBytes(
      "reference-bf16",
      "afterSelfAttention",
      4_608_000,
      PROFILE,
    )).toBe(18_432_000);

    expect(() => planAceDitGpuBackendMemory(
      "reference-bf16",
      SHAPE,
      1_024,
      "portable",
      1,
      false,
      false,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      false,
      PROFILE,
    )).toThrow(/exact mixed-opt-0009 singleton production-eight/);
    expect(() => planAceDitGpuBackendMemory(
      "reference-bf16",
      SHAPE,
      1_024,
      "mixed-opt-0009",
      8,
      false,
      false,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      false,
      PROFILE,
    )).toThrow(/exact mixed-opt-0009 singleton production-eight/);
  });

  it("rejects graph-runtime and prepared-resource profile disagreement both ways", async () => {
    for (const [runtimeProfile, compilationProfile] of [
      [PROFILE, undefined],
      [undefined, PROFILE],
    ] as const) {
      const runtimeDestroy = vi.fn();
      const modelDestroy = vi.fn();
      const runtime = {
        denseInputStorageProfile: runtimeProfile,
        destroy: runtimeDestroy,
      } as unknown as Parameters<typeof AceDitGraphOwner.createWithRuntime>[2];
      await expect(AceDitGraphOwner.createWithRuntime(
        {} as GPUDevice,
        {
          modelProfile: "reference-bf16",
          residentBytes: 0,
          weights: {} as never,
          destroy: modelDestroy,
        },
        runtime,
        SHAPE,
        ACE_DIRECT_DCW_CONFIGURATION,
        {} as never,
        compilationProfile === undefined
          ? {}
          : { ditDenseInputStorageProfile: compilationProfile },
      )).rejects.toThrow(/runtime and dense-input storage profile diverged/);
      expect(runtimeDestroy).toHaveBeenCalledOnce();
      expect(modelDestroy).toHaveBeenCalledOnce();
    }

    const memory = planAceDitGpuBackendMemory(
      "reference-bf16",
      SHAPE,
      1_024,
      "mixed-opt-0009",
    );
    const graphDestroy = vi.fn(async () => undefined);
    const resourceDestroy = vi.fn();
    expect(() => AceDitGpuBackend.fromPreparedResources({
      device: fakeDevice(),
      modelProfile: "reference-bf16",
      denseInputStorageProfile: PROFILE,
      gemmBackend: "mixed-opt-0009",
      shape: planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, SHAPE),
      graph: {
        commandBufferCount: memory.physicalGraphQuantumCount,
        run: vi.fn(),
        destroy: graphDestroy,
      },
      readback: { size: memory.readbackBufferBytes } as GPUBuffer,
      memory,
      destroy: resourceDestroy,
    })).toThrow(/prepared resources violate the backend contract/);
    expect(graphDestroy).toHaveBeenCalledOnce();
    expect(resourceDestroy).toHaveBeenCalledOnce();
  });

  it("compiles both production attention modes through six producers and nine consumers", async () => {
    const fake = fakeDevice();
    const runtime = AceCorrectnessDitRuntime.create(
      fake,
      "reference-bf16",
      GEMM_CONFIGURATION,
      DENSE_CONFIGURATION,
      ATTENTION_CONFIGURATION,
      PROFILE,
    );
    const evenBindings = layerBindings();
    const oddBindings = layerBindings();
    const even = await runtime.createLayerDispatch(
      "ace-dit-eval-0-layer-0",
      ACE_TURBO_DIT_CONFIG,
      {
        batch: 1,
        tokens: 2_250,
        conditionTokens: 98,
        attentionMode: "sliding",
      },
      evenBindings,
    );
    const odd = await runtime.createLayerDispatch(
      "ace-dit-eval-0-layer-1",
      ACE_TURBO_DIT_CONFIG,
      {
        batch: 1,
        tokens: 2_250,
        conditionTokens: 98,
        attentionMode: "full",
      },
      oddBindings,
    );

    expect(runtime.denseInputStorageProfile).toBe(PROFILE);
    expect(runtime.attentionBackend).toBe(
      "opt-0070-fixed32-quad-query32-full-self-production",
    );
    expect(even.primitiveCount).toBe(32);
    expect(odd.primitiveCount).toBe(32);
    expect(even.cooperativeSequence.quantumCount).toBe(11);
    expect(odd.cooperativeSequence.quantumCount).toBe(15);
    expect(1 + 1 + even.cooperativeSequence.quantumCount +
      odd.cooperativeSequence.quantumCount).toBe(28);
    expect(fake.bindGroupLabels.filter((label) =>
      label.includes("-opt-0081-typed-f16-input-bindings")
    )).toHaveLength(18);
    expect(fake.bindGroupLabels.filter((label) =>
      label.endsWith("-opt-0081-bindings")
    )).toHaveLength(12);
    expect(fake.shaderLabels.filter((label) =>
      label.includes("opt-0081")
    )).toHaveLength(8);

    runtime.destroy();
    runtime.destroy();
    expect(() => even.encode(fakePass())).toThrow(/destroyed/);
    await expect(runtime.createLayerDispatch(
      "ace-dit-eval-0-layer-0",
      ACE_TURBO_DIT_CONFIG,
      {
        batch: 1,
        tokens: 2_250,
        conditionTokens: 98,
        attentionMode: "sliding",
      },
      evenBindings,
    )).rejects.toThrow(/runtime was destroyed/);
  });

  it("rejects storage/config disagreement before compiling a pipeline", () => {
    const fake = fakeDevice();
    expect(() => AceCorrectnessDitRuntime.create(
      fake,
      "reference-bf16",
      GEMM_CONFIGURATION,
      { backend: "opt-0037-k4-fp16-partials", capability: FIXED32 },
      ATTENTION_CONFIGURATION,
      PROFILE,
    )).toThrow(/exact M2250\/C98 OPT-0009\/OPT-0070 fixed32/);
    expect(() => AceCorrectnessDitRuntime.create(
      fake,
      "reference-bf16",
      GEMM_CONFIGURATION,
      DENSE_CONFIGURATION,
      { backend: "fixed32-subgroup-query8", capability: FIXED32 },
      PROFILE,
    )).toThrow(/exact M2250\/C98 OPT-0009\/OPT-0070 fixed32/);
    expect(() => AceCorrectnessDitRuntime.create(
      fake,
      "reference-bf16",
      {
        backend: "subgroups",
        capability: { subgroupMinSize: 16, subgroupMaxSize: 32 },
      },
      DENSE_CONFIGURATION,
      ATTENTION_CONFIGURATION,
      PROFILE,
    )).toThrow(/exact M2250\/C98 OPT-0009\/OPT-0070 fixed32/);
    expect(fake.createComputePipelineAsyncMock).not.toHaveBeenCalled();

    expect(() => AceCorrectnessDitRuntime.create(
      fakeDevice(["subgroups"]),
      "reference-bf16",
      GEMM_CONFIGURATION,
      DENSE_CONFIGURATION,
      ATTENTION_CONFIGURATION,
      PROFILE,
    )).toThrow(/shader-f16/);
  });

  it("keeps the profile isolated from production and legacy capture paths", () => {
    expect(pipelineSource).not.toContain("ditDenseInputStorageProfile");
    expect(aceDitSource.match(/\.createSelfModulatedDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(aceDitSource.match(/\.createSelfMergedAttentionDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(aceDitSource.match(/\.createCrossNormalizedDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(aceDitSource.match(/\.createCrossMergedAttentionDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(aceDitSource.match(/\.createMlpModulatedDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(aceDitSource.match(/\.createGatedActivationDispatch\(/g) ?? [])
      .toHaveLength(1);
    expect(backendSource).toContain("ditSubmissionPolicy !== undefined");
    expect(backendSource).toContain(
      "options.onCommandProfile !== undefined",
    );
    expect(backendSource).toContain(
      "options.opt0080SchedulingProfile !== undefined",
    );
    expect(backendSource).toContain(
      "options.captureOpt0062AttentionIdentity !== undefined",
    );
    expect(backendSource).toContain("slot,\n      0,\n      role.byteLength");
    expect(graphSource).toContain(
      "compilation.capturePhysicalCommandDescriptors !== undefined",
    );
    expect(graphSource).toContain(
      "OPT-0081 dense-input graph storage requires matching fixed32",
    );
  });
});

interface FakeDeviceDiagnostics {
  readonly shaderLabels: string[];
  readonly bindGroupLabels: string[];
  readonly createComputePipelineAsyncMock: ReturnType<typeof vi.fn>;
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(
  features: readonly string[] = ["shader-f16", "subgroups"],
): FakeDevice {
  const shaderLabels: string[] = [];
  const bindGroupLabels: string[] = [];
  const pipeline = {
    getBindGroupLayout: () => ({}) as GPUBindGroupLayout,
  } as unknown as GPUComputePipeline;
  const createComputePipelineAsyncMock = vi.fn(async () => pipeline);
  return {
    features: new Set(features),
    limits: {
      maxComputeInvocationsPerWorkgroup: 1_024,
      maxComputeWorkgroupSizeX: 1_024,
      maxComputeWorkgroupSizeY: 1_024,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeWorkgroupStorageSize: 32_768,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
    },
    shaderLabels,
    bindGroupLabels,
    createComputePipelineAsyncMock,
    createShaderModule: vi.fn((descriptor: GPUShaderModuleDescriptor) => {
      shaderLabels.push(descriptor.label ?? "");
      return {
        getCompilationInfo: vi.fn(async () => ({ messages: [] })),
      } as unknown as GPUShaderModule;
    }),
    createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    createPipelineLayout: vi.fn(() => ({} as GPUPipelineLayout)),
    createComputePipelineAsync: createComputePipelineAsyncMock,
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const mapped = new ArrayBuffer(Number(descriptor.size));
      return {
        size: Number(descriptor.size),
        getMappedRange: () => mapped,
        unmap: vi.fn(),
        destroy: vi.fn(),
      } as unknown as GPUBuffer;
    }),
    createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => {
      bindGroupLabels.push(descriptor.label ?? "");
      return {} as GPUBindGroup;
    }),
  } as unknown as FakeDevice;
}

function layerBindings(): AceDitLayerBindings {
  const hidden = 2_250 * 2_048;
  const query = hidden;
  const keyValue = 2_250 * 1_024;
  const intermediate = 2_250 * 6_144;
  const scratchElements: Readonly<Record<keyof AceDitLayerScratch, number>> = {
    modulation: 6 * 2_048,
    selfNormalized: hidden,
    selfModulated: hidden,
    selfQueryFlat: query,
    selfKeyFlat: keyValue,
    selfValueFlat: keyValue,
    selfQueryHeads: query,
    selfKeyHeads: keyValue,
    selfValueHeads: keyValue,
    selfNormalizedQueryHeads: query,
    selfNormalizedKeyHeads: keyValue,
    selfRotatedQueryHeads: query,
    selfRotatedKeyHeads: keyValue,
    selfAttentionHeads: query,
    selfMergedAttention: query,
    selfProjectedAttention: hidden,
    afterSelfAttention: hidden,
    crossNormalized: hidden,
    crossQueryFlat: query,
    crossQueryHeads: query,
    crossNormalizedQueryHeads: query,
    crossAttentionHeads: query,
    crossMergedAttention: query,
    crossProjectedAttention: hidden,
    afterCrossAttention: hidden,
    mlpNormalized: hidden,
    mlpModulated: hidden,
    gate: intermediate,
    up: intermediate,
    gatedActivation: intermediate,
    projectedMlp: hidden,
  };
  const scratch = Object.fromEntries(Object.entries(scratchElements).map(
    ([role, elements]) => [
      role,
      fakeBinding(aceDitLayerScratchBytes(
        "reference-bf16",
        role as keyof AceDitLayerScratch,
        elements,
        PROFILE,
      )),
    ],
  )) as unknown as AceDitLayerScratch;
  const packedWeight = (elements: number): GPUBufferBinding =>
    fakeBinding(elements * 2);

  return {
    input: fakeBinding(hidden * 4),
    output: fakeBinding(hidden * 4),
    weights: {
      scaleShiftTable: packedWeight(6 * 2_048),
      selfAttentionNorm: packedWeight(2_048),
      selfQueryProjection: packedWeight(2_048 * 2_048),
      selfKeyProjection: packedWeight(2_048 * 1_024),
      selfValueProjection: packedWeight(2_048 * 1_024),
      selfQueryNorm: packedWeight(128),
      selfKeyNorm: packedWeight(128),
      selfOutputProjection: packedWeight(2_048 * 2_048),
      crossAttentionNorm: packedWeight(2_048),
      crossQueryProjection: packedWeight(2_048 * 2_048),
      crossQueryNorm: packedWeight(128),
      crossOutputProjection: packedWeight(2_048 * 2_048),
      mlpNorm: packedWeight(2_048),
      gateProjection: packedWeight(2_048 * 6_144),
      upProjection: packedWeight(2_048 * 6_144),
      downProjection: packedWeight(6_144 * 2_048),
    },
    scratch,
    timestepProjection: fakeBinding(6 * 2_048 * 4),
    crossKey: fakeBinding(8 * 98 * 128 * 4),
    crossValue: fakeBinding(8 * 98 * 128 * 4),
    selfValidLengths: fakeBinding(2 * 4),
    crossValidLengths: fakeBinding(2 * 4),
    cosine: fakeBinding(2_250 * 128 * 4),
    sine: fakeBinding(2_250 * 128 * 4),
  };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}

function fakePass(): GPUComputePassEncoder {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder;
}
