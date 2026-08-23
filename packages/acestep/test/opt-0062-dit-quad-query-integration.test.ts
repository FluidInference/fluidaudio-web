import { describe, expect, it, vi } from "vitest";
import pipelineSource from "../src/runtime/webgpu-pipeline.ts?raw";
import graphSource from "../src/webgpu/dit-graph.ts?raw";

import { ACE_DIT_LAYER_TYPES } from "../src/model/graph-contract.js";
import { isAceClientMessage } from "../src/runtime/protocol.js";
import { ACE_REFERENCE_SUBGROUP_PROFILE } from
  "../src/webgpu/capabilities.js";
import {
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  requireAceDitAttentionRuntimeProfile,
} from "../src/webgpu/dit-attention-profile.js";
import { planAceDitGpuBackendMemory, resolveAceDitMixedGemmSelection } from
  "../src/webgpu/dit-backend.js";
import {
  classifyAceOpt0018DitCommandMember,
  type AceDitGraphQuantum,
} from "../src/webgpu/dit-graph.js";
import {
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
} from "../src/webgpu/dit-fp16-package.js";
import { testInitializeMessage } from "./runtime-fixtures.js";
import {
  ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
  ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256,
  ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
  ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
  ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES,
  ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS,
  AceOpt0062QuadQueryAttentionKernel,
  planAceOpt0062Attention,
  selectAceOpt0062AttentionRoute,
} from "../src/webgpu/kernels/attention-quad-query-production.js";
import {
  exactOpt0062TrajectoryIdentity,
  summarizeOpt0062BalancedGate,
  type Opt0062TimingSample,
} from "./browser/opt-0062-dit-quad-query-contract.js";

const FULL_SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});
const CAPABILITY = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

describe("OPT-0062 exact quad-query production integration", () => {
  it("authenticates an independent attention-only runtime/kernel-set profile", () => {
    expect(requireAceDitAttentionRuntimeProfile()).toMatchObject({
      id: ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      fullSelfAttentionOwner: "fixed32-subgroup-query8",
    });
    expect(requireAceDitAttentionRuntimeProfile(
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toEqual({
      id: ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      fullSelfAttentionOwner: "opt-0062-fixed32-quad-query32",
      slidingSelfAttentionOwner: "fixed32-subgroup-query8",
      crossAttentionOwner: "fixed32-subgroup-query8",
    });
    expect(() => requireAceDitAttentionRuntimeProfile(
      "future-profile" as never,
    )).toThrow(/not authenticated/);
  });

  it("selects quad only as an opt-in rev7 profile and preserves query8 default", () => {
    const current = resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
    );
    expect(current).toMatchObject({
      attentionRuntimeProfile: ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      attentionConfiguration: { backend: "fixed32-subgroup-query8" },
    });
    const candidate = resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    );
    expect(candidate).toMatchObject({
      backend: "mixed-opt-0009",
      denseRuntimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      attentionRuntimeProfile:
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      attentionConfiguration: {
        backend: "opt-0062-fixed32-quad-query32-full-self",
        runtimeProfileId:
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      },
    });
    for (const denseProfile of [
      ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
    ]) {
      expect(() => resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_SUBGROUP_PROFILE,
        32,
        32,
        denseProfile,
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      )).toThrow(/cannot combine/);
    }
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      "future-attention-profile" as never,
    )).toThrow(/not authenticated/);
  });

  it("keeps protocol omission query8 and authenticates only the rev7 OPT-0062 pair", () => {
    const base = testInitializeMessage();
    expect(base.configuration).not.toHaveProperty("ditAttentionRuntimeProfile");
    expect(isAceClientMessage(base)).toBe(true);
    const candidate = {
      ...base,
      configuration: {
        ...base.configuration,
        ditAttentionRuntimeProfile:
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      },
    };
    expect(isAceClientMessage(candidate)).toBe(true);
    expect(candidate.configuration.schedulingProfile).toBe(
      base.configuration.schedulingProfile,
    );
    expect(candidate.configuration.vaePackage).toBe(
      base.configuration.vaePackage,
    );
    expect(candidate.configuration.ditDensePackage).toBe(
      base.configuration.ditDensePackage,
    );
    expect(isAceClientMessage({
      ...candidate,
      configuration: {
        ...candidate.configuration,
        ditDensePackage: {
          ...candidate.configuration.ditDensePackage,
          manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
          runtimeProfile: ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
        },
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...candidate,
      configuration: {
        ...candidate.configuration,
        ditAttentionRuntimeProfile: "future-profile",
      },
    })).toBe(false);
  });

  it("keeps the full-graph checkpoint seam additive and sampler-copy bounded", () => {
    expect(pipelineSource).toContain(
      "ditAttentionRuntimeProfile:\n                ready.configuration.ditAttentionRuntimeProfile",
    );
    expect(pipelineSource).toContain("context.opt0062DitRun === undefined");
    expect(pipelineSource).toContain("captureEvaluationLatents: true as const");
    expect(pipelineSource).toContain(
      'schema: "ace-dit-opt0062-m2250-trajectory-checkpoint-v1"',
    );
    expect(pipelineSource).toContain("snapshotExtraCommandBufferCount: 0 as const");
    expect(pipelineSource).toContain("snapshotExtraQueueDrainCount: 0 as const");
    expect(pipelineSource).toContain("quadDescriptorCount !== 480");
    expect(pipelineSource).toContain(
      "OPT-0062 diagnostic checkpoint cannot continue into VAE",
    );
    expect(pipelineSource).toContain(
      "captureOpt0062AttentionIdentity: true as const",
    );
    expect(pipelineSource).toContain(
      "actualLayerIdentity.totalComparedElements !== 442_368_000",
    );
    expect(graphSource).toContain(
      "encodedAttentionIdentityCopies += 1",
    );
    expect(graphSource).toContain(
      "encodedAttentionIdentityCopies !== 1",
    );
  });

  it("enforces exact trajectory and balanced directional performance gates", () => {
    const trajectory = Array.from({ length: 8 }, (_, evaluation) => {
      const values = new Float32Array(288_000);
      values[0] = evaluation + 1;
      return values;
    });
    const repeat = trajectory.map((values) => values.slice());
    expect(exactOpt0062TrajectoryIdentity(trajectory, repeat)).toBe(true);
    new Uint32Array(repeat[4]!.buffer)[17] = 1;
    expect(exactOpt0062TrajectoryIdentity(trajectory, repeat)).toBe(false);

    const sample = (
      direction: "forward" | "reverse",
      order: 0 | 1,
      arm: "query8" | "quad",
      graphWallMs: number,
      fullSelfMs: number,
    ): Opt0062TimingSample => Object.freeze({
      direction,
      order,
      arm,
      fullSelfMs,
      sliceEvaluation0FullSelfMs: fullSelfMs / 8,
      graphWallMs,
      ditStageWallMs: graphWallMs + 100,
      commandDrainMs: graphWallMs - 50,
      requestedIdleMs: 2_553,
      readbackMs: 3,
      residualMs: 50,
      familyMs: Object.freeze({
        "self-full": fullSelfMs,
        "feed-forward": 10_000,
      }),
    });
    const passed = summarizeOpt0062BalancedGate([
      sample("forward", 0, "query8", 20_000, 12_000),
      sample("forward", 1, "quad", 18_000, 8_000),
      sample("reverse", 0, "quad", 17_900, 7_900),
      sample("reverse", 1, "query8", 20_100, 12_100),
    ]);
    expect(passed).toMatchObject({
      forwardGraphImproved: true,
      reverseGraphImproved: true,
      aggregateGraphSavingsMs: 4_200,
      passed: true,
    });
    expect(() => summarizeOpt0062BalancedGate([
      passed.samples[1]!,
      passed.samples[0]!,
      passed.samples[2]!,
      passed.samples[3]!,
    ])).toThrow(/inventory/);
  });

  it("bounds actual-layer identity storage without changing graph topology", () => {
    const args = [
      "reference-bf16" as const,
      { batch: 1, latentFrames: 4_500, conditionTokens: 98 },
      1_024,
      "mixed-opt-0009" as const,
      1 as const,
      true,
    ] as const;
    const control = planAceDitGpuBackendMemory(...args, false);
    const exact = planAceDitGpuBackendMemory(...args, true);
    expect(ACE_OPT_0062_IDENTITY_COUNTER_BYTES).toBe(24_576);
    expect(exact).toMatchObject({
      opt0062AttentionIdentityCounterBytes: 24_576,
      opt0062AttentionIdentityReadbackBytes: 24_576,
      detachedOpt0062AttentionIdentityBytes: 24_576,
      graphCommandBufferCount: control.graphCommandBufferCount,
      commandBufferCount: control.commandBufferCount,
    });
    expect(exact.accountedGpuBytes - control.accountedGpuBytes).toBe(49_152);
  });

  it("attributes only full-self commands to the actual quad kernel", () => {
    const quantum = Object.freeze({
      index: 27,
      kind: "layer",
      evaluation: 0,
      layer: 1,
      label: "ace-dit-eval-0-layer-1",
    }) satisfies AceDitGraphQuantum;
    const classify = (suffix: string) => classifyAceOpt0018DitCommandMember(
      quantum,
      `${quantum.label}-${suffix}`,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    );
    expect(classify("self-full-attention")).toEqual({
      family: "self-full",
      backend: "opt-0062-fixed32-quad-query32-full-self",
      kernel: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
    });
    expect(classify("self-sliding-attention")).toEqual({
      family: "self-sliding",
      backend: "fixed32-subgroup-query8",
      kernel: "fixed32-subgroup-query8",
    });
    expect(classify("cross-attention")).toEqual({
      family: "cross-attention",
      backend: "fixed32-subgroup-query8",
      kernel: "fixed32-subgroup-query8",
    });
    expect(classify("mlp-down-projection")).toMatchObject({
      family: "feed-forward",
      backend: "opt-0009-fp16-fp32",
    });
  });

  it("adds bounded cooperative ranges without changing quad-query ownership", () => {
    const plan = planAceOpt0062Attention(FULL_SHAPE);
    expect(plan).toMatchObject({
      backend: "opt-0062-fixed32-quad-query32-full-self",
      workgroupSize: 256,
      workgroupStorageBytes: 1_024,
      queriesPerWorkgroup: 32,
      queryTokensPerTile: 16,
      workgroupCount: 1_128,
      query8WorkgroupCount: 4_504,
      outputElements: 4_608_000,
      outputRangeCount: 5,
      maximumMultiplyAddsPerWorkgroup: 9_216_000,
      totalMultiplyAdds: 10_368_000_000,
    });
    expect(plan.outputRanges.map((range) => range.workgroupCount)).toEqual([
      233,
      233,
      233,
      233,
      196,
    ]);
    expect(plan.outputRanges.reduce(
      (total, range) => total + range.outputCount,
      0,
    )).toBe(plan.outputElements);
    expect(plan.outputRanges.reduce(
      (total, range) => total + range.multiplyAdds,
      0,
    )).toBe(plan.totalMultiplyAdds);
    expect(plan.outputRanges.every((range) =>
      range.multiplyAdds <= 2 * 1024 * 1024 * 1024
    )).toBe(true);

    const owners = new Uint8Array(16 * 2_250);
    for (let kvHead = 0; kvHead < 8; kvHead += 1) {
      for (let tile = 0; tile < Math.ceil(2_250 / 16); tile += 1) {
        for (let subgroup = 0; subgroup < 8; subgroup += 1) {
          const head = kvHead * 2 + Math.floor(subgroup / 4);
          const firstToken = tile * 16 + subgroup % 4;
          for (const token of [
            firstToken,
            firstToken + 4,
            firstToken + 8,
            firstToken + 12,
          ]) {
            if (token < 2_250) {
              const index = head * 2_250 + token;
              owners[index] = owners[index]! + 1;
            }
          }
        }
      }
    }
    expect([...owners].every((ownerCount) => ownerCount === 1)).toBe(true);
  });

  it("routes only exact production full-self labels and fails closed", () => {
    expect(selectAceOpt0062AttentionRoute(
      "ace-dit-eval-7-layer-23-self-full-attention",
      FULL_SHAPE,
    )).toEqual({
      route: "quad-query32-full-self",
      label: "ace-dit-eval-7-layer-23-self-full-attention",
      kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
    });
    expect(selectAceOpt0062AttentionRoute(
      "ace-dit-eval-0-layer-0-self-sliding-attention",
      { ...FULL_SHAPE, mode: "sliding", slidingRadius: 128 },
    ).route).toBe("query8-self-sliding");
    expect(selectAceOpt0062AttentionRoute(
      "ace-dit-eval-0-layer-1-cross-attention",
      { ...FULL_SHAPE, keyValueTokens: 98 },
    ).route).toBe("query8-cross");
    for (const [label, shape] of [
      ["slice-self-full-attention", FULL_SHAPE],
      ["ace-dit-eval-8-layer-1-self-full-attention", FULL_SHAPE],
      ["ace-dit-eval-0-layer-0-self-full-attention", FULL_SHAPE],
      [
        "ace-dit-eval-0-layer-1-self-full-attention",
        { ...FULL_SHAPE, queryTokens: 2_249, keyValueTokens: 2_249 },
      ],
      [
        "ace-dit-eval-0-layer-1-self-full-attention",
        { ...FULL_SHAPE, mode: "sliding" as const, slidingRadius: 128 },
      ],
    ] as const) {
      expect(() => selectAceOpt0062AttentionRoute(label, shape), label)
        .toThrow(/OPT-0062/);
    }
  });

  it("reports the actual kernel, encodes five ranges, and destroys once", async () => {
    const device = fakeFixed32Device();
    const kernel = createKernel(device);
    const dispatch = await kernel.createDispatch(
      "ace-dit-eval-0-layer-1-self-full-attention",
      FULL_SHAPE,
      bindingsFor(FULL_SHAPE.queryTokens, FULL_SHAPE.keyValueTokens),
    );
    expect(dispatch).toMatchObject({
      backend: "opt-0062-fixed32-quad-query32-full-self",
      kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
      rangeCount: 5,
    });
    if (dispatch.backend !== "opt-0062-fixed32-quad-query32-full-self") {
      throw new Error("expected OPT-0062 quad-query dispatch");
    }
    const pass = fakeComputePass();
    dispatch.encode(pass);
    expect(pass.dispatchWorkgroups.mock.calls.map((call) => call[0])).toEqual([
      233,
      233,
      233,
      233,
      196,
    ]);
    expect(device.createShaderModule).toHaveBeenCalledWith(
      expect.objectContaining({
        code: expect.stringContaining("subgroupAdd(dot_partial_3)"),
      }),
    );
    expect(device.createBindGroup).toHaveBeenCalledTimes(5);
    expect(() => dispatch.encodeRange(pass, 5)).toThrow(/outside/);
    kernel.destroy();
    kernel.destroy();
    await Promise.resolve();
    expect(device.buffers).toHaveLength(1);
    expect(device.buffers[0]!.destroy).toHaveBeenCalledTimes(1);
    await expect(kernel.createDispatch(
      "ace-dit-eval-0-layer-3-self-full-attention",
      FULL_SHAPE,
      bindingsFor(FULL_SHAPE.queryTokens, FULL_SHAPE.keyValueTokens),
    )).rejects.toThrow(/destroyed/);
  });

  it("runs the query8 oracle and exhaustive raw-U32 comparison inside existing ranges", async () => {
    const device = fakeFixed32Device();
    const kernel = createKernel(device);
    const base = bindingsFor(FULL_SHAPE.queryTokens, FULL_SHAPE.keyValueTokens);
    const dispatch = await kernel.createDispatch(
      "ace-dit-eval-0-layer-1-self-full-attention",
      FULL_SHAPE,
      {
        ...base,
        opt0062Identity: {
          oracleOutput: fakeBinding(
            ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS * 4,
          ),
          counters: fakeBinding(ACE_OPT_0062_IDENTITY_COUNTER_BYTES),
          routeIndex: 0,
        },
      },
    );
    if (dispatch.backend !== "opt-0062-fixed32-quad-query32-full-self") {
      throw new Error("expected OPT-0062 quad-query dispatch");
    }
    const first = fakeComputePass();
    dispatch.encodeRange(first, 0);
    expect(first.dispatchWorkgroups.mock.calls.map((call) => call[0])).toEqual([
      18_000,
      932,
      233,
    ]);
    const last = fakeComputePass();
    dispatch.encodeRange(last, 4);
    expect(last.dispatchWorkgroups.mock.calls.map((call) => call[0])).toEqual([
      776,
      196,
      18_000,
    ]);
    expect(device.createShaderModule).toHaveBeenCalledWith(
      expect.objectContaining({
        code: expect.stringContaining("atomicAdd(&counters[1], 1u)"),
      }),
    );
    const compare = device.createBindGroup.mock.calls.find(
      ([descriptor]) => descriptor.label.endsWith("identity-compare-bindings"),
    )?.[0] as GPUBindGroupDescriptor | undefined;
    expect(compare === undefined ? undefined : [...compare.entries][2]).toEqual({
      binding: 2,
      resource: expect.objectContaining({
        offset: 0,
        size: 24,
      }),
    });
    kernel.destroy();

    const mismatched = createKernel(fakeFixed32Device());
    await expect(mismatched.createDispatch(
      "ace-dit-eval-0-layer-1-self-full-attention",
      FULL_SHAPE,
      {
        ...base,
        opt0062Identity: {
          oracleOutput: fakeBinding(
            ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS * 4,
          ),
          counters: fakeBinding(ACE_OPT_0062_IDENTITY_COUNTER_BYTES),
          routeIndex: ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES / 256,
        },
      },
    )).rejects.toThrow(/route index/);
    mismatched.destroy();
  });

  it("proves all 96 selected routes and zero unintended production routes", async () => {
    const device = fakeFixed32Device();
    const kernel = createKernel(device);
    try {
      for (let evaluation = 0; evaluation < 8; evaluation += 1) {
        for (let layer = 0; layer < ACE_DIT_LAYER_TYPES.length; layer += 1) {
          const mode = ACE_DIT_LAYER_TYPES[layer] === "full_attention"
            ? "full" as const
            : "sliding" as const;
          await kernel.createDispatch(
            `ace-dit-eval-${evaluation}-layer-${layer}-self-${mode}-attention`,
            {
              ...FULL_SHAPE,
              mode,
              ...(mode === "sliding" ? { slidingRadius: 128 } : {}),
            },
            bindingsFor(2_250, 2_250),
          );
          await kernel.createDispatch(
            `ace-dit-eval-${evaluation}-layer-${layer}-cross-attention`,
            { ...FULL_SHAPE, keyValueTokens: 98 },
            bindingsFor(2_250, 98),
          );
        }
      }
      const profile = kernel.finalizeRoutes();
      expect(profile).toMatchObject({
        expectedQuadQueryRoutes: ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
        quadQueryRoutes: 96,
        query8SlidingRoutes: 96,
        query8CrossRoutes: 192,
        query8OtherRoutes: 0,
        unintendedQuadQueryRoutes: 0,
        authenticatedWgslSha256: ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256,
      });
      expect(profile.uniqueQuadQueryRouteIds).toHaveLength(96);
      expect(new Set(profile.uniqueQuadQueryRouteIds).size).toBe(96);
      expect(profile.uniqueQuadQueryRouteIds[0]).toBe(
        "ace-dit-eval-0-layer-1-self-full-attention",
      );
      expect(profile.uniqueQuadQueryRouteIds.at(-1)).toBe(
        "ace-dit-eval-7-layer-23-self-full-attention",
      );
    } finally {
      kernel.destroy();
    }
    await Promise.resolve();
    expect(device.buffers).toHaveLength(3);
    expect(device.buffers.every((buffer) =>
      buffer.destroy.mock.calls.length === 1
    )).toBe(true);
  });

  it("refuses to finalize a partial graph inventory", () => {
    const kernel = createKernel(fakeFixed32Device());
    try {
      expect(() => kernel.finalizeRoutes()).toThrow(/incomplete or unintended/);
    } finally {
      kernel.destroy();
    }
  });
});

function createKernel(device: GPUDevice): AceOpt0062QuadQueryAttentionKernel {
  return AceOpt0062QuadQueryAttentionKernel.create(
    device,
    "reference-bf16",
    {
      backend: "opt-0062-fixed32-quad-query32-full-self",
      runtimeProfileId:
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      capability: CAPABILITY,
    },
  );
}

function bindingsFor(
  queryTokens: number,
  keyValueTokens: number,
) {
  return {
    query: fakeBinding(16 * queryTokens * 128 * 4),
    key: fakeBinding(8 * keyValueTokens * 128 * 4),
    value: fakeBinding(8 * keyValueTokens * 128 * 4),
    validLengths: fakeBinding(8),
    output: fakeBinding(16 * queryTokens * 128 * 4),
  } as const;
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}

function fakeFixed32Device(): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
  readonly buffers: Array<ReturnType<typeof fakeMappedBuffer>>;
} {
  const buffers: Array<ReturnType<typeof fakeMappedBuffer>> = [];
  return {
    features: new Set<GPUFeatureName>(["subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createShaderModule: vi.fn(() => ({ label: "attention-module" })),
    createComputePipelineAsync: vi.fn(async () => fakePipeline()),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = fakeMappedBuffer(Number(descriptor.size));
      buffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({ label: "attention-bindings" })),
    buffers,
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
    readonly buffers: Array<ReturnType<typeof fakeMappedBuffer>>;
  };
}

function fakePipeline(): GPUComputePipeline {
  return {
    getBindGroupLayout: vi.fn(() => ({ label: "attention-layout" })),
  } as unknown as GPUComputePipeline;
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

function fakeComputePass(): GPUComputePassEncoder & {
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

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });
