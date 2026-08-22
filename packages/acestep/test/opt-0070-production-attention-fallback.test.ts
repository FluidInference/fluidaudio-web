import { describe, expect, it, vi } from "vitest";

import { ACE_DIT_LAYER_TYPES } from "../src/model/graph-contract.js";
import { resolveAceDitMixedGemmSelection } from
  "../src/webgpu/dit-backend.js";
import { ACE_REFERENCE_SUBGROUP_PROFILE } from
  "../src/webgpu/capabilities.js";
import {
  ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import type {
  AceAttentionBindings,
  AceAttentionShape,
} from "../src/webgpu/kernels/attention.js";
import { ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE } from
  "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
} from "../src/webgpu/kernels/attention-quad-query-production.js";
import {
  ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT,
  AceOpt0070ProductionAttentionKernel,
  finalizeAceOpt0070ProductionAttentionRoutes,
  resolveAceOpt0070ProductionAttentionOwnerMode,
  selectAceOpt0070ProductionAttentionRoute,
  type AceOpt0070ProductionAttentionConfiguration,
  type AceOpt0070ProductionAttentionRouteDecision,
} from "../src/webgpu/kernels/attention-opt0070-production.js";

const CAPABILITY = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

describe("OPT-0070 public non-M2250 attention fallback", () => {
  it("freezes exact M2250 quad and non-M2250 query8 ownership", () => {
    expect(ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY).toEqual({
      schema: "ace-opt-0070-attention-shape-policy-v1",
      exactQuadQueryTokens: 2_250,
      exactQuadKeyValueTokens: 2_250,
      exactM2250FullSelfOwner: "opt-0062-fixed32-quad-query32",
      otherFullSelfOwner: "fixed32-subgroup-query8",
      slidingSelfAttentionOwner: "fixed32-subgroup-query8",
      crossAttentionOwner: "fixed32-subgroup-query8",
    });
    expect(resolveAceOpt0070ProductionAttentionOwnerMode(2_250)).toBe(
      "exact-m2250-opt0062-quad",
    );
    for (const tokens of [125, 138, 150, 375, 2_249, 3_000]) {
      expect(resolveAceOpt0070ProductionAttentionOwnerMode(tokens)).toBe(
        "non-m2250-query8",
      );
    }
    for (const tokens of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => resolveAceOpt0070ProductionAttentionOwnerMode(tokens))
        .toThrow(/positive safe integer/);
    }
  });

  it("routes only the exact M2250 full-self shape through physical OPT-0062", () => {
    const m2250 = selectAceOpt0070ProductionAttentionRoute(
      "ace-dit-eval-7-layer-23-self-full-attention",
      fullShape(2_250),
      2_250,
      98,
    );
    expect(m2250).toEqual({
      label: "ace-dit-eval-7-layer-23-self-full-attention",
      evaluation: 7,
      layer: 23,
      route: "quad-query32-full-self",
      kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
    });
    for (const tokens of [125, 138, 150, 375, 2_249, 3_000]) {
      expect(selectAceOpt0070ProductionAttentionRoute(
        "ace-dit-eval-0-layer-1-self-full-attention",
        fullShape(tokens),
        tokens,
        97,
      )).toMatchObject({
        evaluation: 0,
        layer: 1,
        route: "query8-full-self",
        kernelId: "fixed32-subgroup-query8",
      });
    }
  });

  it("binds the public backend to authenticated graph token counts", () => {
    for (const [queryTokens, conditionTokens] of [
      [150, 97],
      [2_250, 98],
    ] as const) {
      expect(resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_SUBGROUP_PROFILE,
        32,
        32,
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        queryTokens,
        conditionTokens,
      ).attentionConfiguration).toEqual({
        backend: "opt-0070-fixed32-quad-query32-full-self-production",
        runtimeProfileId:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        capability: CAPABILITY,
        expectedQueryTokens: queryTokens,
        expectedConditionTokens: conditionTokens,
      });
    }
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toThrow(/graph token counts/);
  });

  it("validates registered layer roles and the complete graph-plan geometry", () => {
    const cases: readonly [string, AceAttentionShape, number, number][] = [
      ["slice-self-full-attention", fullShape(150), 150, 97],
      [
        "ace-dit-eval-8-layer-1-self-full-attention",
        fullShape(150),
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-0-self-full-attention",
        fullShape(150),
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-1-self-full-attention",
        fullShape(149),
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-1-self-full-attention",
        { ...fullShape(150), keyValueTokens: 149 },
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-0-self-sliding-attention",
        { ...slidingShape(150), slidingRadius: 127 },
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-1-self-sliding-attention",
        slidingShape(150),
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-1-cross-attention",
        crossShape(150, 96),
        150,
        97,
      ],
      [
        "ace-dit-eval-0-layer-1-cross-attention",
        { ...crossShape(150, 97), queryHeads: 8 },
        150,
        97,
      ],
    ];
    for (const [label, shape, queryTokens, conditionTokens] of cases) {
      expect(() => selectAceOpt0070ProductionAttentionRoute(
        label,
        shape,
        queryTokens,
        conditionTokens,
      ), label).toThrow(/OPT-0070/);
    }
  });

  it("reconciles all 384 M2250 public routes without changing OPT-0062 counts", () => {
    const routes = completeRoutes(2_250, 98);
    const profile = finalizeAceOpt0070ProductionAttentionRoutes(
      2_250,
      98,
      routes,
    );
    expect(profile).toMatchObject({
      schema: "ace-opt-0070-production-attention-routes-v1",
      runtimeProfileId:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      ownerMode: "exact-m2250-opt0062-quad",
      expectedQueryTokens: 2_250,
      expectedConditionTokens: 98,
      routeCount: ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT,
      quadQueryRoutes: 96,
      query8FullSelfRoutes: 0,
      query8SlidingRoutes: 96,
      query8CrossRoutes: 192,
      query8OtherRoutes: 0,
      unintendedQuadQueryRoutes: 0,
    });
    expect(profile.fullSelfRouteIds).toHaveLength(96);
    expect(profile.slidingSelfRouteIds).toHaveLength(96);
    expect(profile.crossRouteIds).toHaveLength(192);
  });

  it("reconciles all 384 short-product routes with zero unintended quad", () => {
    for (const [queryTokens, conditionTokens] of [
      [125, 91],
      [138, 93],
      [150, 97],
      [375, 98],
      [3_000, 98],
    ] as const) {
      const routes = completeRoutes(queryTokens, conditionTokens);
      expect(finalizeAceOpt0070ProductionAttentionRoutes(
        queryTokens,
        conditionTokens,
        routes,
      )).toMatchObject({
        ownerMode: "non-m2250-query8",
        expectedQueryTokens: queryTokens,
        expectedConditionTokens: conditionTokens,
        routeCount: 384,
        quadQueryRoutes: 0,
        query8FullSelfRoutes: 96,
        query8SlidingRoutes: 96,
        query8CrossRoutes: 192,
        query8OtherRoutes: 0,
        unintendedQuadQueryRoutes: 0,
      });
    }
  });

  it("fails closed on missing, duplicate, or forged route inventory", () => {
    const routes = completeRoutes(150, 97);
    expect(() => finalizeAceOpt0070ProductionAttentionRoutes(
      150,
      97,
      routes.slice(1),
    )).toThrow(/missing|incomplete/);
    expect(() => finalizeAceOpt0070ProductionAttentionRoutes(
      150,
      97,
      [...routes, routes[0]!],
    )).toThrow(/duplicated/);
    const forged = routes.map((route, index) => index === 0
      ? Object.freeze({
          ...route,
          route: "quad-query32-full-self" as const,
          kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
        })
      : route);
    expect(() => finalizeAceOpt0070ProductionAttentionRoutes(
      150,
      97,
      forged,
    )).toThrow(/missing/);
  });

  it("constructs exactly one geometry-selected owner and rejects diagnostics", async () => {
    const device = fakeFixed32Device();
    const short = AceOpt0070ProductionAttentionKernel.create(
      device,
      "reference-bf16",
      configuration(150, 97),
    );
    expect(short.ownerMode).toBe("non-m2250-query8");
    await expect(short.createDispatch(
      "ace-dit-eval-0-layer-1-self-full-attention",
      fullShape(150),
      { opt0062Identity: {} } as unknown as AceAttentionBindings,
    )).rejects.toThrow(/rejects diagnostic/);
    short.destroy();
    short.destroy();
    await expect(short.createDispatch(
      "ace-dit-eval-0-layer-1-self-full-attention",
      fullShape(150),
      {} as AceAttentionBindings,
    )).rejects.toThrow(/destroyed/);

    const m2250 = AceOpt0070ProductionAttentionKernel.create(
      device,
      "reference-bf16",
      configuration(2_250, 98),
    );
    expect(m2250.ownerMode).toBe("exact-m2250-opt0062-quad");
    m2250.destroy();
    expect(() => AceOpt0070ProductionAttentionKernel.create(
      device,
      "reference-bf16",
      {
        ...configuration(150, 97),
        runtimeProfileId: "opt-0062-fixed32-quad-query32-full-self-v1",
      } as unknown as AceOpt0070ProductionAttentionConfiguration,
    )).toThrow(/public fixed32 profile/);
  });

  it("delegates a short full-self route to the real query8 owner", async () => {
    const device = fakeFixed32Device();
    const kernel = AceOpt0070ProductionAttentionKernel.create(
      device,
      "reference-bf16",
      configuration(150, 97),
    );
    const label = "ace-dit-eval-0-layer-1-self-full-attention";
    const dispatch = await kernel.createDispatch(
      label,
      fullShape(150),
      bindingsFor(150, 150),
    );
    expect(dispatch).toMatchObject({
      label,
      backend: "fixed32-subgroup-query8",
    });
    await expect(kernel.createDispatch(
      label,
      fullShape(150),
      bindingsFor(150, 150),
    )).rejects.toThrow(/duplicated/);
    expect(() => kernel.finalizeRoutes()).toThrow(/missing|incomplete/);
    kernel.destroy();
    await Promise.resolve();
    expect(device.buffers).toHaveLength(1);
    expect(device.buffers[0]!.destroy).toHaveBeenCalledTimes(1);
  });
});

function completeRoutes(
  queryTokens: number,
  conditionTokens: number,
): readonly AceOpt0070ProductionAttentionRouteDecision[] {
  const routes: AceOpt0070ProductionAttentionRouteDecision[] = [];
  for (let evaluation = 0; evaluation < 8; evaluation += 1) {
    for (let layer = 0; layer < 24; layer += 1) {
      const full = ACE_DIT_LAYER_TYPES[layer] === "full_attention";
      routes.push(selectAceOpt0070ProductionAttentionRoute(
        `ace-dit-eval-${evaluation}-layer-${layer}-self-${
          full ? "full" : "sliding"
        }-attention`,
        full ? fullShape(queryTokens) : slidingShape(queryTokens),
        queryTokens,
        conditionTokens,
      ));
      routes.push(selectAceOpt0070ProductionAttentionRoute(
        `ace-dit-eval-${evaluation}-layer-${layer}-cross-attention`,
        crossShape(queryTokens, conditionTokens),
        queryTokens,
        conditionTokens,
      ));
    }
  }
  return Object.freeze(routes);
}

function fullShape(tokens: number): AceAttentionShape {
  return Object.freeze({
    batch: 1,
    queryHeads: 16,
    keyValueHeads: 8,
    queryTokens: tokens,
    keyValueTokens: tokens,
    headDimension: 128,
    mode: "full",
  });
}

function slidingShape(tokens: number): AceAttentionShape {
  return Object.freeze({
    ...fullShape(tokens),
    mode: "sliding",
    slidingRadius: 128,
  });
}

function crossShape(
  queryTokens: number,
  conditionTokens: number,
): AceAttentionShape {
  return Object.freeze({
    ...fullShape(queryTokens),
    keyValueTokens: conditionTokens,
  });
}

function configuration(
  queryTokens: number,
  conditionTokens: number,
): AceOpt0070ProductionAttentionConfiguration {
  return Object.freeze({
    backend: "opt-0070-fixed32-quad-query32-full-self-production",
    runtimeProfileId:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    capability: CAPABILITY,
    expectedQueryTokens: queryTokens,
    expectedConditionTokens: conditionTokens,
  });
}

function fakeFixed32Device(): GPUDevice & {
  readonly buffers: Array<ReturnType<typeof fakeMappedBuffer>>;
} {
  const buffers: Array<ReturnType<typeof fakeMappedBuffer>> = [];
  return {
    features: new Set<GPUFeatureName>(["subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16 * 1_024,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createShaderModule: vi.fn(() => ({ label: "attention-module" })),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({ label: "attention-layout" })),
    })),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = fakeMappedBuffer(Number(descriptor.size));
      buffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({ label: "attention-bindings" })),
    buffers,
  } as unknown as GPUDevice & {
    readonly buffers: Array<ReturnType<typeof fakeMappedBuffer>>;
  };
}

function bindingsFor(
  queryTokens: number,
  keyValueTokens: number,
): AceAttentionBindings {
  return Object.freeze({
    query: fakeBinding(16 * queryTokens * 128 * 4),
    key: fakeBinding(8 * keyValueTokens * 128 * 4),
    value: fakeBinding(8 * keyValueTokens * 128 * 4),
    validLengths: fakeBinding(8),
    output: fakeBinding(16 * queryTokens * 128 * 4),
  });
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
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

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });
