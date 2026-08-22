import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../dit-attention-profile.js";
import { createAceScopedBuffers } from "../scoped-buffer-allocation.js";
import { ACE_DIT_LAYER_TYPES } from "../../model/graph-contract.js";
import { aceSha256Hex } from "../../model/sha256.js";
import {
  AceCorrectnessAttentionKernel,
  type AceAttentionBindings,
  type AceAttentionDispatch,
  type AceAttentionOutputRange,
  type AceAttentionRuntimeConfiguration,
  type AceAttentionShape,
  type AceOpt0062QuadQueryAttentionDispatch,
  planAceAttention,
} from "./attention.js";
import {
  ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES,
  ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE,
  aceOpt0061AttentionWgsl,
  planAceOpt0061Attention,
  type AceOpt0061AttentionPlan,
} from "./attention-multi-query.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import { ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE } from "./gemm.js";

export const ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256 =
  "7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6" as const;
export const ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES = 96 as const;
export const ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES = 96 as const;
export const ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES = 192 as const;
export const ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS = 4_608_000 as const;
export const ACE_OPT_0062_IDENTITY_COUNTER_WORDS_PER_ROUTE = 6 as const;
export const ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES = 256 as const;
export const ACE_OPT_0062_IDENTITY_COUNTER_BYTES =
  ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES *
  ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES;

const RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;
const DENOISING_EVALUATIONS = 8;
const AUTHENTICATED_QUAD_QUERY_SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});
const AUTHENTICATED_QUAD_QUERY_WGSL = aceOpt0061AttentionWgsl(
  AUTHENTICATED_QUAD_QUERY_SHAPE,
  4,
);
const FULL_LAYER_INDICES = Object.freeze(
  ACE_DIT_LAYER_TYPES.flatMap((type, layer) =>
    type === "full_attention" ? [layer] : []
  ),
);
const EXPECTED_QUAD_ROUTE_IDS = Object.freeze(
  Array.from({ length: DENOISING_EVALUATIONS }, (_, evaluation) =>
    FULL_LAYER_INDICES.map((layer) =>
      `ace-dit-eval-${evaluation}-layer-${layer}-self-full-attention`
    )
  ).flat(),
);
const EXPECTED_QUAD_ROUTE_SET = new Set(EXPECTED_QUAD_ROUTE_IDS);
const PRODUCTION_SELF_FULL_LABEL =
  /^ace-dit-eval-([0-7])-layer-(\d+)-self-full-attention$/u;
const PRODUCTION_SELF_SLIDING_LABEL =
  /^ace-dit-eval-([0-7])-layer-(\d+)-self-sliding-attention$/u;
const PRODUCTION_CROSS_LABEL =
  /^ace-dit-eval-([0-7])-layer-(\d+)-cross-attention$/u;

export type AceOpt0062AttentionRoute =
  | "quad-query32-full-self"
  | "query8-self-sliding"
  | "query8-cross"
  | "query8-other";

export interface AceOpt0062AttentionRouteDecision {
  readonly route: AceOpt0062AttentionRoute;
  readonly label: string;
  readonly kernelId:
    | typeof ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID
    | "fixed32-subgroup-query8";
}

export interface AceOpt0062AttentionPlan extends AceOpt0061AttentionPlan {
  readonly backend: "opt-0062-fixed32-quad-query32-full-self";
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceAttentionOutputRange[];
  readonly maximumMultiplyAddsPerWorkgroup: number;
  readonly totalMultiplyAdds: number;
}

export interface AceOpt0062AttentionRouteProfile {
  readonly runtimeProfileId:
    typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  readonly kernelSetId:
    typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID;
  readonly quadQueryKernelId:
    typeof ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID;
  readonly authenticatedWgslSha256:
    typeof ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256;
  readonly expectedQuadQueryRoutes:
    typeof ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES;
  readonly quadQueryRoutes: typeof ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES;
  readonly query8SlidingRoutes:
    typeof ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES;
  readonly query8CrossRoutes: typeof ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES;
  readonly query8OtherRoutes: 0;
  readonly unintendedQuadQueryRoutes: 0;
  readonly uniqueQuadQueryRouteIds: readonly string[];
}

type AceOpt0062AttentionConfiguration = Extract<
  AceAttentionRuntimeConfiguration,
  { backend: "opt-0062-fixed32-quad-query32-full-self" }
>;

interface CompiledOpt0062Attention {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly rangeParameters: GPUBuffer;
  readonly rangeParameterStride: number;
  destroy(): void;
}

interface CompiledOpt0062Identity {
  readonly initializePipeline: GPUComputePipeline;
  readonly initializeBindGroupLayout: GPUBindGroupLayout;
  readonly comparePipeline: GPUComputePipeline;
  readonly compareBindGroupLayout: GPUBindGroupLayout;
  readonly workgroupCount: number;
}

/** Exact route/shape selector; a malformed full-self operation never falls back. */
export function selectAceOpt0062AttentionRoute(
  label: string,
  shape: AceAttentionShape,
): AceOpt0062AttentionRouteDecision {
  const plan = planAceAttention(shape);
  if (label.endsWith("-self-full-attention")) {
    const match = PRODUCTION_SELF_FULL_LABEL.exec(label);
    const layer = match === null ? -1 : Number(match[2]);
    if (
      match === null ||
      !EXPECTED_QUAD_ROUTE_SET.has(label) ||
      ACE_DIT_LAYER_TYPES[layer] !== "full_attention"
    ) {
      throw new Error(
        `OPT-0062 full-self attention label is not an exact production route: ${label}`,
      );
    }
    try {
      planAceOpt0061Attention(plan, 4);
    } catch (error) {
      throw new Error(
        `OPT-0062 full-self attention shape changed for ${label}`,
        { cause: error },
      );
    }
    return Object.freeze({
      route: "quad-query32-full-self",
      label,
      kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
    });
  }
  if (PRODUCTION_SELF_SLIDING_LABEL.test(label)) {
    return Object.freeze({
      route: "query8-self-sliding",
      label,
      kernelId: "fixed32-subgroup-query8",
    });
  }
  if (PRODUCTION_CROSS_LABEL.test(label)) {
    return Object.freeze({
      route: "query8-cross",
      label,
      kernelId: "fixed32-subgroup-query8",
    });
  }
  return Object.freeze({
    route: "query8-other",
    label,
    kernelId: "fixed32-subgroup-query8",
  });
}

/** Production scheduler ranges for the qualified four-stream OPT-0061 owner. */
export function planAceOpt0062Attention(
  shape: AceAttentionShape,
): AceOpt0062AttentionPlan {
  const primitive = planAceOpt0061Attention(shape, 4);
  const maximumMultiplyAddsPerWorkgroup =
    primitive.queriesPerWorkgroup *
    primitive.shape.keyValueTokens *
    primitive.shape.headDimension;
  const workgroupsPerRange = Math.max(
    1,
    Math.floor(
      ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE /
        maximumMultiplyAddsPerWorkgroup,
    ),
  );
  const outputRanges: AceAttentionOutputRange[] = [];
  let firstOutput = 0;
  for (
    let firstWorkgroup = 0;
    firstWorkgroup < primitive.workgroupCount;
    firstWorkgroup += workgroupsPerRange
  ) {
    const workgroupCount = Math.min(
      workgroupsPerRange,
      primitive.workgroupCount - firstWorkgroup,
    );
    const outputCount = quadQueryOutputsInWorkgroupRange(
      primitive,
      firstWorkgroup,
      workgroupCount,
    );
    const multiplyAdds = outputCount * primitive.shape.keyValueTokens;
    outputRanges.push(Object.freeze({
      firstOutput,
      outputCount,
      firstWorkgroup,
      workgroupCount,
      multiplyAdds,
    }));
    firstOutput += outputCount;
  }
  if (
    firstOutput !== primitive.outputElements ||
    outputRanges.some((range) =>
      range.multiplyAdds > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
    )
  ) {
    throw new Error("OPT-0062 cooperative range accounting changed");
  }
  return Object.freeze({
    ...primitive,
    backend: "opt-0062-fixed32-quad-query32-full-self",
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
    maximumMultiplyAddsPerWorkgroup,
    totalMultiplyAdds: firstOutput * primitive.shape.keyValueTokens,
  });
}

/**
 * Attention-only OPT-0062 owner. Query8 remains the owner of sliding/cross;
 * the quad-query pipeline exists only for exact production full-self routes.
 */
export class AceOpt0062QuadQueryAttentionKernel {
  readonly configuration: AceOpt0062AttentionConfiguration;
  readonly modelProfile: "reference-bf16";

  private readonly query8: AceCorrectnessAttentionKernel;
  private compiled: Promise<CompiledOpt0062Attention> | undefined;
  private identityCompiled: Promise<CompiledOpt0062Identity> | undefined;
  private readonly quadRouteIds = new Set<string>();
  private query8SlidingRoutes = 0;
  private query8CrossRoutes = 0;
  private query8OtherRoutes = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    configuration: AceOpt0062AttentionConfiguration,
    query8: AceCorrectnessAttentionKernel,
  ) {
    this.configuration = Object.freeze(configuration);
    this.modelProfile = "reference-bf16";
    this.query8 = query8;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    configuration: AceOpt0062AttentionConfiguration,
  ): AceOpt0062QuadQueryAttentionKernel {
    if (
      modelProfile !== "reference-bf16" ||
      configuration.runtimeProfileId !==
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
    ) {
      throw new Error(
        "OPT-0062 requires its authenticated reference-BF16 attention profile",
      );
    }
    const query8 = AceCorrectnessAttentionKernel.create(
      device,
      modelProfile,
      Object.freeze({
        backend: "fixed32-subgroup-query8" as const,
        capability: configuration.capability,
      }),
    );
    return new AceOpt0062QuadQueryAttentionKernel(
      device,
      configuration,
      query8,
    );
  }

  async createDispatch(
    label: string,
    shape: AceAttentionShape,
    bindings: AceAttentionBindings,
  ): Promise<AceAttentionDispatch> {
    this.requireLive();
    const route = selectAceOpt0062AttentionRoute(label, shape);
    if (route.route !== "quad-query32-full-self") {
      const dispatch = await this.query8.createDispatch(label, shape, bindings);
      this.requireLive(" while compiling");
      if (dispatch.backend !== "fixed32-subgroup-query8") {
        throw new Error(`OPT-0062 query8 route ${label} selected ${dispatch.backend}`);
      }
      if (route.route === "query8-self-sliding") this.query8SlidingRoutes += 1;
      else if (route.route === "query8-cross") this.query8CrossRoutes += 1;
      else this.query8OtherRoutes += 1;
      return dispatch;
    }
    if (this.quadRouteIds.has(label)) {
      throw new Error(`OPT-0062 duplicate full-self route ${label}`);
    }
    const plan = planAceOpt0062Attention(shape);
    requireOpt0062Bindings(bindings, plan, label);
    const identity = bindings.opt0062Identity;
    const [compiled, identityCompiled, query8Oracle] = await Promise.all([
      this.pipelineFor(plan),
      identity === undefined ? undefined : this.identityPipelineFor(),
      identity === undefined
        ? undefined
        : this.query8.createDispatch(
            `${label}-opt0062-query8-oracle`,
            shape,
            {
              query: bindings.query,
              key: bindings.key,
              value: bindings.value,
              validLengths: bindings.validLengths,
              output: identity.oracleOutput,
            },
          ),
    ]);
    this.requireLive(" while compiling");
    if (
      query8Oracle !== undefined &&
      (query8Oracle.backend !== "fixed32-subgroup-query8" ||
        query8Oracle.rangeCount !== plan.outputRangeCount)
    ) {
      throw new Error(`OPT-0062 query8 identity owner changed for ${label}`);
    }
    const commonEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: bindings.query },
      { binding: 1, resource: bindings.key },
      { binding: 2, resource: bindings.value },
      { binding: 3, resource: bindings.validLengths },
      { binding: 4, resource: bindings.output },
    ];
    const bindGroups = plan.outputRanges.map((_, rangeIndex) =>
      this.device.createBindGroup({
        label: `${label}-opt0062-range-${rangeIndex}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          ...commonEntries,
          {
            binding: 7,
            resource: {
              buffer: compiled.rangeParameters,
              offset: rangeIndex * compiled.rangeParameterStride,
              size: RANGE_PARAMETER_BYTES,
            },
          },
        ],
      })
    );
    const initializeBindGroup = identity === undefined
      ? undefined
      : this.device.createBindGroup({
          label: `${label}-opt0062-identity-initialize-bindings`,
          layout: identityCompiled!.initializeBindGroupLayout,
          entries: [
            { binding: 0, resource: identity.oracleOutput },
            { binding: 1, resource: bindings.output },
          ],
        });
    const compareBindGroup = identity === undefined
      ? undefined
      : this.device.createBindGroup({
          label: `${label}-opt0062-identity-compare-bindings`,
          layout: identityCompiled!.compareBindGroupLayout,
          entries: [
            { binding: 0, resource: identity.oracleOutput },
            { binding: 1, resource: bindings.output },
            {
              binding: 2,
              resource: {
                buffer: identity.counters.buffer,
                offset: (identity.counters.offset ?? 0) +
                  identity.routeIndex *
                    ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES,
                size: ACE_OPT_0062_IDENTITY_COUNTER_WORDS_PER_ROUTE *
                  Uint32Array.BYTES_PER_ELEMENT,
              },
            },
          ],
        });
    const encodeInitialize = (pass: GPUComputePassEncoder): void => {
      if (identityCompiled === undefined || initializeBindGroup === undefined) {
        return;
      }
      pass.setPipeline(identityCompiled.initializePipeline);
      pass.setBindGroup(0, initializeBindGroup);
      pass.dispatchWorkgroups(identityCompiled.workgroupCount, 1, 1);
    };
    const encodeCompare = (pass: GPUComputePassEncoder): void => {
      if (identityCompiled === undefined || compareBindGroup === undefined) {
        return;
      }
      pass.setPipeline(identityCompiled.comparePipeline);
      pass.setBindGroup(0, compareBindGroup);
      pass.dispatchWorkgroups(identityCompiled.workgroupCount, 1, 1);
    };
    const encodeQuadRange = (
      pass: GPUComputePassEncoder,
      rangeIndex: number,
    ): void => {
      const range = plan.outputRanges[rangeIndex]!;
      pass.setPipeline(compiled.pipeline);
      pass.setBindGroup(0, bindGroups[rangeIndex]!);
      pass.dispatchWorkgroups(range.workgroupCount, 1, 1);
    };
    this.quadRouteIds.add(label);
    const dispatch: AceOpt0062QuadQueryAttentionDispatch = Object.freeze({
      label,
      backend: "opt-0062-fixed32-quad-query32-full-self",
      kernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} OPT-0062 attention range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        if (rangeIndex === 0) encodeInitialize(pass);
        query8Oracle?.encodeRange(pass, rangeIndex);
        encodeQuadRange(pass, rangeIndex);
        if (rangeIndex + 1 === plan.outputRangeCount) encodeCompare(pass);
      },
      encode(pass: GPUComputePassEncoder): void {
        encodeInitialize(pass);
        query8Oracle?.encode(pass);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          encodeQuadRange(pass, index);
        }
        encodeCompare(pass);
      },
    });
    return dispatch;
  }

  finalizeRoutes(): AceOpt0062AttentionRouteProfile {
    this.requireLive(" while finalizing routes");
    const uniqueQuadQueryRouteIds = Object.freeze(
      [...this.quadRouteIds].sort(routeIdOrder),
    );
    if (
      uniqueQuadQueryRouteIds.length !==
        ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES ||
      uniqueQuadQueryRouteIds.some((id, index) =>
        id !== EXPECTED_QUAD_ROUTE_IDS[index]
      ) ||
      this.query8SlidingRoutes !==
        ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES ||
      this.query8CrossRoutes !== ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES ||
      this.query8OtherRoutes !== 0
    ) {
      throw new Error(
        "OPT-0062 production attention route inventory is incomplete or unintended",
      );
    }
    return Object.freeze({
      runtimeProfileId:
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      quadQueryKernelId: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
      authenticatedWgslSha256: ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256,
      expectedQuadQueryRoutes: ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
      quadQueryRoutes: ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
      query8SlidingRoutes: ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES,
      query8CrossRoutes: ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES,
      query8OtherRoutes: 0,
      unintendedQuadQueryRoutes: 0,
      uniqueQuadQueryRouteIds,
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.query8.destroy();
    if (this.compiled !== undefined) {
      void this.compiled.then(
        (resources) => resources.destroy(),
        () => undefined,
      );
    }
    this.compiled = undefined;
    this.identityCompiled = undefined;
  }

  private pipelineFor(
    plan: AceOpt0062AttentionPlan,
  ): Promise<CompiledOpt0062Attention> {
    if (this.compiled !== undefined) return this.compiled;
    const created = compileAceOpt0062Attention(this.device, plan);
    this.compiled = created;
    void created.catch(() => {
      if (this.compiled === created) this.compiled = undefined;
    });
    return created;
  }

  private identityPipelineFor(): Promise<CompiledOpt0062Identity> {
    if (this.identityCompiled !== undefined) return this.identityCompiled;
    const created = compileAceOpt0062Identity(this.device);
    this.identityCompiled = created;
    void created.catch(() => {
      if (this.identityCompiled === created) this.identityCompiled = undefined;
    });
    return created;
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`OPT-0062 attention kernel was destroyed${suffix}`);
    }
  }
}

function quadQueryOutputsInWorkgroupRange(
  plan: AceOpt0061AttentionPlan,
  firstWorkgroup: number,
  workgroupCount: number,
): number {
  let outputCount = 0;
  for (
    let workgroup = firstWorkgroup;
    workgroup < firstWorkgroup + workgroupCount;
    workgroup += 1
  ) {
    const withinBatch = workgroup %
      (plan.shape.keyValueHeads * plan.queryTokenTiles);
    const queryTile = withinBatch % plan.queryTokenTiles;
    const validTokens = Math.max(0, Math.min(
      plan.queryTokensPerTile,
      plan.shape.queryTokens - queryTile * plan.queryTokensPerTile,
    ));
    outputCount += validTokens *
      plan.shape.queryHeadsPerKeyValueHead *
      plan.shape.headDimension;
  }
  return outputCount;
}

async function compileAceOpt0062Attention(
  device: GPUDevice,
  plan: AceOpt0062AttentionPlan,
): Promise<CompiledOpt0062Attention> {
  const label = "ace-opt-0062-fixed32-quad-query32-full-self";
  const module = device.createShaderModule({
    label: `${label}-module`,
    code: AUTHENTICATED_QUAD_QUERY_WGSL,
  });
  const pipeline = await device.createComputePipelineAsync({
    label: `${label}-pipeline`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const rangeParameterStride = Math.max(
    MINIMUM_UNIFORM_STRIDE,
    device.limits.minUniformBufferOffsetAlignment ?? MINIMUM_UNIFORM_STRIDE,
  );
  const allocated = await createAceScopedBuffers(
    device,
    [{
      label: `${label}-range-parameters`,
      size: Math.max(
        rangeParameterStride,
        plan.outputRangeCount * rangeParameterStride,
      ),
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }],
    "OPT-0062 attention range parameters",
  );
  const rangeParameters = allocated[0];
  if (rangeParameters === undefined) {
    throw new Error("OPT-0062 attention range allocation returned no buffer");
  }
  try {
    const mapped = rangeParameters.getMappedRange();
    for (let index = 0; index < plan.outputRanges.length; index += 1) {
      new Uint32Array(
        mapped,
        index * rangeParameterStride,
        RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
      )[0] = plan.outputRanges[index]!.firstWorkgroup;
    }
    rangeParameters.unmap();
    let destroyed = false;
    return Object.freeze({
      pipeline,
      bindGroupLayout: pipeline.getBindGroupLayout(0),
      rangeParameters,
      rangeParameterStride,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        rangeParameters.destroy();
      },
    });
  } catch (error) {
    rangeParameters.destroy();
    throw error;
  }
}

async function compileAceOpt0062Identity(
  device: GPUDevice,
): Promise<CompiledOpt0062Identity> {
  const label = "ace-opt-0062-query8-quad-raw-u32-identity";
  const code = /* wgsl */ `
const OUTPUT_ELEMENTS: u32 = ${ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS}u;
const CONTROL_CANARY: u32 = 0x7fc12345u;
const CANDIDATE_CANARY: u32 = 0x7fc54321u;

@group(0) @binding(0) var<storage, read_write> control: array<u32>;
@group(0) @binding(1) var<storage, read_write> candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

fn non_finite(word: u32) -> bool {
  return (word & 0x7f800000u) == 0x7f800000u;
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= OUTPUT_ELEMENTS) { return; }
  control[id.x] = CONTROL_CANARY;
  candidate[id.x] = CANDIDATE_CANARY;
}

@compute @workgroup_size(256)
fn compare(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= OUTPUT_ELEMENTS) { return; }
  let expected = control[id.x];
  let actual = candidate[id.x];
  atomicAdd(&counters[0], 1u);
  if (expected != actual) { atomicAdd(&counters[1], 1u); }
  if (non_finite(expected)) { atomicAdd(&counters[2], 1u); }
  if (non_finite(actual)) { atomicAdd(&counters[3], 1u); }
  if (expected == CONTROL_CANARY) { atomicAdd(&counters[4], 1u); }
  if (actual == CANDIDATE_CANARY) { atomicAdd(&counters[5], 1u); }
}
`;
  const module = device.createShaderModule({ label: `${label}-module`, code });
  const [initializePipeline, comparePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: `${label}-initialize-pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "initialize" },
    }),
    device.createComputePipelineAsync({
      label: `${label}-compare-pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "compare" },
    }),
  ]);
  return Object.freeze({
    initializePipeline,
    initializeBindGroupLayout: initializePipeline.getBindGroupLayout(0),
    comparePipeline,
    compareBindGroupLayout: comparePipeline.getBindGroupLayout(0),
    workgroupCount: Math.ceil(
      ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS /
        ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE,
    ),
  });
}

function requireOpt0062Bindings(
  bindings: AceAttentionBindings,
  plan: AceOpt0062AttentionPlan,
  label: string,
): void {
  const elementBytes = Float32Array.BYTES_PER_ELEMENT;
  requireBindingBytes(
    bindings.query,
    plan.shape.queryElements * elementBytes,
    `${label} query`,
  );
  requireBindingBytes(
    bindings.key,
    plan.shape.keyValueElements * elementBytes,
    `${label} key`,
  );
  requireBindingBytes(
    bindings.value,
    plan.shape.keyValueElements * elementBytes,
    `${label} value`,
  );
  requireBindingBytes(
    bindings.output,
    plan.shape.outputElements * elementBytes,
    `${label} output`,
  );
  requireBindingBytes(bindings.validLengths, plan.shape.batch * 2 * 4, `${label} lengths`);
  if (bindings.keyValidity !== undefined || bindings.queryPositions !== undefined) {
    throw new TypeError("OPT-0062 full-self attention cannot bind a mask");
  }
  requireAceDisjointOutput(
    bindings.output,
    [bindings.query, bindings.key, bindings.value, bindings.validLengths],
    label,
  );
  const identity = bindings.opt0062Identity;
  if (identity !== undefined) {
    if (
      !Number.isSafeInteger(identity.routeIndex) ||
      identity.routeIndex < 0 ||
      identity.routeIndex >= ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES ||
      EXPECTED_QUAD_ROUTE_IDS[identity.routeIndex] !== label
    ) {
      throw new RangeError(
        `OPT-0062 identity route index does not own ${label}`,
      );
    }
    requireBindingBytes(
      identity.oracleOutput,
      plan.shape.outputElements * elementBytes,
      `${label} query8 identity output`,
    );
    requireBindingBytes(
      identity.counters,
      ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
      `${label} identity counters`,
    );
    requireAceDisjointOutput(
      identity.oracleOutput,
      [
        bindings.query,
        bindings.key,
        bindings.value,
        bindings.validLengths,
        bindings.output,
        identity.counters,
      ],
      `${label} query8 identity`,
    );
    requireAceDisjointOutput(
      identity.counters,
      [
        bindings.query,
        bindings.key,
        bindings.value,
        bindings.validLengths,
        bindings.output,
        identity.oracleOutput,
      ],
      `${label} identity counters`,
    );
  }
}

function requireBindingBytes(
  binding: GPUBufferBinding,
  required: number,
  label: string,
): void {
  const offset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - offset;
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(available) ||
    available < required ||
    offset + available > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${required} bytes`);
  }
}

function routeIdOrder(left: string, right: string): number {
  const leftMatch = PRODUCTION_SELF_FULL_LABEL.exec(left)!;
  const rightMatch = PRODUCTION_SELF_FULL_LABEL.exec(right)!;
  const leftEvaluation = Number(leftMatch[1]);
  const rightEvaluation = Number(rightMatch[1]);
  return leftEvaluation - rightEvaluation ||
    Number(leftMatch[2]) - Number(rightMatch[2]);
}

if (
  FULL_LAYER_INDICES.length !== 12 ||
  EXPECTED_QUAD_ROUTE_IDS.length !== ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES ||
  ACE_OPT_0062_IDENTITY_COUNTER_BYTES !== 24_576 ||
  AUTHENTICATED_QUAD_QUERY_SHAPE.queryHeads *
      AUTHENTICATED_QUAD_QUERY_SHAPE.queryTokens *
      AUTHENTICATED_QUAD_QUERY_SHAPE.headDimension !==
    ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS ||
  ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE !== 256 ||
  ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES !== 1_024 ||
  aceSha256Hex(new TextEncoder().encode(AUTHENTICATED_QUAD_QUERY_WGSL)) !==
    ACE_OPT_0062_QUAD_QUERY_WGSL_SHA256
) {
  throw new Error("OPT-0062 frozen production attention identity changed");
}
