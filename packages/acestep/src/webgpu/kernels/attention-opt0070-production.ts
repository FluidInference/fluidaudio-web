import { ACE_DIT_LAYER_TYPES } from "../../model/graph-contract.js";
import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../dit-attention-profile.js";
import {
  AceCorrectnessAttentionKernel,
  planAceAttention,
  type AceAttentionBindings,
  type AceAttentionDispatch,
  type AceAttentionRuntimeConfiguration,
  type AceAttentionShape,
} from "./attention.js";
import {
  ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
  ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES,
  ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES,
  ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
  AceOpt0062QuadQueryAttentionKernel,
  type AceOpt0062AttentionRouteProfile,
} from "./attention-quad-query-production.js";

export const ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT = 384 as const;
export const ACE_OPT_0070_PRODUCTION_FULL_SELF_ROUTE_COUNT = 96 as const;
export const ACE_OPT_0070_PRODUCTION_SLIDING_ROUTE_COUNT = 96 as const;
export const ACE_OPT_0070_PRODUCTION_CROSS_ROUTE_COUNT = 192 as const;

const FIXED32_QUERY8_KERNEL_ID = "fixed32-subgroup-query8" as const;
const FULL_SELF_SUFFIX = "self-full-attention" as const;
const SLIDING_SELF_SUFFIX = "self-sliding-attention" as const;
const CROSS_SUFFIX = "cross-attention" as const;
const ROUTE_PATTERN =
  /^ace-dit-eval-([0-7])-layer-(0|[1-9]|1[0-9]|2[0-3])-(self-full-attention|self-sliding-attention|cross-attention)$/u;

const FULL_LAYERS = Object.freeze(ACE_DIT_LAYER_TYPES.flatMap(
  (type, layer) => type === "full_attention" ? [layer] : [],
));
const SLIDING_LAYERS = Object.freeze(ACE_DIT_LAYER_TYPES.flatMap(
  (type, layer) => type === "sliding_attention" ? [layer] : [],
));
const EXPECTED_FULL_SELF_ROUTE_IDS = routeIds(FULL_LAYERS, FULL_SELF_SUFFIX);
const EXPECTED_SLIDING_SELF_ROUTE_IDS = routeIds(
  SLIDING_LAYERS,
  SLIDING_SELF_SUFFIX,
);
const EXPECTED_CROSS_ROUTE_IDS = routeIds(
  Object.freeze(Array.from({ length: 24 }, (_, layer) => layer)),
  CROSS_SUFFIX,
);

export type AceOpt0070ProductionAttentionConfiguration = Extract<
  AceAttentionRuntimeConfiguration,
  { backend: "opt-0070-fixed32-quad-query32-full-self-production" }
>;

export type AceOpt0070ProductionAttentionOwnerMode =
  | "exact-m2250-opt0062-quad"
  | "non-m2250-query8";

export type AceOpt0070ProductionAttentionRoute =
  | "quad-query32-full-self"
  | "query8-full-self"
  | "query8-self-sliding"
  | "query8-cross";

export interface AceOpt0070ProductionAttentionRouteDecision {
  readonly label: string;
  readonly evaluation: number;
  readonly layer: number;
  readonly route: AceOpt0070ProductionAttentionRoute;
  readonly kernelId:
    | typeof ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID
    | typeof FIXED32_QUERY8_KERNEL_ID;
}

export interface AceOpt0070ProductionAttentionRouteProfile {
  readonly schema: "ace-opt-0070-production-attention-routes-v1";
  readonly runtimeProfileId:
    typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  readonly kernelSetId:
    typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID;
  readonly ownerMode: AceOpt0070ProductionAttentionOwnerMode;
  readonly expectedQueryTokens: number;
  readonly expectedConditionTokens: number;
  readonly routeCount: typeof ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT;
  readonly quadQueryRoutes: number;
  readonly query8FullSelfRoutes: number;
  readonly query8SlidingRoutes:
    typeof ACE_OPT_0070_PRODUCTION_SLIDING_ROUTE_COUNT;
  readonly query8CrossRoutes:
    typeof ACE_OPT_0070_PRODUCTION_CROSS_ROUTE_COUNT;
  readonly query8OtherRoutes: 0;
  readonly unintendedQuadQueryRoutes: 0;
  readonly fullSelfRouteIds: readonly string[];
  readonly slidingSelfRouteIds: readonly string[];
  readonly crossRouteIds: readonly string[];
  readonly physicalOpt0062RouteProfile?: AceOpt0062AttentionRouteProfile;
}

type ProductionOwner =
  | AceOpt0062QuadQueryAttentionKernel
  | AceCorrectnessAttentionKernel;

/**
 * Public OPT-0070 owner. The measured OPT-0062 object owns the complete exact
 * M2250 graph. Every other graph owns exactly one ordinary query8 kernel.
 * Diagnostic OPT-0062 buffers are rejected at this public boundary.
 */
export class AceOpt0070ProductionAttentionKernel {
  readonly modelProfile = "reference-bf16" as const;
  readonly configuration: AceOpt0070ProductionAttentionConfiguration;
  readonly ownerMode: AceOpt0070ProductionAttentionOwnerMode;

  private readonly routes = new Map<
    string,
    AceOpt0070ProductionAttentionRouteDecision
  >();
  private readonly pendingRoutes = new Set<string>();
  private finalized = false;
  private destroyed = false;

  private constructor(
    configuration: AceOpt0070ProductionAttentionConfiguration,
    private readonly owner: ProductionOwner,
  ) {
    this.configuration = freezeConfiguration(configuration);
    this.ownerMode = resolveAceOpt0070ProductionAttentionOwnerMode(
      configuration.expectedQueryTokens,
    );
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    configuration: AceOpt0070ProductionAttentionConfiguration,
  ): AceOpt0070ProductionAttentionKernel {
    requireProductionConfiguration(modelProfile, configuration);
    const ownerMode = resolveAceOpt0070ProductionAttentionOwnerMode(
      configuration.expectedQueryTokens,
    );
    const owner = ownerMode === "exact-m2250-opt0062-quad"
      ? AceOpt0062QuadQueryAttentionKernel.create(
          device,
          modelProfile,
          Object.freeze({
            backend: "opt-0062-fixed32-quad-query32-full-self" as const,
            runtimeProfileId:
              ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
            capability: configuration.capability,
          }),
        )
      : AceCorrectnessAttentionKernel.create(
          device,
          modelProfile,
          Object.freeze({
            backend: "fixed32-subgroup-query8" as const,
            capability: configuration.capability,
          }),
        );
    return new AceOpt0070ProductionAttentionKernel(configuration, owner);
  }

  async createDispatch(
    label: string,
    shape: AceAttentionShape,
    bindings: AceAttentionBindings,
  ): Promise<AceAttentionDispatch> {
    this.requireMutable();
    if (bindings.opt0062Identity !== undefined) {
      throw new Error(
        "OPT-0070 production attention rejects diagnostic OPT-0062 identity buffers",
      );
    }
    const route = selectAceOpt0070ProductionAttentionRoute(
      label,
      shape,
      this.configuration.expectedQueryTokens,
      this.configuration.expectedConditionTokens,
    );
    if (this.routes.has(label) || this.pendingRoutes.has(label)) {
      throw new Error(`OPT-0070 production attention route ${label} duplicated`);
    }
    this.pendingRoutes.add(label);
    try {
      const dispatch = await this.owner.createDispatch(label, shape, bindings);
      this.requireMutable(" while compiling");
      requireDispatchOwner(route, dispatch);
      this.routes.set(label, route);
      return dispatch;
    } finally {
      this.pendingRoutes.delete(label);
    }
  }

  finalizeRoutes(): AceOpt0070ProductionAttentionRouteProfile {
    this.requireMutable(" while finalizing routes");
    if (this.pendingRoutes.size !== 0) {
      throw new Error(
        "OPT-0070 production attention cannot finalize pending routes",
      );
    }
    const publicProfile = finalizeAceOpt0070ProductionAttentionRoutes(
      this.configuration.expectedQueryTokens,
      this.configuration.expectedConditionTokens,
      [...this.routes.values()],
    );
    const physicalOpt0062RouteProfile =
      this.ownerMode === "exact-m2250-opt0062-quad"
        ? (this.owner as AceOpt0062QuadQueryAttentionKernel).finalizeRoutes()
        : undefined;
    if (
      physicalOpt0062RouteProfile !== undefined &&
      (physicalOpt0062RouteProfile.quadQueryRoutes !==
          ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES ||
        physicalOpt0062RouteProfile.query8SlidingRoutes !==
          ACE_OPT_0062_EXPECTED_QUERY8_SLIDING_ROUTES ||
        physicalOpt0062RouteProfile.query8CrossRoutes !==
          ACE_OPT_0062_EXPECTED_QUERY8_CROSS_ROUTES ||
        physicalOpt0062RouteProfile.query8OtherRoutes !== 0 ||
        physicalOpt0062RouteProfile.unintendedQuadQueryRoutes !== 0)
    ) {
      throw new Error("OPT-0070 physical OPT-0062 route inventory changed");
    }
    this.finalized = true;
    return Object.freeze({
      ...publicProfile,
      ...(physicalOpt0062RouteProfile === undefined
        ? {}
        : { physicalOpt0062RouteProfile }),
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.owner.destroy();
    this.pendingRoutes.clear();
    this.routes.clear();
  }

  private requireMutable(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`OPT-0070 production attention was destroyed${suffix}`);
    }
    if (this.finalized) {
      throw new Error(`OPT-0070 production attention routes were finalized${suffix}`);
    }
  }
}

export function resolveAceOpt0070ProductionAttentionOwnerMode(
  expectedQueryTokens: number,
): AceOpt0070ProductionAttentionOwnerMode {
  requirePositiveSafeInteger(expectedQueryTokens, "query tokens");
  return expectedQueryTokens ===
      ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY.exactQuadQueryTokens
    ? "exact-m2250-opt0062-quad"
    : "non-m2250-query8";
}

export function selectAceOpt0070ProductionAttentionRoute(
  label: string,
  shape: AceAttentionShape,
  expectedQueryTokens: number,
  expectedConditionTokens: number,
): AceOpt0070ProductionAttentionRouteDecision {
  requirePositiveSafeInteger(expectedQueryTokens, "query tokens");
  requirePositiveSafeInteger(expectedConditionTokens, "condition tokens");
  const match = ROUTE_PATTERN.exec(label);
  if (match === null) {
    throw new Error(`OPT-0070 rejected unregistered attention label ${label}`);
  }
  const evaluation = Number(match[1]);
  const layer = Number(match[2]);
  const suffix = match[3] as
    | typeof FULL_SELF_SUFFIX
    | typeof SLIDING_SELF_SUFFIX
    | typeof CROSS_SUFFIX;
  let plan: ReturnType<typeof planAceAttention>;
  try {
    plan = planAceAttention(shape);
  } catch (error) {
    throw new Error(`OPT-0070 rejected invalid attention shape for ${label}`, {
      cause: error,
    });
  }
  if (
    plan.batch !== 1 ||
    plan.queryHeads !== 16 ||
    plan.keyValueHeads !== 8 ||
    plan.queryTokens !== expectedQueryTokens ||
    plan.headDimension !== 128 ||
    plan.keyValidity !== "none"
  ) {
    throw shapeError(label, plan, expectedQueryTokens, expectedConditionTokens);
  }
  if (suffix === FULL_SELF_SUFFIX) {
    if (
      ACE_DIT_LAYER_TYPES[layer] !== "full_attention" ||
      plan.mode !== "full" ||
      plan.keyValueTokens !== expectedQueryTokens
    ) {
      throw shapeError(label, plan, expectedQueryTokens, expectedConditionTokens);
    }
    const quad = expectedQueryTokens ===
      ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY.exactQuadQueryTokens;
    return Object.freeze({
      label,
      evaluation,
      layer,
      route: quad
        ? "quad-query32-full-self" as const
        : "query8-full-self" as const,
      kernelId: quad
        ? ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID
        : FIXED32_QUERY8_KERNEL_ID,
    });
  }
  if (suffix === SLIDING_SELF_SUFFIX) {
    if (
      ACE_DIT_LAYER_TYPES[layer] !== "sliding_attention" ||
      plan.mode !== "sliding" ||
      plan.keyValueTokens !== expectedQueryTokens ||
      plan.slidingRadius !== 128
    ) {
      throw shapeError(label, plan, expectedQueryTokens, expectedConditionTokens);
    }
    return Object.freeze({
      label,
      evaluation,
      layer,
      route: "query8-self-sliding",
      kernelId: FIXED32_QUERY8_KERNEL_ID,
    });
  }
  if (
    plan.mode !== "full" ||
    plan.keyValueTokens !== expectedConditionTokens
  ) {
    throw shapeError(label, plan, expectedQueryTokens, expectedConditionTokens);
  }
  return Object.freeze({
    label,
    evaluation,
    layer,
    route: "query8-cross",
    kernelId: FIXED32_QUERY8_KERNEL_ID,
  });
}

export function finalizeAceOpt0070ProductionAttentionRoutes(
  expectedQueryTokens: number,
  expectedConditionTokens: number,
  routes: readonly AceOpt0070ProductionAttentionRouteDecision[],
): AceOpt0070ProductionAttentionRouteProfile {
  const ownerMode = resolveAceOpt0070ProductionAttentionOwnerMode(
    expectedQueryTokens,
  );
  requirePositiveSafeInteger(expectedConditionTokens, "condition tokens");
  const byLabel = new Map<string, AceOpt0070ProductionAttentionRouteDecision>();
  for (const route of routes) {
    if (byLabel.has(route.label)) {
      throw new Error(`OPT-0070 production attention route ${route.label} duplicated`);
    }
    byLabel.set(route.label, route);
  }
  const expectedFullRoute = ownerMode === "exact-m2250-opt0062-quad"
    ? "quad-query32-full-self" as const
    : "query8-full-self" as const;
  requireRouteInventory(
    byLabel,
    EXPECTED_FULL_SELF_ROUTE_IDS,
    expectedFullRoute,
    ownerMode === "exact-m2250-opt0062-quad"
      ? ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID
      : FIXED32_QUERY8_KERNEL_ID,
  );
  requireRouteInventory(
    byLabel,
    EXPECTED_SLIDING_SELF_ROUTE_IDS,
    "query8-self-sliding",
    FIXED32_QUERY8_KERNEL_ID,
  );
  requireRouteInventory(
    byLabel,
    EXPECTED_CROSS_ROUTE_IDS,
    "query8-cross",
    FIXED32_QUERY8_KERNEL_ID,
  );
  if (
    byLabel.size !== ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT ||
    routes.length !== ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT
  ) {
    throw new Error("OPT-0070 production attention route inventory is incomplete");
  }
  const quadQueryRoutes = ownerMode === "exact-m2250-opt0062-quad"
    ? ACE_OPT_0070_PRODUCTION_FULL_SELF_ROUTE_COUNT
    : 0;
  return Object.freeze({
    schema: "ace-opt-0070-production-attention-routes-v1",
    runtimeProfileId:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    ownerMode,
    expectedQueryTokens,
    expectedConditionTokens,
    routeCount: ACE_OPT_0070_PRODUCTION_ATTENTION_ROUTE_COUNT,
    quadQueryRoutes,
    query8FullSelfRoutes:
      ACE_OPT_0070_PRODUCTION_FULL_SELF_ROUTE_COUNT - quadQueryRoutes,
    query8SlidingRoutes: ACE_OPT_0070_PRODUCTION_SLIDING_ROUTE_COUNT,
    query8CrossRoutes: ACE_OPT_0070_PRODUCTION_CROSS_ROUTE_COUNT,
    query8OtherRoutes: 0,
    unintendedQuadQueryRoutes: 0,
    fullSelfRouteIds: EXPECTED_FULL_SELF_ROUTE_IDS,
    slidingSelfRouteIds: EXPECTED_SLIDING_SELF_ROUTE_IDS,
    crossRouteIds: EXPECTED_CROSS_ROUTE_IDS,
  });
}

function requireProductionConfiguration(
  modelProfile: AceModelProfileId,
  configuration: AceOpt0070ProductionAttentionConfiguration,
): void {
  if (
    modelProfile !== "reference-bf16" ||
    configuration.backend !==
      "opt-0070-fixed32-quad-query32-full-self-production" ||
    configuration.runtimeProfileId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    configuration.capability.subgroupMinSize !== 32 ||
    configuration.capability.subgroupMaxSize !== 32
  ) {
    throw new Error(
      "OPT-0070 production attention requires its public fixed32 profile",
    );
  }
  requirePositiveSafeInteger(
    configuration.expectedQueryTokens,
    "query tokens",
  );
  requirePositiveSafeInteger(
    configuration.expectedConditionTokens,
    "condition tokens",
  );
}

function freezeConfiguration(
  configuration: AceOpt0070ProductionAttentionConfiguration,
): AceOpt0070ProductionAttentionConfiguration {
  return Object.freeze({
    ...configuration,
    capability: Object.freeze({ ...configuration.capability }),
  });
}

function requireDispatchOwner(
  route: AceOpt0070ProductionAttentionRouteDecision,
  dispatch: AceAttentionDispatch,
): void {
  if (
    (route.route === "quad-query32-full-self" &&
      (dispatch.backend !== "opt-0062-fixed32-quad-query32-full-self" ||
        dispatch.kernelId !== ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID)) ||
    (route.route === "query8-full-self" &&
      dispatch.backend !== "fixed32-subgroup-query8") ||
    (route.route !== "quad-query32-full-self" &&
      dispatch.backend === "opt-0062-fixed32-quad-query32-full-self")
  ) {
    throw new Error(
      `OPT-0070 production attention route ${route.label} selected ${dispatch.backend}`,
    );
  }
}

function requireRouteInventory(
  routes: ReadonlyMap<string, AceOpt0070ProductionAttentionRouteDecision>,
  expectedIds: readonly string[],
  expectedRoute: AceOpt0070ProductionAttentionRoute,
  expectedKernelId:
    | typeof ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID
    | typeof FIXED32_QUERY8_KERNEL_ID,
): void {
  for (const label of expectedIds) {
    const route = routes.get(label);
    const match = ROUTE_PATTERN.exec(label)!;
    if (
      route === undefined ||
      route.evaluation !== Number(match[1]) ||
      route.layer !== Number(match[2]) ||
      route.route !== expectedRoute ||
      route.kernelId !== expectedKernelId
    ) {
      throw new Error(`OPT-0070 production attention route ${label} is missing`);
    }
  }
}

function routeIds(
  layers: readonly number[],
  suffix: typeof FULL_SELF_SUFFIX | typeof SLIDING_SELF_SUFFIX | typeof CROSS_SUFFIX,
): readonly string[] {
  return Object.freeze(Array.from({ length: 8 }, (_, evaluation) =>
    layers.map((layer) =>
      `ace-dit-eval-${evaluation}-layer-${layer}-${suffix}`
    )
  ).flat());
}

function shapeError(
  label: string,
  plan: ReturnType<typeof planAceAttention>,
  expectedQueryTokens: number,
  expectedConditionTokens: number,
): Error {
  return new Error(
    `OPT-0070 ${label} rejected B${plan.batch}/Hq${plan.queryHeads}/` +
      `Hkv${plan.keyValueHeads}/Q${plan.queryTokens}/KV${plan.keyValueTokens}/` +
      `D${plan.headDimension}/${plan.mode}; expected query ${expectedQueryTokens} ` +
      `and condition ${expectedConditionTokens}`,
  );
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`OPT-0070 ${label} must be a positive safe integer`);
  }
}
