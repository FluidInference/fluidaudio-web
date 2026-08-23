import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";
import {
  ACE_OPT_0037_DENSE_K4_KERNEL_ID,
  ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT,
  AceOpt0037DenseK4ProductionKernel,
} from "./dit-dense-fp16-k4-production.js";
import {
  ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
  ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT,
  AceOpt0056DenseK4ExactKernel,
} from "./dit-dense-fp16-k4-exact.js";

export const ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID =
  "ace-opt-0056-selective-k4-exact-down-fixed32-v1";
export const ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT = 216 as const;
export const ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT = 192 as const;
export const ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT = 24 as const;
export const ACE_OPT_0056_REPEATED_DENSE_DISPATCH_COUNT = 1_728 as const;

export type AceOpt0056DenseRouteOwner =
  | "opt-0032-k4-fp16-partials"
  | "opt-0056-k4-exact-fp32";

export interface AceOpt0056DenseDispatchRoute {
  readonly label: string;
  readonly evaluation: number;
  readonly layer: number;
  readonly operation: AceOpt0056DenseOperation;
  readonly rows: 2_250;
  readonly inner: 2_048 | 6_144;
  readonly columns: 1_024 | 2_048 | 6_144;
  readonly owner: AceOpt0056DenseRouteOwner;
  readonly kernelId:
    | typeof ACE_OPT_0037_DENSE_K4_KERNEL_ID
      | typeof ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID;
}

export interface AceOpt0056DenseRoute {
  readonly routeKey: string;
  readonly layer: number;
  readonly operation: AceOpt0056DenseOperation;
  readonly rows: 2_250;
  readonly inner: 2_048 | 6_144;
  readonly columns: 1_024 | 2_048 | 6_144;
  readonly owner: AceOpt0056DenseRouteOwner;
  readonly kernelId:
    | typeof ACE_OPT_0037_DENSE_K4_KERNEL_ID
    | typeof ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID;
  readonly evaluationDispatchCount: 8;
  readonly evaluationLabels: readonly string[];
}

export interface AceOpt0056DenseRouteProfile {
  readonly schema: "ace-opt-0056-selective-dense-routes-v1";
  readonly kernelSetId: typeof ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID;
  readonly routeCount: typeof ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT;
  readonly dispatchCount: typeof ACE_OPT_0056_REPEATED_DENSE_DISPATCH_COUNT;
  readonly approximateRouteCount: typeof ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT;
  readonly exactDownRouteCount: typeof ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT;
  readonly routes: readonly AceOpt0056DenseRoute[];
}

const OPERATIONS = Object.freeze([
  "self-query-projection",
  "self-key-projection",
  "self-value-projection",
  "self-output-projection",
  "cross-query-projection",
  "cross-output-projection",
  "mlp-gate-projection",
  "mlp-up-projection",
  "mlp-down-projection",
] as const);

export type AceOpt0056DenseOperation = (typeof OPERATIONS)[number];

const ROUTE_PATTERN =
  /^ace-dit-eval-([0-7])-layer-(0|[1-9]|1[0-9]|2[0-3])-(self-query-projection|self-key-projection|self-value-projection|self-output-projection|cross-query-projection|cross-output-projection|mlp-gate-projection|mlp-up-projection|mlp-down-projection)$/u;

/**
 * Benchmark-only selector for the authenticated revision-8 package. Exactly
 * the K6144/N2048 MLP down projection uses the exact owner; every other
 * repeated dense route uses the measured OPT-0032 arithmetic. Labels, shapes,
 * uniqueness, complete 8x24x9 coverage, and lifecycle all fail closed.
 */
export class AceOpt0056SelectiveDenseKernel implements AceGemmKernel {
  private readonly routes = new Map<string, AceOpt0056DenseDispatchRoute>();
  private finalized = false;
  private destroyed = false;

  private constructor(
    private readonly approximate: AceOpt0037DenseK4ProductionKernel,
    private readonly exactDown: AceOpt0056DenseK4ExactKernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0056SelectiveDenseKernel {
    let approximate: AceOpt0037DenseK4ProductionKernel | undefined;
    let exactDown: AceOpt0056DenseK4ExactKernel | undefined;
    try {
      approximate = AceOpt0037DenseK4ProductionKernel.create(
        device,
        capability,
      );
      exactDown = AceOpt0056DenseK4ExactKernel.create(device, capability);
      return new AceOpt0056SelectiveDenseKernel(approximate, exactDown);
    } catch (error) {
      approximate?.destroy();
      exactDown?.destroy();
      throw error;
    }
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch> {
    this.requireMutable();
    const route = resolveAceOpt0056DenseRoute(label, shape);
    if (this.routes.has(route.label)) {
      throw new Error(`OPT-0056 repeated dense route ${route.label} duplicated`);
    }
    this.routes.set(route.label, route);
    const owner = route.owner === "opt-0056-k4-exact-fp32"
      ? this.exactDown
      : this.approximate;
    const dispatch = await owner.createDispatch(label, shape, bindings);
    if (this.destroyed) {
      throw new Error("OPT-0056 selective dense selector was destroyed while compiling");
    }
    if (
      dispatch.weightLayout !== ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT ||
      dispatch.weightLayout !== ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT
    ) {
      throw new Error(`OPT-0056 ${label} escaped the authenticated rev8 layout`);
    }
    return dispatch;
  }

  finalizeRoutes(): AceOpt0056DenseRouteProfile {
    if (this.destroyed) {
      throw new Error("OPT-0056 selective dense selector was destroyed");
    }
    if (this.finalized) {
      throw new Error("OPT-0056 selective dense routes were already finalized");
    }
    const ordered: AceOpt0056DenseRoute[] = [];
    for (let layer = 0; layer < 24; layer += 1) {
      for (const operation of OPERATIONS) {
        const dispatches: AceOpt0056DenseDispatchRoute[] = [];
        for (let evaluation = 0; evaluation < 8; evaluation += 1) {
          const label =
            `ace-dit-eval-${evaluation}-layer-${layer}-${operation}`;
          const route = this.routes.get(label);
          if (route === undefined) {
            throw new Error(`OPT-0056 repeated dense route ${label} is missing`);
          }
          dispatches.push(route);
        }
        const first = dispatches[0]!;
        if (dispatches.some(({ owner, kernelId, rows, inner, columns }) =>
          owner !== first.owner ||
          kernelId !== first.kernelId ||
          rows !== first.rows ||
          inner !== first.inner ||
          columns !== first.columns
        )) {
          throw new Error(
            `OPT-0056 layer ${layer} ${operation} changed owner across evaluations`,
          );
        }
        ordered.push(Object.freeze({
          routeKey: `ace-dit-layer-${layer}-${operation}`,
          layer,
          operation,
          rows: first.rows,
          inner: first.inner,
          columns: first.columns,
          owner: first.owner,
          kernelId: first.kernelId,
          evaluationDispatchCount: 8 as const,
          evaluationLabels: Object.freeze(dispatches.map(({ label }) => label)),
        }));
      }
    }
    const exactDownRouteCount = ordered.filter(
      ({ owner }) => owner === "opt-0056-k4-exact-fp32",
    ).length;
    const approximateRouteCount = ordered.length - exactDownRouteCount;
    if (
      this.routes.size !== ACE_OPT_0056_REPEATED_DENSE_DISPATCH_COUNT ||
      ordered.length !== ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT ||
      exactDownRouteCount !== ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT ||
      approximateRouteCount !== ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT
    ) {
      throw new Error("OPT-0056 repeated dense route inventory is incomplete");
    }
    this.finalized = true;
    return Object.freeze({
      schema: "ace-opt-0056-selective-dense-routes-v1",
      kernelSetId: ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
      routeCount: ACE_OPT_0056_REPEATED_DENSE_ROUTE_COUNT,
      dispatchCount: ACE_OPT_0056_REPEATED_DENSE_DISPATCH_COUNT,
      approximateRouteCount: ACE_OPT_0056_APPROXIMATE_ROUTE_COUNT,
      exactDownRouteCount: ACE_OPT_0056_EXACT_DOWN_ROUTE_COUNT,
      routes: Object.freeze(ordered),
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.approximate.destroy();
    this.exactDown.destroy();
    this.routes.clear();
  }

  private requireMutable(): void {
    if (this.destroyed) {
      throw new Error("OPT-0056 selective dense selector was destroyed");
    }
    if (this.finalized) {
      throw new Error("OPT-0056 selective dense routes were finalized");
    }
  }
}

export function resolveAceOpt0056DenseRoute(
  label: string,
  shape: AceGemmShape,
): AceOpt0056DenseDispatchRoute {
  const match = ROUTE_PATTERN.exec(label);
  if (match === null) {
    throw new Error(`OPT-0056 rejected unregistered dense label ${label}`);
  }
  const evaluation = Number(match[1]);
  const layer = Number(match[2]);
  const operation = match[3] as AceOpt0056DenseOperation;
  const expected = expectedShape(operation);
  if (
    shape.rows !== 2_250 ||
    shape.inner !== expected.inner ||
    shape.columns !== expected.columns
  ) {
    throw new Error(
      `OPT-0056 ${label} expected M2250/K${expected.inner}/N${expected.columns}, ` +
        `got M${shape.rows}/K${shape.inner}/N${shape.columns}`,
    );
  }
  const exact = operation === "mlp-down-projection";
  return Object.freeze({
    label,
    evaluation,
    layer,
    operation,
    rows: 2_250,
    inner: expected.inner,
    columns: expected.columns,
    owner: exact
      ? "opt-0056-k4-exact-fp32"
      : "opt-0032-k4-fp16-partials",
    kernelId: exact
      ? ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID
      : ACE_OPT_0037_DENSE_K4_KERNEL_ID,
  });
}

function expectedShape(
  operation: AceOpt0056DenseOperation,
): Readonly<{
  inner: 2_048 | 6_144;
  columns: 1_024 | 2_048 | 6_144;
}> {
  if (operation === "self-key-projection" || operation === "self-value-projection") {
    return Object.freeze({ inner: 2_048, columns: 1_024 });
  }
  if (operation === "mlp-gate-projection" || operation === "mlp-up-projection") {
    return Object.freeze({ inner: 2_048, columns: 6_144 });
  }
  if (operation === "mlp-down-projection") {
    return Object.freeze({ inner: 6_144, columns: 2_048 });
  }
  return Object.freeze({ inner: 2_048, columns: 2_048 });
}
