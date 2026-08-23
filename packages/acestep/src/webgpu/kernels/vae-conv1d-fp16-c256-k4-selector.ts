import {
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
} from "./vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from "./vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0057_VAE_K7_ROUTES,
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0057VaeK7ShapeSelectorKernel,
  selectAceOpt0057VaeK7,
  type AceOpt0057VaeK7Owner,
  type AceOpt0057VaeK7Route,
} from "./vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  type AceFp16VaeConv1dFixedSubgroupCapability,
} from "./vae-conv1d-fp16-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID =
  "ace-opt-0076-vae-k7-c256-native-k4-selector-v1" as const;

export const ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS = Object.freeze([
  "block-2-res-1-conv1",
  "block-2-res-2-conv1",
  "block-2-res-3-conv1",
] as const);

export const ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT = 17 as const;
export const ACE_OPT_0076_VAE_ROW_REUSE_K4_ROUTE_COUNT = 12 as const;
export const ACE_OPT_0076_VAE_NATIVE_K4_ROUTE_COUNT = 3 as const;
export const ACE_OPT_0076_VAE_NATIVE_SCALAR_ROUTE_COUNT = 2 as const;

export type AceOpt0076VaeC256K4OperationLabel =
  typeof ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS[number];

export type AceOpt0076VaeC256K4Owner =
  | AceOpt0057VaeK7Owner
  | "native-k4";

export type AceOpt0076VaeC256K4KernelId =
  | typeof ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID
  | typeof ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID;

type AceOpt0057VaeK7KernelId =
  | typeof ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID;

export interface AceOpt0076VaeC256K4Route {
  readonly operationLabel: string;
  readonly owner: AceOpt0076VaeC256K4Owner;
  readonly kernelId: AceOpt0076VaeC256K4KernelId;
  readonly literalOwner: AceOpt0057VaeK7Owner;
  readonly literalKernelId: AceOpt0057VaeK7KernelId;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly dilation: number;
  readonly hasBias: boolean;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
}

export const ACE_OPT_0076_VAE_C256_K4_ROUTES = buildRoutes();

export interface AceOpt0076VaeC256K4Selection {
  readonly selectorKernelId:
    typeof ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID;
  readonly literalSelectorKernelId:
    typeof ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: string;
  readonly owner: AceOpt0076VaeC256K4Owner;
  readonly kernelId: AceOpt0076VaeC256K4KernelId;
  readonly route: Readonly<AceOpt0076VaeC256K4Route>;
}

export interface AceOpt0076VaeC256K4Dispatch {
  readonly label: string;
  readonly selectorKernelId:
    typeof ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID;
  readonly literalSelectorKernelId:
    typeof ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: string;
  readonly owner: AceOpt0076VaeC256K4Owner;
  readonly kernelId: AceOpt0076VaeC256K4KernelId;
  readonly route: Readonly<AceOpt0076VaeC256K4Route>;
  readonly plan: AceFp16VaeConv1dPlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface DelegatedDispatch {
  readonly label: string;
  readonly kernelId: AceOpt0076VaeC256K4KernelId;
  readonly plan: AceFp16VaeConv1dPlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Benchmark-only OPT-0076 selector. It delegates the literal OPT-0057 owner
 * for fourteen routes and substitutes the unchanged native-layout OPT-0024
 * owner only for the three authenticated C256 residual K7 labels.
 */
export class AceOpt0076VaeC256K4SelectorKernel {
  private destroyed = false;

  private constructor(
    private readonly literal: AceOpt0057VaeK7ShapeSelectorKernel,
    private readonly nativeK4: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceOpt0076VaeC256K4SelectorKernel {
    const literal = AceOpt0057VaeK7ShapeSelectorKernel.create(
      device,
      capability,
    );
    try {
      const nativeK4 = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
        device,
        capability,
      );
      return new AceOpt0076VaeC256K4SelectorKernel(literal, nativeK4);
    } catch (error) {
      literal.destroy();
      throw error;
    }
  }

  async createDispatch(
    label: string,
    operationLabel: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0076VaeC256K4Dispatch> {
    this.requireLive();
    const selection = selectAceOpt0076VaeC256K4(
      operationLabel,
      shape,
      bindings.bias !== undefined,
      outputStorage,
    );
    let dispatch: DelegatedDispatch;
    if (selection.owner === "native-k4") {
      dispatch = await this.nativeK4.createDispatch(
        label,
        shape,
        bindings,
        outputStorage,
        range,
      );
    } else {
      const literalDispatch = await this.literal.createDispatch(
        label,
        operationLabel,
        shape,
        bindings,
        outputStorage,
        range,
      );
      if (
        literalDispatch.selectorKernelId !==
          ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID ||
        literalDispatch.operationLabel !== selection.operationLabel ||
        literalDispatch.owner !== selection.owner
      ) {
        throw new Error(
          `OPT-0076 ${operationLabel} escaped the literal OPT-0057 route`,
        );
      }
      dispatch = literalDispatch;
    }
    this.requireLive();
    if (dispatch.label !== label || dispatch.kernelId !== selection.kernelId) {
      throw new Error(
        `OPT-0076 ${operationLabel} resolved an unexpected nested owner`,
      );
    }
    const selector = this;
    return Object.freeze({
      label,
      selectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
      literalSelectorKernelId:
        ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: selection.operationLabel,
      owner: selection.owner,
      kernelId: selection.kernelId,
      route: selection.route,
      plan: dispatch.plan,
      encode(pass: GPUComputePassEncoder): void {
        selector.requireLive();
        dispatch.encode(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.nativeK4.destroy();
    this.literal.destroy();
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0076 VAE C256 K4 selector was destroyed");
    }
  }
}

export function selectAceOpt0076VaeC256K4(
  operationLabel: string,
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): AceOpt0076VaeC256K4Selection {
  const literal = selectAceOpt0057VaeK7(
    operationLabel,
    shape,
    hasBias,
    outputStorage,
  );
  const route = ROUTE_BY_LABEL.get(operationLabel);
  if (route === undefined || !matchesLiteralRoute(route, literal.route)) {
    throw new Error(
      `OPT-0076 ${operationLabel} diverged from the literal OPT-0057 inventory`,
    );
  }
  return Object.freeze({
    selectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
    literalSelectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    operationLabel: route.operationLabel,
    owner: route.owner,
    kernelId: route.kernelId,
    route,
  });
}

const ROUTE_BY_LABEL = new Map(
  ACE_OPT_0076_VAE_C256_K4_ROUTES.map((route) => [
    route.operationLabel,
    route,
  ]),
);

function buildRoutes(): readonly Readonly<AceOpt0076VaeC256K4Route>[] {
  const labels = new Set<string>();
  const routes = ACE_OPT_0057_VAE_K7_ROUTES.map((literal) => {
    if (labels.has(literal.operationLabel)) {
      throw new Error(`OPT-0076 duplicate K7 route ${literal.operationLabel}`);
    }
    labels.add(literal.operationLabel);
    const literalKernelId = kernelIdForLiteralOwner(literal.owner);
    const nativeK4 = isC256K4OperationLabel(literal.operationLabel);
    if (nativeK4) requireC256K4LiteralRoute(literal);
    return Object.freeze({
      operationLabel: literal.operationLabel,
      owner: nativeK4 ? "native-k4" as const : literal.owner,
      kernelId: nativeK4
        ? ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID
        : literalKernelId,
      literalOwner: literal.owner,
      literalKernelId,
      inputChannels: literal.inputChannels,
      outputChannels: literal.outputChannels,
      dilation: literal.dilation,
      hasBias: literal.hasBias,
      outputStorage: literal.outputStorage,
    });
  });
  for (const operationLabel of ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS) {
    if (!labels.has(operationLabel)) {
      throw new Error(`OPT-0076 missing C256 K4 route ${operationLabel}`);
    }
  }
  const rowReuse = routes.filter(({ owner }) => owner === "row-reuse-k4").length;
  const nativeK4 = routes.filter(({ owner }) => owner === "native-k4").length;
  const nativeScalar = routes.filter(
    ({ owner }) => owner === "native-scalar-fp32",
  ).length;
  if (
    routes.length !== ACE_OPT_0076_VAE_C256_K4_ROUTE_COUNT ||
    rowReuse !== ACE_OPT_0076_VAE_ROW_REUSE_K4_ROUTE_COUNT ||
    nativeK4 !== ACE_OPT_0076_VAE_NATIVE_K4_ROUTE_COUNT ||
    nativeScalar !== ACE_OPT_0076_VAE_NATIVE_SCALAR_ROUTE_COUNT
  ) {
    throw new Error("OPT-0076 K7 route inventory is incomplete");
  }
  return Object.freeze(routes);
}

function kernelIdForLiteralOwner(
  owner: AceOpt0057VaeK7Owner,
): AceOpt0057VaeK7KernelId {
  return owner === "row-reuse-k4"
    ? ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
    : ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID;
}

function isC256K4OperationLabel(
  operationLabel: string,
): operationLabel is AceOpt0076VaeC256K4OperationLabel {
  return ACE_OPT_0076_VAE_C256_K4_OPERATION_LABELS.some(
    (candidate) => candidate === operationLabel,
  );
}

function requireC256K4LiteralRoute(route: AceOpt0057VaeK7Route): void {
  const expectedDilation = route.operationLabel === "block-2-res-1-conv1"
    ? 1
    : route.operationLabel === "block-2-res-2-conv1"
    ? 3
    : 9;
  if (
    route.owner !== "native-scalar-fp32" ||
    route.inputChannels !== 256 ||
    route.outputChannels !== 256 ||
    route.dilation !== expectedDilation ||
    !route.hasBias ||
    route.outputStorage !== "float16"
  ) {
    throw new Error(
      `OPT-0076 ${route.operationLabel} changed its literal C256 K7 contract`,
    );
  }
}

function matchesLiteralRoute(
  route: AceOpt0076VaeC256K4Route,
  literal: AceOpt0057VaeK7Route,
): boolean {
  return route.operationLabel === literal.operationLabel &&
    route.literalOwner === literal.owner &&
    route.literalKernelId === kernelIdForLiteralOwner(literal.owner) &&
    route.inputChannels === literal.inputChannels &&
    route.outputChannels === literal.outputChannels &&
    route.dilation === literal.dilation &&
    route.hasBias === literal.hasBias &&
    route.outputStorage === literal.outputStorage;
}
