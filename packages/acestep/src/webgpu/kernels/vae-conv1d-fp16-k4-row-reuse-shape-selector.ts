import {
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
} from "../../model/manifest.js";
import {
  ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
  AceFp16VaeConv1dKernel,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
  type AceFp16VaeConv1dFixedSubgroupCapability,
} from "./vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
  AceOpt0051VaeConv1dK4RowReuse16x64Kernel,
} from "./vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID,
  AceOpt0088VaeConv1dK4RowReusePortableKernel,
} from "./vae-conv1d-fp16-k4-row-reuse-portable.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID =
  "ace-opt-0057-vae-k7-row-reuse-shape-selector-v1" as const;
export const ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID =
  "opt-0088-vae-k7-portable-shape-selector-v1" as const;

export type AceOpt0057VaeK7Owner =
  | "row-reuse-k4"
  | "native-scalar-fp32";

export interface AceOpt0057VaeK7Route {
  readonly operationLabel: string;
  readonly owner: AceOpt0057VaeK7Owner;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly dilation: number;
  readonly hasBias: boolean;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
}

const ROW_REUSE_ROUTES = ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(
  (contract): Readonly<AceOpt0057VaeK7Route> => Object.freeze({
    operationLabel: contract.operationLabel,
    owner: "row-reuse-k4",
    inputChannels: contract.channels,
    outputChannels: contract.channels,
    dilation: contract.dilation,
    hasBias: true,
    outputStorage: "float16",
  }),
);

const NATIVE_ROUTES = Object.freeze([
  nativeRoute("conv1", 64, 2_048, 1, true, "float16"),
  nativeRoute("block-2-res-1-conv1", 256, 256, 1, true, "float16"),
  nativeRoute("block-2-res-2-conv1", 256, 256, 3, true, "float16"),
  nativeRoute("block-2-res-3-conv1", 256, 256, 9, true, "float16"),
  nativeRoute("conv2", 128, 2, 1, false, "float32"),
] as const);

export const ACE_OPT_0057_VAE_K7_ROUTES = Object.freeze([
  ...ROW_REUSE_ROUTES,
  ...NATIVE_ROUTES,
]);

export interface AceOpt0057VaeK7Selection {
  readonly selectorKernelId:
    typeof ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID;
  readonly route: Readonly<AceOpt0057VaeK7Route>;
}

export interface AceOpt0057VaeK7Dispatch {
  readonly label: string;
  readonly selectorKernelId:
    typeof ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: string;
  readonly owner: AceOpt0057VaeK7Owner;
  readonly kernelId:
    | typeof ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID
    | typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Revision-7 K7 owner. Selected biased C1024/C512/C128 operations use the
 * OPT-0051 K4 arithmetic/layout; C256, ingress, and the unbiased final layer
 * remain on revision 6's native-layout scalar-FP32 subgroup owner.
 */
export class AceOpt0057VaeK7ShapeSelectorKernel {
  private destroyed = false;

  private constructor(
    private readonly native: AceFp16VaeConv1dSubgroupKernel,
    private readonly rowReuse: AceOpt0051VaeConv1dK4RowReuse16x64Kernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceOpt0057VaeK7ShapeSelectorKernel {
    const native = AceFp16VaeConv1dSubgroupKernel.create(device, capability);
    try {
      const rowReuse = AceOpt0051VaeConv1dK4RowReuse16x64Kernel.create(
        device,
        capability,
      );
      return new AceOpt0057VaeK7ShapeSelectorKernel(native, rowReuse);
    } catch (error) {
      native.destroy();
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
  ): Promise<AceOpt0057VaeK7Dispatch> {
    this.requireLive();
    const selection = selectAceOpt0057VaeK7(
      operationLabel,
      shape,
      bindings.bias !== undefined,
      outputStorage,
    );
    const dispatch = selection.route.owner === "row-reuse-k4"
      ? await this.rowReuse.createDispatch(
          label,
          shape,
          bindings,
          outputStorage,
          range,
        )
      : await this.native.createDispatch(
          label,
          shape,
          bindings,
          outputStorage,
          range,
        );
    this.requireLive();
    return Object.freeze({
      label,
      selectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: selection.route.operationLabel,
      owner: selection.route.owner,
      kernelId: dispatch.kernelId,
      plan: dispatch.plan,
      encode: (pass: GPUComputePassEncoder): void => dispatch.encode(pass),
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.rowReuse.destroy();
    this.native.destroy();
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0057 VAE K7 selector was destroyed");
    }
  }
}

export interface AceOpt0088VaeK7PortableDispatch {
  readonly label: string;
  readonly selectorKernelId:
    typeof ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: string;
  readonly owner: AceOpt0057VaeK7Owner;
  readonly kernelId:
    | typeof ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID
    | typeof ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Portable no-subgroups twin of the OPT-0057 revision-7 K7 owner. The frozen
 * OPT-0057 route table and selection function are reused verbatim; only the
 * owners change transports: row-reuse routes go to the OPT-0088 portable
 * workgroup-staging kernel and native routes to the portable scalar-FP32
 * kernel. Both owners are bit-identical to their subgroup siblings.
 */
export class AceOpt0088VaeK7PortableShapeSelectorKernel {
  private destroyed = false;

  private constructor(
    private readonly native: AceFp16VaeConv1dKernel,
    private readonly rowReuse: AceOpt0088VaeConv1dK4RowReusePortableKernel,
  ) {}

  static create(device: GPUDevice): AceOpt0088VaeK7PortableShapeSelectorKernel {
    const native = AceFp16VaeConv1dKernel.create(device);
    try {
      const rowReuse = AceOpt0088VaeConv1dK4RowReusePortableKernel.create(
        device,
      );
      return new AceOpt0088VaeK7PortableShapeSelectorKernel(native, rowReuse);
    } catch (error) {
      native.destroy();
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
  ): Promise<AceOpt0088VaeK7PortableDispatch> {
    this.requireLive();
    const selection = selectAceOpt0057VaeK7(
      operationLabel,
      shape,
      bindings.bias !== undefined,
      outputStorage,
    );
    const dispatch = selection.route.owner === "row-reuse-k4"
      ? await this.rowReuse.createDispatch(
          label,
          shape,
          bindings,
          outputStorage,
          range,
        )
      : await this.native.createDispatch(
          label,
          shape,
          bindings,
          outputStorage,
          range,
        );
    this.requireLive();
    return Object.freeze({
      label,
      selectorKernelId: ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: selection.route.operationLabel,
      owner: selection.route.owner,
      kernelId: dispatch.kernelId,
      plan: dispatch.plan,
      encode: (pass: GPUComputePassEncoder): void => dispatch.encode(pass),
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.rowReuse.destroy();
    this.native.destroy();
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0088 portable VAE K7 selector was destroyed");
    }
  }
}

export function selectAceOpt0057VaeK7(
  operationLabel: string,
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): AceOpt0057VaeK7Selection {
  const route = ACE_OPT_0057_VAE_K7_ROUTES.find((candidate) =>
    candidate.operationLabel === operationLabel
  );
  if (route === undefined) {
    throw new RangeError(`OPT-0057 has no K7 route for ${operationLabel}`);
  }
  if (
    shape.batch !== 1 ||
    shape.inputChannels !== route.inputChannels ||
    shape.outputChannels !== route.outputChannels ||
    shape.kernelSize !== 7 ||
    shape.stride !== 1 ||
    shape.dilation !== route.dilation ||
    shape.padding !== 3 * route.dilation ||
    hasBias !== route.hasBias ||
    outputStorage !== route.outputStorage
  ) {
    throw new RangeError(
      `OPT-0057 ${operationLabel} changed its authenticated K7 contract`,
    );
  }
  return Object.freeze({
    selectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    route,
  });
}

function nativeRoute(
  operationLabel: string,
  inputChannels: number,
  outputChannels: number,
  dilation: number,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): Readonly<AceOpt0057VaeK7Route> {
  return Object.freeze({
    operationLabel,
    owner: "native-scalar-fp32",
    inputChannels,
    outputChannels,
    dilation,
    hasBias,
    outputStorage,
  });
}
