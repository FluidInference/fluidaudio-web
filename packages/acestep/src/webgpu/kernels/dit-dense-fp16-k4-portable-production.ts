import { ACE_DIT_DENSE_K4_FP16_LAYOUT } from "../../model/manifest.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";
import { AceOpt0088DenseK4PortableKernel } from "./dit-dense-fp16-k4-portable.js";

/**
 * Production adapter: rev8 package layout plus the portable OPT-0088 owner.
 * Unlike the OPT-0037 wrapper this carries its own kernel id because the
 * portable owner is a distinct WGSL module, not the measured OPT-0032 one.
 */
export const ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_KERNEL_ID =
  "opt-0088-dense-k4-fp16-portable-production-v1";
export const ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_WEIGHT_LAYOUT =
  ACE_DIT_DENSE_K4_FP16_LAYOUT;

/**
 * Keeps the portable kernel identity immutable while binding its bit-exact
 * OPT-0032 arithmetic to the authenticated rev8 package layout. The adapter
 * owns no GPU resources of its own and destroys its single OPT-0088 owner
 * exactly once. Same public surface as the OPT-0037 wrapper so backend wiring
 * can swap the two on a backend string; the subgroup capability hints are
 * accepted for signature compatibility and ignored.
 */
export class AceOpt0088DenseK4PortableProductionKernel implements AceGemmKernel {
  private destroyed = false;

  private constructor(
    private readonly owner: AceOpt0088DenseK4PortableKernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }> = {},
  ): AceOpt0088DenseK4PortableProductionKernel {
    return new AceOpt0088DenseK4PortableProductionKernel(
      AceOpt0088DenseK4PortableKernel.create(device, capability),
    );
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0088 portable production K4 kernel was destroyed");
    }
    const dispatch = await this.owner.createDispatch(label, shape, bindings);
    if (this.destroyed) {
      throw new Error(
        "OPT-0088 portable production K4 kernel was destroyed while compiling",
      );
    }
    return Object.freeze({
      label: dispatch.label,
      weightLayout: ACE_OPT_0088_DENSE_K4_PORTABLE_PRODUCTION_WEIGHT_LAYOUT,
      plan: dispatch.plan,
      rangeCount: dispatch.rangeCount,
      encodeRange: dispatch.encodeRange,
      encode: dispatch.encode,
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.owner.destroy();
  }
}
