import { ACE_DIT_DENSE_K4_FP16_LAYOUT } from "../../model/manifest.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  AceOpt0032DenseK4PartialsKernel,
} from "./dit-dense-fp16-k4-partials.js";

/** Production adapter: rev8 package layout plus the measured OPT-0032 owner. */
export const ACE_OPT_0037_DENSE_K4_KERNEL_ID =
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID;
export const ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT =
  ACE_DIT_DENSE_K4_FP16_LAYOUT;

/**
 * Keeps the benchmark identity immutable while binding its exact arithmetic to
 * the authenticated rev8 package layout. The adapter owns no GPU resources of
 * its own and destroys its single OPT-0032 owner exactly once.
 */
export class AceOpt0037DenseK4ProductionKernel implements AceGemmKernel {
  private destroyed = false;

  private constructor(
    private readonly owner: AceOpt0032DenseK4PartialsKernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0037DenseK4ProductionKernel {
    return new AceOpt0037DenseK4ProductionKernel(
      AceOpt0032DenseK4PartialsKernel.create(device, capability),
    );
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0037 production K4 kernel was destroyed");
    }
    const dispatch = await this.owner.createDispatch(label, shape, bindings);
    if (this.destroyed) {
      throw new Error("OPT-0037 production K4 kernel was destroyed while compiling");
    }
    return Object.freeze({
      label: dispatch.label,
      weightLayout: ACE_OPT_0037_DENSE_K4_WEIGHT_LAYOUT,
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
