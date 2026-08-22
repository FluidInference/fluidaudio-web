import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessGemmKernel,
  type AceGemmBufferBindings,
  type AceGemmDispatch,
  type AceGemmKernel,
  type AceGemmShape,
} from "./kernels/gemm.js";
import {
  AceOpt0083PlannerDirectLowRowBf16GemvKernel,
  planAceOpt0083PlannerDirectLowRowBf16Gemv,
} from "./kernels/planner-low-row-bf16-gemv.js";

/** @internal Frozen package-native A/B selector for OPT-0087 only. */
export type AceOpt0087PlannerDenseArm = "generic-a" | "direct-b";

/** @internal Every dense role whose physical planner weight may select B. */
export type AceOpt0087PlannerDenseRole =
  | "query-projection"
  | "key-projection"
  | "value-projection"
  | "attention-output-projection"
  | "gate-projection"
  | "up-projection"
  | "down-projection"
  | "tied-lm-head";

/** @internal Immutable invocation facts used by the fail-closed selector. */
export interface AceOpt0087PlannerDenseInvocation {
  readonly owner: "planner" | "non-planner";
  readonly kind: "prefill" | "decode";
  readonly batch: number;
  readonly tokens: number;
  readonly requestedArm: AceOpt0087PlannerDenseArm;
}

/** @internal Why one OPT-0087 dense request resolved to A or B. */
export type AceOpt0087PlannerDenseSelectionReason =
  | "control-requested"
  | "direct-selected"
  | "non-planner-owner"
  | "non-reference-profile"
  | "candidate-owner-unavailable"
  | "not-single-token-decode"
  | "unsupported-physical-rows"
  | "physical-row-mismatch"
  | "bias-present"
  | "direct-shape-rejected";

/** @internal Clone-free evidence captured before a kernel compiles. */
export interface AceOpt0087PlannerDenseSelection {
  readonly label: string;
  readonly role: AceOpt0087PlannerDenseRole;
  readonly requestedArm: AceOpt0087PlannerDenseArm;
  readonly selectedArm: AceOpt0087PlannerDenseArm;
  readonly reason: AceOpt0087PlannerDenseSelectionReason;
  readonly shape: Readonly<AceGemmShape>;
  /** Exact caller-owned binding object; the dense owner never copies weights. */
  readonly weightBinding: GPUBufferBinding;
  readonly activationBinding: GPUBufferBinding;
  readonly outputBinding: GPUBufferBinding;
}

/** @internal One planner dense compilation request. */
export interface AceOpt0087PlannerDenseRequest {
  readonly label: string;
  readonly role: AceOpt0087PlannerDenseRole;
  readonly invocation: AceOpt0087PlannerDenseInvocation;
  readonly shape: AceGemmShape;
  readonly bindings: AceGemmBufferBindings;
}

/** @internal Kernel factories exposed only for deterministic ownership tests. */
export interface AceOpt0087PlannerDenseKernelFactories {
  readonly generic: () => AceGemmKernel;
  readonly direct: () => AceGemmKernel;
}

/** @internal Result keeps the evidence beside the actual compiled dispatch. */
export interface AceOpt0087PlannerDenseDispatchResult {
  readonly dispatch: AceGemmDispatch;
  readonly selection: AceOpt0087PlannerDenseSelection;
}

/**
 * @internal
 * Planner-only owner for the frozen OPT-0087 package A/B gate. Ordinary Qwen,
 * ACE encoders, DiT, and every other caller continue to construct only the
 * generic correctness GEMM. This owner never owns or destroys graph weights.
 */
export class AceOpt0087PlannerDenseOwner {
  private destroyed = false;

  private constructor(
    readonly modelProfile: AceModelProfileId,
    private readonly generic: AceGemmKernel,
    private readonly direct: AceGemmKernel | undefined,
  ) {}

  static createGeneric(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceOpt0087PlannerDenseOwner {
    return new AceOpt0087PlannerDenseOwner(
      modelProfile,
      AceCorrectnessGemmKernel.create(device, modelProfile),
      undefined,
    );
  }

  /** @internal Construct both frozen pipeline owners without graph weights. */
  static createPairedForOpt0087(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceOpt0087PlannerDenseOwner {
    return AceOpt0087PlannerDenseOwner.createFromFactoriesForOpt0087(
      modelProfile,
      {
        generic: () => AceCorrectnessGemmKernel.create(device, modelProfile),
        direct: () => AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
          device,
          modelProfile,
        ),
      },
    );
  }

  /** @internal Deterministic partial-construction and lifecycle seam. */
  static createFromFactoriesForOpt0087(
    modelProfile: AceModelProfileId,
    factories: AceOpt0087PlannerDenseKernelFactories,
  ): AceOpt0087PlannerDenseOwner {
    let generic: AceGemmKernel | undefined;
    let direct: AceGemmKernel | undefined;
    try {
      generic = factories.generic();
      // Raw-FP16 must not even instantiate the packed-BF16 candidate owner.
      if (modelProfile === "reference-bf16") direct = factories.direct();
      return new AceOpt0087PlannerDenseOwner(
        modelProfile,
        generic,
        direct,
      );
    } catch (error) {
      direct?.destroy();
      generic?.destroy();
      throw error;
    }
  }

  createGenericDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch> {
    this.requireLive();
    return this.generic.createDispatch(label, shape, bindings);
  }

  /** @internal Compile one explicit gate arm and capture clone-free evidence. */
  async createDispatchForOpt0087(
    request: AceOpt0087PlannerDenseRequest,
    onSelection?: (selection: AceOpt0087PlannerDenseSelection) => void,
  ): Promise<AceOpt0087PlannerDenseDispatchResult> {
    this.requireLive();
    const resolved = selectAceOpt0087PlannerDenseArm(
      this.modelProfile,
      this.direct !== undefined,
      request.invocation,
      request.shape,
      request.bindings.bias !== undefined,
    );
    const selection = Object.freeze({
      label: request.label,
      role: request.role,
      requestedArm: request.invocation.requestedArm,
      selectedArm: resolved.arm,
      reason: resolved.reason,
      shape: Object.freeze({ ...request.shape }),
      weightBinding: request.bindings.weight,
      activationBinding: request.bindings.activation,
      outputBinding: request.bindings.output,
    });
    onSelection?.(selection);
    const kernel = resolved.arm === "direct-b" ? this.direct : this.generic;
    if (kernel === undefined) {
      throw new Error("OPT-0087 selected an unavailable direct dense owner");
    }
    const dispatch = await kernel.createDispatch(
      request.label,
      request.shape,
      request.bindings,
    );
    this.requireLive(" while compiling");
    return Object.freeze({ dispatch, selection });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.direct?.destroy();
    this.generic.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`OPT-0087 planner dense owner was destroyed${suffix}`);
    }
  }
}

/** @internal Pure, exception-free fail-closed OPT-0087 selector. */
export function selectAceOpt0087PlannerDenseArm(
  modelProfile: AceModelProfileId,
  candidateAvailable: boolean,
  invocation: AceOpt0087PlannerDenseInvocation,
  shape: AceGemmShape,
  hasBias = false,
): Readonly<{
  readonly arm: AceOpt0087PlannerDenseArm;
  readonly reason: AceOpt0087PlannerDenseSelectionReason;
}> {
  const generic = (
    reason: Exclude<AceOpt0087PlannerDenseSelectionReason, "direct-selected">,
  ): Readonly<{
    readonly arm: "generic-a";
    readonly reason: AceOpt0087PlannerDenseSelectionReason;
  }> => Object.freeze({ arm: "generic-a", reason });
  if (invocation.requestedArm !== "direct-b") {
    return generic("control-requested");
  }
  if (invocation.owner !== "planner") return generic("non-planner-owner");
  if (modelProfile !== "reference-bf16") {
    return generic("non-reference-profile");
  }
  if (!candidateAvailable) return generic("candidate-owner-unavailable");
  if (invocation.kind !== "decode" || invocation.tokens !== 1) {
    return generic("not-single-token-decode");
  }
  if (invocation.batch !== 1 && invocation.batch !== 2) {
    return generic("unsupported-physical-rows");
  }
  if (shape.rows !== invocation.batch) return generic("physical-row-mismatch");
  if (hasBias) return generic("bias-present");
  try {
    planAceOpt0083PlannerDirectLowRowBf16Gemv(shape);
  } catch {
    return generic("direct-shape-rejected");
  }
  return Object.freeze({ arm: "direct-b", reason: "direct-selected" });
}
