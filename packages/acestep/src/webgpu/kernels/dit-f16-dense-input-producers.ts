import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  aceCorrectnessRmsNormWgsl,
  planAceRmsNorm,
  type AceRmsNormPlan,
  type AceRmsNormShape,
} from "./rmsnorm.js";
import {
  aceCorrectnessElementwiseWgsl,
  aceCorrectnessHeadTransformWgsl,
  planAceHeadTransform,
  planAceTransformerTensor,
  type AceHeadTransformPlan,
  type AceHeadTransformShape,
  type AceTransformerTensorPlan,
  type AceTransformerTensorShape,
} from "./transformer-plumbing.js";

const FP16_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const FP32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const PACKED_BF16_PAIR_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const GPU_BUFFER_ALIGNMENT = 4;
const RMSNORM_REDUCTION_LANES = 256;
const RMSNORM_WORKGROUP_BYTES =
  RMSNORM_REDUCTION_LANES * Float32Array.BYTES_PER_ELEMENT;
const ACE_DIT_HIDDEN_SIZE = 2_048;
const ACE_DIT_INTERMEDIATE_SIZE = 6_144;
const ACE_DIT_QUERY_HEADS = 16;
const ACE_DIT_HEAD_DIMENSION = 128;
const ACE_DIT_RMSNORM_EPSILON = 1e-6;
const ACE_OPT_0081_TOKENS = 2_250;

export const ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE =
  "opt-0081-six-dense-input-f16-storage-v1" as const;

export const ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID =
  "opt-0081-six-f32-producer-f16-storage-v1" as const;

export const ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES = Object.freeze([
  "selfModulated",
  "selfMergedAttention",
  "crossNormalized",
  "crossMergedAttention",
  "mlpModulated",
  "gatedActivation",
] as const);

export type AceOpt0081F16DenseInputRole =
  (typeof ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES)[number];

export type AceOpt0081F16DenseInputOperation =
  | "adaln"
  | "merge-heads"
  | "cross-rmsnorm"
  | "swiglu";

export type AceOpt0081AdaLnRole = "selfModulated" | "mlpModulated";
export type AceOpt0081MergeHeadsRole =
  | "selfMergedAttention"
  | "crossMergedAttention";

export interface AceOpt0081AdaLnBindings {
  readonly normalized: GPUBufferBinding;
  readonly scale: GPUBufferBinding;
  readonly shift: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0081MergeHeadsBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0081CrossRmsNormBindings {
  readonly input: GPUBufferBinding;
  /** Packed source-order BF16 pairs, identical to the reference graph. */
  readonly weight: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0081SwiGluBindings {
  readonly gate: GPUBufferBinding;
  readonly up: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0081F16DenseInputProducerDispatch<Plan> {
  readonly label: string;
  readonly role: AceOpt0081F16DenseInputRole;
  readonly operation: AceOpt0081F16DenseInputOperation;
  readonly plan: Plan;
  readonly inputStorage: "f32";
  readonly outputStorage: "f16";
  readonly arithmetic: "f32";
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * OPT-0081's exact six producer boundaries.
 *
 * This is deliberately not a model profile. Inputs, packed-BF16 RMS weights,
 * reductions, nonlinear arithmetic, and all residual state retain the current
 * reference graph's FP32 behavior. Only the final assignment is rounded once
 * into scalar F16 storage for the immediately following dense consumer.
 */
export class AceOpt0081F16DenseInputProducerKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0081F16DenseInputProducerKernel {
    if (!device.features.has("shader-f16")) {
      throw new Error("OPT-0081 F16 dense-input producers require shader-f16");
    }
    const limits = device.limits;
    if (
      !supportsAtLeast(
        limits.maxComputeInvocationsPerWorkgroup,
        ACE_CORRECTNESS_WORKGROUP_SIZE,
      ) ||
      !supportsAtLeast(
        limits.maxComputeWorkgroupSizeX,
        ACE_CORRECTNESS_WORKGROUP_SIZE,
      ) ||
      !supportsAtLeast(
        limits.maxComputeWorkgroupStorageSize,
        RMSNORM_WORKGROUP_BYTES,
      )
    ) {
      throw new Error(
        "OPT-0081 F16 dense-input producers require WG256 and 1024 bytes of workgroup storage",
      );
    }
    if (
      !isValidGpuAlignment(limits.minStorageBufferOffsetAlignment) ||
      !supportsAtLeast(limits.maxStorageBufferBindingSize, 1) ||
      !supportsAtLeast(limits.maxBufferSize, 1)
    ) {
      throw new Error(
        "OPT-0081 F16 dense-input producers require valid storage-buffer binding limits",
      );
    }
    return new AceOpt0081F16DenseInputProducerKernel(device);
  }

  async createSelfModulatedDispatch(
    label: string,
    role: "selfModulated",
    operation: "adaln",
    shape: AceTransformerTensorShape,
    bindings: AceOpt0081AdaLnBindings,
  ): Promise<
    AceOpt0081F16DenseInputProducerDispatch<AceTransformerTensorPlan>
  > {
    this.requireLive();
    requireRoleOperation(role, operation, "selfModulated", "adaln");
    return this.createAdaLnF16OutputDispatch(
      label,
      role,
      operation,
      shape,
      bindings,
    );
  }

  async createSelfMergedAttentionDispatch(
    label: string,
    role: "selfMergedAttention",
    operation: "merge-heads",
    shape: AceHeadTransformShape,
    bindings: AceOpt0081MergeHeadsBindings,
  ): Promise<AceOpt0081F16DenseInputProducerDispatch<AceHeadTransformPlan>> {
    this.requireLive();
    requireRoleOperation(
      role,
      operation,
      "selfMergedAttention",
      "merge-heads",
    );
    return this.createMergeHeadsF16OutputDispatch(
      label,
      role,
      operation,
      shape,
      bindings,
    );
  }

  async createCrossNormalizedDispatch(
    label: string,
    role: "crossNormalized",
    operation: "cross-rmsnorm",
    shape: AceRmsNormShape,
    bindings: AceOpt0081CrossRmsNormBindings,
  ): Promise<AceOpt0081F16DenseInputProducerDispatch<AceRmsNormPlan>> {
    this.requireLive();
    requireRoleOperation(
      role,
      operation,
      "crossNormalized",
      "cross-rmsnorm",
    );
    const plan = planAceRmsNorm(shape);
    requireCrossRmsNormShape(plan, `${label} cross RMSNorm`);
    const inputBytes = checkedBytes(plan.elements, FP32_BYTES, `${label} input`);
    const weightBytes = checkedBytes(
      Math.ceil(plan.width / 2),
      PACKED_BF16_PAIR_BYTES,
      `${label} weight`,
    );
    const outputBytes = checkedBytes(
      plan.elements,
      FP16_BYTES,
      `${label} output`,
    );
    this.requireExactBinding(bindings.input, inputBytes, `${label} input`);
    this.requireExactBinding(bindings.weight, weightBytes, `${label} weight`);
    this.requireExactBinding(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      bindings.output,
      [bindings.input, bindings.weight],
      label,
    );
    const pipeline = await this.pipelineFor(
      `cross-rmsnorm:${rmsNormShapeKey(plan)}`,
      `${label}-opt-0081-cross-rmsnorm-f32-to-f16`,
      aceOpt0081CrossRmsNormF16OutputWgsl(plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, role, operation, plan, pipeline, [
      bindings.input,
      bindings.weight,
      bindings.output,
    ]);
  }

  async createCrossMergedAttentionDispatch(
    label: string,
    role: "crossMergedAttention",
    operation: "merge-heads",
    shape: AceHeadTransformShape,
    bindings: AceOpt0081MergeHeadsBindings,
  ): Promise<AceOpt0081F16DenseInputProducerDispatch<AceHeadTransformPlan>> {
    this.requireLive();
    requireRoleOperation(
      role,
      operation,
      "crossMergedAttention",
      "merge-heads",
    );
    return this.createMergeHeadsF16OutputDispatch(
      label,
      role,
      operation,
      shape,
      bindings,
    );
  }

  async createMlpModulatedDispatch(
    label: string,
    role: "mlpModulated",
    operation: "adaln",
    shape: AceTransformerTensorShape,
    bindings: AceOpt0081AdaLnBindings,
  ): Promise<
    AceOpt0081F16DenseInputProducerDispatch<AceTransformerTensorPlan>
  > {
    this.requireLive();
    requireRoleOperation(role, operation, "mlpModulated", "adaln");
    return this.createAdaLnF16OutputDispatch(
      label,
      role,
      operation,
      shape,
      bindings,
    );
  }

  async createGatedActivationDispatch(
    label: string,
    role: "gatedActivation",
    operation: "swiglu",
    shape: AceTransformerTensorShape,
    bindings: AceOpt0081SwiGluBindings,
  ): Promise<
    AceOpt0081F16DenseInputProducerDispatch<AceTransformerTensorPlan>
  > {
    this.requireLive();
    requireRoleOperation(
      role,
      operation,
      "gatedActivation",
      "swiglu",
    );
    const plan = planAceTransformerTensor(shape);
    requireIntermediateTensorShape(plan, `${label} SwiGLU`);
    const inputBytes = checkedBytes(plan.elements, FP32_BYTES, `${label} input`);
    const outputBytes = checkedBytes(
      plan.elements,
      FP16_BYTES,
      `${label} output`,
    );
    this.requireExactBinding(bindings.gate, inputBytes, `${label} gate`);
    this.requireExactBinding(bindings.up, inputBytes, `${label} up`);
    this.requireExactBinding(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(bindings.output, [bindings.gate, bindings.up], label);
    const pipeline = await this.pipelineFor(
      `swiglu:${tensorShapeKey(plan)}`,
      `${label}-opt-0081-swiglu-f32-to-f16`,
      aceOpt0081SwiGluF16OutputWgsl(plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, role, operation, plan, pipeline, [
      bindings.gate,
      bindings.up,
      bindings.output,
    ]);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private async createAdaLnF16OutputDispatch(
    label: string,
    role: AceOpt0081AdaLnRole,
    operation: "adaln",
    shape: AceTransformerTensorShape,
    bindings: AceOpt0081AdaLnBindings,
  ): Promise<
    AceOpt0081F16DenseInputProducerDispatch<AceTransformerTensorPlan>
  > {
    this.requireLive();
    requireRoleOperation(role, operation);
    const plan = planAceTransformerTensor(shape);
    requireHiddenTensorShape(plan, `${label} AdaLN`);
    const inputBytes = checkedBytes(plan.elements, FP32_BYTES, `${label} input`);
    const broadcastBytes = checkedBytes(
      plan.broadcastElements,
      FP32_BYTES,
      `${label} broadcast`,
    );
    const outputBytes = checkedBytes(
      plan.elements,
      FP16_BYTES,
      `${label} output`,
    );
    this.requireExactBinding(
      bindings.normalized,
      inputBytes,
      `${label} normalized`,
    );
    this.requireExactBinding(bindings.scale, broadcastBytes, `${label} scale`);
    this.requireExactBinding(bindings.shift, broadcastBytes, `${label} shift`);
    this.requireExactBinding(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      bindings.output,
      [bindings.normalized, bindings.scale, bindings.shift],
      label,
    );
    const pipeline = await this.pipelineFor(
      `adaln:${tensorShapeKey(plan)}`,
      `${label}-opt-0081-adaln-f32-to-f16`,
      aceOpt0081AdaLnF16OutputWgsl(plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, role, operation, plan, pipeline, [
      bindings.normalized,
      bindings.scale,
      bindings.shift,
      bindings.output,
    ]);
  }

  private async createMergeHeadsF16OutputDispatch(
    label: string,
    role: AceOpt0081MergeHeadsRole,
    operation: "merge-heads",
    shape: AceHeadTransformShape,
    bindings: AceOpt0081MergeHeadsBindings,
  ): Promise<AceOpt0081F16DenseInputProducerDispatch<AceHeadTransformPlan>> {
    this.requireLive();
    requireRoleOperation(role, operation);
    const plan = planAceHeadTransform(shape);
    requireQueryHeadShape(plan, `${label} merge-heads`);
    const inputBytes = checkedBytes(plan.elements, FP32_BYTES, `${label} input`);
    const outputBytes = checkedBytes(
      plan.elements,
      FP16_BYTES,
      `${label} output`,
    );
    this.requireExactBinding(bindings.input, inputBytes, `${label} input`);
    this.requireExactBinding(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(bindings.output, [bindings.input], label);
    const pipeline = await this.pipelineFor(
      `merge-heads:${headShapeKey(plan)}`,
      `${label}-opt-0081-merge-heads-f32-to-f16`,
      aceOpt0081MergeHeadsF16OutputWgsl(plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, role, operation, plan, pipeline, [
      bindings.input,
      bindings.output,
    ]);
  }

  private dispatch<Plan extends {
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>(
    label: string,
    role: AceOpt0081F16DenseInputRole,
    operation: AceOpt0081F16DenseInputOperation,
    plan: Plan,
    pipeline: GPUComputePipeline,
    bindings: readonly GPUBufferBinding[],
  ): AceOpt0081F16DenseInputProducerDispatch<Plan> {
    const owner = this;
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0081-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((resource, binding) => ({ binding, resource })),
    });
    return Object.freeze({
      label,
      role,
      operation,
      plan,
      inputStorage: "f32" as const,
      outputStorage: "f16" as const,
      arithmetic: "f32" as const,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive(" before encoding");
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY, 1);
      },
    });
  }

  private pipelineFor(
    key: string,
    label: string,
    source: string,
  ): Promise<GPUComputePipeline> {
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const module = this.device.createShaderModule({ label, code: source });
    const created = this.device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }

  private requireExactBinding(
    binding: GPUBufferBinding,
    requiredBytes: number,
    label: string,
  ): void {
    const offset = binding.offset ?? 0;
    const bufferBytes = binding.buffer.size;
    const exposed = binding.size ?? bufferBytes - offset;
    const end = offset + exposed;
    const alignment = this.device.limits.minStorageBufferOffsetAlignment;
    if (
      !Number.isSafeInteger(bufferBytes) ||
      bufferBytes < 0 ||
      bufferBytes % GPU_BUFFER_ALIGNMENT !== 0 ||
      bufferBytes > this.device.limits.maxBufferSize ||
      !Number.isSafeInteger(offset) ||
      offset < 0 ||
      !Number.isSafeInteger(exposed) ||
      exposed !== requiredBytes ||
      !Number.isSafeInteger(end) ||
      end > bufferBytes ||
      offset % alignment !== 0 ||
      exposed > this.device.limits.maxStorageBufferBindingSize
    ) {
      throw new RangeError(
        `${label} binding must expose exactly ${requiredBytes} aligned bytes`,
      );
    }
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(
        `OPT-0081 F16 dense-input producer kernel was destroyed${suffix}`,
      );
    }
  }
}

export function aceOpt0081AdaLnF16OutputWgsl(
  shape: AceTransformerTensorShape,
): string {
  const plan = planAceTransformerTensor(shape);
  requireHiddenTensorShape(plan, "OPT-0081 AdaLN");
  return deriveF16FinalStoreWgsl(
    aceCorrectnessElementwiseWgsl("reference-bf16", "adaln", plan),
    "output[index] = normalized[index] * (1.0 + scale[broadcast_index]) + shift[broadcast_index];",
    "AdaLN",
  );
}

export function aceOpt0081MergeHeadsF16OutputWgsl(
  shape: AceHeadTransformShape,
): string {
  const plan = planAceHeadTransform(shape);
  requireQueryHeadShape(plan, "OPT-0081 merge-heads");
  return deriveF16FinalStoreWgsl(
    aceCorrectnessHeadTransformWgsl(
      "reference-bf16",
      "merge-heads",
      plan,
    ),
    "output[index] = input[source_index];",
    "merge-heads",
  );
}

export function aceOpt0081CrossRmsNormF16OutputWgsl(
  shape: AceRmsNormShape,
): string {
  const plan = planAceRmsNorm(shape);
  requireCrossRmsNormShape(plan, "OPT-0081 cross RMSNorm");
  return deriveF16FinalStoreWgsl(
    aceCorrectnessRmsNormWgsl("reference-bf16", plan),
    "output[index] = value * inverse_rms * load_weight(column);",
    "cross RMSNorm",
  );
}

export function aceOpt0081SwiGluF16OutputWgsl(
  shape: AceTransformerTensorShape,
): string {
  const plan = planAceTransformerTensor(shape);
  requireIntermediateTensorShape(plan, "OPT-0081 SwiGLU");
  return deriveF16FinalStoreWgsl(
    aceCorrectnessElementwiseWgsl("reference-bf16", "swiglu", plan),
    "output[index] = (value / (1.0 + exp(-value))) * up[index];",
    "SwiGLU",
  );
}

function requireRoleOperation(
  role: unknown,
  operation: unknown,
  requiredRole?: AceOpt0081F16DenseInputRole,
  requiredOperation?: AceOpt0081F16DenseInputOperation,
): asserts role is AceOpt0081F16DenseInputRole {
  const expected = role === "selfModulated" || role === "mlpModulated"
    ? "adaln"
    : role === "selfMergedAttention" || role === "crossMergedAttention"
      ? "merge-heads"
      : role === "crossNormalized"
        ? "cross-rmsnorm"
        : role === "gatedActivation"
          ? "swiglu"
          : undefined;
  if (
    expected === undefined ||
    operation !== expected ||
    (requiredRole !== undefined && role !== requiredRole) ||
    (requiredOperation !== undefined && operation !== requiredOperation)
  ) {
    throw new TypeError(
      `OPT-0081 rejects producer role/operation ${String(role)}/${String(operation)}` +
        (requiredRole === undefined
          ? ""
          : `; expected ${requiredRole}/${String(requiredOperation)}`),
    );
  }
}

function supportsAtLeast(value: number, minimum: number): boolean {
  return Number.isSafeInteger(value) && value >= minimum;
}

function isValidGpuAlignment(value: number): boolean {
  return supportsAtLeast(value, GPU_BUFFER_ALIGNMENT) &&
    Number.isInteger(Math.log2(value));
}

function requireHiddenTensorShape(
  plan: AceTransformerTensorPlan,
  label: string,
): void {
  if (
    plan.batch !== 1 ||
    plan.tokens !== ACE_OPT_0081_TOKENS ||
    plan.width !== ACE_DIT_HIDDEN_SIZE
  ) {
    throw new RangeError(`${label} requires batch 1, tokens 2250, and width 2048`);
  }
}

function requireIntermediateTensorShape(
  plan: AceTransformerTensorPlan,
  label: string,
): void {
  if (
    plan.batch !== 1 ||
    plan.tokens !== ACE_OPT_0081_TOKENS ||
    plan.width !== ACE_DIT_INTERMEDIATE_SIZE
  ) {
    throw new RangeError(`${label} requires batch 1, tokens 2250, and width 6144`);
  }
}

function requireQueryHeadShape(
  plan: AceHeadTransformPlan,
  label: string,
): void {
  if (
    plan.batch !== 1 ||
    plan.tokens !== ACE_OPT_0081_TOKENS ||
    plan.heads !== ACE_DIT_QUERY_HEADS ||
    plan.headDimension !== ACE_DIT_HEAD_DIMENSION ||
    plan.width !== ACE_DIT_HIDDEN_SIZE
  ) {
    throw new RangeError(
      `${label} requires batch 1, tokens 2250, and 16 heads of dimension 128`,
    );
  }
}

function requireCrossRmsNormShape(
  plan: AceRmsNormPlan,
  label: string,
): void {
  if (
    plan.rows !== ACE_OPT_0081_TOKENS ||
    plan.width !== ACE_DIT_HIDDEN_SIZE ||
    plan.epsilon !== ACE_DIT_RMSNORM_EPSILON
  ) {
    throw new RangeError(
      `${label} requires rows 2250, width 2048, and epsilon 1e-6`,
    );
  }
}

function deriveF16FinalStoreWgsl(
  referenceSource: string,
  referenceAssignment: string,
  operation: string,
): string {
  if (referenceSource.includes("enable f16;")) {
    throw new Error(
      `OPT-0081 ${operation} derivation requires the FP32 reference shader`,
    );
  }
  const outputDeclaration =
    "var<storage, read_write> output: array<f32>;";
  let source = replaceExactOnce(
    referenceSource,
    outputDeclaration,
    "var<storage, read_write> output: array<f16>;",
    `${operation} output declaration`,
  );
  const assignmentPrefix = "output[index] = ";
  if (
    !referenceAssignment.startsWith(assignmentPrefix) ||
    !referenceAssignment.endsWith(";")
  ) {
    throw new Error(`OPT-0081 ${operation} has an invalid reference assignment`);
  }
  const rightHandSide = referenceAssignment.slice(
    assignmentPrefix.length,
    -1,
  );
  source = replaceExactOnce(
    source,
    referenceAssignment,
    `${assignmentPrefix}f16(${rightHandSide});`,
    `${operation} final assignment`,
  );
  return `enable f16;\n${source}`;
}

function replaceExactOnce(
  source: string,
  expected: string,
  replacement: string,
  label: string,
): string {
  const first = source.indexOf(expected);
  if (first < 0 || source.indexOf(expected, first + expected.length) >= 0) {
    throw new Error(`OPT-0081 ${label} must occur exactly once`);
  }
  return source.slice(0, first) + replacement + source.slice(first + expected.length);
}

function checkedBytes(
  elements: number,
  bytesPerElement: number,
  label: string,
): number {
  const bytes = elements * bytesPerElement;
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new RangeError(`${label} byte count is not a positive safe integer`);
  }
  return bytes;
}

function tensorShapeKey(shape: AceTransformerTensorPlan): string {
  return `${shape.batch}x${shape.tokens}x${shape.width}`;
}

function headShapeKey(shape: AceHeadTransformPlan): string {
  return `${shape.batch}x${shape.tokens}x${shape.heads}x${shape.headDimension}`;
}

function rmsNormShapeKey(shape: AceRmsNormPlan): string {
  return `${shape.rows}x${shape.width}:${shape.epsilon}`;
}
