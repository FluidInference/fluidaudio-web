import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  checkedAceProduct,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
} from "./correctness-utils.js";

export type AceHeadTransform = "split-heads" | "merge-heads";

export type AceElementwiseOperation =
  | "residual-add"
  | "broadcast-add"
  | "broadcast-multiply"
  | "silu"
  | "swiglu"
  | "adaln"
  | "gated-residual";

export interface AceTransformerTensorShape {
  readonly batch: number;
  readonly tokens: number;
  readonly width: number;
}

export interface AceTransformerTensorPlan extends AceTransformerTensorShape {
  readonly elements: number;
  readonly broadcastElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceHeadTransformShape {
  readonly batch: number;
  readonly tokens: number;
  readonly heads: number;
  readonly headDimension: number;
}

export interface AceHeadTransformPlan extends AceHeadTransformShape {
  readonly width: number;
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceUnaryBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceBinaryBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceBroadcastBindings {
  readonly input: GPUBufferBinding;
  /** `[batch,width]`, broadcast over the token dimension. */
  readonly broadcast: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceSwiGluBindings {
  readonly gate: GPUBufferBinding;
  readonly up: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceAdaLnBindings {
  readonly normalized: GPUBufferBinding;
  /** `[batch,width]`, broadcast over the token dimension. */
  readonly scale: GPUBufferBinding;
  /** `[batch,width]`, broadcast over the token dimension. */
  readonly shift: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceGatedResidualBindings {
  readonly residual: GPUBufferBinding;
  readonly branch: GPUBufferBinding;
  /** `[batch,width]`, broadcast over the token dimension. */
  readonly gate: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceTransformerDispatch<Plan> {
  readonly label: string;
  readonly operation: AceHeadTransform | AceElementwiseOperation;
  readonly plan: Plan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Direct Stage 1 transformer plumbing.
 *
 * The packed-BF16 reference graph stores activations and every operand here as
 * FP32. The raw-FP16 graph stores them as FP16. FP16 add/multiply expressions
 * retain FP16 intermediate rounding; SiLU evaluates `exp` in FP32, rounds its
 * result once, and (for SwiGLU) then performs the final FP16 multiply.
 */
export class AceCorrectnessTransformerPlumbingKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessTransformerPlumbingKernel {
    requireAceModelProfile(device, modelProfile, "transformer plumbing");
    return new AceCorrectnessTransformerPlumbingKernel(device, modelProfile);
  }

  async createHeadTransformDispatch(
    label: string,
    operation: AceHeadTransform,
    shape: AceHeadTransformShape,
    bindings: AceUnaryBindings,
  ): Promise<AceTransformerDispatch<AceHeadTransformPlan>> {
    this.requireLive();
    const plan = planAceHeadTransform(shape);
    const bytes = aceActivationBytes(this.modelProfile, plan.elements);
    requireAceBindingBytes(bindings.input, bytes, `${label} input`);
    requireAceBindingBytes(bindings.output, bytes, `${label} output`);
    requireAceDisjointOutput(bindings.output, [bindings.input], label);
    const pipeline = await this.pipelineFor(
      `head:${operation}:${headShapeKey(plan)}`,
      `${label}-${operation}`,
      aceCorrectnessHeadTransformWgsl(this.modelProfile, operation, plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, operation, plan, pipeline, [
      bindings.input,
      bindings.output,
    ]);
  }

  async createResidualAddDispatch(
    label: string,
    shape: AceTransformerTensorShape,
    bindings: AceBinaryBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      "residual-add",
      shape,
      [bindings.left, bindings.right],
      bindings.output,
    );
  }

  async createBroadcastDispatch(
    label: string,
    operation: "broadcast-add" | "broadcast-multiply",
    shape: AceTransformerTensorShape,
    bindings: AceBroadcastBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      operation,
      shape,
      [bindings.input, bindings.broadcast],
      bindings.output,
      ["full", "broadcast"],
    );
  }

  async createSiluDispatch(
    label: string,
    shape: AceTransformerTensorShape,
    bindings: AceUnaryBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      "silu",
      shape,
      [bindings.input],
      bindings.output,
    );
  }

  async createSwiGluDispatch(
    label: string,
    shape: AceTransformerTensorShape,
    bindings: AceSwiGluBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      "swiglu",
      shape,
      [bindings.gate, bindings.up],
      bindings.output,
    );
  }

  async createAdaLnDispatch(
    label: string,
    shape: AceTransformerTensorShape,
    bindings: AceAdaLnBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      "adaln",
      shape,
      [bindings.normalized, bindings.scale, bindings.shift],
      bindings.output,
      ["full", "broadcast", "broadcast"],
    );
  }

  async createGatedResidualDispatch(
    label: string,
    shape: AceTransformerTensorShape,
    bindings: AceGatedResidualBindings,
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    return this.createElementwiseDispatch(
      label,
      "gated-residual",
      shape,
      [bindings.residual, bindings.branch, bindings.gate],
      bindings.output,
      ["full", "full", "broadcast"],
    );
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private async createElementwiseDispatch(
    label: string,
    operation: AceElementwiseOperation,
    shape: AceTransformerTensorShape,
    inputs: readonly GPUBufferBinding[],
    output: GPUBufferBinding,
    inputLayouts: readonly ("full" | "broadcast")[] = inputs.map(() => "full"),
  ): Promise<AceTransformerDispatch<AceTransformerTensorPlan>> {
    this.requireLive();
    const plan = planAceTransformerTensor(shape);
    if (inputLayouts.length !== inputs.length) {
      throw new Error("ACE transformer plumbing internal binding-layout mismatch");
    }
    for (let index = 0; index < inputs.length; index += 1) {
      const layout = inputLayouts[index];
      const elements = layout === "broadcast" ? plan.broadcastElements : plan.elements;
      requireAceBindingBytes(
        inputs[index]!,
        aceActivationBytes(this.modelProfile, elements),
        `${label} input ${index}`,
      );
    }
    requireAceBindingBytes(
      output,
      aceActivationBytes(this.modelProfile, plan.elements),
      `${label} output`,
    );
    requireAceDisjointOutput(output, inputs, label);
    const pipeline = await this.pipelineFor(
      `elementwise:${operation}:${shapeKey(plan)}`,
      `${label}-${operation}`,
      aceCorrectnessElementwiseWgsl(this.modelProfile, operation, plan),
    );
    this.requireLive(" while compiling");
    return this.dispatch(label, operation, plan, pipeline, [...inputs, output]);
  }

  private dispatch<Plan>(
    label: string,
    operation: AceHeadTransform | AceElementwiseOperation,
    plan: Plan & { readonly workgroupsX: number; readonly workgroupsY: number },
    pipeline: GPUComputePipeline,
    bindings: readonly GPUBufferBinding[],
  ): AceTransformerDispatch<Plan> {
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((resource, binding) => ({ binding, resource })),
    });
    return Object.freeze({
      label,
      operation,
      plan,
      encode(pass: GPUComputePassEncoder): void {
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

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE transformer plumbing kernel was destroyed${suffix}`);
    }
  }
}

export function planAceTransformerTensor(
  shape: AceTransformerTensorShape,
): AceTransformerTensorPlan {
  const elements = checkedAceProduct(
    [shape.batch, shape.tokens, shape.width],
    "ACE transformer tensor",
  );
  const broadcastElements = checkedAceProduct(
    [shape.batch, shape.width],
    "ACE transformer broadcast",
  );
  const dispatch = planAceLinearDispatch(elements, "ACE transformer tensor");
  return Object.freeze({
    ...shape,
    elements,
    broadcastElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function planAceHeadTransform(
  shape: AceHeadTransformShape,
): AceHeadTransformPlan {
  const width = checkedAceProduct(
    [shape.heads, shape.headDimension],
    "ACE head width",
  );
  const elements = checkedAceProduct(
    [shape.batch, shape.tokens, width],
    "ACE head transform",
  );
  const dispatch = planAceLinearDispatch(elements, "ACE head transform");
  return Object.freeze({
    ...shape,
    width,
    elements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function aceCorrectnessHeadTransformWgsl(
  modelProfile: AceModelProfileId,
  operation: AceHeadTransform,
  shape: AceHeadTransformShape,
): string {
  const plan = planAceHeadTransform(shape);
  const type = profileActivationType(modelProfile, "head transform");
  if (operation !== "split-heads" && operation !== "merge-heads") {
    throw new TypeError(`Unknown ACE head transform ${String(operation)}`);
  }
  const sourceIndex = operation === "split-heads"
    ? /* wgsl */ `
  let dimension = index % HEAD_DIMENSION;
  let token = (index / HEAD_DIMENSION) % TOKENS;
  let head = (index / (HEAD_DIMENSION * TOKENS)) % HEADS;
  let batch = index / (HEAD_DIMENSION * TOKENS * HEADS);
  let source_index = ((batch * TOKENS + token) * HEADS + head) * HEAD_DIMENSION + dimension;`
    : /* wgsl */ `
  let dimension = index % HEAD_DIMENSION;
  let head = (index / HEAD_DIMENSION) % HEADS;
  let token = (index / (HEAD_DIMENSION * HEADS)) % TOKENS;
  let batch = index / (HEAD_DIMENSION * HEADS * TOKENS);
  let source_index = ((batch * HEADS + head) * TOKENS + token) * HEAD_DIMENSION + dimension;`;
  return /* wgsl */ `${type.enable}
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
const TOKENS: u32 = ${plan.tokens}u;
const HEADS: u32 = ${plan.heads}u;
const HEAD_DIMENSION: u32 = ${plan.headDimension}u;

@group(0) @binding(0) var<storage, read> input: array<${type.name}>;
@group(0) @binding(1) var<storage, read_write> output: array<${type.name}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
${sourceIndex}
  output[index] = input[source_index];
}
`;
}

export function aceCorrectnessElementwiseWgsl(
  modelProfile: AceModelProfileId,
  operation: AceElementwiseOperation,
  shape: AceTransformerTensorShape,
): string {
  const plan = planAceTransformerTensor(shape);
  const type = profileActivationType(modelProfile, "elementwise");
  const broadcastIndex = /* wgsl */ `
  let feature = index % WIDTH;
  let batch = index / (TOKENS * WIDTH);
  let broadcast_index = batch * WIDTH + feature;`;
  let declarations: string;
  let body: string;
  switch (operation) {
    case "residual-add":
      declarations = binaryDeclarations(type.name, "left", "right");
      body = "output[index] = left[index] + right[index];";
      break;
    case "broadcast-add":
      declarations = binaryDeclarations(type.name, "input", "broadcast");
      body = `${broadcastIndex}\n  output[index] = input[index] + broadcast[broadcast_index];`;
      break;
    case "broadcast-multiply":
      declarations = binaryDeclarations(type.name, "input", "broadcast");
      body = `${broadcastIndex}\n  output[index] = input[index] * broadcast[broadcast_index];`;
      break;
    case "silu":
      declarations = unaryDeclarations(type.name, "input");
      body = type.name === "f32"
        ? "let value = input[index];\n  output[index] = value / (1.0 + exp(-value));"
        : "let value = f32(input[index]);\n  output[index] = f16(value / (1.0 + exp(-value)));";
      break;
    case "swiglu":
      declarations = binaryDeclarations(type.name, "gate", "up");
      body = type.name === "f32"
        ? "let value = gate[index];\n  output[index] = (value / (1.0 + exp(-value))) * up[index];"
        : "let value = f32(gate[index]);\n  let activated = f16(value / (1.0 + exp(-value)));\n  output[index] = activated * up[index];";
      break;
    case "adaln":
      declarations = ternaryDeclarations(
        type.name,
        "normalized",
        "scale",
        "shift",
      );
      body = `${broadcastIndex}\n  output[index] = normalized[index] * (${type.name === "f16" ? "f16(1.0)" : "1.0"} + scale[broadcast_index]) + shift[broadcast_index];`;
      break;
    case "gated-residual":
      declarations = ternaryDeclarations(
        type.name,
        "residual",
        "branch",
        "gate",
      );
      body = `${broadcastIndex}\n  output[index] = residual[index] + branch[index] * gate[broadcast_index];`;
      break;
    default:
      throw new TypeError(`Unknown ACE elementwise operation ${String(operation)}`);
  }
  return /* wgsl */ `${type.enable}
${aceLinearInvocationWgsl(plan.elements, plan.workgroupsX)}
const TOKENS: u32 = ${plan.tokens}u;
const WIDTH: u32 = ${plan.width}u;

${declarations}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  ${body}
}
`;
}

function unaryDeclarations(type: string, input: string): string {
  return `@group(0) @binding(0) var<storage, read> ${input}: array<${type}>;\n@group(0) @binding(1) var<storage, read_write> output: array<${type}>;`;
}

function binaryDeclarations(type: string, left: string, right: string): string {
  return `@group(0) @binding(0) var<storage, read> ${left}: array<${type}>;\n@group(0) @binding(1) var<storage, read> ${right}: array<${type}>;\n@group(0) @binding(2) var<storage, read_write> output: array<${type}>;`;
}

function ternaryDeclarations(
  type: string,
  first: string,
  second: string,
  third: string,
): string {
  return `@group(0) @binding(0) var<storage, read> ${first}: array<${type}>;\n@group(0) @binding(1) var<storage, read> ${second}: array<${type}>;\n@group(0) @binding(2) var<storage, read> ${third}: array<${type}>;\n@group(0) @binding(3) var<storage, read_write> output: array<${type}>;`;
}

function profileActivationType(
  modelProfile: AceModelProfileId,
  operation: string,
): { readonly enable: string; readonly name: "f32" | "f16" } {
  if (modelProfile === "reference-bf16") {
    return { enable: "", name: "f32" };
  }
  if (modelProfile === "raw-fp16") {
    return { enable: "enable f16;", name: "f16" };
  }
  throw new TypeError(
    `Unknown ACE ${operation} model profile ${String(modelProfile)}`,
  );
}

function shapeKey(shape: AceTransformerTensorShape): string {
  return `${shape.batch}x${shape.tokens}x${shape.width}`;
}

function headShapeKey(shape: AceHeadTransformShape): string {
  return `${shape.batch}x${shape.tokens}x${shape.heads}x${shape.headDimension}`;
}
