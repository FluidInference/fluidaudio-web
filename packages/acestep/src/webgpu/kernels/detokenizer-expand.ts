import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  acePackedWeightBytes,
  checkedAceProduct,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

export const ACE_DETOKENIZER_PATCHES = 5;

export interface AceDetokenizerExpandShape {
  readonly codeCount: number;
  readonly width: number;
}

export interface AceDetokenizerExpandPlan extends AceDetokenizerExpandShape {
  readonly patches: typeof ACE_DETOKENIZER_PATCHES;
  readonly codeElements: number;
  readonly specialElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceDetokenizerExpandBindings {
  /** Profile activation `[codeCount,width]`. */
  readonly embeddedCodes: GPUBufferBinding;
  /** Packed model weight `[1,5,width]`. */
  readonly specialTokens: GPUBufferBinding;
  /** Profile activation `[codeCount,5,width]`. */
  readonly output: GPUBufferBinding;
}

export interface AceDetokenizerExpandDispatch {
  readonly label: string;
  readonly plan: AceDetokenizerExpandPlan;
  encode(pass: GPUComputePassEncoder): void;
}

/** Decodes the retained special-token weight while expanding each code row. */
export class AceCorrectnessDetokenizerExpandKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessDetokenizerExpandKernel {
    requireAceModelProfile(device, modelProfile, "detokenizer expand");
    return new AceCorrectnessDetokenizerExpandKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceDetokenizerExpandShape,
    bindings: AceDetokenizerExpandBindings,
  ): Promise<AceDetokenizerExpandDispatch> {
    this.requireLive();
    const plan = planAceDetokenizerExpand(shape);
    requireAceBindingBytes(
      bindings.embeddedCodes,
      aceActivationBytes(this.modelProfile, plan.codeElements),
      `${label} embedded codes`,
    );
    requireAceBindingBytes(
      bindings.specialTokens,
      acePackedWeightBytes(this.modelProfile, plan.specialElements),
      `${label} special tokens`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.embeddedCodes, bindings.specialTokens],
      label,
    );
    const key = `${this.modelProfile}:${plan.codeCount}:${plan.width}`;
    const pipeline = await this.pipelineFor(
      key,
      label,
      aceCorrectnessDetokenizerExpandWgsl(this.modelProfile, plan),
    );
    this.requireLive(" while compiling");
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.embeddedCodes },
        { binding: 1, resource: bindings.specialTokens },
        { binding: 2, resource: bindings.output },
      ],
    });
    return Object.freeze({
      label,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY, 1);
      },
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
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
      throw new Error(`ACE detokenizer-expand kernel was destroyed${suffix}`);
    }
  }
}

export function planAceDetokenizerExpand(
  shape: AceDetokenizerExpandShape,
): AceDetokenizerExpandPlan {
  requirePositiveSafeInteger(shape.codeCount, "ACE detokenizer-expand codeCount");
  requirePositiveSafeInteger(shape.width, "ACE detokenizer-expand width");
  const codeElements = checkedAceProduct(
    [shape.codeCount, shape.width],
    "ACE detokenizer-expand codes",
  );
  const specialElements = checkedAceProduct(
    [ACE_DETOKENIZER_PATCHES, shape.width],
    "ACE detokenizer-expand specials",
  );
  const outputElements = checkedAceProduct(
    [shape.codeCount, ACE_DETOKENIZER_PATCHES, shape.width],
    "ACE detokenizer-expand output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE detokenizer-expand");
  return Object.freeze({
    ...shape,
    patches: ACE_DETOKENIZER_PATCHES,
    codeElements,
    specialElements,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function aceCorrectnessDetokenizerExpandWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDetokenizerExpandShape,
): string {
  const plan = planAceDetokenizerExpand(shape);
  const f16 = modelProfile === "raw-fp16";
  if (modelProfile !== "reference-bf16" && !f16) {
    throw new TypeError(
      `Unknown ACE detokenizer-expand profile ${String(modelProfile)}`,
    );
  }
  const specialType = f16 ? "f16" : "u32";
  const loadSpecial = f16
    ? "special_tokens[special_index]"
    : "load_bf16(special_index)";
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const WIDTH: u32 = ${plan.width}u;
const PATCHES: u32 = ${ACE_DETOKENIZER_PATCHES}u;

@group(0) @binding(0) var<storage, read> embedded_codes: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> special_tokens: array<${specialType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;
${f16 ? "" : `
fn load_bf16(index: u32) -> f32 {
  let pair = special_tokens[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}`}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let column = index % WIDTH;
  let patch_index = (index / WIDTH) % PATCHES;
  let code = index / (WIDTH * PATCHES);
  let code_index = code * WIDTH + column;
  let special_index = patch_index * WIDTH + column;
  output[index] = embedded_codes[code_index] + ${loadSpecial};
}
`;
}
