import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceActivationBytes,
  aceLinearInvocationWgsl,
  checkedAceProduct,
  checkedAceSum,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requireAceModelProfile,
  requireAceU32,
  requireNonNegativeSafeInteger,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

export interface AceGatherRowsShape {
  readonly outer: number;
  readonly sourceRows: number;
  readonly outputRows: number;
  readonly width: number;
}

export interface AceGatherRowsPlan extends AceGatherRowsShape {
  readonly sourceElements: number;
  readonly indexElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceGatherRowsBindings {
  readonly input: GPUBufferBinding;
  /** Outer-local unsigned row indices in stable destination order. */
  readonly indices: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceAxisCopyShape {
  readonly outer: number;
  readonly inputLength: number;
  readonly outputLength: number;
  readonly inner: number;
}

export interface AceCropShape {
  readonly outer: number;
  readonly inputLength: number;
  readonly offset: number;
  readonly outputLength: number;
  readonly inner: number;
}

export interface AceRepeatShape {
  readonly outer: number;
  readonly inputLength: number;
  readonly repeats: number;
  readonly inner: number;
}

export interface AceConcatShape {
  readonly outer: number;
  readonly leftLength: number;
  readonly rightLength: number;
  readonly inner: number;
}

export interface AceAxisCopyPlan {
  readonly operation: AceAxisCopyOperation;
  readonly outer: number;
  readonly inputLength: number;
  readonly secondLength: number;
  readonly offset: number;
  readonly outputLength: number;
  readonly repeats: number;
  readonly inner: number;
  readonly inputElements: number;
  readonly secondElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export type AceAxisCopyOperation = "right-pad" | "crop" | "repeat" | "concat";

export interface AceCopyBindings {
  readonly input: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceConcatBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceStablePackShape {
  readonly batch: number;
  readonly leftLength: number;
  readonly rightLength: number;
  readonly width: number;
}

export interface AceStablePackPlan extends AceStablePackShape {
  readonly packedLength: number;
  readonly leftElements: number;
  readonly rightElements: number;
  readonly packedRows: number;
  readonly outputElements: number;
  readonly indexWorkgroupsX: number;
  readonly indexWorkgroupsY: number;
  readonly gatherWorkgroupsX: number;
  readonly gatherWorkgroupsY: number;
}

export interface AceStablePackBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  /** Unsigned 0/1 mask `[batch,leftLength]`. */
  readonly leftMask: GPUBufferBinding;
  /** Unsigned 0/1 mask `[batch,rightLength]`. */
  readonly rightMask: GPUBufferBinding;
  /** `batch * (leftLength + rightLength)` u32 entries. */
  readonly indicesScratch: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  /** Unsigned 0/1 packed mask `[batch,leftLength + rightLength]`. */
  readonly outputMask: GPUBufferBinding;
}

export interface AceCopyDispatch<Plan> {
  readonly label: string;
  readonly operation: "gather-rows" | AceAxisCopyOperation | "stable-pack";
  readonly plan: Plan;
  readonly cooperativeQuanta?: readonly Readonly<{
    readonly id: string;
    readonly primitiveCount: number;
    encode(pass: GPUComputePassEncoder): void;
  }>[];
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Untuned, layout-explicit tensor movement used by the Stage 1 graph.
 *
 * Gather indices are independent of KV-cache layout. `repeat` follows
 * PyTorch's axis tiling semantics (`destination % inputLength`), not
 * repeat-interleave. Stable pack exactly implements ACE's concatenate then
 * stable mask partition without a dense mask or JavaScript activation copy.
 */
export class AceCorrectnessTensorCopyKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessTensorCopyKernel {
    requireAceModelProfile(device, modelProfile, "tensor copy");
    return new AceCorrectnessTensorCopyKernel(device, modelProfile);
  }

  async createGatherRowsDispatch(
    label: string,
    shape: AceGatherRowsShape,
    bindings: AceGatherRowsBindings,
  ): Promise<AceCopyDispatch<AceGatherRowsPlan>> {
    this.requireLive();
    const plan = planAceGatherRows(shape);
    requireAceBindingBytes(
      bindings.input,
      aceActivationBytes(this.modelProfile, plan.sourceElements),
      `${label} input`,
    );
    requireAceBindingBytes(
      bindings.indices,
      plan.indexElements * Uint32Array.BYTES_PER_ELEMENT,
      `${label} indices`,
    );
    requireAceBindingBytes(
      bindings.output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.input, bindings.indices],
      label,
    );
    const pipeline = await this.pipelineFor(
      `gather:${gatherKey(plan)}`,
      `${label}-gather-rows`,
      aceCorrectnessGatherRowsWgsl(this.modelProfile, plan),
    );
    this.requireLive(" while compiling");
    return this.singleDispatch(
      label,
      "gather-rows",
      plan,
      pipeline,
      [bindings.input, bindings.indices, bindings.output],
    );
  }

  async createRightPadDispatch(
    label: string,
    shape: AceAxisCopyShape,
    bindings: AceCopyBindings,
  ): Promise<AceCopyDispatch<AceAxisCopyPlan>> {
    this.requireLive();
    const plan = planAceRightPad(shape);
    return this.createAxisDispatch(label, plan, [bindings.input], bindings.output);
  }

  async createCropDispatch(
    label: string,
    shape: AceCropShape,
    bindings: AceCopyBindings,
  ): Promise<AceCopyDispatch<AceAxisCopyPlan>> {
    this.requireLive();
    const plan = planAceCrop(shape);
    return this.createAxisDispatch(label, plan, [bindings.input], bindings.output);
  }

  async createRepeatDispatch(
    label: string,
    shape: AceRepeatShape,
    bindings: AceCopyBindings,
  ): Promise<AceCopyDispatch<AceAxisCopyPlan>> {
    this.requireLive();
    const plan = planAceRepeat(shape);
    return this.createAxisDispatch(label, plan, [bindings.input], bindings.output);
  }

  async createConcatDispatch(
    label: string,
    shape: AceConcatShape,
    bindings: AceConcatBindings,
  ): Promise<AceCopyDispatch<AceAxisCopyPlan>> {
    this.requireLive();
    const plan = planAceConcat(shape);
    return this.createAxisDispatch(
      label,
      plan,
      [bindings.left, bindings.right],
      bindings.output,
    );
  }

  async createStablePackDispatch(
    label: string,
    shape: AceStablePackShape,
    bindings: AceStablePackBindings,
  ): Promise<AceCopyDispatch<AceStablePackPlan>> {
    this.requireLive();
    const plan = planAceStablePack(shape);
    const activationBytes = (elements: number): number =>
      aceActivationBytes(this.modelProfile, elements);
    requireAceBindingBytes(
      bindings.left,
      activationBytes(plan.leftElements),
      `${label} left`,
    );
    requireAceBindingBytes(
      bindings.right,
      activationBytes(plan.rightElements),
      `${label} right`,
    );
    requireAceBindingBytes(
      bindings.leftMask,
      plan.batch * plan.leftLength * 4,
      `${label} left mask`,
    );
    requireAceBindingBytes(
      bindings.rightMask,
      plan.batch * plan.rightLength * 4,
      `${label} right mask`,
    );
    requireAceBindingBytes(
      bindings.indicesScratch,
      plan.packedRows * 4,
      `${label} index scratch`,
    );
    requireAceBindingBytes(
      bindings.output,
      activationBytes(plan.outputElements),
      `${label} output`,
    );
    requireAceBindingBytes(
      bindings.outputMask,
      plan.packedRows * 4,
      `${label} output mask`,
    );
    const readBindings = [
      bindings.left,
      bindings.right,
      bindings.leftMask,
      bindings.rightMask,
    ];
    requireAceDisjointOutput(bindings.indicesScratch, readBindings, `${label} index scratch`);
    requireAceDisjointOutput(
      bindings.outputMask,
      [...readBindings, bindings.indicesScratch],
      `${label} output mask`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [...readBindings, bindings.indicesScratch, bindings.outputMask],
      label,
    );

    const [indexPipeline, gatherPipeline] = await Promise.all([
      this.pipelineFor(
        `stable-pack-indices:${stablePackKey(plan)}`,
        `${label}-stable-pack-indices`,
        aceCorrectnessStablePackIndicesWgsl(plan),
      ),
      this.pipelineFor(
        `stable-pack-gather:${stablePackKey(plan)}`,
        `${label}-stable-pack-gather`,
        aceCorrectnessStablePackGatherWgsl(this.modelProfile, plan),
      ),
    ]);
    this.requireLive(" while compiling");
    const indexBindGroup = this.device.createBindGroup({
      label: `${label}-stable-pack-index-bindings`,
      layout: indexPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.leftMask },
        { binding: 1, resource: bindings.rightMask },
        { binding: 2, resource: bindings.indicesScratch },
        { binding: 3, resource: bindings.outputMask },
      ],
    });
    const gatherBindGroup = this.device.createBindGroup({
      label: `${label}-stable-pack-gather-bindings`,
      layout: gatherPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.left },
        { binding: 1, resource: bindings.right },
        { binding: 2, resource: bindings.indicesScratch },
        { binding: 3, resource: bindings.output },
      ],
    });
    const cooperativeQuanta = Object.freeze([
      Object.freeze({
        id: `${label}-indices`,
        primitiveCount: 1,
        encode(pass: GPUComputePassEncoder): void {
          pass.setPipeline(indexPipeline);
          pass.setBindGroup(0, indexBindGroup);
          pass.dispatchWorkgroups(
            plan.indexWorkgroupsX,
            plan.indexWorkgroupsY,
            1,
          );
        },
      }),
      Object.freeze({
        id: `${label}-gather`,
        primitiveCount: 1,
        encode(pass: GPUComputePassEncoder): void {
          pass.setPipeline(gatherPipeline);
          pass.setBindGroup(0, gatherBindGroup);
          pass.dispatchWorkgroups(
            plan.gatherWorkgroupsX,
            plan.gatherWorkgroupsY,
            1,
          );
        },
      }),
    ]);
    return Object.freeze({
      label,
      operation: "stable-pack" as const,
      plan,
      cooperativeQuanta,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(indexPipeline);
        pass.setBindGroup(0, indexBindGroup);
        pass.dispatchWorkgroups(
          plan.indexWorkgroupsX,
          plan.indexWorkgroupsY,
          1,
        );
        pass.setPipeline(gatherPipeline);
        pass.setBindGroup(0, gatherBindGroup);
        pass.dispatchWorkgroups(
          plan.gatherWorkgroupsX,
          plan.gatherWorkgroupsY,
          1,
        );
      },
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private async createAxisDispatch(
    label: string,
    plan: AceAxisCopyPlan,
    inputs: readonly GPUBufferBinding[],
    output: GPUBufferBinding,
  ): Promise<AceCopyDispatch<AceAxisCopyPlan>> {
    this.requireLive();
    if (inputs.length !== (plan.operation === "concat" ? 2 : 1)) {
      throw new Error("ACE tensor copy internal binding count mismatch");
    }
    requireAceBindingBytes(
      inputs[0]!,
      aceActivationBytes(this.modelProfile, plan.inputElements),
      `${label} input`,
    );
    if (plan.operation === "concat") {
      requireAceBindingBytes(
        inputs[1]!,
        aceActivationBytes(this.modelProfile, plan.secondElements),
        `${label} second input`,
      );
    }
    requireAceBindingBytes(
      output,
      aceActivationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    requireAceDisjointOutput(output, inputs, label);
    const pipeline = await this.pipelineFor(
      `axis:${axisKey(plan)}`,
      `${label}-${plan.operation}`,
      aceCorrectnessAxisCopyWgsl(this.modelProfile, plan),
    );
    this.requireLive(" while compiling");
    return this.singleDispatch(
      label,
      plan.operation,
      plan,
      pipeline,
      [...inputs, output],
    );
  }

  private singleDispatch<Plan extends {
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>(
    label: string,
    operation: "gather-rows" | AceAxisCopyOperation,
    plan: Plan,
    pipeline: GPUComputePipeline,
    bindings: readonly GPUBufferBinding[],
  ): AceCopyDispatch<Plan> {
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
      throw new Error(`ACE tensor copy kernel was destroyed${suffix}`);
    }
  }
}

export function planAceGatherRows(shape: AceGatherRowsShape): AceGatherRowsPlan {
  const sourceElements = checkedAceProduct(
    [shape.outer, shape.sourceRows, shape.width],
    "ACE gather source",
  );
  const indexElements = checkedAceProduct(
    [shape.outer, shape.outputRows],
    "ACE gather indices",
  );
  const outputElements = checkedAceProduct(
    [shape.outer, shape.outputRows, shape.width],
    "ACE gather output",
  );
  const dispatch = planAceLinearDispatch(outputElements, "ACE gather rows");
  requireAceU32(sourceElements, "ACE gather source elements");
  requireAceU32(indexElements, "ACE gather index elements");
  return Object.freeze({
    ...shape,
    sourceElements,
    indexElements,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function planAceRightPad(shape: AceAxisCopyShape): AceAxisCopyPlan {
  if (shape.outputLength < shape.inputLength) {
    throw new RangeError("ACE right-pad outputLength must cover inputLength");
  }
  return planAxisCopy("right-pad", {
    ...shape,
    secondLength: 0,
    offset: 0,
    repeats: 1,
  });
}

export function planAceCrop(shape: AceCropShape): AceAxisCopyPlan {
  requireNonNegativeSafeInteger(shape.offset, "ACE crop offset");
  if (
    !Number.isSafeInteger(shape.offset + shape.outputLength) ||
    shape.offset + shape.outputLength > shape.inputLength
  ) {
    throw new RangeError("ACE crop range exceeds inputLength");
  }
  return planAxisCopy("crop", {
    ...shape,
    secondLength: 0,
    repeats: 1,
  });
}

export function planAceRepeat(shape: AceRepeatShape): AceAxisCopyPlan {
  requirePositiveSafeInteger(shape.repeats, "ACE repeat count");
  const outputLength = checkedAceProduct(
    [shape.inputLength, shape.repeats],
    "ACE repeated axis",
  );
  return planAxisCopy("repeat", {
    ...shape,
    secondLength: 0,
    offset: 0,
    outputLength,
  });
}

export function planAceConcat(shape: AceConcatShape): AceAxisCopyPlan {
  requirePositiveSafeInteger(shape.leftLength, "ACE concat leftLength");
  requirePositiveSafeInteger(shape.rightLength, "ACE concat rightLength");
  const outputLength = checkedAceSum(
    shape.leftLength,
    shape.rightLength,
    "ACE concat axis",
  );
  return planAxisCopy("concat", {
    outer: shape.outer,
    inputLength: shape.leftLength,
    secondLength: shape.rightLength,
    offset: 0,
    outputLength,
    repeats: 1,
    inner: shape.inner,
  });
}

export function planAceStablePack(shape: AceStablePackShape): AceStablePackPlan {
  const packedLength = checkedAceSum(
    shape.leftLength,
    shape.rightLength,
    "ACE stable-pack length",
  );
  requirePositiveSafeInteger(shape.batch, "ACE stable-pack batch");
  requirePositiveSafeInteger(shape.leftLength, "ACE stable-pack leftLength");
  requirePositiveSafeInteger(shape.rightLength, "ACE stable-pack rightLength");
  requirePositiveSafeInteger(shape.width, "ACE stable-pack width");
  const leftElements = checkedAceProduct(
    [shape.batch, shape.leftLength, shape.width],
    "ACE stable-pack left",
  );
  const rightElements = checkedAceProduct(
    [shape.batch, shape.rightLength, shape.width],
    "ACE stable-pack right",
  );
  const packedRows = checkedAceProduct(
    [shape.batch, packedLength],
    "ACE stable-pack rows",
  );
  const outputElements = checkedAceProduct(
    [packedRows, shape.width],
    "ACE stable-pack output",
  );
  const indexDispatch = planAceLinearDispatch(
    packedRows,
    "ACE stable-pack indices",
  );
  const gatherDispatch = planAceLinearDispatch(
    outputElements,
    "ACE stable-pack gather",
  );
  return Object.freeze({
    ...shape,
    packedLength,
    leftElements,
    rightElements,
    packedRows,
    outputElements,
    indexWorkgroupsX: indexDispatch.workgroupsX,
    indexWorkgroupsY: indexDispatch.workgroupsY,
    gatherWorkgroupsX: gatherDispatch.workgroupsX,
    gatherWorkgroupsY: gatherDispatch.workgroupsY,
  });
}

export function aceCorrectnessGatherRowsWgsl(
  modelProfile: AceModelProfileId,
  shape: AceGatherRowsShape,
): string {
  const plan = planAceGatherRows(shape);
  const type = activationType(modelProfile, "gather rows");
  return /* wgsl */ `${type.enable}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const SOURCE_ROWS: u32 = ${plan.sourceRows}u;
const OUTPUT_ROWS: u32 = ${plan.outputRows}u;
const WIDTH: u32 = ${plan.width}u;

@group(0) @binding(0) var<storage, read> input: array<${type.name}>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<${type.name}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let feature = index % WIDTH;
  let output_row = (index / WIDTH) % OUTPUT_ROWS;
  let outer = index / (WIDTH * OUTPUT_ROWS);
  let source_row = indices[outer * OUTPUT_ROWS + output_row];
  if (source_row >= SOURCE_ROWS) {
    output[index] = ${type.zero};
    return;
  }
  let source_index = (outer * SOURCE_ROWS + source_row) * WIDTH + feature;
  output[index] = input[source_index];
}
`;
}

export function aceCorrectnessAxisCopyWgsl(
  modelProfile: AceModelProfileId,
  planOrShape: AceAxisCopyPlan,
): string {
  const plan = canonicalAxisCopyPlan(planOrShape);
  const type = activationType(modelProfile, `tensor ${plan.operation}`);
  const source = axisCopySource(plan, type.zero);
  const secondBinding = plan.operation === "concat"
    ? `@group(0) @binding(1) var<storage, read> second: array<${type.name}>;\n@group(0) @binding(2) var<storage, read_write> output: array<${type.name}>;`
    : `@group(0) @binding(1) var<storage, read_write> output: array<${type.name}>;`;
  return /* wgsl */ `${type.enable}
${aceLinearInvocationWgsl(plan.outputElements, plan.workgroupsX)}
const INPUT_LENGTH: u32 = ${plan.inputLength}u;
const SECOND_LENGTH: u32 = ${plan.secondLength}u;
const OUTPUT_LENGTH: u32 = ${plan.outputLength}u;
const INNER: u32 = ${plan.inner}u;
const OFFSET: u32 = ${plan.offset}u;

@group(0) @binding(0) var<storage, read> input: array<${type.name}>;
${secondBinding}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let inner = index % INNER;
  let output_axis = (index / INNER) % OUTPUT_LENGTH;
  let outer = index / (INNER * OUTPUT_LENGTH);
  ${source}
}
`;
}

export function aceCorrectnessStablePackIndicesWgsl(
  shape: AceStablePackShape,
): string {
  const plan = planAceStablePack(shape);
  return /* wgsl */ `${aceLinearInvocationWgsl(plan.packedRows, plan.indexWorkgroupsX)}
const LEFT_LENGTH: u32 = ${plan.leftLength}u;
const RIGHT_LENGTH: u32 = ${plan.rightLength}u;
const PACKED_LENGTH: u32 = ${plan.packedLength}u;

@group(0) @binding(0) var<storage, read> left_mask: array<u32>;
@group(0) @binding(1) var<storage, read> right_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_mask: array<u32>;

fn mask_at(batch: u32, source: u32) -> bool {
  if (source < LEFT_LENGTH) {
    return left_mask[batch * LEFT_LENGTH + source] != 0u;
  }
  return right_mask[batch * RIGHT_LENGTH + source - LEFT_LENGTH] != 0u;
}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let batch = index / PACKED_LENGTH;
  let destination = index % PACKED_LENGTH;
  var valid_count = 0u;
  for (var source = 0u; source < PACKED_LENGTH; source += 1u) {
    if (mask_at(batch, source)) { valid_count += 1u; }
  }
  let wants_valid = destination < valid_count;
  let rank = select(destination - valid_count, destination, wants_valid);
  var seen = 0u;
  var chosen = 0u;
  for (var source = 0u; source < PACKED_LENGTH; source += 1u) {
    if (mask_at(batch, source) == wants_valid) {
      if (seen == rank) {
        chosen = source;
        break;
      }
      seen += 1u;
    }
  }
  indices[index] = chosen;
  output_mask[index] = select(0u, 1u, wants_valid);
}
`;
}

export function aceCorrectnessStablePackGatherWgsl(
  modelProfile: AceModelProfileId,
  shape: AceStablePackShape,
): string {
  const plan = planAceStablePack(shape);
  const type = activationType(modelProfile, "stable pack");
  return /* wgsl */ `${type.enable}
${aceLinearInvocationWgsl(plan.outputElements, plan.gatherWorkgroupsX)}
const LEFT_LENGTH: u32 = ${plan.leftLength}u;
const RIGHT_LENGTH: u32 = ${plan.rightLength}u;
const PACKED_LENGTH: u32 = ${plan.packedLength}u;
const WIDTH: u32 = ${plan.width}u;

@group(0) @binding(0) var<storage, read> left: array<${type.name}>;
@group(0) @binding(1) var<storage, read> right: array<${type.name}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<${type.name}>;

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = linear_index(workgroup_id, lane);
  if (index >= ELEMENTS) { return; }
  let feature = index % WIDTH;
  let packed_row = (index / WIDTH) % PACKED_LENGTH;
  let batch = index / (WIDTH * PACKED_LENGTH);
  let source = indices[batch * PACKED_LENGTH + packed_row];
  if (source < LEFT_LENGTH) {
    output[index] = left[(batch * LEFT_LENGTH + source) * WIDTH + feature];
  } else {
    let right_row = source - LEFT_LENGTH;
    output[index] = right[(batch * RIGHT_LENGTH + right_row) * WIDTH + feature];
  }
}
`;
}

function planAxisCopy(
  operation: AceAxisCopyOperation,
  shape: {
    readonly outer: number;
    readonly inputLength: number;
    readonly secondLength: number;
    readonly offset: number;
    readonly outputLength: number;
    readonly repeats: number;
    readonly inner: number;
  },
): AceAxisCopyPlan {
  requirePositiveSafeInteger(shape.outer, `ACE ${operation} outer`);
  requirePositiveSafeInteger(shape.inputLength, `ACE ${operation} inputLength`);
  requireNonNegativeSafeInteger(shape.secondLength, `ACE ${operation} secondLength`);
  requireNonNegativeSafeInteger(shape.offset, `ACE ${operation} offset`);
  requirePositiveSafeInteger(shape.outputLength, `ACE ${operation} outputLength`);
  requirePositiveSafeInteger(shape.repeats, `ACE ${operation} repeats`);
  requirePositiveSafeInteger(shape.inner, `ACE ${operation} inner`);
  const inputElements = checkedAceProduct(
    [shape.outer, shape.inputLength, shape.inner],
    `ACE ${operation} input`,
  );
  requireAceU32(inputElements, `ACE ${operation} input elements`);
  const secondElements = shape.secondLength === 0
    ? 0
    : checkedAceProduct(
      [shape.outer, shape.secondLength, shape.inner],
      `ACE ${operation} second input`,
    );
  requireAceU32(secondElements, `ACE ${operation} second-input elements`);
  const outputElements = checkedAceProduct(
    [shape.outer, shape.outputLength, shape.inner],
    `ACE ${operation} output`,
  );
  const dispatch = planAceLinearDispatch(outputElements, `ACE ${operation}`);
  return Object.freeze({
    operation,
    ...shape,
    inputElements,
    secondElements,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

function axisCopySource(plan: AceAxisCopyPlan, zero: string): string {
  switch (plan.operation) {
    case "right-pad":
      return /* wgsl */ `if (output_axis < INPUT_LENGTH) {
    output[index] = input[(outer * INPUT_LENGTH + output_axis) * INNER + inner];
  } else {
    output[index] = ${zero};
  }`;
    case "crop":
      return "let source_axis = output_axis + OFFSET;\n  output[index] = input[(outer * INPUT_LENGTH + source_axis) * INNER + inner];";
    case "repeat":
      return "let source_axis = output_axis % INPUT_LENGTH;\n  output[index] = input[(outer * INPUT_LENGTH + source_axis) * INNER + inner];";
    case "concat":
      return /* wgsl */ `if (output_axis < INPUT_LENGTH) {
    output[index] = input[(outer * INPUT_LENGTH + output_axis) * INNER + inner];
  } else {
    let second_axis = output_axis - INPUT_LENGTH;
    output[index] = second[(outer * SECOND_LENGTH + second_axis) * INNER + inner];
  }`;
    default:
      throw new TypeError(
        `Unknown ACE axis-copy operation ${String(plan.operation)}`,
      );
  }
}

function canonicalAxisCopyPlan(plan: AceAxisCopyPlan): AceAxisCopyPlan {
  switch (plan.operation) {
    case "right-pad":
      return planAceRightPad({
        outer: plan.outer,
        inputLength: plan.inputLength,
        outputLength: plan.outputLength,
        inner: plan.inner,
      });
    case "crop":
      return planAceCrop({
        outer: plan.outer,
        inputLength: plan.inputLength,
        offset: plan.offset,
        outputLength: plan.outputLength,
        inner: plan.inner,
      });
    case "repeat":
      return planAceRepeat({
        outer: plan.outer,
        inputLength: plan.inputLength,
        repeats: plan.repeats,
        inner: plan.inner,
      });
    case "concat":
      return planAceConcat({
        outer: plan.outer,
        leftLength: plan.inputLength,
        rightLength: plan.secondLength,
        inner: plan.inner,
      });
    default:
      throw new TypeError(
        `Unknown ACE axis-copy operation ${String(plan.operation)}`,
      );
  }
}

function activationType(
  modelProfile: AceModelProfileId,
  operation: string,
): { readonly enable: string; readonly name: "f32" | "f16"; readonly zero: string } {
  if (modelProfile === "reference-bf16") {
    return { enable: "", name: "f32", zero: "0.0" };
  }
  if (modelProfile === "raw-fp16") {
    return { enable: "enable f16;", name: "f16", zero: "f16(0.0)" };
  }
  throw new TypeError(
    `Unknown ACE ${operation} model profile ${String(modelProfile)}`,
  );
}

function gatherKey(shape: AceGatherRowsShape): string {
  return `${shape.outer}x${shape.sourceRows}x${shape.outputRows}x${shape.width}`;
}

function axisKey(plan: AceAxisCopyPlan): string {
  return `${plan.operation}:${plan.outer}x${plan.inputLength}x${plan.secondLength}x${plan.offset}x${plan.outputLength}x${plan.repeats}x${plan.inner}`;
}

function stablePackKey(shape: AceStablePackShape): string {
  return `${shape.batch}x${shape.leftLength}x${shape.rightLength}x${shape.width}`;
}
