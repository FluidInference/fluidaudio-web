import type { AceDynamicConditionalWeightingConfiguration } from "../../api.js";
import type { AceModelProfileId } from "../capabilities.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";

const WORKGROUP_SIZE = 256;
const MAX_DISPATCH_DIMENSION = 65_535;

export interface AceDcwShape {
  readonly batch: number;
  readonly time: number;
  readonly channels: number;
}

export interface AceDcwPlan extends AceDcwShape {
  readonly timePairs: number;
  readonly pairChannels: number;
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceDcwStepScales {
  readonly currentTimestep: number;
  readonly lowBandScale: number;
  readonly highBandScale: number;
}

export interface AceDcwBindings {
  readonly steppedLatent: GPUBufferBinding;
  readonly predictedCleanLatent: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceDcwDispatch {
  readonly label: string;
  readonly plan: AceDcwPlan;
  readonly scales: AceDcwStepScales;
  encode(pass: GPUComputePassEncoder): void;
}

/** One-level zero-padded Haar DCW from the pinned upstream sampler. */
export class AceCorrectnessDcwKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessDcwKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE raw-FP16 DCW requires WebGPU shader-f16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE DCW model profile ${String(modelProfile)}`);
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
    ) {
      throw new Error("ACE DCW requires 256 compute lanes");
    }
    return new AceCorrectnessDcwKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceDcwShape,
    configuration: AceDynamicConditionalWeightingConfiguration,
    currentTimestep: number,
    bindings: AceDcwBindings,
  ): Promise<AceDcwDispatch> {
    if (this.destroyed) throw new Error("ACE DCW kernel was destroyed");
    const plan = planAceDcw(shape);
    const scales = deriveAceDcwStepScales(configuration, currentTimestep);
    const elementBytes = this.modelProfile === "raw-fp16" ? 2 : 4;
    const requiredBytes = plan.elements * elementBytes;
    requireBindingBytes(bindings.steppedLatent, requiredBytes, `${label} stepped latent`);
    requireBindingBytes(
      bindings.predictedCleanLatent,
      requiredBytes,
      `${label} predicted-clean latent`,
    );
    requireBindingBytes(bindings.output, requiredBytes, `${label} output`);
    requireAceDisjointOutput(
      bindings.output,
      [bindings.steppedLatent, bindings.predictedCleanLatent],
      label,
    );
    const pipeline = await this.pipelineFor(plan, scales);
    if (this.destroyed) throw new Error("ACE DCW kernel was destroyed while compiling");
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.steppedLatent },
        { binding: 1, resource: bindings.predictedCleanLatent },
        { binding: 2, resource: bindings.output },
      ],
    });
    return Object.freeze({
      label,
      plan,
      scales,
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
    plan: AceDcwPlan,
    scales: AceDcwStepScales,
  ): Promise<GPUComputePipeline> {
    const key = [
      plan.batch,
      plan.time,
      plan.channels,
      f32Hex(scales.lowBandScale),
      f32Hex(scales.highBandScale),
    ].join("x");
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-dcw-${this.modelProfile}-${key}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessDcwWgsl(this.modelProfile, plan, scales),
    });
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
}

export function planAceDcw(shape: AceDcwShape): AceDcwPlan {
  const { batch, time, channels } = shape;
  for (const [name, value] of Object.entries({ batch, time, channels })) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`ACE DCW ${name} must be a positive safe integer`);
    }
  }
  const timePairs = Math.ceil(time / 2);
  const pairChannels = batch * timePairs * channels;
  const elements = batch * time * channels;
  if (!Number.isSafeInteger(pairChannels) || !Number.isSafeInteger(elements)) {
    throw new RangeError("ACE DCW element count is not a safe integer");
  }
  const totalWorkgroups = Math.ceil(pairChannels / WORKGROUP_SIZE);
  const workgroupsX = Math.min(totalWorkgroups, MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(totalWorkgroups / workgroupsX);
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("ACE DCW exceeds the portable 2D dispatch domain");
  }
  return Object.freeze({
    batch,
    time,
    channels,
    timePairs,
    pairChannels,
    elements,
    workgroupsX,
    workgroupsY,
  });
}

export function deriveAceDcwStepScales(
  configuration: AceDynamicConditionalWeightingConfiguration,
  currentTimestep: number,
): AceDcwStepScales {
  if (
    configuration.enabled !== true ||
    configuration.mode !== "double" ||
    configuration.wavelet !== "haar" ||
    typeof configuration.lowBandScale !== "number" ||
    !Number.isFinite(configuration.lowBandScale) ||
    configuration.lowBandScale < 0 ||
    typeof configuration.highBandScale !== "number" ||
    !Number.isFinite(configuration.highBandScale) ||
    configuration.highBandScale < 0
  ) {
    throw new TypeError("ACE v1 requires enabled double-band Haar DCW with finite scales");
  }
  if (
    typeof currentTimestep !== "number" ||
    !Number.isFinite(currentTimestep) ||
    currentTimestep < 0 ||
    currentTimestep > 1
  ) {
    throw new RangeError("ACE DCW timestep must be finite in [0, 1]");
  }
  return Object.freeze({
    currentTimestep,
    lowBandScale: Math.fround(currentTimestep * configuration.lowBandScale),
    highBandScale: Math.fround(
      (1 - currentTimestep) * configuration.highBandScale,
    ),
  });
}

export function aceCorrectnessDcwWgsl(
  modelProfile: AceModelProfileId,
  shape: AceDcwShape,
  scales: Pick<AceDcwStepScales, "lowBandScale" | "highBandScale">,
): string {
  const plan = planAceDcw(shape);
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE DCW model profile ${String(modelProfile)}`);
  }
  for (const [name, value] of Object.entries(scales)) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new RangeError(`ACE DCW ${name} must be finite and non-negative`);
    }
  }
  const f16 = modelProfile === "raw-fp16";
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
const TIME: u32 = ${plan.time}u;
const CHANNELS: u32 = ${plan.channels}u;
const TIME_PAIRS: u32 = ${plan.timePairs}u;
const PAIR_CHANNELS: u32 = ${plan.pairChannels}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;
const INV_SQRT_TWO: f32 = 0.7071067690849304;
const LOW_SCALE: f32 = ${Math.fround(scales.lowBandScale)};
const HIGH_SCALE: f32 = ${Math.fround(scales.highBandScale)};

@group(0) @binding(0) var<storage, read> stepped: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> clean: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(2) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let flat_pair_channel =
    (workgroup_id.y * DISPATCH_X + workgroup_id.x) * ${WORKGROUP_SIZE}u + lane;
  if (flat_pair_channel >= PAIR_CHANNELS) { return; }
  let channel = flat_pair_channel % CHANNELS;
  let flat_pair = flat_pair_channel / CHANNELS;
  let time_pair = flat_pair % TIME_PAIRS;
  let batch = flat_pair / TIME_PAIRS;
  let even_time = time_pair * 2u;
  let odd_time = even_time + 1u;
  let even_index = (batch * TIME + even_time) * CHANNELS + channel;
  let odd_index = (batch * TIME + odd_time) * CHANNELS + channel;

  let x_even = f32(stepped[even_index]);
  let y_even = f32(clean[even_index]);
  var x_odd = 0.0;
  var y_odd = 0.0;
  if (odd_time < TIME) {
    x_odd = f32(stepped[odd_index]);
    y_odd = f32(clean[odd_index]);
  }

  var x_low = (x_even + x_odd) * INV_SQRT_TWO;
  var x_high = (x_even - x_odd) * INV_SQRT_TWO;
  let y_low = (y_even + y_odd) * INV_SQRT_TWO;
  let y_high = (y_even - y_odd) * INV_SQRT_TWO;
  x_low = x_low + LOW_SCALE * (x_low - y_low);
  x_high = x_high + HIGH_SCALE * (x_high - y_high);

  let corrected_even = (x_low + x_high) * INV_SQRT_TWO;
  let corrected_odd = (x_low - x_high) * INV_SQRT_TWO;
  output[even_index] = ${f16 ? "f16(corrected_even)" : "corrected_even"};
  if (odd_time < TIME) {
    output[odd_index] = ${f16 ? "f16(corrected_odd)" : "corrected_odd"};
  }
}
`;
}

function requireBindingBytes(
  binding: GPUBufferBinding,
  required: number,
  label: string,
): void {
  const offset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - offset;
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(available) ||
    available < required ||
    offset + available > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${required} bytes`);
  }
}

function f32Hex(value: number): string {
  const floats = new Float32Array([value]);
  return new Uint32Array(floats.buffer)[0]!.toString(16).padStart(8, "0");
}
