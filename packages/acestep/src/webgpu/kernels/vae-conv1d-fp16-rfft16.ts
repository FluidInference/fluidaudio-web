import {
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import type {
  AceFp16VaeConv1dFixedSubgroupCapability,
} from "./vae-conv1d-fp16-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";
import {
  ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
  ACE_OPT_0077_RFFT16_LENGTH,
  ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
  ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
  ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32,
  ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32,
  planAceOpt0077Rfft16Range,
  type AceOpt0077Rfft16Dilation,
} from "./vae-conv1d-fp16-rfft16-math.js";

export {
  ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32,
  ACE_OPT_0077_RFFT16_LENGTH,
  ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
  ACE_OPT_0077_RFFT16_PAIR_SCALE_F32,
  ACE_OPT_0077_RFFT16_COORDINATE_ORDER,
  ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32,
  ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32,
  ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT,
  aceOpt0077Rfft16ForwardF32,
  aceOpt0077Rfft16NativeWeightIndex,
  aceOpt0077Rfft16PackedWeightCoordinate,
  aceOpt0077Rfft16PackedWeightIndex,
  aceOpt0077TransformK7WeightF32,
  packAceOpt0077Rfft16WeightU16,
  packAceOpt0077Rfft16WeightU16 as transformAceOpt0077VaeK7WeightU16,
  planAceOpt0077Rfft16Range,
} from "./vae-conv1d-fp16-rfft16-math.js";

export const ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID =
  "ace-opt-0077-vae-k7-rfft16-real-basis-overlap-save-v1" as const;
export const ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES =
  ACE_OPT_0077_RFFT16_LENGTH;
export const ACE_OPT_0077_RFFT16_WORKGROUP_SIZE = 128;
export const ACE_OPT_0077_RFFT16_SUBGROUP_SIZE = 32;
export const ACE_OPT_0077_RFFT16_TRANSFORMS_PER_WORKGROUP = 8;
export const ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP = 16;
export const ACE_OPT_0077_RFFT16_OUTPUT_CHANNELS_PER_DOMAIN_WORKGROUP = 128;
export const ACE_OPT_0077_RFFT16_OUTPUT_CHANNELS_PER_INVERSE_WORKGROUP = 32;
export const ACE_OPT_0077_RFFT16_STAGE_DISPATCH_COUNT = 3;
export const ACE_OPT_0077_RFFT16_REDUCTION_SEMANTICS =
  "real-basis-rfft16-fp16-dot4-partials-fp32-domain-and-inverse" as const;

const GPU_BUFFER_ALIGNMENT = 4;
const F16_VEC4_ALIGNMENT = 8;
const F32_ALIGNMENT = 4;
const OUTPUT_RANGE_CONTROL_BYTES = 16;
const MAX_WGSL_U32 = 0xffff_ffff;
const MAX_WGSL_I32 = 0x7fff_ffff;

export interface AceOpt0077VaeK7Rfft16Bindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 `[coord16, Cin4, Cout, CinElement4]`. */
  readonly transformedWeight: GPUBufferBinding;
  /** FP16 `[Cout]`. */
  readonly bias: GPUBufferBinding;
  /** FP16 frame-major NLC output. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0077VaeK7Rfft16ScratchBindings {
  /** FP16 `[coord16, tile, Cin4, CinElement4]`. */
  readonly inputSpectrum: GPUBufferBinding;
  /** FP32 `[coord16, tile, Cout]`. */
  readonly contractionSpectrum: GPUBufferBinding;
}

export interface AceOpt0077VaeK7Rfft16ScratchBytes {
  readonly inputSpectrum: number;
  readonly contractionSpectrum: number;
  readonly total: number;
}

export interface AceOpt0077VaeK7Rfft16RangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly firstTileGroup: number;
  readonly tileGroupCount: number;
  readonly tileCount: number;
  readonly forwardWorkgroupsX: number;
  readonly forwardWorkgroupsY: number;
  readonly domainWorkgroupsX: number;
  readonly domainWorkgroupsY: number;
  readonly domainWorkgroupsZ: 16;
  readonly inverseWorkgroupsX: number;
  readonly inverseWorkgroupsY: number;
  readonly scratchBytes: AceOpt0077VaeK7Rfft16ScratchBytes;
}

export interface AceOpt0077VaeK7Rfft16Dispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly outputRange: AceOpt0077VaeK7Rfft16RangePlan;
  readonly scratchBytes: AceOpt0077VaeK7Rfft16ScratchBytes;
  readonly stageDispatchCount: 3;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0077VaeK7Rfft16 {
  readonly forward: GPUComputePipeline;
  readonly domain: GPUComputePipeline;
  readonly inverse: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name:
    | "input"
    | "transformed weight"
    | "bias"
    | "output"
    | "input spectrum"
    | "contraction spectrum"
    | "range control";
  readonly binding: GPUBufferBinding;
}

/** Isolated OPT-0077 owner. It never allocates or owns caller scratch. */
export class AceOpt0077VaeK7Rfft16Kernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceOpt0077VaeK7Rfft16>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceOpt0077VaeK7Rfft16Kernel {
    requireFixed32Device(device, capability);
    return new AceOpt0077VaeK7Rfft16Kernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0077VaeK7Rfft16Bindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
    scratch: AceOpt0077VaeK7Rfft16ScratchBindings,
  ): Promise<AceOpt0077VaeK7Rfft16Dispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireCandidateBoundary(plan, true);
    const outputRange = planAceOpt0077VaeK7Rfft16Range(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = this.requireBindings(
      label,
      plan,
      outputRange,
      bindings,
      scratch,
      range,
    );
    const compiled = await this.pipelineFor(plan);
    this.requireLive();

    const rangeBindingIndex = normalized.length - 1;
    const controlOffset = normalized[rangeBindingIndex]!.binding.offset ?? 0;
    const resources = normalized.map(({ binding }, index) =>
      index === rangeBindingIndex
        ? Object.freeze({
            buffer: binding.buffer,
            offset: 0,
            size: OUTPUT_RANGE_CONTROL_BYTES,
          })
        : binding
    );
    const bindGroupKey = `${convKey(plan)}:${resources.map(
      (binding) => this.bindingKey(binding),
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0077-rfft16-bindings`,
        layout: compiled.bindGroupLayout,
        entries: resources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID,
      plan,
      outputRange,
      scratchBytes: outputRange.scratchBytes,
      stageDispatchCount: ACE_OPT_0077_RFFT16_STAGE_DISPATCH_COUNT,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.setPipeline(compiled.forward);
        pass.dispatchWorkgroups(
          outputRange.forwardWorkgroupsX,
          outputRange.forwardWorkgroupsY,
          1,
        );
        pass.setPipeline(compiled.domain);
        pass.dispatchWorkgroups(
          outputRange.domainWorkgroupsX,
          outputRange.domainWorkgroupsY,
          outputRange.domainWorkgroupsZ,
        );
        pass.setPipeline(compiled.inverse);
        pass.dispatchWorkgroups(
          outputRange.inverseWorkgroupsX,
          outputRange.inverseWorkgroupsY,
          1,
        );
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private pipelineFor(
    plan: AceFp16VaeConv1dPlan,
  ): Promise<CompiledAceOpt0077VaeK7Rfft16> {
    const key = convKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0077VaeK7Rfft16(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireBindings(
    label: string,
    plan: AceFp16VaeConv1dPlan,
    outputRange: AceOpt0077VaeK7Rfft16RangePlan,
    bindings: AceOpt0077VaeK7Rfft16Bindings,
    scratch: AceOpt0077VaeK7Rfft16ScratchBindings,
    range: AceVaeOutputRangeBinding,
  ): readonly NamedBinding[] {
    const weightBytes = transformedWeightBytes(plan);
    const normalized = Object.freeze([
      namedStorage(
        this.device,
        "input",
        bindings.input,
        plan.inputStorageBytes,
        plan.inputBindingBytes,
        F16_VEC4_ALIGNMENT,
        label,
      ),
      namedStorage(
        this.device,
        "transformed weight",
        bindings.transformedWeight,
        weightBytes,
        weightBytes,
        F16_VEC4_ALIGNMENT,
        label,
      ),
      namedStorage(
        this.device,
        "bias",
        bindings.bias,
        plan.biasStorageBytes,
        plan.biasBindingBytes,
        GPU_BUFFER_ALIGNMENT,
        label,
      ),
      namedStorage(
        this.device,
        "output",
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        F16_VEC4_ALIGNMENT,
        label,
      ),
      namedStorage(
        this.device,
        "input spectrum",
        scratch.inputSpectrum,
        outputRange.scratchBytes.inputSpectrum,
        outputRange.scratchBytes.inputSpectrum,
        F16_VEC4_ALIGNMENT,
        label,
      ),
      namedStorage(
        this.device,
        "contraction spectrum",
        scratch.contractionSpectrum,
        outputRange.scratchBytes.contractionSpectrum,
        outputRange.scratchBytes.contractionSpectrum,
        F32_ALIGNMENT,
        label,
      ),
      Object.freeze({
        name: "range control" as const,
        binding: requireRangeControlBinding(
          this.device,
          range.control,
          `${label} range control`,
        ),
      }),
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    return normalized;
  }

  private requireDeviceLimits(
    plan: AceFp16VaeConv1dPlan,
    range: AceOpt0077VaeK7Rfft16RangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    for (const [name, value] of [
      ["forward x", range.forwardWorkgroupsX],
      ["forward y", range.forwardWorkgroupsY],
      ["domain x", range.domainWorkgroupsX],
      ["domain y", range.domainWorkgroupsY],
      ["domain z", range.domainWorkgroupsZ],
      ["inverse x", range.inverseWorkgroupsX],
      ["inverse y", range.inverseWorkgroupsY],
    ] as const) {
      if (!Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
        value > maximumDispatch) {
        throw new RangeError(`OPT-0077 ${name} dispatch exceeds the device limit`);
      }
    }
    const maximumBinding = Number(this.device.limits.maxStorageBufferBindingSize);
    const maximumBuffer = Number(this.device.limits.maxBufferSize);
    for (const [name, bytes] of [
      ["input", plan.inputBindingBytes],
      ["transformed weight", transformedWeightBytes(plan)],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
      ["input spectrum", range.scratchBytes.inputSpectrum],
      ["contraction spectrum", range.scratchBytes.contractionSpectrum],
    ] as const) {
      if (!Number.isSafeInteger(maximumBinding) || maximumBinding < bytes ||
        !Number.isSafeInteger(maximumBuffer) || maximumBuffer < bytes) {
        throw new RangeError(`OPT-0077 ${name} exceeds the device buffer limits`);
      }
    }
  }

  private bindingKey(binding: GPUBufferBinding): string {
    let id = this.bufferIds.get(binding.buffer);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(binding.buffer, id);
    }
    return `${id}:${binding.offset ?? 0}:${binding.size ?? -1}`;
  }

  private requireLive(): void {
    if (this.destroyed) throw new Error("OPT-0077 RFFT16 kernel was destroyed");
  }
}

export function planAceOpt0077VaeK7Rfft16Range(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0077VaeK7Rfft16RangePlan {
  requireCandidateBoundary(plan, true);
  const portable = planAceFp16VaeConv1dRange(plan, range);
  const tileSpan = ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE * plan.dilation;
  const firstTileGroup = Math.floor(portable.firstOutputTime / tileSpan);
  const endOutputTime = portable.firstOutputTime + portable.outputRowCount;
  const endTileGroup = Math.ceil(endOutputTime / tileSpan);
  const tileGroupCount = endTileGroup - firstTileGroup;
  const spectralRange = planAceOpt0077Rfft16Range(
    plan.outputFrames,
    plan.dilation as AceOpt0077Rfft16Dilation,
    {
      firstOutputTime: portable.firstOutputTime,
      outputTimeCount: portable.outputRowCount,
    },
  );
  const tileCount = spectralRange.tiles.length;
  const inputSpectrum = checkedBytes(
    ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES * tileCount * plan.inputChannels * 2,
    "input spectrum",
  );
  const contractionSpectrum = checkedBytes(
    ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES * tileCount * plan.outputChannels * 4,
    "contraction spectrum",
  );
  const scratchBytes = Object.freeze({
    inputSpectrum,
    contractionSpectrum,
    total: checkedBytes(inputSpectrum + contractionSpectrum, "total scratch"),
  });
  const result = Object.freeze({
    base: portable.base,
    count: portable.count,
    batch: portable.batch,
    firstOutputTime: portable.firstOutputTime,
    firstOutputRow: portable.firstOutputRow,
    outputRowCount: portable.outputRowCount,
    firstTileGroup,
    tileGroupCount,
    tileCount,
    forwardWorkgroupsX: tileCount,
    forwardWorkgroupsY: Math.ceil(
      (plan.inputChannels / 4) /
        ACE_OPT_0077_RFFT16_TRANSFORMS_PER_WORKGROUP,
    ),
    domainWorkgroupsX: Math.ceil(
      tileCount / ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP,
    ),
    domainWorkgroupsY: plan.outputChannels /
      ACE_OPT_0077_RFFT16_OUTPUT_CHANNELS_PER_DOMAIN_WORKGROUP,
    domainWorkgroupsZ:
      ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES as 16,
    inverseWorkgroupsX: tileCount,
    inverseWorkgroupsY: plan.outputChannels /
      ACE_OPT_0077_RFFT16_OUTPUT_CHANNELS_PER_INVERSE_WORKGROUP,
    scratchBytes,
  });
  for (const value of [
    firstTileGroup,
    tileGroupCount,
    tileCount,
    endOutputTime,
  ]) {
    if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
      throw new RangeError("OPT-0077 RFFT16 range exceeds safe WGSL indexing");
    }
  }
  return result;
}

export function aceOpt0077VaeK7Rfft16Wgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  requireCandidateBoundary(plan, true);
  const domainInitializers = Array.from(
    { length: ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP },
    (_, tile) => `  var sum${tile}: f32 = 0.0;`,
  ).join("\n");
  const domainUpdates = Array.from(
    { length: ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP },
    (_, tile) => /* wgsl */ `
      let x0_${tile} = subgroupBroadcast(lane_x0, ${tile}u);
      let x1_${tile} = subgroupBroadcast(lane_x1, ${tile}u);
      if (result_coord < 2u) {
        sum${tile} += f32(dot(x0_${tile}, weight0));
      } else if ((result_coord & 1u) == 0u) {
        sum${tile} += f32(dot(x0_${tile}, weight0));
        sum${tile} += f32(dot(x1_${tile}, weight1));
      } else {
        sum${tile} += f32(dot(x0_${tile}, weight1));
        sum${tile} -= f32(dot(x1_${tile}, weight0));
      }`,
  ).join("");
  const domainStores = Array.from(
    { length: ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP },
    (_, tile) => /* wgsl */ `
  let tile${tile} = tile_base + ${tile}u;
  if (tile${tile} < tile_count) {
    contraction_spectrum[
      (result_coord * tile_count + tile${tile}) * OUTPUT_CHANNELS +
        output_channel
    ] = sum${tile};
  }`,
  ).join("");
  const forwardTwiddleCases = ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32
    .slice(0, 7)
    .map((real, exponent) =>
      `    case ${exponent}u: { return vec2<f32>(${wgslF32(real)}, ${
        wgslF32(ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32[exponent]!)
      }); }`
    )
    .join("\n");
  const finalTwiddle = `    default: { return vec2<f32>(${
    wgslF32(ACE_OPT_0077_RFFT16_TWIDDLE_REAL_F32[7])
  }, ${wgslF32(ACE_OPT_0077_RFFT16_TWIDDLE_IMAG_F32[7])}); }`;
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID}
// reduction-semantics: ${ACE_OPT_0077_RFFT16_REDUCTION_SEMANTICS}
// coordinate-order: DC,NYQUIST,COS1,SIN1,...,COS7,SIN7
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const INPUT_CHANNEL_VEC4S: u32 = ${plan.inputChannels / 4}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_VEC4S: u32 = ${plan.outputChannels / 4}u;
const DILATION: u32 = ${plan.dilation}u;
const TILE_TIME_SPAN: u32 = ${
    ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE * plan.dilation
  }u;
const ENDPOINT_SCALE: f32 = ${wgslF32(ACE_OPT_0077_RFFT16_ENDPOINT_SCALE_F32)};
const PAIR_SCALE: f32 = ${wgslF32(ACE_OPT_0077_RFFT16_PAIR_SCALE_F32)};

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>
  transformed_weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write>
  input_spectrum: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write>
  contraction_spectrum: array<f32>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(6) var<uniform>
  output_range: OutputRangeParameters;

struct Complex4 {
  re: vec4<f32>,
  im: vec4<f32>,
}

fn bit_reverse4(value: u32) -> u32 {
  return ((value & 1u) << 3u) | ((value & 2u) << 1u) |
    ((value & 4u) >> 1u) | ((value & 8u) >> 3u);
}

fn forward_twiddle(exponent: u32) -> vec2<f32> {
  switch exponent {
${forwardTwiddleCases}
${finalTwiddle}
  }
}

fn inverse_twiddle(exponent: u32) -> vec2<f32> {
  let value = forward_twiddle(exponent);
  return vec2<f32>(value.x, -value.y);
}

fn butterfly_forward(
  value: Complex4,
  local_lane: u32,
  mask: u32,
  twiddle_stride: u32,
) -> Complex4 {
  let partner_re = subgroupShuffleXor(value.re, mask);
  let partner_im = subgroupShuffleXor(value.im, mask);
  let twiddle = forward_twiddle((local_lane & (mask - 1u)) * twiddle_stride);
  var lower_re: vec4<f32>;
  var lower_im: vec4<f32>;
  var upper_re: vec4<f32>;
  var upper_im: vec4<f32>;
  if ((local_lane & mask) == 0u) {
    lower_re = value.re;
    lower_im = value.im;
    upper_re = partner_re;
    upper_im = partner_im;
  } else {
    lower_re = partner_re;
    lower_im = partner_im;
    upper_re = value.re;
    upper_im = value.im;
  }
  let product_re = upper_re * twiddle.x - upper_im * twiddle.y;
  let product_im = upper_re * twiddle.y + upper_im * twiddle.x;
  if ((local_lane & mask) == 0u) {
    return Complex4(lower_re + product_re, lower_im + product_im);
  }
  return Complex4(lower_re - product_re, lower_im - product_im);
}

fn butterfly_inverse(
  value: Complex4,
  local_lane: u32,
  mask: u32,
  twiddle_stride: u32,
) -> Complex4 {
  let partner_re = subgroupShuffleXor(value.re, mask);
  let partner_im = subgroupShuffleXor(value.im, mask);
  let twiddle = inverse_twiddle((local_lane & (mask - 1u)) * twiddle_stride);
  var lower_re: vec4<f32>;
  var lower_im: vec4<f32>;
  var upper_re: vec4<f32>;
  var upper_im: vec4<f32>;
  if ((local_lane & mask) == 0u) {
    lower_re = value.re;
    lower_im = value.im;
    upper_re = partner_re;
    upper_im = partner_im;
  } else {
    lower_re = partner_re;
    lower_im = partner_im;
    upper_re = value.re;
    upper_im = value.im;
  }
  let product_re = upper_re * twiddle.x - upper_im * twiddle.y;
  let product_im = upper_re * twiddle.y + upper_im * twiddle.x;
  if ((local_lane & mask) == 0u) {
    return Complex4(lower_re + product_re, lower_im + product_im);
  }
  return Complex4(lower_re - product_re, lower_im - product_im);
}

fn first_output_time() -> u32 {
  return (output_range.first_output / OUTPUT_CHANNELS) % OUTPUT_FRAMES;
}

fn range_output_count() -> u32 {
  return output_range.output_count / OUTPUT_CHANNELS;
}

fn is_full_output_range() -> bool {
  return first_output_time() == 0u && range_output_count() == OUTPUT_FRAMES;
}

fn first_requested_stream_position(residue: u32) -> u32 {
  let first = first_output_time();
  if (first <= residue) { return 0u; }
  return (first - residue + DILATION - 1u) / DILATION;
}

fn end_requested_stream_position(residue: u32) -> u32 {
  let end = first_output_time() + range_output_count();
  if (end <= residue) { return 0u; }
  return (end - residue + DILATION - 1u) / DILATION;
}

fn first_tile_group_for_residue(residue: u32) -> u32 {
  return first_requested_stream_position(residue) / 10u;
}

fn tile_count_for_residue(residue: u32) -> u32 {
  let first = first_requested_stream_position(residue);
  let end = end_requested_stream_position(residue);
  if (first >= end) { return 0u; }
  return (end - 1u) / 10u - first / 10u + 1u;
}

fn tile_count_value() -> u32 {
  if (is_full_output_range()) {
    let full_groups = OUTPUT_FRAMES / TILE_TIME_SPAN;
    let tail_residues = min(DILATION, OUTPUT_FRAMES % TILE_TIME_SPAN);
    return full_groups * DILATION + tail_residues;
  }
  var total = 0u;
  for (var residue = 0u; residue < DILATION; residue += 1u) {
    total += tile_count_for_residue(residue);
  }
  return total;
}

fn tile_first_time(tile: u32) -> u32 {
  if (is_full_output_range()) {
    return (tile / DILATION) * TILE_TIME_SPAN + tile % DILATION;
  }
  var remaining = tile;
  for (var residue = 0u; residue < DILATION; residue += 1u) {
    let count = tile_count_for_residue(residue);
    if (remaining < count) {
      let group = first_tile_group_for_residue(residue) + remaining;
      return residue + group * TILE_TIME_SPAN;
    }
    remaining -= count;
  }
  return 0u;
}

fn contraction4(
  coord: u32,
  tile: u32,
  output4: u32,
  tile_count: u32,
) -> vec4<f32> {
  let base = (coord * tile_count + tile) * OUTPUT_CHANNELS + output4 * 4u;
  return vec4<f32>(
    contraction_spectrum[base],
    contraction_spectrum[base + 1u],
    contraction_spectrum[base + 2u],
    contraction_spectrum[base + 3u]
  );
}

@compute @workgroup_size(${ACE_OPT_0077_RFFT16_WORKGROUP_SIZE})
fn forward_main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size != ${ACE_OPT_0077_RFFT16_SUBGROUP_SIZE}u) { return; }
  let local_lane = subgroup_lane & 15u;
  let transform = subgroup * 2u + subgroup_lane / 16u;
  let input_channel4 = group.y * 8u + transform;
  let sample = bit_reverse4(local_lane);
  let first_time = tile_first_time(group.x);
  let signed_time = i32(first_time) - i32(3u * DILATION) +
    i32(sample * DILATION);
  var initial = vec4<f32>(0.0);
  if (input_channel4 < INPUT_CHANNEL_VEC4S && signed_time >= 0 &&
    signed_time < i32(INPUT_FRAMES)) {
    let batch = (output_range.first_output / OUTPUT_CHANNELS) / OUTPUT_FRAMES;
    initial = vec4<f32>(input[
      (batch * INPUT_FRAMES + u32(signed_time)) * INPUT_CHANNEL_VEC4S +
        input_channel4
    ]);
  }
  var value = Complex4(initial, vec4<f32>(0.0));
  value = butterfly_forward(value, local_lane, 1u, 8u);
  value = butterfly_forward(value, local_lane, 2u, 4u);
  value = butterfly_forward(value, local_lane, 4u, 2u);
  value = butterfly_forward(value, local_lane, 8u, 1u);
  if (input_channel4 < INPUT_CHANNEL_VEC4S && local_lane <= 8u) {
    let tile_count = tile_count_value();
    if (local_lane == 0u || local_lane == 8u) {
      let coord = select(0u, 1u, local_lane == 8u);
      input_spectrum[(coord * tile_count + group.x) *
        INPUT_CHANNEL_VEC4S + input_channel4] =
          vec4<f16>(value.re * ENDPOINT_SCALE);
    } else {
      let coord = 2u + (local_lane - 1u) * 2u;
      input_spectrum[(coord * tile_count + group.x) *
        INPUT_CHANNEL_VEC4S + input_channel4] =
          vec4<f16>(value.re * PAIR_SCALE);
      input_spectrum[((coord + 1u) * tile_count + group.x) *
        INPUT_CHANNEL_VEC4S + input_channel4] =
          vec4<f16>(-value.im * PAIR_SCALE);
    }
  }
}

@compute @workgroup_size(${ACE_OPT_0077_RFFT16_WORKGROUP_SIZE})
fn domain_main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size != ${ACE_OPT_0077_RFFT16_SUBGROUP_SIZE}u) { return; }
  let result_coord = group.z;
  let pair_coord = select(result_coord, result_coord - (result_coord & 1u),
    result_coord >= 2u);
  let tile_count = tile_count_value();
  let tile_base = group.x * 16u;
  let output_channel = group.y * 128u + subgroup * 32u + subgroup_lane;
${domainInitializers}
  for (var input_channel4 = 0u; input_channel4 < INPUT_CHANNEL_VEC4S;
    input_channel4 += 1u) {
    var lane_x0 = vec4<f16>(0.0h);
    var lane_x1 = vec4<f16>(0.0h);
    let lane_tile = tile_base + subgroup_lane;
    if (subgroup_lane < 16u && lane_tile < tile_count) {
      lane_x0 = input_spectrum[(pair_coord * tile_count + lane_tile) *
        INPUT_CHANNEL_VEC4S + input_channel4];
      if (result_coord >= 2u) {
        lane_x1 = input_spectrum[((pair_coord + 1u) * tile_count + lane_tile) *
          INPUT_CHANNEL_VEC4S + input_channel4];
      }
    }
    let weight0 = transformed_weight[
      (pair_coord * INPUT_CHANNEL_VEC4S + input_channel4) * OUTPUT_CHANNELS +
        output_channel
    ];
    var weight1 = vec4<f16>(0.0h);
    if (result_coord >= 2u) {
      weight1 = transformed_weight[
        ((pair_coord + 1u) * INPUT_CHANNEL_VEC4S + input_channel4) *
          OUTPUT_CHANNELS + output_channel
      ];
    }
${domainUpdates}
  }
${domainStores}
}

@compute @workgroup_size(${ACE_OPT_0077_RFFT16_WORKGROUP_SIZE})
fn inverse_main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size != ${ACE_OPT_0077_RFFT16_SUBGROUP_SIZE}u) { return; }
  let local_lane = subgroup_lane & 15u;
  let transform = subgroup * 2u + subgroup_lane / 16u;
  let output4 = group.y * 8u + transform;
  let frequency = bit_reverse4(local_lane);
  let tile_count = tile_count_value();
  var value = Complex4(vec4<f32>(0.0), vec4<f32>(0.0));
  if (frequency == 0u) {
    value.re = 4.0 * contraction4(0u, group.x, output4, tile_count);
  } else if (frequency == 8u) {
    value.re = 4.0 * contraction4(1u, group.x, output4, tile_count);
  } else {
    let source_frequency = select(frequency, 16u - frequency, frequency > 8u);
    let coord = 2u + (source_frequency - 1u) * 2u;
    value.re = 2.0 * contraction4(coord, group.x, output4, tile_count);
    value.im = 2.0 * contraction4(coord + 1u, group.x, output4, tile_count);
    if (frequency > 8u) { value.im = -value.im; }
  }
  value = butterfly_inverse(value, local_lane, 1u, 8u);
  value = butterfly_inverse(value, local_lane, 2u, 4u);
  value = butterfly_inverse(value, local_lane, 4u, 2u);
  value = butterfly_inverse(value, local_lane, 8u, 1u);
  value.re *= 0.25;
  let output_time = tile_first_time(group.x) + local_lane * DILATION;
  let range_first = first_output_time();
  let range_end = range_first + range_output_count();
  if (local_lane < 10u && output_time >= range_first &&
    output_time < range_end && output_time < OUTPUT_FRAMES) {
    let output_channel = output4 * 4u;
    let bias4 = vec4<f32>(
      f32(bias[output_channel]), f32(bias[output_channel + 1u]),
      f32(bias[output_channel + 2u]), f32(bias[output_channel + 3u])
    );
    let batch = (output_range.first_output / OUTPUT_CHANNELS) / OUTPUT_FRAMES;
    output[(batch * OUTPUT_FRAMES + output_time) * OUTPUT_CHANNEL_VEC4S +
      output4] = vec4<f16>(value.re + bias4);
  }
}
`;
}

async function compileAceOpt0077VaeK7Rfft16(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
): Promise<CompiledAceOpt0077VaeK7Rfft16> {
  const label = `ace-opt-0077-rfft16-${convKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0077VaeK7Rfft16Wgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(`OPT-0077 RFFT16 WGSL failed: ${errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`).join("; ")}`);
  }
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[
        plan.inputBindingBytes,
        transformedWeightBytes(plan),
        plan.biasBindingBytes,
        plan.outputBindingBytes,
        0,
        0,
      ].map((minBindingSize, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: binding === 3 || binding === 4 || binding === 5
            ? "storage" as const
            : "read-only-storage" as const,
          minBindingSize,
        },
      })),
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform" as const,
          hasDynamicOffset: true,
          minBindingSize: OUTPUT_RANGE_CONTROL_BYTES,
        },
      },
    ],
  });
  const layout = device.createPipelineLayout({
    label: `${label}-layout`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const [forward, domain, inverse] = await Promise.all([
    device.createComputePipelineAsync({
      label: `${label}-forward`, layout,
      compute: { module, entryPoint: "forward_main" },
    }),
    device.createComputePipelineAsync({
      label: `${label}-domain`, layout,
      compute: { module, entryPoint: "domain_main" },
    }),
    device.createComputePipelineAsync({
      label: `${label}-inverse`, layout,
      compute: { module, entryPoint: "inverse_main" },
    }),
  ]);
  return Object.freeze({ forward, domain, inverse, bindGroupLayout });
}

function requireCandidateBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.family !== "k7" || plan.kernelSize !== 7) {
    throw new RangeError("OPT-0077 RFFT16 supports only K7 Conv1D");
  }
  if (!hasBias) throw new RangeError("OPT-0077 RFFT16 requires bias");
  if (plan.outputStorage !== "float16") {
    throw new RangeError("OPT-0077 RFFT16 requires FP16 output");
  }
  if (plan.inputChannels % 4 !== 0) {
    throw new RangeError("OPT-0077 RFFT16 requires Cin divisible by four");
  }
  if (plan.dilation !== 1 && plan.dilation !== 3 && plan.dilation !== 9) {
    throw new RangeError("OPT-0077 RFFT16 requires dilation 1, 3, or 9");
  }
  if (plan.outputChannels < 128 || plan.outputChannels % 128 !== 0) {
    throw new RangeError("OPT-0077 RFFT16 requires Cout divisible by 128");
  }
  if (plan.outputFrames > MAX_WGSL_I32) {
    throw new RangeError("OPT-0077 RFFT16 frames exceed signed shader indexing");
  }
}

function requireFixed32Device(
  device: GPUDevice,
  capability: AceFp16VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16") || !device.features.has("subgroups")) {
    throw new Error("OPT-0077 RFFT16 requires shader-f16 and subgroups");
  }
  if (capability.subgroupMinSize !== 32 || capability.subgroupMaxSize !== 32) {
    throw new Error("OPT-0077 RFFT16 requires fixed 32-lane subgroups");
  }
  if (device.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    device.limits.maxComputeWorkgroupSizeX < 128) {
    throw new Error("OPT-0077 RFFT16 requires WG128");
  }
}

function transformedWeightBytes(plan: AceFp16VaeConv1dPlan): number {
  return checkedBytes(
    ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES * plan.inputChannels *
      plan.outputChannels * 2,
    "transformed weight",
  );
}

function checkedBytes(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value % 4 !== 0) {
    throw new RangeError(`OPT-0077 ${label} byte count is invalid`);
  }
  return value;
}

function namedStorage(
  device: GPUDevice,
  name: NamedBinding["name"],
  binding: GPUBufferBinding,
  storageBytes: number,
  bindingBytes: number,
  viewAlignment: number,
  label: string,
): NamedBinding {
  return Object.freeze({
    name,
    binding: requireStorageBinding(
      device,
      binding,
      storageBytes,
      bindingBytes,
      viewAlignment,
      `${label} ${name}`,
    ),
  });
}

function requireStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredStorageBytes: number,
  requiredBindingBytes: number,
  requiredViewAlignment: number,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error("OPT-0077 device reported invalid storage alignment");
  }
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (!Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(available) || available < requiredBindingBytes ||
    offset + available > bufferBytes || offset % alignment !== 0 ||
    offset % requiredViewAlignment !== 0 ||
    requiredStorageBytes % requiredViewAlignment !== 0 ||
    requiredBindingBytes % requiredViewAlignment !== 0) {
    throw new RangeError(
      `${label} binding does not expose an aligned ${requiredStorageBytes}-byte payload`,
    );
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: requiredBindingBytes });
}

function requireRangeControlBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error("OPT-0077 device reported invalid uniform alignment");
  }
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (!Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 || offset > MAX_WGSL_U32 ||
    !Number.isSafeInteger(available) || available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > bufferBytes || offset % alignment !== 0) {
    throw new RangeError(`${label} must expose one aligned immutable record`);
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: OUTPUT_RANGE_CONTROL_BYTES });
}

function requireDisjointBindings(
  bindings: readonly NamedBinding[],
  label: string,
): void {
  for (let left = 0; left < bindings.length; left++) {
    const a = bindings[left]!;
    const a0 = a.binding.offset ?? 0;
    const a1 = a0 + (a.binding.size ?? 0);
    for (let right = left + 1; right < bindings.length; right++) {
      const b = bindings[right]!;
      if (a.binding.buffer !== b.binding.buffer) continue;
      const b0 = b.binding.offset ?? 0;
      const b1 = b0 + (b.binding.size ?? 0);
      if (a0 < b1 && b0 < a1) {
        throw new RangeError(`${label} ${a.name} and ${b.name} must not overlap`);
      }
    }
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 4 &&
    Number.isInteger(Math.log2(value));
}

function wgslF32(value: number): string {
  if (!Number.isFinite(value)) {
    throw new RangeError("OPT-0077 WGSL constants must be finite");
  }
  if (Object.is(value, -0)) return "-0.0";
  return Number.isInteger(value) ? `${value}.0` : String(value);
}

function convKey(plan: AceFp16VaeConv1dPlan): string {
  return [
    "opt-0077-rfft16-real-basis",
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputStorage,
    "bias",
  ].join("x");
}
