/// <reference types="@webgpu/types" />

import {
  AceFp16VaeConv1dKernel,
  planAceFp16VaeConv1d,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
  AceOpt0025VaeK1SubgroupGemmKernel,
  packAceOpt0025VaeK1WeightU16,
  planAceOpt0025VaeK1SubgroupGemm,
} from "../../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import type { AceVaeConv1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

export type Opt0025Arm = "current" | "candidate";

export interface Opt0025ShapeSpec {
  readonly id: "c1024" | "c512" | "c256" | "c128-a" | "c128-b";
  readonly frames: number;
  readonly channels: 1024 | 512 | 256 | 128;
  readonly operationMultiplicity: 3;
  readonly shape: AceVaeConv1dShape;
}

export interface Opt0025TimingInput {
  readonly id: Opt0025ShapeSpec["id"];
  readonly samples: Readonly<Record<Opt0025Arm, readonly number[]>>;
}

interface Encodable {
  encode(pass: GPUComputePassEncoder): void;
}

interface PreparedShape {
  readonly spec: Opt0025ShapeSpec;
  readonly current: Encodable;
  readonly candidate: Encodable;
  readonly fillCurrent: GPUBindGroup;
  readonly fillCandidate: GPUBindGroup;
  readonly compare: GPUBindGroup;
  readonly wordCount: number;
  readonly dispatchX: number;
  readonly dispatchY: number;
  readonly macs: number;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly currentKernel: AceFp16VaeConv1dKernel;
  readonly candidateKernel: AceOpt0025VaeK1SubgroupGemmKernel;
  readonly fillPipeline: GPUComputePipeline;
  readonly comparePipeline: GPUComputePipeline;
  readonly shapes: readonly PreparedShape[];
  readonly mismatch: GPUBuffer;
  readonly mismatchReadback: GPUBuffer;
  readonly buffers: readonly GPUBuffer[];
  readonly correctness: readonly Readonly<Record<string, unknown>>[];
  destroy(): void;
}

const TIMING_ROUNDS = 4;
const WORKGROUP_SIZE = 256;
const MAX_DISPATCH_X = 65_535;
const PREFILL_PAIR = 0x7e55_7e55;
const CURRENT_C4500_K1_MILLISECONDS = 25_772.300002217293;
const INPUT_PAIR_PATTERN = 0x2800_a800;
const BIAS_PAIR_PATTERN = 0x1800_9800;
const WEIGHT_PATTERN = Object.freeze([
  0x1c00, 0x9c00, 0x2000, 0xa000,
  0x2200, 0xa200, 0x2400, 0xa400,
] as const);

const SHAPES = Object.freeze([
  shapeSpec("c1024", 5_120, 1_024),
  shapeSpec("c512", 30_720, 512),
  shapeSpec("c256", 122_880, 256),
  shapeSpec("c128-a", 491_520, 128),
  shapeSpec("c128-b", 983_040, 128),
]);

export function buildOpt0025ShapeSpecs(): readonly Opt0025ShapeSpec[] {
  return SHAPES;
}

export function buildOpt0025TimingOrders(): readonly Readonly<{
  roundIndex: number;
  shapeIndex: number;
  order: readonly Opt0025Arm[];
}>[] {
  return Object.freeze(Array.from({ length: TIMING_ROUNDS }, (_, roundIndex) =>
    SHAPES.map((_, shapeIndex) => {
      const reverse = (roundIndex + shapeIndex) % 2 === 1;
      return Object.freeze({
        roundIndex,
        shapeIndex,
        order: Object.freeze(reverse
          ? ["candidate", "current"] as const
          : ["current", "candidate"] as const),
      });
    })
  ).flat());
}

export function summarizeOpt0025Timing(
  inputs: readonly Opt0025TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPES.length) {
    throw new Error("OPT-0025 timing requires all five production shapes");
  }
  let currentWindowMilliseconds = 0;
  let candidateWindowMilliseconds = 0;
  let windowMacs = 0;
  const shapes = inputs.map((input, index) => {
    const spec = SHAPES[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0025 timing shape order changed");
    }
    const currentMedian = median4(input.samples.current);
    const candidateMedian = median4(input.samples.candidate);
    const macs = spec.frames * spec.channels * spec.channels;
    currentWindowMilliseconds += currentMedian * spec.operationMultiplicity;
    candidateWindowMilliseconds += candidateMedian * spec.operationMultiplicity;
    windowMacs += macs * spec.operationMultiplicity;
    return Object.freeze({
      id: spec.id,
      frames: spec.frames,
      channels: spec.channels,
      operationMultiplicity: spec.operationMultiplicity,
      macs,
      samples: input.samples,
      medians: Object.freeze({ current: currentMedian, candidate: candidateMedian }),
      speedup: currentMedian / candidateMedian,
      currentTflops: 2 * macs / (currentMedian / 1_000) / 1e12,
      candidateTflops: 2 * macs / (candidateMedian / 1_000) / 1e12,
    });
  });
  const speedup = currentWindowMilliseconds / candidateWindowMilliseconds;
  return Object.freeze({
    samplesPerArmPerShape: TIMING_ROUNDS,
    c512DecoderWindow: Object.freeze({
      operationCount: 15,
      macs: windowMacs,
      currentMilliseconds: currentWindowMilliseconds,
      candidateMilliseconds: candidateWindowMilliseconds,
      savingMilliseconds: currentWindowMilliseconds - candidateWindowMilliseconds,
      speedup,
      currentTflops:
        2 * windowMacs / (currentWindowMilliseconds / 1_000) / 1e12,
      candidateTflops:
        2 * windowMacs / (candidateWindowMilliseconds / 1_000) / 1e12,
    }),
    c4500Projection: Object.freeze({
      measuredCurrentMilliseconds: CURRENT_C4500_K1_MILLISECONDS,
      projectedCandidateMilliseconds: CURRENT_C4500_K1_MILLISECONDS / speedup,
      projectedSavingMilliseconds:
        CURRENT_C4500_K1_MILLISECONDS - CURRENT_C4500_K1_MILLISECONDS / speedup,
      method: "scale-authoritative-OPT-0023-K1-wall-by-paired-shape-speedup",
    }),
    shapes: Object.freeze(shapes),
  });
}

async function prepareGate(
  update: (message: string) => void,
): Promise<PreparedGate> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("core-features-and-limits") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32
  ) {
    throw new Error(
      "OPT-0025 requires shader-f16, core limits, and reported 32/32 subgroups",
    );
  }
  const maximumPlan = planAceFp16VaeConv1d(SHAPES.at(-1)!.shape, "float16");
  const device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16", "subgroups", "core-features-and-limits"],
    requiredLimits: {
      maxBufferSize: maximumPlan.inputBindingBytes,
      maxStorageBufferBindingSize: maximumPlan.inputBindingBytes,
    },
  });
  const buffers: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    buffers.push(buffer);
    return buffer;
  };
  if (
    maximumPlan.inputBindingBytes > Number(device.limits.maxBufferSize) ||
    maximumPlan.inputBindingBytes > Number(device.limits.maxStorageBufferBindingSize)
  ) {
    throw new Error("OPT-0025 maximum production K1 buffer exceeds adapter limits");
  }

  update("allocating one shared production-sized activation and two outputs");
  const input = own(device.createBuffer({
    label: "opt-0025-shared-input",
    size: maximumPlan.inputBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  }));
  new Uint32Array(input.getMappedRange()).fill(INPUT_PAIR_PATTERN);
  input.unmap();
  const bias = own(device.createBuffer({
    label: "opt-0025-shared-bias",
    size: 1_024 * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  }));
  new Uint32Array(bias.getMappedRange()).fill(BIAS_PAIR_PATTERN);
  bias.unmap();
  const currentOutput = own(device.createBuffer({
    label: "opt-0025-current-output",
    size: maximumPlan.outputBindingBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  }));
  const candidateOutput = own(device.createBuffer({
    label: "opt-0025-candidate-output",
    size: maximumPlan.outputBindingBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  }));
  const mismatch = own(device.createBuffer({
    label: "opt-0025-mismatch",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  }));
  const mismatchReadback = own(device.createBuffer({
    label: "opt-0025-mismatch-readback",
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  }));

  const fillPipeline = await compilePipeline(device, "fill", fillWgsl());
  const comparePipeline = await compilePipeline(device, "compare", compareWgsl());
  const currentKernel = AceFp16VaeConv1dKernel.create(device);
  const candidateKernel = AceOpt0025VaeK1SubgroupGemmKernel.create(device, {
    subgroupMinSize: adapter.info.subgroupMinSize,
    subgroupMaxSize: adapter.info.subgroupMaxSize,
  });

  const preparedShapes: PreparedShape[] = [];
  for (const spec of SHAPES) {
    update(`packing and compiling ${spec.id} (${spec.frames}x${spec.channels})`);
    const currentPlan = planAceFp16VaeConv1d(spec.shape, "float16");
    const candidatePlan = planAceOpt0025VaeK1SubgroupGemm(spec.shape);
    const nativeWeight = patternedWeight(candidatePlan.weightElements);
    const packedWeight = packAceOpt0025VaeK1WeightU16(
      nativeWeight,
      spec.channels,
      spec.channels,
    );
    const nativeWeightBuffer = own(bufferFromU16(
      device,
      `${spec.id}-native-weight`,
      nativeWeight,
    ));
    const packedWeightBuffer = own(bufferFromU16(
      device,
      `${spec.id}-packed-weight`,
      packedWeight,
    ));
    const rangeControl = own(device.createBuffer({
      label: `${spec.id}-range-control`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }));
    new Uint32Array(rangeControl.getMappedRange()).set([
      0,
      currentPlan.outputElements,
      0,
      0,
    ]);
    rangeControl.unmap();
    const wordCount = currentPlan.outputElements / 2;
    const compareParameters = own(device.createBuffer({
      label: `${spec.id}-compare-parameters`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }));
    new Uint32Array(compareParameters.getMappedRange()).set([wordCount, 0, 0, 0]);
    compareParameters.unmap();
    const inputBinding = {
      buffer: input,
      size: currentPlan.inputBindingBytes,
    } satisfies GPUBufferBinding;
    const biasBinding = {
      buffer: bias,
      size: currentPlan.biasBindingBytes,
    } satisfies GPUBufferBinding;
    const currentOutputBinding = {
      buffer: currentOutput,
      size: currentPlan.outputBindingBytes,
    } satisfies GPUBufferBinding;
    const candidateOutputBinding = {
      buffer: candidateOutput,
      size: candidatePlan.outputBytes,
    } satisfies GPUBufferBinding;
    const current = await currentKernel.createDispatch(
      `opt-0025-${spec.id}-current`,
      spec.shape,
      {
        input: inputBinding,
        weight: { buffer: nativeWeightBuffer, size: currentPlan.weightBindingBytes },
        bias: biasBinding,
        output: currentOutputBinding,
      },
      "float16",
      {
        base: 0,
        count: currentPlan.outputElements,
        control: { buffer: rangeControl, size: 16 },
      },
    );
    const candidate = await candidateKernel.createDispatch(
      `opt-0025-${spec.id}-candidate`,
      spec.shape,
      {
        input: inputBinding,
        packedWeight: {
          buffer: packedWeightBuffer,
          size: candidatePlan.weightBytes,
        },
        bias: biasBinding,
        output: candidateOutputBinding,
      },
    );
    const parametersBinding = { buffer: compareParameters, size: 16 };
    const fillCurrent = device.createBindGroup({
      layout: fillPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: currentOutputBinding },
        { binding: 1, resource: parametersBinding },
      ],
    });
    const fillCandidate = device.createBindGroup({
      layout: fillPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: candidateOutputBinding },
        { binding: 1, resource: parametersBinding },
      ],
    });
    const compare = device.createBindGroup({
      layout: comparePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: currentOutputBinding },
        { binding: 1, resource: candidateOutputBinding },
        { binding: 2, resource: { buffer: mismatch, size: 4 } },
        { binding: 3, resource: parametersBinding },
      ],
    });
    const dispatchGroups = Math.ceil(wordCount / WORKGROUP_SIZE);
    preparedShapes.push(Object.freeze({
      spec,
      current,
      candidate,
      fillCurrent,
      fillCandidate,
      compare,
      wordCount,
      dispatchX: Math.min(dispatchGroups, MAX_DISPATCH_X),
      dispatchY: Math.ceil(dispatchGroups / MAX_DISPATCH_X),
      macs: spec.frames * spec.channels * spec.channels,
    }));
  }

  const correctness: Readonly<Record<string, unknown>>[] = [];
  for (const shape of preparedShapes) {
    update(`comparing every raw FP16 output for ${shape.spec.id}`);
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(mismatch);
    const pass = encoder.beginComputePass();
    pass.setPipeline(fillPipeline);
    pass.setBindGroup(0, shape.fillCurrent);
    pass.dispatchWorkgroups(shape.dispatchX, shape.dispatchY, 1);
    pass.setBindGroup(0, shape.fillCandidate);
    pass.dispatchWorkgroups(shape.dispatchX, shape.dispatchY, 1);
    shape.current.encode(pass);
    shape.candidate.encode(pass);
    pass.setPipeline(comparePipeline);
    pass.setBindGroup(0, shape.compare);
    pass.dispatchWorkgroups(shape.dispatchX, shape.dispatchY, 1);
    pass.end();
    encoder.copyBufferToBuffer(mismatch, 0, mismatchReadback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await mismatchReadback.mapAsync(GPUMapMode.READ);
    const mismatchCount = new Uint32Array(mismatchReadback.getMappedRange())[0]!;
    mismatchReadback.unmap();
    if (mismatchCount !== 0) {
      throw new Error(`${shape.spec.id} has ${mismatchCount} raw FP16 mismatches`);
    }
    correctness.push(Object.freeze({
      id: shape.spec.id,
      comparedU16: shape.wordCount * 2,
      mismatchCount,
      fullRawFp16Identity: true,
      prefillPairRejectedByComparator: `0x${PREFILL_PAIR.toString(16)}`,
    }));
  }

  update("warming both arms once on all five exact production shapes");
  for (const shape of preparedShapes) {
    await execute(device, shape.current);
    await execute(device, shape.candidate);
  }

  return Object.freeze({
    adapter,
    device,
    currentKernel,
    candidateKernel,
    fillPipeline,
    comparePipeline,
    shapes: Object.freeze(preparedShapes),
    mismatch,
    mismatchReadback,
    buffers: Object.freeze(buffers),
    correctness: Object.freeze(correctness),
    destroy(): void {
      currentKernel.destroy();
      candidateKernel.destroy();
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
    },
  });
}

async function runTiming(gate: PreparedGate): Promise<Readonly<Record<string, unknown>>> {
  const samples = Object.fromEntries(SHAPES.map(({ id }) => [id, {
    current: [] as number[],
    candidate: [] as number[],
  }])) as Record<Opt0025ShapeSpec["id"], Record<Opt0025Arm, number[]>>;
  for (const entry of buildOpt0025TimingOrders()) {
    const shape = gate.shapes[entry.shapeIndex]!;
    for (const arm of entry.order) {
      samples[shape.spec.id][arm].push(await execute(gate.device, shape[arm]));
    }
  }
  const summary = summarizeOpt0025Timing(SHAPES.map(({ id }) => ({
    id,
    samples: samples[id],
  })));
  return Object.freeze({
    schema: "ace-opt-0025-vae-k1-subgroup-gemm-ab-v1",
    experimentId: "OPT-0025",
    completedAt: new Date().toISOString(),
    adapter: Object.freeze({
      info: gate.adapter.info,
      subgroupMinSize: gate.adapter.info.subgroupMinSize,
      subgroupMaxSize: gate.adapter.info.subgroupMaxSize,
      shaderF16: gate.device.features.has("shader-f16"),
      subgroups: gate.device.features.has("subgroups"),
    }),
    kernel: Object.freeze({
      current: "ace-vae-fp16-portable-conv1d-v1",
      candidate: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
      compilationAllocationUploadExcludedFromTiming: true,
      warmupExecutionsPerArmPerShape: 1,
      oneDispatchPerOperation: true,
      pairedRounds: TIMING_ROUNDS,
    }),
    correctness: gate.correctness,
    summary,
  });
}

async function execute(device: GPUDevice, dispatch: Encodable): Promise<number> {
  const started = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

async function compilePipeline(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `opt-0025-${label}`, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(errors.map(({ lineNum, linePos, message }) =>
      `${lineNum}:${linePos} ${message}`
    ).join("; "));
  }
  return device.createComputePipelineAsync({
    label: `opt-0025-${label}`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function fillWgsl(): string {
  return /* wgsl */ `
struct Parameters { word_count: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let linear_group = group.y * ${MAX_DISPATCH_X}u + group.x;
  let index = linear_group * ${WORKGROUP_SIZE}u + lane;
  if (index < parameters.word_count) { output[index] = ${PREFILL_PAIR}u; }
}`;
}

function compareWgsl(): string {
  return /* wgsl */ `
struct Parameters { word_count: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> current: array<u32>;
@group(0) @binding(1) var<storage, read> candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> mismatch: atomic<u32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let linear_group = group.y * ${MAX_DISPATCH_X}u + group.x;
  let index = linear_group * ${WORKGROUP_SIZE}u + lane;
  if (index < parameters.word_count) {
    let left = current[index];
    if (left != candidate[index] || left == ${PREFILL_PAIR}u) {
      atomicAdd(&mismatch, 1u);
    }
  }
}`;
}

function patternedWeight(elements: number): Uint16Array {
  const result = new Uint16Array(elements);
  for (let index = 0; index < result.length; index += 1) {
    result[index] = WEIGHT_PATTERN[index % WEIGHT_PATTERN.length]!;
  }
  return result;
}

function bufferFromU16(
  device: GPUDevice,
  label: string,
  data: Uint16Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint16Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function shapeSpec(
  id: Opt0025ShapeSpec["id"],
  frames: number,
  channels: Opt0025ShapeSpec["channels"],
): Opt0025ShapeSpec {
  return Object.freeze({
    id,
    frames,
    channels,
    operationMultiplicity: 3,
    shape: Object.freeze({
      batch: 1,
      inputFrames: frames,
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    }),
  });
}

function median4(samples: readonly number[]): number {
  if (samples.length !== TIMING_ROUNDS || samples.some((value) =>
    !Number.isFinite(value) || value <= 0
  )) {
    throw new Error("OPT-0025 requires four finite positive samples per arm");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

if (typeof document !== "undefined") installBrowserGate();

function installBrowserGate(): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  const result = document.querySelector<HTMLElement>("#result");
  const runButton = document.querySelector<HTMLButtonElement>("#run");
  if (progress === null || result === null || runButton === null) {
    throw new Error("OPT-0025 browser fixture DOM changed");
  }

  let gate: PreparedGate | undefined;
  const update = (message: string): void => {
    progress.textContent = message;
  };

  void prepareGate(update).then((prepared) => {
    gate = prepared;
    document.body.dataset.status = "ready";
    update("ready — wait 30 seconds, confirm nominal thermal state, then run once");
    runButton.disabled = false;
  }, (error: unknown) => {
    document.body.dataset.status = "failed";
    result.textContent = String(
      error instanceof Error ? error.stack ?? error.message : error,
    );
    update("preparation failed");
  });

  runButton.addEventListener("click", () => {
    if (gate === undefined) return;
    runButton.disabled = true;
    document.body.dataset.status = "running";
    update("running four balanced paired rounds across five production shapes");
    void runTiming(gate).then((receipt) => {
      result.textContent = JSON.stringify(receipt, null, 2);
      document.body.dataset.status = "complete";
      update("complete");
    }, (error: unknown) => {
      result.textContent = String(
        error instanceof Error ? error.stack ?? error.message : error,
      );
      document.body.dataset.status = "failed";
      update("timing failed");
    });
  });

  window.addEventListener("pagehide", () => gate?.destroy(), { once: true });
}
