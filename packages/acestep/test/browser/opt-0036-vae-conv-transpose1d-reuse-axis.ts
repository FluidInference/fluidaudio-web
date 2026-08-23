/// <reference types="@webgpu/types" />

import {
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
  aceOpt0026VaeConvTranspose1dWgsl,
  planAceOpt0026VaeConvTranspose1d,
  planAceOpt0026VaeConvTranspose1dRange,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
  aceOpt0036VaeConvTranspose1dR4C8Wgsl,
  aceOpt0036VaeConvTranspose1dR8C4Wgsl,
  planAceOpt0036VaeConvTranspose1dR4C8,
  planAceOpt0036VaeConvTranspose1dR8C4,
  planAceOpt0036VaeConvTranspose1dRange,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  planAceFp16VaeConvTranspose1d,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

declare global {
  interface Window {
    __ACE_OPT0036_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "control" | "rowReuse" | "channelReuse";

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly dispatch: readonly [number, number, number];
}

interface CompiledArm extends CompiledKernel {
  readonly bindGroup: GPUBindGroup;
}

interface PreparedOperation {
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly outputElements: number;
  readonly arms: Readonly<Record<Arm, CompiledArm>>;
  readonly buffers: readonly GPUBuffer[];
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly requestedLimits: Readonly<Record<string, number>>;
  readonly operations: readonly PreparedOperation[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: string[];
}

const ARMS = Object.freeze([
  "control",
  "rowReuse",
  "channelReuse",
] as const);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["control", "rowReuse", "channelReuse"] as const),
  Object.freeze(["control", "channelReuse", "rowReuse"] as const),
  Object.freeze(["rowReuse", "control", "channelReuse"] as const),
  Object.freeze(["rowReuse", "channelReuse", "control"] as const),
  Object.freeze(["channelReuse", "control", "rowReuse"] as const),
  Object.freeze(["channelReuse", "rowReuse", "control"] as const),
]);
const SHAPES = Object.freeze([
  operation("block-0-conv-t1", 300, 2_048, 1_024, 10),
  operation("block-1-conv-t1", 3_000, 1_024, 512, 6),
  operation("block-2-conv-t1", 18_000, 512, 256, 4),
  operation("block-3-conv-t1", 72_000, 256, 128, 4),
  operation("block-4-conv-t1", 288_000, 128, 128, 2),
]);
const INPUT_PATTERN = new Uint16Array([
  0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
]);
const POLYPHASE_WEIGHT_PATTERN = new Uint16Array([
  0x1800, 0x9800, 0x1c00, 0x9c00, 0x2000, 0xa000, 0x2200, 0xa200,
  0x0000, 0x8000,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2800, 0xa800,
]);
const OUTPUT_PREFILL = 0x7e55;
const REQUIRED_SUMMED_MEDIAN_SPEEDUP = 1.15;

if (typeof document !== "undefined") install();

function install(): void {
  const progress = element<HTMLElement>("#progress");
  const run = element<HTMLButtonElement>("#run");
  let prepared: PreparedGate | undefined;
  void prepare((message) => progress.textContent = message).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — both reuse-axis candidates are raw-U16 exact on all five C512 shapes; timing has not run";
      run.disabled = false;
    },
    (error: unknown) => finish("failed", failure(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    const owned = prepared;
    prepared = undefined;
    run.disabled = true;
    document.body.dataset.status = "running";
    void timeGate(owned, (message) => progress.textContent = message).then(
      (receipt) => finish(receipt.passed === true ? "passed" : "failed", receipt),
      (error: unknown) => {
        cleanup(owned);
        finish("failed", failure(error));
      },
    );
  }, { once: true });
  window.addEventListener("beforeunload", () => {
    if (prepared !== undefined) {
      cleanup(prepared);
      prepared = undefined;
    }
  });
}

async function prepare(
  update: (message: string) => void,
): Promise<PreparedGate> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const adapterInfo = adapter.info;
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    Number(adapterInfo.subgroupMinSize) !== 32 ||
    Number(adapterInfo.subgroupMaxSize) !== 32
  ) {
    throw new Error("OPT-0036 requires shader-f16 and fixed32 subgroups");
  }
  const requestedLimits = Object.freeze({
    maxBufferSize: Number(adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize:
      Number(adapter.limits.maxStorageBufferBindingSize),
    maxComputeWorkgroupsPerDimension:
      Number(adapter.limits.maxComputeWorkgroupsPerDimension),
  });
  const device = await adapter.requestDevice({
    label: "ace-opt-0036-conv-transpose-reuse-axis-abc",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: requestedLimits,
  });
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const operations: PreparedOperation[] = [];
  let sharedPolyphaseWeightU16Count = 0;
  let controlOutputU16Count = 0;
  let comparedOutputU16Count = 0;
  let deterministicRerunOutputU16Count = 0;
  let transientOutputDestroyedCount = 0;
  const started = performance.now();
  try {
    for (const [ordinal, topology] of SHAPES.entries()) {
      update(`compile + full exactness ${ordinal + 1}/5: ${topology.label}`);
      const prepared = await prepareOperation(device, topology, ordinal);
      operations.push(prepared.operation);
      sharedPolyphaseWeightU16Count += prepared.weightElements;
      controlOutputU16Count += prepared.outputElements;
      comparedOutputU16Count += prepared.outputElements * 4;
      deterministicRerunOutputU16Count += prepared.outputElements * 2;
      transientOutputDestroyedCount += prepared.transientOutputDestroyedCount;
      await browserYield();
    }
    await device.queue.onSubmittedWorkDone();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`uncaptured GPU errors: ${uncapturedErrors.join("; ")}`);
    }
    return Object.freeze({
      adapter,
      device,
      requestedLimits,
      operations: Object.freeze(operations),
      correctness: Object.freeze({
        operationCount: operations.length,
        armCount: ARMS.length,
        sharedRevision6PolyphaseWeightU16Count:
          sharedPolyphaseWeightU16Count,
        controlOutputU16Count,
        comparedOutputU16Count,
        deterministicRerunOutputU16Count,
        outputMismatchCount: 0,
        outputPrefillRemainingCount: 0,
        rawU16Exact: true,
        completeDeterministicWrites: true,
        completedBeforeReady: true,
        transientOutputDestroyedCount,
        milliseconds: performance.now() - started,
      }),
      uncapturedErrors,
    });
  } catch (error) {
    for (const operation of operations) destroyBuffers(operation.buffers);
    device.destroy();
    throw error;
  }
}

async function prepareOperation(
  device: GPUDevice,
  topology: (typeof SHAPES)[number],
  ordinal: number,
): Promise<Readonly<{
  operation: PreparedOperation;
  weightElements: number;
  outputElements: number;
  transientOutputDestroyedCount: number;
}>> {
  const shape = topology.shape;
  const basePlan = planAceFp16VaeConvTranspose1d(shape);
  const controlPlan = planAceOpt0026VaeConvTranspose1d(shape);
  const rowPlan = planAceOpt0036VaeConvTranspose1dR8C4(shape);
  const channelPlan = planAceOpt0036VaeConvTranspose1dR4C8(shape);
  const fullRange = Object.freeze({ base: 0, count: basePlan.outputElements });
  const controlRange = planAceOpt0026VaeConvTranspose1dRange(
    controlPlan,
    fullRange,
  );
  const rowRange = planAceOpt0036VaeConvTranspose1dRange(rowPlan, fullRange);
  const channelRange = planAceOpt0036VaeConvTranspose1dRange(
    channelPlan,
    fullRange,
  );
  const retained: GPUBuffer[] = [];
  const transient: GPUBuffer[] = [];
  try {
    const input = patternedStorageSource(
      device,
      `${topology.label}-input`,
      basePlan.inputBindingBytes,
      INPUT_PATTERN,
      ordinal,
    );
    const polyphaseWeight = patternedStorageSource(
      device,
      `${topology.label}-revision6-polyphase-weight`,
      basePlan.weightBindingBytes,
      POLYPHASE_WEIGHT_PATTERN,
      ordinal * 3,
    );
    const bias = patternedStorageSource(
      device,
      `${topology.label}-bias`,
      basePlan.biasBindingBytes,
      BIAS_PATTERN,
      ordinal,
    );
    const control = rangeBuffer(
      device,
      `${topology.label}-full-range`,
      basePlan.outputElements,
    );
    retained.push(input, polyphaseWeight, bias, control);
    const compiled = Object.freeze({
      control: await compileKernel(
        device,
        `${topology.label}-opt0026-control`,
        aceOpt0026VaeConvTranspose1dWgsl(shape),
        [
          controlRange.workgroupsX,
          controlRange.workgroupsY,
          controlRange.workgroupsZ,
        ],
      ),
      rowReuse: await compileKernel(
        device,
        `${topology.label}-opt0036-r8c4`,
        aceOpt0036VaeConvTranspose1dR8C4Wgsl(shape),
        [rowRange.workgroupsX, rowRange.workgroupsY, rowRange.workgroupsZ],
      ),
      channelReuse: await compileKernel(
        device,
        `${topology.label}-opt0036-r4c8`,
        aceOpt0036VaeConvTranspose1dR4C8Wgsl(shape),
        [
          channelRange.workgroupsX,
          channelRange.workgroupsY,
          channelRange.workgroupsZ,
        ],
      ),
    } as const satisfies Readonly<Record<Arm, CompiledKernel>>);

    const controlOutput = prefilledOutput(
      device,
      `${topology.label}-control-correctness-output`,
      basePlan.outputBindingBytes,
    );
    transient.push(controlOutput);
    await executeArm(device, bindArm(
      device,
      `${topology.label}-control-correctness`,
      compiled.control,
      [input, polyphaseWeight, bias, controlOutput, control],
    ));
    const reference = await readWords(
      device,
      controlOutput,
      basePlan.outputBindingBytes,
    );
    requireNoPrefill(reference, basePlan.outputElements, topology.label, "control");
    controlOutput.destroy();
    transient.splice(transient.indexOf(controlOutput), 1);
    let transientOutputDestroyedCount = 1;

    for (const arm of ["rowReuse", "channelReuse"] as const) {
      for (let execution = 0; execution < 2; execution += 1) {
        const output = prefilledOutput(
          device,
          `${topology.label}-${arm}-correctness-${execution + 1}-output`,
          basePlan.outputBindingBytes,
        );
        transient.push(output);
        await executeArm(device, bindArm(
          device,
          `${topology.label}-${arm}-correctness-${execution + 1}`,
          compiled[arm],
          [input, polyphaseWeight, bias, output, control],
        ));
        await requireExactOutput(
          device,
          output,
          basePlan.outputBindingBytes,
          basePlan.outputElements,
          reference,
          topology.label,
          arm,
        );
        output.destroy();
        transient.splice(transient.indexOf(output), 1);
        transientOutputDestroyedCount += 1;
      }
    }

    const timingOutput = device.createBuffer({
      label: `${topology.label}-shared-timing-output`,
      size: basePlan.outputBindingBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    retained.push(timingOutput);
    const arms = Object.freeze(Object.fromEntries(ARMS.map((arm) => [
      arm,
      bindArm(
        device,
        `${topology.label}-${arm}-timing`,
        compiled[arm],
        [input, polyphaseWeight, bias, timingOutput, control],
      ),
    ])) as unknown as Readonly<Record<Arm, CompiledArm>>);
    await executeWarmup(device, arms);
    return Object.freeze({
      operation: Object.freeze({
        label: topology.label,
        shape,
        outputElements: basePlan.outputElements,
        arms,
        buffers: Object.freeze(retained.slice()),
      }),
      weightElements: basePlan.weightElements,
      outputElements: basePlan.outputElements,
      transientOutputDestroyedCount,
    });
  } catch (error) {
    destroyBuffers(transient);
    destroyBuffers(retained);
    throw error;
  }
}

async function compileKernel(
  device: GPUDevice,
  label: string,
  code: string,
  dispatch: readonly [number, number, number],
): Promise<CompiledKernel> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(`${label} WGSL failed: ${errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`
    ).join("; ")}`);
  }
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, dispatch });
}

function bindArm(
  device: GPUDevice,
  label: string,
  compiled: CompiledKernel,
  buffers: readonly GPUBuffer[],
): CompiledArm {
  const bindGroup = device.createBindGroup({
    label: `${label}-bindings`,
    layout: compiled.pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
  return Object.freeze({ ...compiled, bindGroup });
}

async function executeWarmup(
  device: GPUDevice,
  arms: Readonly<Record<Arm, CompiledArm>>,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  for (const arm of ARMS) {
    const pass = encoder.beginComputePass();
    encode(pass, arms[arm]);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeArm(device: GPUDevice, arm: CompiledArm): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encode(pass, arm);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function timeGate(
  prepared: PreparedGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Arm, number[]>>();
  for (const operation of prepared.operations) {
    samples.set(operation.label, {
      control: [],
      rowReuse: [],
      channelReuse: [],
    });
  }
  const timingStarted = performance.now();
  for (const [round, order] of TIMING_ORDERS.entries()) {
    const operations = round % 2 === 0
      ? prepared.operations
      : prepared.operations.slice().reverse();
    for (const [index, operation] of operations.entries()) {
      for (const arm of order) {
        update(
          `timing round ${round + 1}/6, shape ${index + 1}/5, arm ${arm}`,
        );
        samples.get(operation.label)![arm].push(
          await executeTimed(prepared.device, operation.arms[arm]),
        );
      }
      await browserYield();
    }
  }
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error(`uncaptured GPU errors: ${prepared.uncapturedErrors.join("; ")}`);
  }
  const cases = prepared.operations.map((operation) => {
    const values = samples.get(operation.label)!;
    const controlMedian = median6(values.control);
    const rowMedian = median6(values.rowReuse);
    const channelMedian = median6(values.channelReuse);
    return Object.freeze({
      label: operation.label,
      shape: operation.shape,
      samplesMilliseconds: Object.freeze({
        control: Object.freeze(values.control.slice()),
        rowReuse: Object.freeze(values.rowReuse.slice()),
        channelReuse: Object.freeze(values.channelReuse.slice()),
      }),
      mediansMilliseconds: Object.freeze({
        control: controlMedian,
        rowReuse: rowMedian,
        channelReuse: channelMedian,
      }),
      speedups: Object.freeze({
        rowReuse: controlMedian / rowMedian,
        channelReuse: controlMedian / channelMedian,
      }),
    });
  });
  const summedMediansMilliseconds = Object.freeze({
    control: sumCases(cases, "control"),
    rowReuse: sumCases(cases, "rowReuse"),
    channelReuse: sumCases(cases, "channelReuse"),
  });
  const candidates = (["rowReuse", "channelReuse"] as const).map((arm) => {
    const noSlowerEveryShape = cases.every((entry) =>
      entry.mediansMilliseconds[arm] <= entry.mediansMilliseconds.control
    );
    const summedMedianSpeedup = summedMediansMilliseconds.control /
      summedMediansMilliseconds[arm];
    return Object.freeze({
      arm,
      noSlowerEveryShape,
      summedMedianSpeedup,
      requiredSummedMedianSpeedup: REQUIRED_SUMMED_MEDIAN_SPEEDUP,
      promotionEligible: noSlowerEveryShape &&
        summedMedianSpeedup >= REQUIRED_SUMMED_MEDIAN_SPEEDUP,
    });
  });
  const eligible = candidates.filter((candidate) => candidate.promotionEligible)
    .sort((left, right) => right.summedMedianSpeedup - left.summedMedianSpeedup);
  const fastestArm = (Object.entries(summedMediansMilliseconds) as Array<
    [Arm, number]
  >).sort((left, right) => left[1] - right[1])[0]![0];
  const adapterInfo = prepared.adapter.info;
  const cleanupReceipt = cleanup(prepared);
  return Object.freeze({
    schema: "ace-opt-0036-conv-transpose-reuse-axis-abc-v1",
    passed: eligible.length > 0,
    experimentId: "OPT-0036",
    controlKernelId: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    candidateKernelIds: Object.freeze({
      rowReuse: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
      channelReuse: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
    }),
    weightLayout: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    correctness: prepared.correctness,
    timing: Object.freeze({
      rounds: TIMING_ORDERS.length,
      order: "all six A/B/C permutations; shape order alternates forward/reverse",
      orders: TIMING_ORDERS,
      sampleContract: "one submit and matching queue drain per sample",
      milliseconds: performance.now() - timingStarted,
      cases: Object.freeze(cases),
      summedMediansMilliseconds,
      candidates: Object.freeze(candidates),
      fastestArm,
      promotionEligibleArm: eligible[0]?.arm ?? null,
    }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      adapter: Object.freeze({
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        subgroupMinSize: adapterInfo.subgroupMinSize,
        subgroupMaxSize: adapterInfo.subgroupMaxSize,
      }),
      requestedLimits: prepared.requestedLimits,
      deviceLimits: Object.freeze({
        maxBufferSize: Number(prepared.device.limits.maxBufferSize),
        maxStorageBufferBindingSize:
          Number(prepared.device.limits.maxStorageBufferBindingSize),
        maxComputeWorkgroupsPerDimension:
          Number(prepared.device.limits.maxComputeWorkgroupsPerDimension),
      }),
      features: Object.freeze([...prepared.device.features].sort()),
      uncapturedErrors: Object.freeze(prepared.uncapturedErrors.slice()),
    }),
    cleanup: cleanupReceipt,
  });
}

async function executeTimed(
  device: GPUDevice,
  arm: CompiledArm,
): Promise<number> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encode(pass, arm);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function encode(pass: GPUComputePassEncoder, arm: CompiledArm): void {
  pass.setPipeline(arm.pipeline);
  pass.setBindGroup(0, arm.bindGroup);
  pass.dispatchWorkgroups(...arm.dispatch);
}

function operation(
  label: string,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  stride: number,
): Readonly<{ label: string; shape: AceVaeConvTranspose1dShape }> {
  return Object.freeze({
    label,
    shape: Object.freeze({
      batch: 1,
      inputFrames,
      inputChannels,
      outputChannels,
      kernelSize: 2 * stride,
      stride,
      dilation: 1,
      padding: stride / 2,
      outputPadding: 0,
    }),
  });
}

function patternedStorageSource(
  device: GPUDevice,
  label: string,
  size: number,
  pattern: Uint16Array,
  offset: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const words = new Uint16Array(buffer.getMappedRange());
  for (let index = 0; index < words.length; index += 1) {
    words[index] = pattern[(index + offset) % pattern.length]!;
  }
  buffer.unmap();
  return buffer;
}

function rangeBuffer(
  device: GPUDevice,
  label: string,
  outputElements: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set([0, outputElements, 0, 0]);
  buffer.unmap();
  return buffer;
}

function prefilledOutput(
  device: GPUDevice,
  label: string,
  size: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint16Array(buffer.getMappedRange()).fill(OUTPUT_PREFILL);
  buffer.unmap();
  return buffer;
}

async function readWords(
  device: GPUDevice,
  source: GPUBuffer,
  bytes: number,
): Promise<Uint16Array> {
  const readback = await mapReadback(device, source, bytes);
  const copy = new Uint16Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return copy;
}

async function requireExactOutput(
  device: GPUDevice,
  source: GPUBuffer,
  bytes: number,
  elements: number,
  reference: Uint16Array,
  label: string,
  arm: Exclude<Arm, "control">,
): Promise<void> {
  const readback = await mapReadback(device, source, bytes);
  const actual = new Uint16Array(readback.getMappedRange());
  let prefillRemaining = 0;
  let mismatchCount = 0;
  let firstMismatch = -1;
  for (let index = 0; index < elements; index += 1) {
    if (actual[index] === OUTPUT_PREFILL) prefillRemaining += 1;
    if (actual[index] !== reference[index]) {
      mismatchCount += 1;
      if (firstMismatch < 0) firstMismatch = index;
    }
  }
  if (mismatchCount !== 0 || prefillRemaining !== 0) {
    const detail = firstMismatch < 0
      ? ""
      : `; first mismatch ${firstMismatch}: ` +
        `0x${reference[firstMismatch]!.toString(16)} != ` +
        `0x${actual[firstMismatch]!.toString(16)}`;
    readback.unmap();
    readback.destroy();
    throw new Error(
      `${label} ${arm} has ${mismatchCount} mismatches and ` +
        `${prefillRemaining} unwritten outputs${detail}`,
    );
  }
  readback.unmap();
  readback.destroy();
}

async function mapReadback(
  device: GPUDevice,
  source: GPUBuffer,
  bytes: number,
): Promise<GPUBuffer> {
  const readback = device.createBuffer({
    label: "ace-opt-0036-output-readback",
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  return readback;
}

function requireNoPrefill(
  words: Uint16Array,
  elements: number,
  label: string,
  arm: Arm,
): void {
  let count = 0;
  for (let index = 0; index < elements; index += 1) {
    if (words[index] === OUTPUT_PREFILL) count += 1;
  }
  if (count !== 0) {
    throw new Error(`${label} ${arm} left ${count} outputs unwritten`);
  }
}

function median6(values: readonly number[]): number {
  if (values.length !== 6 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("OPT-0036 requires six finite timing samples per arm");
  }
  const sorted = values.slice().sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
}

function sumCases(
  cases: readonly Readonly<{
    mediansMilliseconds: Readonly<Record<Arm, number>>;
  }>[],
  arm: Arm,
): number {
  return cases.reduce(
    (sum, entry) => sum + entry.mediansMilliseconds[arm],
    0,
  );
}

function cleanup(prepared: PreparedGate): Readonly<Record<string, unknown>> {
  let destroyedBufferCount = 0;
  for (const operation of prepared.operations) {
    for (const buffer of operation.buffers) {
      buffer.destroy();
      destroyedBufferCount += 1;
    }
  }
  prepared.device.destroy();
  return Object.freeze({ destroyedBufferCount, deviceDestroyed: true });
}

function destroyBuffers(buffers: readonly GPUBuffer[]): void {
  for (const buffer of buffers) buffer.destroy();
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing element ${selector}`);
  return value;
}

function finish(
  status: "passed" | "failed",
  receipt: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  element<HTMLElement>("#progress").textContent = status;
  element<HTMLElement>("#result").textContent = JSON.stringify(receipt, null, 2);
  window.__ACE_OPT0036_RESULT__ = receipt;
}

function failure(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0036-conv-transpose-reuse-axis-abc-v1",
    passed: false,
    experimentId: "OPT-0036",
    error: error instanceof Error
      ? Object.freeze({
          name: error.name,
          message: error.message,
          stack: error.stack,
        })
      : String(error),
  });
}

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
