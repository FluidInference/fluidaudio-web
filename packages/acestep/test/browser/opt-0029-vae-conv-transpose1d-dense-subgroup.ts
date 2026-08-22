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
  ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
  aceOpt0029VaeConvTranspose1dWgsl,
  planAceOpt0029VaeConvTranspose1d,
  planAceOpt0029VaeConvTranspose1dRange,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-dense-subgroup.js";
import {
  planAceFp16VaeConvTranspose1d,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

declare global {
  interface Window {
    __ACE_OPT0029_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "control" | "candidate";

interface CompiledArm {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly dispatch: readonly [number, number, number];
}

interface PreparedOperation {
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly weight: 1;
  readonly outputElements: number;
  readonly weightElements: number;
  readonly arms: Readonly<Record<Arm, CompiledArm>>;
  readonly buffers: readonly GPUBuffer[];
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly operations: readonly PreparedOperation[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: string[];
}

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
const TIMING_ROUNDS = 4;

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
        "READY — exact C512 outputs match; timing has not run";
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
  const info = adapter.info;
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    Number(info.subgroupMinSize) !== 32 ||
    Number(info.subgroupMaxSize) !== 32
  ) {
    throw new Error("OPT-0029 requires shader-f16 and fixed32 subgroups");
  }
  const device = await adapter.requestDevice({
    label: "ace-opt-0029-dense-conv-transpose-ab",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension:
        adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const operations: PreparedOperation[] = [];
  let comparedOutputWords = 0;
  let sharedPolyphaseWeightWords = 0;
  const started = performance.now();
  try {
    for (const [ordinal, topology] of SHAPES.entries()) {
      update(`compile + exactness ${ordinal + 1}/5: ${topology.label}`);
      const prepared = await prepareOperation(device, topology, ordinal);
      operations.push(prepared.operation);
      comparedOutputWords += prepared.comparedOutputWords;
      sharedPolyphaseWeightWords += prepared.sharedWeightWords;
      await browserYield();
    }
    await device.queue.onSubmittedWorkDone();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`uncaptured GPU errors: ${uncapturedErrors.join("; ")}`);
    }
    return Object.freeze({
      adapter,
      device,
      operations: Object.freeze(operations),
      correctness: Object.freeze({
        operationCount: operations.length,
        sharedRevision6PolyphaseWeightU16Count: sharedPolyphaseWeightWords,
        comparedOutputU16Count: comparedOutputWords,
        outputMismatchCount: 0,
        outputPrefillRemainingCount: 0,
        rawU16Exact: true,
        milliseconds: performance.now() - started,
      }),
      uncapturedErrors,
    });
  } catch (error) {
    for (const prepared of operations) {
      for (const buffer of prepared.buffers) buffer.destroy();
    }
    device.destroy();
    throw error;
  }
}

async function prepareOperation(
  device: GPUDevice,
  topology: Readonly<{
    label: string;
    shape: AceVaeConvTranspose1dShape;
    weight: 1;
  }>,
  ordinal: number,
): Promise<Readonly<{
  operation: PreparedOperation;
  sharedWeightWords: number;
  comparedOutputWords: number;
}>> {
  const shape = topology.shape;
  const basePlan = planAceFp16VaeConvTranspose1d(shape);
  const controlPlan = planAceOpt0026VaeConvTranspose1d(shape);
  const candidatePlan = planAceOpt0029VaeConvTranspose1d(shape);
  const fullRange = { base: 0, count: basePlan.outputElements };
  const controlRange = planAceOpt0026VaeConvTranspose1dRange(
    controlPlan,
    fullRange,
  );
  const candidateRange = planAceOpt0029VaeConvTranspose1dRange(
    candidatePlan,
    fullRange,
  );

  const input = storageSource(
    device,
    `${topology.label}-input`,
    basePlan.inputBindingBytes,
    periodicWords(basePlan.inputElements, INPUT_PATTERN, ordinal),
  );
  const polyphaseWeight = storageSource(
    device,
    `${topology.label}-revision6-polyphase-weight`,
    basePlan.weightBindingBytes,
    periodicWords(
      basePlan.weightElements,
      POLYPHASE_WEIGHT_PATTERN,
      ordinal * 3,
    ),
  );
  const bias = storageSource(
    device,
    `${topology.label}-bias`,
    basePlan.biasBindingBytes,
    periodicWords(basePlan.outputChannels, BIAS_PATTERN, ordinal),
  );
  const controlOutput = outputBuffer(
    device,
    `${topology.label}-opt0026-output`,
    basePlan.outputBindingBytes,
  );
  const candidateOutput = outputBuffer(
    device,
    `${topology.label}-opt0029-output`,
    basePlan.outputBindingBytes,
  );
  const control = device.createBuffer({
    label: `${topology.label}-range`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(control.getMappedRange()).set([
    0,
    basePlan.outputElements,
    0,
    0,
  ]);
  control.unmap();

  const controlArm = await compileArm(
    device,
    `${topology.label}-opt0026-control`,
    aceOpt0026VaeConvTranspose1dWgsl(shape),
    [input, polyphaseWeight, bias, controlOutput, control],
    [
      controlRange.workgroupsX,
      controlRange.workgroupsY,
      controlRange.workgroupsZ,
    ],
  );
  const candidateArm = await compileArm(
    device,
    `${topology.label}-opt0029-candidate`,
    aceOpt0029VaeConvTranspose1dWgsl(shape),
    [input, polyphaseWeight, bias, candidateOutput, control],
    [
      candidateRange.workgroupsX,
      candidateRange.workgroupsY,
      candidateRange.workgroupsZ,
    ],
  );
  const arms = Object.freeze({
    control: controlArm,
    candidate: candidateArm,
  });
  await executeBoth(device, arms);
  const [controlBits, candidateBits] = await Promise.all([
    readWords(device, controlOutput, basePlan.outputBindingBytes),
    readWords(device, candidateOutput, basePlan.outputBindingBytes),
  ]);
  let prefillRemaining = 0;
  let mismatchCount = 0;
  let firstMismatch = -1;
  for (let index = 0; index < basePlan.outputElements; index += 1) {
    const expected = controlBits[index]!;
    const actual = candidateBits[index]!;
    if (expected === OUTPUT_PREFILL || actual === OUTPUT_PREFILL) {
      prefillRemaining += 1;
    }
    if (expected !== actual) {
      mismatchCount += 1;
      if (firstMismatch < 0) firstMismatch = index;
    }
  }
  if (mismatchCount !== 0) {
    throw new Error(
      `${topology.label} has ${mismatchCount} raw-U16 mismatches; first at ` +
        `${firstMismatch}: 0x${controlBits[firstMismatch]!.toString(16)} != ` +
        `0x${candidateBits[firstMismatch]!.toString(16)}`,
    );
  }
  if (prefillRemaining !== 0) {
    throw new Error(`${topology.label} left ${prefillRemaining} outputs unwritten`);
  }
  return Object.freeze({
    operation: Object.freeze({
      label: topology.label,
      shape,
      weight: topology.weight,
      outputElements: basePlan.outputElements,
      weightElements: basePlan.weightElements,
      arms,
      buffers: Object.freeze([
        input,
        polyphaseWeight,
        bias,
        controlOutput,
        candidateOutput,
        control,
      ]),
    }),
    sharedWeightWords: basePlan.weightElements,
    comparedOutputWords: basePlan.outputElements,
  });
}

async function compileArm(
  device: GPUDevice,
  label: string,
  code: string,
  buffers: readonly GPUBuffer[],
  dispatch: readonly [number, number, number],
): Promise<CompiledArm> {
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
  const bindGroup = device.createBindGroup({
    label: `${label}-bindings`,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, bindingIndex) => ({
      binding: bindingIndex,
      resource: { buffer },
    })),
  });
  return Object.freeze({ pipeline, bindGroup, dispatch });
}

async function executeBoth(
  device: GPUDevice,
  arms: Readonly<Record<Arm, CompiledArm>>,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  for (const arm of ["control", "candidate"] as const) {
    const pass = encoder.beginComputePass();
    encode(pass, arms[arm]);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function timeGate(
  prepared: PreparedGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Arm, number[]>>();
  for (const operation of prepared.operations) {
    samples.set(operation.label, { control: [], candidate: [] });
  }
  const timingStarted = performance.now();
  for (let round = 0; round < TIMING_ROUNDS; round += 1) {
    const order: readonly Arm[] = round % 2 === 0
      ? ["control", "candidate"]
      : ["candidate", "control"];
    for (const [index, operation] of prepared.operations.entries()) {
      update(`timing round ${round + 1}/4, shape ${index + 1}/5`);
      for (const arm of order) {
        const elapsed = await executeTimed(prepared.device, operation.arms[arm]);
        samples.get(operation.label)![arm].push(elapsed);
      }
      await browserYield();
    }
  }
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error(`uncaptured GPU errors: ${prepared.uncapturedErrors.join("; ")}`);
  }
  const cases = prepared.operations.map((operation) => {
    const values = samples.get(operation.label)!;
    const controlMedian = median4(values.control);
    const candidateMedian = median4(values.candidate);
    return Object.freeze({
      label: operation.label,
      shape: operation.shape,
      weight: operation.weight,
      controlSamplesMilliseconds: Object.freeze(values.control.slice()),
      candidateSamplesMilliseconds: Object.freeze(values.candidate.slice()),
      controlMedianMilliseconds: controlMedian,
      candidateMedianMilliseconds: candidateMedian,
      speedup: controlMedian / candidateMedian,
    });
  });
  const controlWeightedMedian = cases.reduce(
    (sum, entry) => sum + entry.weight * entry.controlMedianMilliseconds,
    0,
  );
  const candidateWeightedMedian = cases.reduce(
    (sum, entry) => sum + entry.weight * entry.candidateMedianMilliseconds,
    0,
  );
  const weightedSpeedup = controlWeightedMedian / candidateWeightedMedian;
  const passed = weightedSpeedup >= 1.75;
  const adapterInfo = prepared.adapter.info;
  const cleanupReceipt = cleanup(prepared);
  return Object.freeze({
    schema: "ace-opt-0029-conv-transpose-dense-subgroup-ab-v1",
    passed,
    experimentId: "OPT-0029",
    controlKernelId: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    candidateKernelId: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    weightLayout: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    correctness: prepared.correctness,
    timing: Object.freeze({
      rounds: TIMING_ROUNDS,
      order: "AB/BA alternating, one submit and matching drain per sample",
      milliseconds: performance.now() - timingStarted,
      cases: Object.freeze(cases),
      controlWeightedMedianMilliseconds: controlWeightedMedian,
      candidateWeightedMedianMilliseconds: candidateWeightedMedian,
      weightedMedianSpeedup: weightedSpeedup,
      requiredWeightedSpeedup: 1.75,
      projectedC4500ControlMilliseconds: controlWeightedMedian * 15,
      projectedC4500CandidateMilliseconds: candidateWeightedMedian * 15,
      projectedC4500SavingMilliseconds:
        (controlWeightedMedian - candidateWeightedMedian) * 15,
    }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      adapter: Object.freeze({
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        subgroupMinSize: adapterInfo.subgroupMinSize,
        subgroupMaxSize: adapterInfo.subgroupMaxSize,
      }),
      features: Object.freeze([...prepared.device.features].sort()),
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
): Readonly<{
  label: string;
  shape: AceVaeConvTranspose1dShape;
  weight: 1;
}> {
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
    weight: 1,
  });
}

function periodicWords(
  length: number,
  pattern: Uint16Array,
  offset: number,
): Uint16Array {
  return Uint16Array.from(
    { length },
    (_, index) => pattern[(index + offset) % pattern.length]!,
  );
}

function storageSource(
  device: GPUDevice,
  label: string,
  size: number,
  words: Uint16Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint16Array(buffer.getMappedRange()).set(words);
  buffer.unmap();
  return buffer;
}

function outputBuffer(
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
  const readback = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Uint16Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return copy;
}

function median4(values: readonly number[]): number {
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("OPT-0029 requires four finite timing samples per arm");
  }
  const sorted = values.slice().sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function cleanup(prepared: PreparedGate): Readonly<Record<string, unknown>> {
  let count = 0;
  for (const operation of prepared.operations) {
    for (const buffer of operation.buffers) {
      buffer.destroy();
      count += 1;
    }
  }
  prepared.device.destroy();
  return Object.freeze({ destroyedBufferCount: count, deviceDestroyed: true });
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
  window.__ACE_OPT0029_RESULT__ = receipt;
}

function failure(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0029-conv-transpose-dense-subgroup-ab-v1",
    passed: false,
    experimentId: "OPT-0029",
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
