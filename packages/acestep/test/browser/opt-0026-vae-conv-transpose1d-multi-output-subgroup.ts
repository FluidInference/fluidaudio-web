/// <reference types="@webgpu/types" />

import {
  aceFp16VaeCongruentConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
  ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
  aceOpt0026VaeConvTranspose1dWgsl,
  packAceOpt0026VaeConvTranspose1dWeights,
  planAceOpt0026VaeConvTranspose1d,
  planAceOpt0026VaeConvTranspose1dRange,
  unpackAceOpt0026VaeConvTranspose1dWeights,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-multi-output-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

type Arm = "current" | "candidate";

interface CompiledArm {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly dispatch: readonly [number, number, number];
}

interface PreparedOperation {
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
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
const WEIGHT_PATTERN = new Uint16Array([
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
        "ready: wait 30 seconds, confirm nominal thermal state, then run";
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
      (receipt) => finish("passed", receipt),
      (error: unknown) => {
        cleanup(owned);
        finish("failed", failure(error));
      },
    );
  }, { once: true });
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
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    Number(info.subgroupMinSize) !== 32 ||
    Number(info.subgroupMaxSize) !== 32) {
    throw new Error("OPT-0026 requires shader-f16 and fixed32 subgroups");
  }
  const device = await adapter.requestDevice({
    label: "ace-opt-0026-conv-transpose-ab",
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
  let comparedWeightWords = 0;
  const started = performance.now();
  try {
    for (const [ordinal, topology] of SHAPES.entries()) {
      update(`exactness ${ordinal + 1}/5: ${topology.label}`);
      const prepared = await prepareOperation(device, topology, ordinal);
      operations.push(prepared.operation);
      comparedOutputWords += prepared.comparedOutputWords;
      comparedWeightWords += prepared.comparedWeightWords;
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
        comparedWeightU16Count: comparedWeightWords,
        inverseWeightMismatchCount: 0,
        comparedOutputU16Count: comparedOutputWords,
        outputMismatchCount: 0,
        outputPrefillRemainingCount: 0,
        rawU16Exact: true,
        milliseconds: performance.now() - started,
      }),
      uncapturedErrors,
    });
  } catch (error) {
    for (const operation of operations) {
      for (const buffer of operation.buffers) buffer.destroy();
    }
    device.destroy();
    throw error;
  }
}

async function prepareOperation(
  device: GPUDevice,
  topology: Readonly<{ label: string; shape: AceVaeConvTranspose1dShape }>,
  ordinal: number,
): Promise<Readonly<{
  operation: PreparedOperation;
  comparedWeightWords: number;
  comparedOutputWords: number;
}>> {
  const shape = topology.shape;
  const plan = planAceFp16VaeConvTranspose1d(shape);
  const candidatePlan = planAceOpt0026VaeConvTranspose1d(shape);
  const range = { base: 0, count: plan.outputElements };
  const currentRange = planAceFp16VaeConvTranspose1dCongruentRange(plan, range);
  const candidateRange = planAceOpt0026VaeConvTranspose1dRange(
    candidatePlan,
    range,
  );
  const nativeWords = periodicWords(
    plan.weightElements,
    WEIGHT_PATTERN,
    ordinal * 3,
  );
  const packedWords = packAceOpt0026VaeConvTranspose1dWeights(
    nativeWords,
    shape,
  );
  const inverseWords = unpackAceOpt0026VaeConvTranspose1dWeights(
    packedWords,
    shape,
  );
  for (let index = 0; index < nativeWords.length; index += 1) {
    if (nativeWords[index] !== inverseWords[index]) {
      throw new Error(`${topology.label} inverse mismatch at ${index}`);
    }
  }

  const input = storageSource(
    device,
    `${topology.label}-input`,
    plan.inputBindingBytes,
    periodicWords(plan.inputElements, INPUT_PATTERN, ordinal),
  );
  const nativeWeight = storageSource(
    device,
    `${topology.label}-native-weight`,
    plan.weightBindingBytes,
    nativeWords,
  );
  const packedWeight = storageSource(
    device,
    `${topology.label}-polyphase-weight`,
    plan.weightBindingBytes,
    packedWords,
  );
  const bias = storageSource(
    device,
    `${topology.label}-bias`,
    plan.biasBindingBytes,
    periodicWords(plan.outputChannels, BIAS_PATTERN, ordinal),
  );
  const currentOutput = outputBuffer(
    device,
    `${topology.label}-current-output`,
    plan.outputBindingBytes,
  );
  const candidateOutput = outputBuffer(
    device,
    `${topology.label}-candidate-output`,
    plan.outputBindingBytes,
  );
  const control = device.createBuffer({
    label: `${topology.label}-range`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(control.getMappedRange()).set([0, plan.outputElements, 0, 0]);
  control.unmap();
  const current = await compileArm(
    device,
    `${topology.label}-current`,
    aceFp16VaeCongruentConvTranspose1dWgsl(shape),
    [input, nativeWeight, bias, currentOutput, control],
    [currentRange.workgroupsX, currentRange.workgroupsY, currentRange.workgroupsZ],
  );
  const candidate = await compileArm(
    device,
    `${topology.label}-candidate`,
    aceOpt0026VaeConvTranspose1dWgsl(shape),
    [input, packedWeight, bias, candidateOutput, control],
    [
      candidateRange.workgroupsX,
      candidateRange.workgroupsY,
      candidateRange.workgroupsZ,
    ],
  );
  const arms = Object.freeze({ current, candidate });
  await executeBoth(device, arms);
  const [currentBits, candidateBits] = await Promise.all([
    readWords(device, currentOutput, plan.outputBindingBytes),
    readWords(device, candidateOutput, plan.outputBindingBytes),
  ]);
  let prefillRemaining = 0;
  for (let index = 0; index < plan.outputElements; index += 1) {
    const expected = currentBits[index]!;
    const actual = candidateBits[index]!;
    if (expected === OUTPUT_PREFILL || actual === OUTPUT_PREFILL) {
      prefillRemaining += 1;
    }
    if (expected !== actual) {
      throw new Error(
        `${topology.label} output mismatch at ${index}: ` +
          `0x${expected.toString(16)} != 0x${actual.toString(16)}`,
      );
    }
  }
  if (prefillRemaining !== 0) {
    throw new Error(`${topology.label} left ${prefillRemaining} outputs unwritten`);
  }
  return Object.freeze({
    operation: Object.freeze({
      label: topology.label,
      shape,
      outputElements: plan.outputElements,
      weightElements: plan.weightElements,
      arms,
      buffers: Object.freeze([
        input,
        nativeWeight,
        packedWeight,
        bias,
        currentOutput,
        candidateOutput,
        control,
      ]),
    }),
    comparedWeightWords: plan.weightElements,
    comparedOutputWords: plan.outputElements,
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
    entries: buffers.map((buffer, binding) => ({
      binding,
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
  for (const arm of ["current", "candidate"] as const) {
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
    samples.set(operation.label, { current: [], candidate: [] });
  }
  const timingStarted = performance.now();
  for (let round = 0; round < TIMING_ROUNDS; round += 1) {
    const order: readonly Arm[] = round % 2 === 0
      ? ["current", "candidate"]
      : ["candidate", "current"];
    for (const [index, operation] of prepared.operations.entries()) {
      update(`timing round ${round + 1}/${TIMING_ROUNDS}, shape ${index + 1}/5`);
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
    const currentMedian = median(values.current);
    const candidateMedian = median(values.candidate);
    return Object.freeze({
      label: operation.label,
      shape: operation.shape,
      currentSamplesMilliseconds: Object.freeze(values.current),
      candidateSamplesMilliseconds: Object.freeze(values.candidate),
      currentMedianMilliseconds: currentMedian,
      candidateMedianMilliseconds: candidateMedian,
      speedup: currentMedian / candidateMedian,
    });
  });
  const currentTotal = cases.reduce(
    (sum, entry) => sum + entry.currentMedianMilliseconds,
    0,
  );
  const candidateTotal = cases.reduce(
    (sum, entry) => sum + entry.candidateMedianMilliseconds,
    0,
  );
  const adapterInfo = prepared.adapter.info;
  const cleanupReceipt = cleanup(prepared);
  return Object.freeze({
    schema: "ace-opt-0026-conv-transpose-multi-output-subgroup-ab-v1",
    status: "passed",
    experimentId: "OPT-0026",
    kernelId: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    weightLayout: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    correctness: prepared.correctness,
    timing: Object.freeze({
      rounds: TIMING_ROUNDS,
      order: "AB/BA alternating, one submit and matching drain per sample",
      milliseconds: performance.now() - timingStarted,
      cases: Object.freeze(cases),
      currentSummedMedianMilliseconds: currentTotal,
      candidateSummedMedianMilliseconds: candidateTotal,
      summedMedianSpeedup: currentTotal / candidateTotal,
      projectedC4500CurrentMilliseconds: currentTotal * 15,
      projectedC4500CandidateMilliseconds: candidateTotal * 15,
      projectedC4500SavingMilliseconds:
        (currentTotal - candidateTotal) * 15,
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

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
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
  (window as typeof window & {
    __ACE_OPT0026_RESULT__?: Readonly<Record<string, unknown>>;
  }).__ACE_OPT0026_RESULT__ = receipt;
}

function failure(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0026-conv-transpose-multi-output-subgroup-ab-v1",
    status: "failed",
    experimentId: "OPT-0026",
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
