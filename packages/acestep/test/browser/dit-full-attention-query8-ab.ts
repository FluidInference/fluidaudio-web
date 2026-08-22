import {
  AceCorrectnessAttentionKernel,
  aceCorrectnessAttentionWgsl,
  aceFixed32TiledFullAttentionWgsl,
  planAceFixed32TiledFullAttention,
  type AceAttentionDispatch,
} from "../../src/webgpu/kernels/attention.js";

const CHECKPOINT_COMMIT = "a88e1b41c7b127d20c3fc4dbdee63acb77612a8c";
const ATTENTION_SOURCE_SHA256 =
  "5f64e5148ee60f26023faeb99ac72b46354086f3db48f7778800a9061d2b9ed3";
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel";
const THERMAL_POLL_MILLISECONDS = 1_000;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;
const MAXIMUM_ABSOLUTE_ERROR = 1e-4;
const MAXIMUM_NRMSE = 1e-5;

export const FULL_ATTENTION_GATE_SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});

const QUERY_SEED = 0x1357_9bdf;
const KEY_SEED = 0x2468_ace0;
const VALUE_SEED = 0x1020_3040;
const INPUT_GENERATOR_ID = "lcg1664525-u24-f32-v1";

interface ThermalGate {
  readonly source: typeof THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly launchDelayMilliseconds: number;
}

interface TimedArm {
  readonly output: Float32Array<ArrayBuffer>;
  readonly computeWallMilliseconds: number;
  readonly readbackWallMilliseconds: number;
  readonly totalWallMilliseconds: number;
  readonly commandBufferCount: 2;
  readonly queueDrainCount: 2;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly portableKernel: AceCorrectnessAttentionKernel;
  readonly candidateKernel: AceCorrectnessAttentionKernel;
  readonly portableDispatch: AceAttentionDispatch;
  readonly candidateDispatch: AceAttentionDispatch;
  readonly outputBuffer: GPUBuffer;
  readonly readbackBuffer: GPUBuffer;
  readonly ownedBuffers: readonly GPUBuffer[];
  readonly inputSha256: Readonly<{
    query: string;
    key: string;
    value: string;
  }>;
  readonly wgslSha256: Readonly<{
    portable: string;
    query8: string;
  }>;
  readonly warmupWallMilliseconds: number;
  readonly warmupCompletedAtEpochMilliseconds: number;
  destroy(): void;
}

const progressNode = requireElement<HTMLElement>("#progress");
const resultNode = requireElement<HTMLElement>("#result");
const thermalGateNode = requireElement<HTMLFieldSetElement>("#thermal-gate");
const runButton = requireElement<HTMLButtonElement>("#run");
let active: PreparedGate | undefined;
let runStarted = false;

void prepareGate().then(
  (prepared) => {
    active = prepared;
    document.body.dataset.status = "ready";
    thermalGateNode.disabled = false;
    runButton.disabled = false;
    progressNode.textContent =
      "ready; begin one external continuous-nominal 30-second interval";
  },
  (error: unknown) => finishFailure(error),
);

runButton.addEventListener("click", () => {
  if (runStarted || active === undefined) return;
  runStarted = true;
  runButton.disabled = true;
  thermalGateNode.disabled = true;
  document.body.dataset.status = "running";
  progressNode.textContent = "running portable then query8 once";
  const prepared = active;
  active = undefined;
  void runGate(prepared).catch((error: unknown) => {
    prepared.destroy();
    finishFailure(error);
  });
});

window.addEventListener("beforeunload", () => {
  active?.destroy();
  active = undefined;
});

async function prepareGate(): Promise<PreparedGate> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  if (
    !adapter.features.has("subgroups") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32
  ) {
    throw new Error("M2250 query8 gate requires reported fixed 32-lane subgroups");
  }
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 1_024,
    },
  });
  const ownedBuffers: GPUBuffer[] = [];
  let portableKernel: AceCorrectnessAttentionKernel | undefined;
  let candidateKernel: AceCorrectnessAttentionKernel | undefined;
  try {
    progressNode.textContent = "materializing deterministic production-shape inputs";
    const query = deterministicF32(
      FULL_ATTENTION_GATE_SHAPE.queryHeads *
        FULL_ATTENTION_GATE_SHAPE.queryTokens *
        FULL_ATTENTION_GATE_SHAPE.headDimension,
      QUERY_SEED,
      1,
    );
    const keyValueElements =
      FULL_ATTENTION_GATE_SHAPE.keyValueHeads *
      FULL_ATTENTION_GATE_SHAPE.keyValueTokens *
      FULL_ATTENTION_GATE_SHAPE.headDimension;
    const key = deterministicF32(keyValueElements, KEY_SEED, 1);
    const value = deterministicF32(keyValueElements, VALUE_SEED, 0.5);
    const inputSha256 = Object.freeze({
      query: await sha256(query),
      key: await sha256(key),
      value: await sha256(value),
    });

    const queryBuffer = uploadStorageBuffer(device, query, "m2250-query");
    const keyBuffer = uploadStorageBuffer(device, key, "m2250-key");
    const valueBuffer = uploadStorageBuffer(device, value, "m2250-value");
    const validLengthsBuffer = uploadStorageBuffer(
      device,
      new Uint32Array([1, 1]),
      "m2250-valid-lengths",
      GPUBufferUsage.COPY_DST,
    );
    const outputBytes = query.byteLength;
    const outputBuffer = device.createBuffer({
      label: "m2250-attention-output",
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      label: "m2250-attention-readback",
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    ownedBuffers.push(
      queryBuffer,
      keyBuffer,
      valueBuffer,
      validLengthsBuffer,
      outputBuffer,
      readbackBuffer,
    );

    progressNode.textContent = "compiling exact M2250 portable/query8 pipelines";
    portableKernel = AceCorrectnessAttentionKernel.create(
      device,
      "reference-bf16",
      { backend: "portable" },
    );
    candidateKernel = AceCorrectnessAttentionKernel.create(
      device,
      "reference-bf16",
      {
        backend: "fixed32-subgroup-query8",
        capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
      },
    );
    const bindings = {
      query: binding(queryBuffer),
      key: binding(keyBuffer),
      value: binding(valueBuffer),
      validLengths: binding(validLengthsBuffer),
      output: binding(outputBuffer),
    } as const;
    const [portableDispatch, candidateDispatch] = await Promise.all([
      portableKernel.createDispatch(
        "m2250-portable-full-attention",
        FULL_ATTENTION_GATE_SHAPE,
        bindings,
      ),
      candidateKernel.createDispatch(
        "m2250-query8-full-attention",
        FULL_ATTENTION_GATE_SHAPE,
        bindings,
      ),
    ]);
    if (
      portableDispatch.backend !== "portable" ||
      candidateDispatch.backend !== "fixed32-subgroup-query8"
    ) {
      throw new Error("M2250 gate selected the wrong attention backend");
    }
    const wgslSha256 = Object.freeze({
      portable: await sha256Text(
        aceCorrectnessAttentionWgsl(
          "reference-bf16",
          FULL_ATTENTION_GATE_SHAPE,
        ),
      ),
      query8: await sha256Text(
        aceFixed32TiledFullAttentionWgsl(FULL_ATTENTION_GATE_SHAPE),
      ),
    });

    progressNode.textContent = "running exact-pipeline valid-length warmup";
    const warmupStarted = performance.now();
    await executeOnly(device, portableDispatch, "m2250-portable-warmup");
    await executeOnly(device, candidateDispatch, "m2250-query8-warmup");
    const warmupWallMilliseconds = performance.now() - warmupStarted;
    device.queue.writeBuffer(
      validLengthsBuffer,
      0,
      new Uint32Array([
        FULL_ATTENTION_GATE_SHAPE.queryTokens,
        FULL_ATTENTION_GATE_SHAPE.keyValueTokens,
      ]),
    );
    await device.queue.onSubmittedWorkDone();
    const warmupCompletedAtEpochMilliseconds = Date.now();
    const capturedPortable = portableKernel;
    const capturedCandidate = candidateKernel;
    let destroyed = false;
    return Object.freeze({
      adapter,
      device,
      portableKernel: capturedPortable,
      candidateKernel: capturedCandidate,
      portableDispatch,
      candidateDispatch,
      outputBuffer,
      readbackBuffer,
      ownedBuffers: Object.freeze(ownedBuffers.slice()),
      inputSha256,
      wgslSha256,
      warmupWallMilliseconds,
      warmupCompletedAtEpochMilliseconds,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        capturedPortable.destroy();
        capturedCandidate.destroy();
        for (const buffer of ownedBuffers) buffer.destroy();
        device.destroy();
      },
    });
  } catch (error) {
    portableKernel?.destroy();
    candidateKernel?.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
    device.destroy();
    throw error;
  }
}

async function runGate(prepared: PreparedGate): Promise<void> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseThermalGate(
    prepared.warmupCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const portable = await executeAndRead(
    prepared.device,
    prepared.portableDispatch,
    prepared.outputBuffer,
    prepared.readbackBuffer,
    "m2250-portable-timed",
  );
  const query8 = await executeAndRead(
    prepared.device,
    prepared.candidateDispatch,
    prepared.outputBuffer,
    prepared.readbackBuffer,
    "m2250-query8-timed",
  );
  const correctness = compareOutputs(portable.output, query8.output);
  const [portableOutputSha256, query8OutputSha256] = await Promise.all([
    sha256(portable.output),
    sha256(query8.output),
  ]);
  const plan = planAceFixed32TiledFullAttention(FULL_ATTENTION_GATE_SHAPE);
  const passed =
    correctness.portableNonFiniteCount === 0 &&
    correctness.query8NonFiniteCount === 0 &&
    correctness.maximumAbsoluteError <= MAXIMUM_ABSOLUTE_ERROR &&
    correctness.nrmse <= MAXIMUM_NRMSE;
  const receipt = Object.freeze({
    schemaVersion: 1,
    gate: "production-m2250-full-attention-portable-query8-single-order-v1",
    passed,
    identity: Object.freeze({
      checkpointCommit: CHECKPOINT_COMMIT,
      attentionSourceSha256: ATTENTION_SOURCE_SHA256,
      portableWgslSha256: prepared.wgslSha256.portable,
      query8WgslSha256: prepared.wgslSha256.query8,
      browserUserAgent: navigator.userAgent,
    }),
    shape: FULL_ATTENTION_GATE_SHAPE,
    deterministicInputs: Object.freeze({
      generator: INPUT_GENERATOR_ID,
      querySeed: QUERY_SEED,
      keySeed: KEY_SEED,
      valueSeed: VALUE_SEED,
      sha256: prepared.inputSha256,
    }),
    capabilities: capabilityReceipt(prepared.adapter, prepared.device),
    thermal,
    protocol: Object.freeze({
      armOrder: ["portable", "fixed32-subgroup-query8"] as const,
      samplesPerArm: 1,
      compileOutsideTiming: true,
      exactPipelineWarmupOutsideTiming: true,
      warmupValidQueryTokens: 1,
      warmupValidKeyTokens: 1,
      warmupWallMilliseconds: prepared.warmupWallMilliseconds,
      timedWallIncludesComputeDrainCopyMapAndFullCpuReadback: true,
      outputReadbackBytesPerArm: portable.output.byteLength,
      commandBuffersPerArm: 2,
      queueDrainsPerArm: 2,
      unchangedThermalRetryPerformed: false,
    }),
    traffic: Object.freeze({
      portableWorkgroups: FULL_ATTENTION_GATE_SHAPE.queryHeads *
        FULL_ATTENTION_GATE_SHAPE.queryTokens,
      query8Workgroups: plan.workgroupCount,
      query8RangeCount: plan.outputRangeCount,
      query8RangeWorkgroups: plan.outputRanges.map(
        (range) => range.workgroupCount,
      ),
      portableKeyValueScalarLoads: plan.portableKeyValueScalarLoads,
      query8KeyValueScalarLoads: plan.tiledKeyValueScalarLoads,
      portableKeyValueBytes:
        plan.portableKeyValueScalarLoads * Float32Array.BYTES_PER_ELEMENT,
      query8KeyValueBytes:
        plan.tiledKeyValueScalarLoads * Float32Array.BYTES_PER_ELEMENT,
      keyValueLoadReduction: plan.keyValueLoadReduction,
      portableBarriersPerKey: plan.portableBarriersPerKey,
      query8BarriersPerKey: plan.tiledBarriersPerKey,
    }),
    timing: Object.freeze({
      portable: timingReceipt(portable),
      query8: timingReceipt(query8),
      totalWallSpeedup:
        portable.totalWallMilliseconds / query8.totalWallMilliseconds,
      computeWallSpeedup:
        portable.computeWallMilliseconds / query8.computeWallMilliseconds,
    }),
    correctness: Object.freeze({
      ...correctness,
      portableOutputSha256,
      query8OutputSha256,
      maximumAbsoluteErrorThreshold: MAXIMUM_ABSOLUTE_ERROR,
      nrmseThreshold: MAXIMUM_NRMSE,
      comparedEveryOutputF32: true,
    }),
    cleanup: Object.freeze({
      ownedBufferCount: prepared.ownedBuffers.length,
      attentionKernelCount: 2,
      kernelsDestroyed: true,
      ownedBuffersDestroyed: true,
      deviceDestroyed: true,
    }),
  });
  prepared.destroy();
  resultNode.textContent = JSON.stringify(receipt);
  document.body.dataset.status = passed ? "passed" : "failed";
  progressNode.textContent = passed
    ? "passed; production-shape receipt ready"
    : "failed numerical envelope; receipt retained";
}

async function executeOnly(
  device: GPUDevice,
  dispatch: AceAttentionDispatch,
  label: string,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeAndRead(
  device: GPUDevice,
  dispatch: AceAttentionDispatch,
  output: GPUBuffer,
  readback: GPUBuffer,
  label: string,
): Promise<TimedArm> {
  const started = performance.now();
  const computeEncoder = device.createCommandEncoder({ label: `${label}-compute` });
  const pass = computeEncoder.beginComputePass({ label: `${label}-compute` });
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([computeEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const computeCompleted = performance.now();
  const readbackEncoder = device.createCommandEncoder({
    label: `${label}-readback`,
  });
  readbackEncoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
  device.queue.submit([readbackEncoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const detached = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  const completed = performance.now();
  return Object.freeze({
    output: detached,
    computeWallMilliseconds: computeCompleted - started,
    readbackWallMilliseconds: completed - computeCompleted,
    totalWallMilliseconds: completed - started,
    commandBufferCount: 2 as const,
    queueDrainCount: 2 as const,
  });
}

function compareOutputs(
  portable: Float32Array,
  query8: Float32Array,
): Readonly<{
  comparedElementCount: number;
  portableNonFiniteCount: number;
  query8NonFiniteCount: number;
  maximumAbsoluteError: number;
  meanAbsoluteError: number;
  nrmse: number;
}> {
  if (portable.length !== query8.length) {
    throw new Error("M2250 output lengths differ");
  }
  let portableNonFiniteCount = 0;
  let query8NonFiniteCount = 0;
  let maximumAbsoluteError = 0;
  let absoluteErrorSum = 0;
  let squaredError = 0;
  let squaredReference = 0;
  for (let index = 0; index < portable.length; index += 1) {
    const reference = portable[index]!;
    const candidate = query8[index]!;
    if (!Number.isFinite(reference)) portableNonFiniteCount += 1;
    if (!Number.isFinite(candidate)) query8NonFiniteCount += 1;
    if (!Number.isFinite(reference) || !Number.isFinite(candidate)) continue;
    const error = candidate - reference;
    const absoluteError = Math.abs(error);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
    absoluteErrorSum += absoluteError;
    squaredError += error * error;
    squaredReference += reference * reference;
  }
  return Object.freeze({
    comparedElementCount: portable.length,
    portableNonFiniteCount,
    query8NonFiniteCount,
    maximumAbsoluteError,
    meanAbsoluteError: absoluteErrorSum / portable.length,
    nrmse: Math.sqrt(squaredError / squaredReference),
  });
}

function parseThermalGate(
  warmupCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): ThermalGate {
  const source = thermalValue("thermalSource");
  const startedAtEpochMilliseconds = thermalNumber(
    "thermalStartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = thermalNumber(
    "thermalCompletedAtEpochMilliseconds",
  );
  const observationCount = thermalNumber("thermalObservations");
  const pollMilliseconds = thermalNumber("thermalPollMilliseconds");
  const maximumPollGapMilliseconds = thermalNumber(
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = thermalNumber(
    "thermalNonNominalObservations",
  );
  const durationMilliseconds =
    completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  const launchDelayMilliseconds =
    launchedAtEpochMilliseconds - completedAtEpochMilliseconds;
  if (source !== THERMAL_SOURCE) {
    throw new Error("M2250 gate requires the accepted notifyutil thermal source");
  }
  if (
    startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    completedAtEpochMilliseconds > launchedAtEpochMilliseconds ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error("M2250 thermal interval is stale, short, or predates warmup");
  }
  if (
    !Number.isSafeInteger(observationCount) ||
    observationCount < Math.floor(durationMilliseconds / pollMilliseconds) + 1 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0
  ) {
    throw new Error("M2250 thermal observations are incomplete or non-nominal");
  }
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0 as const,
    launchDelayMilliseconds,
  });
}

function capabilityReceipt(adapter: GPUAdapter, device: GPUDevice) {
  return Object.freeze({
    adapterInfo: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    adapterFeatures: Object.freeze([...adapter.features].sort()),
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
      minUniformBufferOffsetAlignment:
        device.limits.minUniformBufferOffsetAlignment,
    }),
  });
}

function timingReceipt(arm: TimedArm) {
  return Object.freeze({
    computeWallMilliseconds: arm.computeWallMilliseconds,
    readbackWallMilliseconds: arm.readbackWallMilliseconds,
    totalWallMilliseconds: arm.totalWallMilliseconds,
    commandBufferCount: arm.commandBufferCount,
    queueDrainCount: arm.queueDrainCount,
  });
}

function deterministicF32(
  length: number,
  seed: number,
  scale: number,
): Float32Array<ArrayBuffer> {
  const output = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < output.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const unit = (state >>> 8) / 0x1_000000;
    output[index] = Math.fround((unit * 2 - 1) * scale);
  }
  return output;
}

function uploadStorageBuffer(
  device: GPUDevice,
  data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
  label: string,
  additionalUsage = 0,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | additionalUsage,
    mappedAtCreation: true,
  });
  const mapped = buffer.getMappedRange();
  if (data instanceof Float32Array) {
    new Float32Array(mapped).set(data);
  } else {
    new Uint32Array(mapped).set(data);
  }
  buffer.unmap();
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

async function sha256(
  value: Float32Array<ArrayBuffer>,
): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", value));
}

async function sha256Text(value: string): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
}

function hex(value: ArrayBuffer): string {
  return [...new Uint8Array(value)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function thermalValue(name: string): string {
  return requireElement<HTMLInputElement>(
    `#thermal-gate input[name="${name}"]`,
  ).value;
}

function thermalNumber(name: string): number {
  const value = Number(thermalValue(name));
  if (!Number.isFinite(value)) {
    throw new Error(`M2250 thermal field ${name} is missing or non-finite`);
  }
  return value;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function finishFailure(error: unknown): void {
  document.body.dataset.status = "failed";
  progressNode.textContent = "failed";
  resultNode.textContent = error instanceof Error
    ? error.stack ?? error.message
    : String(error);
}
