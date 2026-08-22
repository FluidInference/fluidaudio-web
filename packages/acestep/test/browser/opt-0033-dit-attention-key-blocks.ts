/// <reference types="@webgpu/types" />

import {
  aceFixed32TiledFullAttentionWgsl,
  planAceFixed32TiledFullAttention,
} from "../../src/webgpu/kernels/attention.js";
import {
  aceOpt0033AttentionWgsl,
  planAceOpt0033Attention,
} from "../../src/webgpu/kernels/attention-key-blocks.js";

declare global {
  interface Window {
    __ACE_OPT0033_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0033Arm = "query8" | "keyBlock8" | "keyBlock16";

export interface Opt0033TimingInput {
  readonly arm: Opt0033Arm;
  readonly samplesMilliseconds: readonly number[];
}

export interface Opt0033ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly launchDelayMilliseconds: number;
}

interface CompiledArm {
  readonly id: Opt0033Arm;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroupCount: number;
  readonly wgslSha256: string;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly logicalElements: number;
  readonly totalBytes: number;
}

interface Snapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly arms: ReadonlyMap<Opt0033Arm, CompiledArm>;
  readonly buffers: readonly GPUBuffer[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly inputSha256: Readonly<Record<"query" | "key" | "value", string>>;
  readonly uncapturedErrors: readonly string[];
  readonly preparedAtEpochMilliseconds: number;
  destroy(): void;
}

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});
const ARMS = Object.freeze([
  "query8",
  "keyBlock8",
  "keyBlock16",
] as const);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["query8", "keyBlock8", "keyBlock16"] as const),
  Object.freeze(["keyBlock8", "keyBlock16", "query8"] as const),
  Object.freeze(["keyBlock16", "query8", "keyBlock8"] as const),
] as const);
const REQUIRED_SPEEDUP = 1.35;
const QUERY_SEED = 0x1357_9bdf;
const KEY_SEED = 0x2468_ace0;
const VALUE_SEED = 0x1020_3040;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_3355;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
// Account for browser-control round-trip latency while keeping the measured
// completion-to-launch delay explicit in the receipt.
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 30_000;

export function buildOpt0033TimingOrders(): typeof TIMING_ORDERS {
  return TIMING_ORDERS;
}

export function summarizeOpt0033Timing(
  inputs: readonly Opt0033TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== ARMS.length ||
    inputs.some((input, index) => input.arm !== ARMS[index])) {
    throw new Error("OPT-0033 timing arms changed");
  }
  const medians = Object.freeze(Object.fromEntries(inputs.map((input) => [
    input.arm,
    median3(input.samplesMilliseconds),
  ])) as Record<Opt0033Arm, number>);
  const bestCandidate = medians.keyBlock8 <= medians.keyBlock16
    ? "keyBlock8" as const
    : "keyBlock16" as const;
  const bestCandidateMilliseconds = medians[bestCandidate];
  const speedupVersusQuery8 = medians.query8 / bestCandidateMilliseconds;
  const passed = speedupVersusQuery8 >= REQUIRED_SPEEDUP;
  return Object.freeze({
    samplesPerArm: 3,
    mediansMilliseconds: medians,
    bestCandidate,
    bestCandidateMilliseconds,
    speedupVersusQuery8,
    requiredSpeedupVersusQuery8: REQUIRED_SPEEDUP,
    passed,
    decision: passed
      ? "positive-primitive-qualifier"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0033ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0033ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const startedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalCompletedAtEpochMilliseconds",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalObservations",
  );
  const pollMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredFiniteParameter(
    parameters,
    "thermalNonNominalObservations",
  );
  const durationMilliseconds = completedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    completedAtEpochMilliseconds;
  if (
    source !== THERMAL_SOURCE ||
    !Number.isSafeInteger(observationCount) ||
    observationCount < 31 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error("OPT-0033 thermal gate is incomplete, stale, or non-nominal");
  }
  return Object.freeze({
    source: THERMAL_SOURCE,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
    launchDelayMilliseconds,
  });
}

if (typeof document !== "undefined") installBrowserHarness();

function installBrowserHarness(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const runButton = requireElement<HTMLButtonElement>("#run");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let active: PreparedHarness | undefined;
  let started = false;
  void prepareHarness((message) => {
    progress.textContent = message;
  }).then(
    (prepared) => {
      active = prepared;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — all 4,608,000 F32 outputs are bit-exact; timing has not run";
      thermalGate.disabled = false;
      runButton.disabled = false;
    },
    (error: unknown) => fail(error),
  );
  runButton.addEventListener("click", () => {
    if (started || active === undefined) return;
    started = true;
    runButton.disabled = true;
    thermalGate.disabled = true;
    document.body.dataset.status = "running";
    const prepared = active;
    active = undefined;
    void runTiming(prepared).catch((error: unknown) => {
      prepared.destroy();
      fail(error);
    });
  });
  window.addEventListener("beforeunload", () => {
    active?.destroy();
    active = undefined;
  });
}

async function prepareHarness(
  updateProgress: (message: string) => void,
): Promise<PreparedHarness> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const outputBytes = outputElements() * Float32Array.BYTES_PER_ELEMENT;
  const device = await adapter.requestDevice({
    label: "ace-opt-0033-attention-key-blocks-device",
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxBufferSize: outputBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: outputBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16_384,
    },
  });
  const buffers: GPUBuffer[] = [];
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  try {
    updateProgress("materializing deterministic M2250 F32 inputs");
    const query = deterministicF32(outputElements(), QUERY_SEED, 0.125);
    const keyValueElements =
      SHAPE.keyValueHeads * SHAPE.keyValueTokens * SHAPE.headDimension;
    const key = deterministicF32(keyValueElements, KEY_SEED, 0.125);
    const value = deterministicF32(keyValueElements, VALUE_SEED, 0.25);
    const inputSha256 = Object.freeze({
      query: await sha256F32(query),
      key: await sha256F32(key),
      value: await sha256F32(value),
    });
    const queryBuffer = uploadStorage(device, query, "opt0033-query");
    buffers.push(queryBuffer);
    const keyBuffer = uploadStorage(device, key, "opt0033-key");
    buffers.push(keyBuffer);
    const valueBuffer = uploadStorage(device, value, "opt0033-value");
    buffers.push(valueBuffer);
    const validLengthsBuffer = uploadStorage(
      device,
      new Uint32Array([SHAPE.queryTokens, SHAPE.keyValueTokens]),
      "opt0033-valid-lengths",
    );
    buffers.push(validLengthsBuffer);
    const rangeBuffer = uploadUniform(
      device,
      new Uint32Array([0, 0, 0, 0]),
      "opt0033-range",
    );
    buffers.push(rangeBuffer);
    const guarded = createGuardedOutput(device);
    buffers.push(guarded.buffer, guarded.prefill, guarded.readback);

    updateProgress("compiling query8, key-block8, and key-block16");
    const sources = new Map<Opt0033Arm, string>([
      ["query8", aceFixed32TiledFullAttentionWgsl(SHAPE)],
      ["keyBlock8", aceOpt0033AttentionWgsl(SHAPE, 8)],
      ["keyBlock16", aceOpt0033AttentionWgsl(SHAPE, 16)],
    ]);
    const workgroupCount = planAceFixed32TiledFullAttention(SHAPE).workgroupCount;
    const arms = new Map<Opt0033Arm, CompiledArm>();
    for (const id of ARMS) {
      const source = requireMapValue(sources, id);
      const module = device.createShaderModule({
        label: `opt0033-${id}-module`,
        code: source,
      });
      await requireCleanCompilation(module, id);
      const pipeline = await device.createComputePipelineAsync({
        label: `opt0033-${id}-pipeline`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        label: `opt0033-${id}-bindings`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: binding(queryBuffer) },
          { binding: 1, resource: binding(keyBuffer) },
          { binding: 2, resource: binding(valueBuffer) },
          { binding: 3, resource: binding(validLengthsBuffer) },
          { binding: 4, resource: guarded.binding },
          { binding: 7, resource: binding(rangeBuffer, 16) },
        ],
      });
      arms.set(id, Object.freeze({
        id,
        pipeline,
        bindGroup,
        workgroupCount,
        wgslSha256: await sha256Text(source),
      }));
    }

    updateProgress("running untimed complete raw-F32 correctness");
    const snapshots = new Map<Opt0033Arm, Snapshot>();
    for (const id of ARMS) {
      const snapshot = await executeAndRead(
        device,
        requireMapValue(arms, id),
        guarded,
        `opt0033-correctness-${id}`,
      );
      requireCompleteSnapshot(snapshot, id);
      snapshots.set(id, snapshot);
    }
    const query8 = requireMapValue(snapshots, "query8");
    const keyBlock8 = requireMapValue(snapshots, "keyBlock8");
    const keyBlock16 = requireMapValue(snapshots, "keyBlock16");
    const mismatch8 = countBitMismatches(query8.words, keyBlock8.words);
    const mismatch16 = countBitMismatches(query8.words, keyBlock16.words);
    if (mismatch8 !== 0 || mismatch16 !== 0) {
      throw new Error(
        `OPT-0033 raw-F32 mismatch: block8=${mismatch8}, block16=${mismatch16}`,
      );
    }
    if (uncapturedErrors.length !== 0) {
      throw new Error(`OPT-0033 observed ${uncapturedErrors.length} GPU errors`);
    }
    const correctness = Object.freeze({
      outputElements: query8.words.length,
      comparedElementsPerCandidate: query8.words.length,
      totalComparedElements: query8.words.length * 2,
      query8NonFiniteCount: query8.nonFiniteCount,
      keyBlock8NonFiniteCount: keyBlock8.nonFiniteCount,
      keyBlock16NonFiniteCount: keyBlock16.nonFiniteCount,
      keyBlock8BitMismatchCount: mismatch8,
      keyBlock16BitMismatchCount: mismatch16,
      completeWrites: true,
      canariesUntouched: true,
      bitExact: true,
      outputSha256: Object.freeze({
        query8: await sha256U32(query8.words),
        keyBlock8: await sha256U32(keyBlock8.words),
        keyBlock16: await sha256U32(keyBlock16.words),
      }),
    });
    // Correctness executions are the untimed warmups; scratch is no longer
    // needed for the button-triggered timing screen.
    guarded.prefill.destroy();
    guarded.readback.destroy();
    buffers.splice(buffers.indexOf(guarded.prefill), 1);
    buffers.splice(buffers.indexOf(guarded.readback), 1);

    let destroyed = false;
    return Object.freeze({
      adapter,
      device,
      arms,
      buffers: Object.freeze(buffers.slice()),
      correctness,
      inputSha256,
      uncapturedErrors,
      preparedAtEpochMilliseconds: Date.now(),
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of buffers) buffer.destroy();
        device.destroy();
      },
    });
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
    throw error;
  }
}

async function runTiming(prepared: PreparedHarness): Promise<void> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0033ThermalGate(
    collectThermalParameters(),
    prepared.preparedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const samples = new Map<Opt0033Arm, number[]>(
    ARMS.map((arm) => [arm, []]),
  );
  for (let round = 0; round < TIMING_ORDERS.length; round += 1) {
    const order = TIMING_ORDERS[round]!;
    for (let position = 0; position < order.length; position += 1) {
      const id = order[position]!;
      requireElement<HTMLElement>("#progress").textContent =
        `timing round ${round + 1}/3, position ${position + 1}/3: ${id}`;
      samples.get(id)!.push(await timeThroughQueueDrain(
        prepared.device,
        requireMapValue(prepared.arms, id),
        `opt0033-timed-r${round + 1}-${id}`,
      ));
    }
  }
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0033 observed an uncaptured timed GPU error");
  }
  const timing = summarizeOpt0033Timing(ARMS.map((arm) => Object.freeze({
    arm,
    samplesMilliseconds: Object.freeze(samples.get(arm)!.slice()),
  })));
  const receipt = Object.freeze({
    schemaVersion: 1,
    experiment: "OPT-0033",
    passed: timing["passed"],
    identity: Object.freeze({
      browserUserAgent: navigator.userAgent,
      wgslSha256: Object.freeze(Object.fromEntries(ARMS.map((id) => [
        id,
        requireMapValue(prepared.arms, id).wgslSha256,
      ]))),
    }),
    shape: SHAPE,
    capabilities: capabilityReceipt(prepared.adapter, prepared.device),
    deterministicInputs: Object.freeze({
      generator: "lcg1664525-u24-f32-v1",
      querySeed: QUERY_SEED,
      keySeed: KEY_SEED,
      valueSeed: VALUE_SEED,
      sha256: prepared.inputSha256,
    }),
    plans: Object.freeze({
      query8: compactCurrentPlan(),
      keyBlock8: compactCandidatePlan(8),
      keyBlock16: compactCandidatePlan(16),
    }),
    correctness: prepared.correctness,
    protocol: Object.freeze({
      thermal,
      compileOutsideTiming: true,
      fullCorrectnessOutsideTiming: true,
      correctnessAlsoServesAsOneWarmupPerArm: true,
      timedGpuWorkBeforeButton: false,
      rounds: 3,
      balancedOrders: TIMING_ORDERS,
      samplesPerArm: 3,
      oneCommandBufferPerSample: true,
      oneQueueDrainPerSample: true,
      outputReadbackInsideTiming: false,
    }),
    timing,
    decision: Object.freeze({
      disposition: timing["decision"],
      productionIntegrationAuthorized: false,
      productClaimAuthorized: false,
    }),
    cleanup: Object.freeze({
      ownedBufferCount: prepared.buffers.length,
      buffersDestroyed: true,
      deviceDestroyed: true,
    }),
  });
  prepared.destroy();
  window.__ACE_OPT0033_RESULT__ = receipt;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    receipt,
    null,
    2,
  );
  document.body.dataset.status = receipt.passed ? "passed" : "failed";
  requireElement<HTMLElement>("#progress").textContent = receipt.passed
    ? `passed — ${String(timing["bestCandidate"])} reached ${Number(timing["speedupVersusQuery8"]).toFixed(3)}×`
    : `completed — best blocked arm reached only ${Number(timing["speedupVersusQuery8"]).toFixed(3)}×`;
}

async function requireCleanCompilation(
  module: GPUShaderModule,
  id: Opt0033Arm,
): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `OPT-0033 ${id} WGSL failed: ` + errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; "),
    );
  }
}

async function executeAndRead(
  device: GPUDevice,
  arm: CompiledArm,
  output: GuardedOutput,
  label: string,
): Promise<Snapshot> {
  const encoder = device.createCommandEncoder({ label });
  encoder.copyBufferToBuffer(
    output.prefill,
    0,
    output.buffer,
    0,
    output.totalBytes,
  );
  const pass = encoder.beginComputePass({ label });
  encodeArm(pass, arm);
  pass.end();
  encoder.copyBufferToBuffer(
    output.buffer,
    0,
    output.readback,
    0,
    output.totalBytes,
  );
  device.queue.submit([encoder.finish()]);
  await output.readback.mapAsync(GPUMapMode.READ);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const guardWords = STORAGE_GUARD_BYTES / 4;
    const first = guardWords;
    const last = first + output.logicalElements;
    let prefixCanaryIntact = true;
    let suffixCanaryIntact = true;
    for (let index = 0; index < guardWords; index += 1) {
      prefixCanaryIntact &&= all[index] === STORAGE_GUARD_U32;
      suffixCanaryIntact &&= all[last + index] === STORAGE_GUARD_U32;
    }
    const words = all.slice(first, last);
    let nonFiniteCount = 0;
    let qNaNPrefillCount = 0;
    for (const word of words) {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    }
    return Object.freeze({
      words,
      nonFiniteCount,
      qNaNPrefillCount,
      prefixCanaryIntact,
      suffixCanaryIntact,
    });
  } finally {
    output.readback.unmap();
  }
}

async function timeThroughQueueDrain(
  device: GPUDevice,
  arm: CompiledArm,
  label: string,
): Promise<number> {
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  encodeArm(pass, arm);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function encodeArm(pass: GPUComputePassEncoder, arm: CompiledArm): void {
  pass.setPipeline(arm.pipeline);
  pass.setBindGroup(0, arm.bindGroup);
  pass.dispatchWorkgroups(arm.workgroupCount, 1, 1);
}

function createGuardedOutput(device: GPUDevice): GuardedOutput {
  const logicalElements = outputElements();
  const logicalBytes = logicalElements * Float32Array.BYTES_PER_ELEMENT;
  const totalBytes = logicalBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = device.createBuffer({
    label: "opt0033-output-prefill",
    size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(prefill.getMappedRange());
  words.fill(STORAGE_GUARD_U32);
  words.fill(
    OUTPUT_PREFILL_QNAN_U32,
    STORAGE_GUARD_BYTES / 4,
    STORAGE_GUARD_BYTES / 4 + logicalElements,
  );
  prefill.unmap();
  const buffer = device.createBuffer({
    label: "opt0033-guarded-output",
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "opt0033-readback",
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({
      buffer,
      offset: STORAGE_GUARD_BYTES,
      size: logicalBytes,
    }),
    prefill,
    readback,
    logicalElements,
    totalBytes,
  });
}

function uploadStorage(
  device: GPUDevice,
  values: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: values.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
  buffer.unmap();
  return buffer;
}

function uploadUniform(
  device: GPUDevice,
  values: Uint32Array<ArrayBuffer>,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: values.byteLength,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}

function binding(buffer: GPUBuffer, size = buffer.size): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function deterministicF32(
  length: number,
  seed: number,
  scale: number,
): Float32Array<ArrayBuffer> {
  const values = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const unit = (state >>> 8) / 0x1_000000;
    values[index] = Math.fround((unit * 2 - 1) * scale);
  }
  return values;
}

function countBitMismatches(
  expected: Uint32Array<ArrayBuffer>,
  actual: Uint32Array<ArrayBuffer>,
): number {
  if (expected.length !== actual.length) {
    throw new Error("OPT-0033 output length changed");
  }
  let count = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) count += 1;
  }
  return count;
}

function requireCompleteSnapshot(snapshot: Snapshot, id: Opt0033Arm): void {
  if (
    snapshot.nonFiniteCount !== 0 ||
    snapshot.qNaNPrefillCount !== 0 ||
    !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact
  ) {
    throw new Error(`OPT-0033 ${id} failed finite, complete-write, or canary gate`);
  }
}

function compactCurrentPlan(): Readonly<Record<string, unknown>> {
  const plan = planAceFixed32TiledFullAttention(SHAPE);
  return Object.freeze({
    queryTokensPerTile: plan.queryTokensPerTile,
    queriesPerWorkgroup: plan.queriesPerWorkgroup,
    workgroupSize: plan.workgroupSize,
    workgroupCount: plan.workgroupCount,
    keyValueScalarLoads: plan.tiledKeyValueScalarLoads,
    barriersPerWorkgroup: plan.keyValueTokens * 2,
    workgroupStorageBytes: plan.workgroupStorageBytes,
  });
}

function compactCandidatePlan(keyBlock: 8 | 16): Readonly<Record<string, unknown>> {
  const plan = planAceOpt0033Attention(SHAPE, keyBlock);
  return Object.freeze({
    keyBlock,
    queryTokensPerTile: plan.queryTokensPerTile,
    queriesPerWorkgroup: plan.queriesPerWorkgroup,
    workgroupSize: plan.workgroupSize,
    workgroupCount: plan.workgroupCount,
    keyValueScalarLoads: plan.keyValueScalarLoads,
    keyBlocksPerWorkgroup: plan.keyBlocksPerWorkgroup,
    barriersPerWorkgroup: plan.barriersPerWorkgroup,
    barrierReductionVersusRowStaged: plan.barrierReductionVersusRowStaged,
    workgroupStorageBytes: plan.workgroupStorageBytes,
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
    features: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
    }),
  });
}

function requireAdapter(adapter: GPUAdapter): void {
  const outputBytes = outputElements() * Float32Array.BYTES_PER_ELEMENT;
  if (
    !adapter.features.has("subgroups") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 256 ||
    adapter.limits.maxComputeWorkgroupStorageSize < 16_384 ||
    adapter.limits.maxStorageBufferBindingSize < outputBytes ||
    adapter.limits.maxBufferSize < outputBytes + 2 * STORAGE_GUARD_BYTES
  ) {
    throw new Error(
      "OPT-0033 requires fixed32 subgroups, WG256, 16KiB shared storage, and full-M2250 buffers",
    );
  }
}

function outputElements(): number {
  return SHAPE.batch * SHAPE.queryHeads * SHAPE.queryTokens *
    SHAPE.headDimension;
}

function median3(samples: readonly number[]): number {
  if (samples.length !== 3 ||
    samples.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0033 requires three finite positive samples per arm");
  }
  return [...samples].sort((left, right) => left - right)[1]!;
}

function collectThermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0033 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0033 thermal field ${name} is not finite`);
  }
  return value;
}

function requireMapValue<Key, Value>(map: ReadonlyMap<Key, Value>, key: Key): Value {
  const value = map.get(key);
  if (value === undefined) throw new Error(`OPT-0033 missing ${String(key)}`);
  return value;
}

async function sha256F32(values: Float32Array<ArrayBuffer>): Promise<string> {
  return sha256Bytes(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256U32(values: Uint32Array<ArrayBuffer>): Promise<string> {
  return sha256Bytes(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function fail(error: unknown): void {
  document.body.dataset.status = "failed";
  requireElement<HTMLElement>("#progress").textContent = "failed";
  const receipt = Object.freeze({
    schemaVersion: 1,
    experiment: "OPT-0033",
    passed: false,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    receipt,
    null,
    2,
  );
  window.__ACE_OPT0033_RESULT__ = receipt;
}

async function settlePostDrainEvents(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
