/// <reference types="@webgpu/types" />

import {
  aceFixed32TiledFullAttentionWgsl,
  planAceFixed32TiledFullAttention,
} from "../../src/webgpu/kernels/attention.js";
import {
  aceOpt0039AttentionWgsl,
  planAceOpt0039Attention,
} from "../../src/webgpu/kernels/attention-dual-query.js";

declare global {
  interface Window {
    __ACE_OPT0039_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0039Arm = "query8" | "dualQuery16";

export interface Opt0039TimingInput {
  readonly arm: Opt0039Arm;
  readonly samplesMilliseconds: readonly number[];
}

export interface Opt0039ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly protocol: "wait-30s-then-one-level0-check";
  readonly startedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: 1;
  readonly observedLevel: 0;
  readonly maximumObservationGapMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

interface CompiledArm {
  readonly id: Opt0039Arm;
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
  readonly arms: ReadonlyMap<Opt0039Arm, CompiledArm>;
  readonly buffers: readonly GPUBuffer[];
  readonly createdBufferCount: number;
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
const ARMS = Object.freeze(["query8", "dualQuery16"] as const);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["query8", "dualQuery16"] as const),
  Object.freeze(["dualQuery16", "query8"] as const),
  Object.freeze(["query8", "dualQuery16"] as const),
  Object.freeze(["dualQuery16", "query8"] as const),
] as const);
const REQUIRED_SPEEDUP = 1.25;
const QUERY_SEED = 0x1357_9bdf;
const KEY_SEED = 0x2468_ace0;
const VALUE_SEED = 0x1020_3040;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_3955;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 30_000;

export function buildOpt0039TimingOrders(): typeof TIMING_ORDERS {
  return TIMING_ORDERS;
}

export function summarizeOpt0039Timing(
  inputs: readonly Opt0039TimingInput[],
): Readonly<Record<string, unknown>> {
  if (
    inputs.length !== ARMS.length ||
    inputs.some((input, index) => input.arm !== ARMS[index])
  ) {
    throw new Error("OPT-0039 timing arms changed");
  }
  const medians = Object.freeze(Object.fromEntries(inputs.map((input) => [
    input.arm,
    median4(input.samplesMilliseconds),
  ])) as Record<Opt0039Arm, number>);
  const speedupVersusQuery8 = medians.query8 / medians.dualQuery16;
  const passed = speedupVersusQuery8 >= REQUIRED_SPEEDUP;
  return Object.freeze({
    samplesPerArm: 4,
    mediansMilliseconds: medians,
    speedupVersusQuery8,
    requiredSpeedupVersusQuery8: REQUIRED_SPEEDUP,
    passed,
    decision: passed
      ? "positive-primitive-qualifier"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0039ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0039ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const startedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalCheckedAtEpochMilliseconds",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalObservations",
  );
  const observedLevel = requiredFiniteParameter(
    parameters,
    "thermalObservedLevel",
  );
  const durationMilliseconds = checkedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const maximumGapText = parameters.get(
    "thermalMaximumObservationGapMilliseconds",
  );
  const maximumObservationGapMilliseconds =
    maximumGapText === null || maximumGapText.trim() === ""
      ? durationMilliseconds
      : Number(maximumGapText);
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    source !== THERMAL_SOURCE ||
    observationCount !== 1 ||
    observedLevel !== 0 ||
    !Number.isFinite(maximumObservationGapMilliseconds) ||
    maximumObservationGapMilliseconds !== durationMilliseconds ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    checkedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0039 requires one truthful level-0 notifyutil check after a 30-second wait",
    );
  }
  return Object.freeze({
    source: THERMAL_SOURCE,
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    protocol: "wait-30s-then-one-level0-check",
    startedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount: 1 as const,
    observedLevel: 0 as const,
    maximumObservationGapMilliseconds,
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
        "READY — dual-query16 is deterministic and bit-exact over all 4,608,000 F32 outputs; timing has not run";
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
    label: "ace-opt-0039-attention-dual-query-device",
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxBufferSize: outputBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: outputBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 1_024,
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
    const queryBuffer = uploadStorage(device, query, "opt0039-query");
    buffers.push(queryBuffer);
    const keyBuffer = uploadStorage(device, key, "opt0039-key");
    buffers.push(keyBuffer);
    const valueBuffer = uploadStorage(device, value, "opt0039-value");
    buffers.push(valueBuffer);
    const validLengthsBuffer = uploadStorage(
      device,
      new Uint32Array([SHAPE.queryTokens, SHAPE.keyValueTokens]),
      "opt0039-valid-lengths",
    );
    buffers.push(validLengthsBuffer);
    const rangeBuffer = uploadUniform(
      device,
      new Uint32Array([0, 0, 0, 0]),
      "opt0039-range",
    );
    buffers.push(rangeBuffer);
    const guarded = createGuardedOutput(device);
    buffers.push(guarded.buffer, guarded.prefill, guarded.readback);
    const createdBufferCount = buffers.length;

    updateProgress("compiling query8 and fixed-WG256 dual-query16");
    const sources = new Map<Opt0039Arm, string>([
      ["query8", aceFixed32TiledFullAttentionWgsl(SHAPE)],
      ["dualQuery16", aceOpt0039AttentionWgsl(SHAPE)],
    ]);
    const plans = Object.freeze({
      query8: planAceFixed32TiledFullAttention(SHAPE),
      dualQuery16: planAceOpt0039Attention(SHAPE),
    });
    const arms = new Map<Opt0039Arm, CompiledArm>();
    for (const id of ARMS) {
      const source = requireMapValue(sources, id);
      const module = device.createShaderModule({
        label: `opt0039-${id}-module`,
        code: source,
      });
      await requireCleanCompilation(module, id);
      const pipeline = await device.createComputePipelineAsync({
        label: `opt0039-${id}-pipeline`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        label: `opt0039-${id}-bindings`,
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
        workgroupCount: id === "query8"
          ? plans.query8.workgroupCount
          : plans.dualQuery16.workgroupCount,
        wgslSha256: await sha256Text(source),
      }));
    }

    updateProgress("running two untimed complete raw-F32 passes per arm");
    const snapshots = new Map<string, Snapshot>();
    for (const [name, id] of [
      ["query8A", "query8"],
      ["dualQuery16A", "dualQuery16"],
      ["dualQuery16B", "dualQuery16"],
      ["query8B", "query8"],
    ] as const) {
      const snapshot = await executeAndRead(
        device,
        requireMapValue(arms, id),
        guarded,
        `opt0039-correctness-${name}`,
      );
      requireCompleteSnapshot(snapshot, name);
      snapshots.set(name, snapshot);
    }
    const query8A = requireMapValue(snapshots, "query8A");
    const query8B = requireMapValue(snapshots, "query8B");
    const dualA = requireMapValue(snapshots, "dualQuery16A");
    const dualB = requireMapValue(snapshots, "dualQuery16B");
    const candidateMismatchCount = countBitMismatches(
      query8A.words,
      dualA.words,
    );
    const query8RepeatMismatchCount = countBitMismatches(
      query8A.words,
      query8B.words,
    );
    const dualQueryRepeatMismatchCount = countBitMismatches(
      dualA.words,
      dualB.words,
    );
    if (
      candidateMismatchCount !== 0 ||
      query8RepeatMismatchCount !== 0 ||
      dualQueryRepeatMismatchCount !== 0
    ) {
      throw new Error(
        "OPT-0039 raw-F32 bit gate failed: " + JSON.stringify({
          candidateMismatchCount,
          query8RepeatMismatchCount,
          dualQueryRepeatMismatchCount,
        }),
      );
    }
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`OPT-0039 observed ${uncapturedErrors.length} GPU errors`);
    }
    const correctness = Object.freeze({
      outputElements: query8A.words.length,
      comparedElementsPerCheck: query8A.words.length,
      totalComparedElements: query8A.words.length * 3,
      query8NonFiniteCount: query8A.nonFiniteCount + query8B.nonFiniteCount,
      dualQuery16NonFiniteCount: dualA.nonFiniteCount + dualB.nonFiniteCount,
      candidateBitMismatchCount: candidateMismatchCount,
      query8RepeatBitMismatchCount: query8RepeatMismatchCount,
      dualQuery16RepeatBitMismatchCount: dualQueryRepeatMismatchCount,
      completeWritesAcrossFourRuns: true,
      canariesUntouchedAcrossFourRuns: true,
      deterministic: true,
      bitExact: true,
      outputSha256: Object.freeze({
        query8A: await sha256U32(query8A.words),
        query8B: await sha256U32(query8B.words),
        dualQuery16A: await sha256U32(dualA.words),
        dualQuery16B: await sha256U32(dualB.words),
      }),
    });

    // Correctness provides two untimed warmups per arm. Readback and prefill
    // scratch are released before the externally triggered timing screen.
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
      createdBufferCount,
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
  const thermal = parseOpt0039ThermalGate(
    collectThermalParameters(),
    prepared.preparedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const samples = new Map<Opt0039Arm, number[]>(
    ARMS.map((arm) => [arm, []]),
  );
  for (let round = 0; round < TIMING_ORDERS.length; round += 1) {
    const order = TIMING_ORDERS[round]!;
    for (let position = 0; position < order.length; position += 1) {
      const id = order[position]!;
      requireElement<HTMLElement>("#progress").textContent =
        `timing round ${round + 1}/4, position ${position + 1}/2: ${id}`;
      samples.get(id)!.push(await timeThroughQueueDrain(
        prepared.device,
        requireMapValue(prepared.arms, id),
        `opt0039-timed-r${round + 1}-${id}`,
      ));
    }
  }
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0039 observed an uncaptured timed GPU error");
  }
  const timing = summarizeOpt0039Timing(ARMS.map((arm) => Object.freeze({
    arm,
    samplesMilliseconds: Object.freeze(samples.get(arm)!.slice()),
  })));
  const receipt = Object.freeze({
    schemaVersion: 1,
    experiment: "OPT-0039",
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
      dualQuery16: compactCandidatePlan(),
    }),
    correctness: prepared.correctness,
    protocol: Object.freeze({
      thermal,
      compileOutsideTiming: true,
      fullCorrectnessOutsideTiming: true,
      correctnessAlsoServesAsTwoWarmupsPerArm: true,
      timedGpuWorkBeforeButton: false,
      rounds: 4,
      balancedOrders: TIMING_ORDERS,
      samplesPerArm: 4,
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
      createdBufferCount: prepared.createdBufferCount,
      preTimingBuffersDestroyed: 2,
      finalBuffersDestroyed: prepared.buffers.length,
      totalBuffersDestroyed: prepared.createdBufferCount,
      deviceDestroyed: true,
    }),
  });
  prepared.destroy();
  window.__ACE_OPT0039_RESULT__ = receipt;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    receipt,
    null,
    2,
  );
  document.body.dataset.status = receipt.passed ? "passed" : "failed";
  requireElement<HTMLElement>("#progress").textContent = receipt.passed
    ? `passed — dual-query16 reached ${Number(timing["speedupVersusQuery8"]).toFixed(3)}x query8`
    : `completed — dual-query16 reached only ${Number(timing["speedupVersusQuery8"]).toFixed(3)}x query8`;
}

async function requireCleanCompilation(
  module: GPUShaderModule,
  id: Opt0039Arm,
): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `OPT-0039 ${id} WGSL failed: ` + errors.map((message) =>
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
    const guardWords = STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
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
    label: "opt0039-output-prefill",
    size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(prefill.getMappedRange());
  words.fill(STORAGE_GUARD_U32);
  words.fill(
    OUTPUT_PREFILL_QNAN_U32,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT + logicalElements,
  );
  prefill.unmap();
  const buffer = device.createBuffer({
    label: "opt0039-guarded-output",
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "opt0039-readback",
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
    throw new Error("OPT-0039 output length changed");
  }
  let count = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) count += 1;
  }
  return count;
}

function requireCompleteSnapshot(snapshot: Snapshot, id: string): void {
  if (
    snapshot.nonFiniteCount !== 0 ||
    snapshot.qNaNPrefillCount !== 0 ||
    !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact
  ) {
    throw new Error(`OPT-0039 ${id} failed finite, complete-write, or canary gate`);
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
    barrierEvents: plan.workgroupCount * plan.keyValueTokens * 2,
    workgroupStorageBytes: plan.workgroupStorageBytes,
  });
}

function compactCandidatePlan(): Readonly<Record<string, unknown>> {
  const plan = planAceOpt0039Attention(SHAPE);
  return Object.freeze({
    queryTokensPerTile: plan.queryTokensPerTile,
    queryTokensPerSubgroup: plan.queryTokensPerSubgroup,
    queriesPerWorkgroup: plan.queriesPerWorkgroup,
    workgroupSize: plan.workgroupSize,
    workgroupCount: plan.workgroupCount,
    keyValueScalarLoads: plan.keyValueScalarLoads,
    keyValueLoadReductionVersusQuery8:
      plan.keyValueLoadReductionVersusQuery8,
    barriersPerWorkgroup: plan.barriersPerWorkgroup,
    barrierEvents: plan.barrierEvents,
    barrierEventReductionVersusQuery8:
      plan.barrierEventReductionVersusQuery8,
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
    adapter.limits.maxComputeWorkgroupStorageSize < 1_024 ||
    adapter.limits.maxStorageBufferBindingSize < outputBytes ||
    adapter.limits.maxBufferSize < outputBytes + 2 * STORAGE_GUARD_BYTES
  ) {
    throw new Error(
      "OPT-0039 requires fixed32 subgroups, WG256, 1KiB shared storage, and full-M2250 buffers",
    );
  }
}

function outputElements(): number {
  return SHAPE.batch * SHAPE.queryHeads * SHAPE.queryTokens *
    SHAPE.headDimension;
}

function median4(samples: readonly number[]): number {
  if (
    samples.length !== 4 ||
    samples.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("OPT-0039 requires four finite positive samples per arm");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
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
    throw new Error(`OPT-0039 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0039 thermal field ${name} is not finite`);
  }
  return value;
}

function requireMapValue<Key, Value>(
  map: ReadonlyMap<Key, Value>,
  key: Key,
): Value {
  const value = map.get(key);
  if (value === undefined) throw new Error(`OPT-0039 missing ${String(key)}`);
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

function requireElement<ElementType extends Element>(
  selector: string,
): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function fail(error: unknown): void {
  document.body.dataset.status = "failed";
  requireElement<HTMLElement>("#progress").textContent = "failed";
  const receipt = Object.freeze({
    schemaVersion: 1,
    experiment: "OPT-0039",
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
  window.__ACE_OPT0039_RESULT__ = receipt;
}

async function settlePostDrainEvents(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
