/// <reference types="@webgpu/types" />

import {
  aceFixed32TiledFullAttentionWgsl,
  planAceFixed32TiledFullAttention,
} from "../../src/webgpu/kernels/attention.js";
import {
  aceOpt0039AttentionWgsl,
  planAceOpt0039Attention,
} from "../../src/webgpu/kernels/attention-dual-query.js";
import {
  aceOpt0061AttentionWgsl,
  planAceOpt0061Attention,
} from "../../src/webgpu/kernels/attention-multi-query.js";

declare global {
  interface Window {
    __ACE_OPT0061_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0061Arm =
  | "query8"
  | "dualQuery16"
  | "tripleQuery24"
  | "quadQuery32";

export interface Opt0061TimingInput {
  readonly arm: Opt0061Arm;
  readonly samplesMilliseconds: readonly number[];
}

export interface Opt0061ThermalGate {
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
  readonly id: Opt0061Arm;
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
  readonly arms: ReadonlyMap<Opt0061Arm, CompiledArm>;
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
const ARMS = Object.freeze([
  "query8",
  "dualQuery16",
  "tripleQuery24",
  "quadQuery32",
] as const);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["query8", "dualQuery16", "tripleQuery24", "quadQuery32"] as const),
  Object.freeze(["dualQuery16", "tripleQuery24", "quadQuery32", "query8"] as const),
  Object.freeze(["tripleQuery24", "quadQuery32", "query8", "dualQuery16"] as const),
  Object.freeze(["quadQuery32", "query8", "dualQuery16", "tripleQuery24"] as const),
  Object.freeze(["quadQuery32", "tripleQuery24", "dualQuery16", "query8"] as const),
  Object.freeze(["query8", "quadQuery32", "tripleQuery24", "dualQuery16"] as const),
  Object.freeze(["dualQuery16", "query8", "quadQuery32", "tripleQuery24"] as const),
  Object.freeze(["tripleQuery24", "dualQuery16", "query8", "quadQuery32"] as const),
] as const);
const REQUIRED_SPEEDUP_VERSUS_DUAL = 1.12;
const REQUIRED_SPEEDUP_VERSUS_QUERY8 = 1.50;
const QUERY_SEED = 0x1357_9bdf;
const KEY_SEED = 0x2468_ace0;
const VALUE_SEED = 0x1020_3040;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_6155;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 30_000;

export function buildOpt0061TimingOrders(): typeof TIMING_ORDERS {
  return TIMING_ORDERS;
}

export function summarizeOpt0061Timing(
  inputs: readonly Opt0061TimingInput[],
): Readonly<Record<string, unknown>> {
  if (
    inputs.length !== ARMS.length ||
    inputs.some((input, index) => input.arm !== ARMS[index])
  ) {
    throw new Error("OPT-0061 timing arms changed");
  }
  const medians = Object.freeze(Object.fromEntries(inputs.map((input) => [
    input.arm,
    median8(input.samplesMilliseconds),
  ])) as Record<Opt0061Arm, number>);
  const candidates = Object.freeze([
    Object.freeze({
      arm: "tripleQuery24" as const,
      speedupVersusQuery8: medians.query8 / medians.tripleQuery24,
      speedupVersusDual: medians.dualQuery16 / medians.tripleQuery24,
    }),
    Object.freeze({
      arm: "quadQuery32" as const,
      speedupVersusQuery8: medians.query8 / medians.quadQuery32,
      speedupVersusDual: medians.dualQuery16 / medians.quadQuery32,
    }),
  ].map((candidate) => Object.freeze({
    ...candidate,
    eligible:
      candidate.speedupVersusQuery8 >= REQUIRED_SPEEDUP_VERSUS_QUERY8 &&
      candidate.speedupVersusDual >= REQUIRED_SPEEDUP_VERSUS_DUAL,
  })));
  const winner = candidates
    .filter(({ eligible }) => eligible)
    .sort((left, right) =>
      medians[left.arm] - medians[right.arm]
    )[0] ?? null;
  return Object.freeze({
    samplesPerArm: 8,
    samplesMilliseconds: Object.freeze(Object.fromEntries(inputs.map(
      (input) => [input.arm, Object.freeze(input.samplesMilliseconds.slice())],
    )) as Record<Opt0061Arm, readonly number[]>),
    mediansMilliseconds: medians,
    dualQuerySpeedupVersusQuery8: medians.query8 / medians.dualQuery16,
    candidates,
    requiredSpeedupVersusDual: REQUIRED_SPEEDUP_VERSUS_DUAL,
    requiredSpeedupVersusQuery8: REQUIRED_SPEEDUP_VERSUS_QUERY8,
    winner: winner?.arm ?? null,
    passed: winner !== null,
    decision: winner === null
      ? "negative-stop-wider-stream-counts"
      : "positive-primitive-qualifier",
  });
}

export function parseOpt0061ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0061ThermalGate {
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
      "OPT-0061 requires one truthful level-0 notifyutil check after a 30-second wait",
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

if (
  typeof document !== "undefined" &&
  document.querySelector("#opt-0061") !== null
) installBrowserHarness();

function installBrowserHarness(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const runButton = requireElement<HTMLButtonElement>("#run");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let active: PreparedHarness | undefined;
  let started = false;
  void prepareHarness((message) => progress.textContent = message).then(
    (prepared) => {
      active = prepared;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — query8/dual/triple/quad outputs are deterministic and raw-bit exact; timing has not run";
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
  }, { once: true });
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
    label: "ace-opt-0061-attention-multi-query-device",
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
    const queryBuffer = uploadStorage(device, query, "opt0061-query");
    const keyBuffer = uploadStorage(device, key, "opt0061-key");
    const valueBuffer = uploadStorage(device, value, "opt0061-value");
    const validLengthsBuffer = uploadStorage(
      device,
      new Uint32Array([SHAPE.queryTokens, SHAPE.keyValueTokens]),
      "opt0061-valid-lengths",
    );
    const rangeBuffer = uploadUniform(
      device,
      new Uint32Array([0, 0, 0, 0]),
      "opt0061-range",
    );
    buffers.push(
      queryBuffer,
      keyBuffer,
      valueBuffer,
      validLengthsBuffer,
      rangeBuffer,
    );
    const guarded = createGuardedOutput(device);
    buffers.push(guarded.buffer, guarded.prefill, guarded.readback);
    const createdBufferCount = buffers.length;

    updateProgress("compiling query8, dual, triple, and quad WG256 owners");
    const sources = new Map<Opt0061Arm, string>([
      ["query8", aceFixed32TiledFullAttentionWgsl(SHAPE)],
      ["dualQuery16", aceOpt0039AttentionWgsl(SHAPE)],
      ["tripleQuery24", aceOpt0061AttentionWgsl(SHAPE, 3)],
      ["quadQuery32", aceOpt0061AttentionWgsl(SHAPE, 4)],
    ]);
    const workgroups = new Map<Opt0061Arm, number>([
      ["query8", planAceFixed32TiledFullAttention(SHAPE).workgroupCount],
      ["dualQuery16", planAceOpt0039Attention(SHAPE).workgroupCount],
      ["tripleQuery24", planAceOpt0061Attention(SHAPE, 3).workgroupCount],
      ["quadQuery32", planAceOpt0061Attention(SHAPE, 4).workgroupCount],
    ]);
    const arms = new Map<Opt0061Arm, CompiledArm>();
    for (const id of ARMS) {
      const source = requireMapValue(sources, id);
      const module = device.createShaderModule({
        label: `opt0061-${id}-module`,
        code: source,
      });
      await requireCleanCompilation(module, id);
      const pipeline = await device.createComputePipelineAsync({
        label: `opt0061-${id}-pipeline`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        label: `opt0061-${id}-bindings`,
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
        workgroupCount: requireMapValue(workgroups, id),
        wgslSha256: await sha256Text(source),
      }));
    }

    updateProgress("running two full untimed guarded passes per arm");
    const snapshots = new Map<string, Snapshot>();
    const correctnessOrder = Object.freeze([
      "query8",
      "dualQuery16",
      "tripleQuery24",
      "quadQuery32",
      "quadQuery32",
      "tripleQuery24",
      "dualQuery16",
      "query8",
    ] as const);
    for (let index = 0; index < correctnessOrder.length; index += 1) {
      const id = correctnessOrder[index]!;
      const suffix = index < ARMS.length ? "A" : "B";
      const keyName = `${id}${suffix}`;
      const snapshot = await executeAndRead(
        device,
        requireMapValue(arms, id),
        guarded,
        `opt0061-correctness-${keyName}`,
      );
      requireCompleteSnapshot(snapshot, keyName);
      snapshots.set(keyName, snapshot);
    }
    const query8A = requireMapValue(snapshots, "query8A");
    const mismatchCounts = Object.freeze(Object.fromEntries(ARMS.flatMap(
      (id) => {
        const a = requireMapValue(snapshots, `${id}A`);
        const b = requireMapValue(snapshots, `${id}B`);
        return [
          [`${id}VersusQuery8`, countBitMismatches(query8A.words, a.words)],
          [`${id}Repeat`, countBitMismatches(a.words, b.words)],
        ];
      },
    )) as Record<string, number>);
    if (Object.values(mismatchCounts).some((count) => count !== 0)) {
      throw new Error(
        `OPT-0061 raw-F32 bit gate failed: ${JSON.stringify(mismatchCounts)}`,
      );
    }
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`OPT-0061 observed ${uncapturedErrors.length} GPU errors`);
    }
    const resolvedOutputSha256: Record<string, string> = {};
    for (const [id, snapshot] of snapshots) {
      resolvedOutputSha256[id] = await sha256U32(snapshot.words);
    }
    const correctness = Object.freeze({
      outputElements: query8A.words.length,
      completeRunCount: correctnessOrder.length,
      totalComparedElements: query8A.words.length * 7,
      nonFiniteCount: [...snapshots.values()].reduce(
        (sum, snapshot) => sum + snapshot.nonFiniteCount,
        0,
      ),
      mismatchCounts,
      completeWritesAcrossEightRuns: true,
      canariesUntouchedAcrossEightRuns: true,
      deterministic: true,
      bitExact: true,
      outputSha256: Object.freeze(resolvedOutputSha256),
    });

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
  const thermal = parseOpt0061ThermalGate(
    collectThermalParameters(),
    prepared.preparedAtEpochMilliseconds,
    Date.now(),
  );
  const samples = new Map<Opt0061Arm, number[]>(
    ARMS.map((arm) => [arm, []]),
  );
  for (let round = 0; round < TIMING_ORDERS.length; round += 1) {
    const order = TIMING_ORDERS[round]!;
    for (let position = 0; position < order.length; position += 1) {
      const id = order[position]!;
      requireElement<HTMLElement>("#progress").textContent =
        `timing round ${round + 1}/8, position ${position + 1}/4: ${id}`;
      samples.get(id)!.push(await timeThroughQueueDrain(
        prepared.device,
        requireMapValue(prepared.arms, id),
        `opt0061-timed-r${round + 1}-${id}`,
      ));
    }
  }
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0061 observed an uncaptured timed GPU error");
  }
  const timing = summarizeOpt0061Timing(ARMS.map((arm) => Object.freeze({
    arm,
    samplesMilliseconds: Object.freeze(samples.get(arm)!.slice()),
  })));
  const receipt = Object.freeze({
    schema: "ace-opt-0061-attention-multi-query-v1",
    experimentId: "OPT-0061",
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
    plans: compactPlans(),
    correctness: prepared.correctness,
    protocol: Object.freeze({
      thermal,
      compileOutsideTiming: true,
      fullCorrectnessOutsideTiming: true,
      correctnessAlsoServesAsTwoWarmupsPerArm: true,
      timedGpuWorkBeforeButton: false,
      rounds: 8,
      balancedForwardReverseLatinOrders: TIMING_ORDERS,
      samplesPerArm: 8,
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
  window.__ACE_OPT0061_RESULT__ = receipt;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    receipt,
    null,
    2,
  );
  document.body.dataset.status = receipt.passed ? "passed" : "failed";
  requireElement<HTMLElement>("#progress").textContent = receipt.passed
    ? `passed — ${String(timing["winner"])} qualified over query8 and dual-query`
    : "completed — neither wider stream count cleared both gates";
}

async function requireCleanCompilation(
  module: GPUShaderModule,
  id: Opt0061Arm,
): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `OPT-0061 ${id} WGSL failed: ` + errors.map((message) =>
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
    label: "opt0061-output-prefill",
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
    label: "opt0061-guarded-output",
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "opt0061-readback",
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
    throw new Error("OPT-0061 output length changed");
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
    throw new Error(`OPT-0061 ${id} failed finite, write, or canary gate`);
  }
}

function compactPlans(): Readonly<Record<Opt0061Arm, unknown>> {
  const query8 = planAceFixed32TiledFullAttention(SHAPE);
  const dual = planAceOpt0039Attention(SHAPE);
  const compactMulti = (streams: 3 | 4): Readonly<Record<string, unknown>> => {
    const plan = planAceOpt0061Attention(SHAPE, streams);
    return Object.freeze({
      streamCount: streams,
      queryTokensPerTile: plan.queryTokensPerTile,
      queriesPerWorkgroup: plan.queriesPerWorkgroup,
      workgroupSize: plan.workgroupSize,
      workgroupCount: plan.workgroupCount,
      keyValueScalarLoads: plan.keyValueScalarLoads,
      keyValueLoadReductionVersusQuery8:
        plan.keyValueLoadReductionVersusQuery8,
      keyValueLoadReductionVersusDual:
        plan.keyValueLoadReductionVersusDualQuery,
      barrierEvents: plan.barrierEvents,
      retainedPrivateFp32ValuesPerLane:
        plan.retainedPrivateFp32ValuesPerLane,
      workgroupStorageBytes: plan.workgroupStorageBytes,
    });
  };
  return Object.freeze({
    query8: Object.freeze({
      queryTokensPerTile: query8.queryTokensPerTile,
      queriesPerWorkgroup: query8.queriesPerWorkgroup,
      workgroupSize: query8.workgroupSize,
      workgroupCount: query8.workgroupCount,
      keyValueScalarLoads: query8.tiledKeyValueScalarLoads,
      barrierEvents: query8.workgroupCount * query8.keyValueTokens * 2,
      workgroupStorageBytes: query8.workgroupStorageBytes,
    }),
    dualQuery16: Object.freeze({
      queryTokensPerTile: dual.queryTokensPerTile,
      queriesPerWorkgroup: dual.queriesPerWorkgroup,
      workgroupSize: dual.workgroupSize,
      workgroupCount: dual.workgroupCount,
      keyValueScalarLoads: dual.keyValueScalarLoads,
      barrierEvents: dual.barrierEvents,
      workgroupStorageBytes: dual.workgroupStorageBytes,
    }),
    tripleQuery24: compactMulti(3),
    quadQuery32: compactMulti(4),
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
      "OPT-0061 requires fixed32 subgroups, WG256, 1KiB shared storage, and M2250 buffers",
    );
  }
}

function outputElements(): number {
  return SHAPE.batch * SHAPE.queryHeads * SHAPE.queryTokens *
    SHAPE.headDimension;
}

function median8(samples: readonly number[]): number {
  if (
    samples.length !== 8 ||
    samples.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("OPT-0061 requires eight finite positive samples per arm");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[3]! + sorted[4]!) / 2;
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
    throw new Error(`OPT-0061 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0061 thermal field ${name} is not finite`);
  }
  return value;
}

function requireMapValue<Key, Value>(
  map: ReadonlyMap<Key, Value>,
  key: Key,
): Value {
  const value = map.get(key);
  if (value === undefined) throw new Error(`OPT-0061 missing ${String(key)}`);
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
    schema: "ace-opt-0061-attention-multi-query-v1",
    experimentId: "OPT-0061",
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
  window.__ACE_OPT0061_RESULT__ = receipt;
}

async function settlePostDrainEvents(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
