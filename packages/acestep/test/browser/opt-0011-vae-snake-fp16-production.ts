/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import productionCoreSource from
  "../../src/webgpu/kernels/vae-snake-fp16.ts?raw";
import {
  ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
  AceFp16VaeSnakeKernel,
  aceFp16VaeSnakeWgsl,
  planAceFp16VaeSnake,
  planAceFp16VaeSnakeRange,
  type AceFp16VaeSnakeDispatch,
  type AceFp16VaeSnakePlan,
} from "../../src/webgpu/kernels/vae-snake-fp16.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderQuantumPlan,
  type AceVaeDecoderSnakeOperation,
} from "../../src/webgpu/vae-decoder.js";

export const OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT =
  "ae2106c9d5834a3cd5cb836cad484665752230e3" as const;
export const OPT_0011_PRODUCTION_SNAKE_CORE_SOURCE_SHA256 =
  "0e0cc8d1974e6f36942a98777e43c6b48b27c00a8cb0d912ff1f510be426601f" as const;
export const OPT_0011_PRODUCTION_SNAKE_GRAPH_TOPOLOGY_SHA256 =
  "ec79060be88fba5d0a2579826f1ca50730dfba16410da09ffc048963f2623bf3" as const;

/** SHA-256 over every exact WGSL source admitted by this browser gate. */
export const OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256 =
  Object.freeze({
    "1:256:2048":
      "c3bd59f4bf75bdf90a49240e9c78e50e33d7f1b4f44e054529df2acdba1e3cfd",
    "1:2560:1024":
      "d1ee7bc58ea77f8c69ae1ea0a9608392309d07a69e1e0ae4ed3b29c0d0e37615",
    "1:15360:512":
      "4e9cb2b2f65bb98cf72c8d73a0a4ea422f2342f67c86e2cf5e025b6a60a4c0cc",
    "1:61440:256":
      "6ed918bfdcd8fc101044ac5ea0ed2d9c939c6d86eb16724f78e08eefe74d006c",
    "1:245760:128":
      "872a9978bd76ebc9e96b23e6b72520f84391b212513fa06dceeb648a46b629f6",
    "1:491520:128":
      "872a9978bd76ebc9e96b23e6b72520f84391b212513fa06dceeb648a46b629f6",
    "1:2:17":
      "f7845832619097877ac31b7e8c801cff73daa433a2a16cb81871f516be8926ec",
    "1:1:257":
      "80ffc022aa8362f5edc4a6fdbedaebdf1e1127f6801deabd78357a153fa45b41",
  } as const);

export interface Opt0011ProductionSnakeRunIdentity {
  readonly harnessCommit: string;
  readonly coreCommit: typeof OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT;
}

export interface Opt0011ProductionSnakeRawBitComparison {
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
}

export interface Opt0011ProductionSnakeGraphRange {
  readonly quantumIndex: number;
  readonly operationQuantumIndex: number;
  readonly controlRecordIndex: number;
  readonly base: number;
  readonly count: number;
}

export interface Opt0011ProductionSnakeGraphCase {
  readonly id: string;
  readonly operationIndex: number;
  readonly snakeOrdinal: number;
  readonly shapeFamilyIndex: number;
  readonly shape: AceVaePointwiseShape;
  readonly ranges: readonly Opt0011ProductionSnakeGraphRange[];
}

export interface Opt0011ProductionSnakeFixture {
  readonly id: string;
  readonly fixtureKind: "graph" | "arithmetic";
  readonly shape: AceVaePointwiseShape;
  readonly sourcePatternSalt: number;
  readonly graphOperationIndex: number | null;
  readonly graphSnakeOrdinal: number | null;
  readonly ranges: readonly Readonly<{
    readonly base: number;
    readonly count: number;
    readonly quantumIndex?: number;
    readonly operationQuantumIndex?: number;
    readonly controlRecordIndex?: number;
  }>[];
  readonly coverage: readonly string[];
}

interface PreparedFixture {
  readonly fixture: Opt0011ProductionSnakeFixture;
  readonly plan: AceFp16VaeSnakePlan;
  readonly dispatches: readonly AceFp16VaeSnakeDispatch[];
  readonly output: OutputTarget;
  readonly prefill: PrefillTemplate;
  readonly sourceIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface OutputTarget {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

interface PrefillTemplate {
  readonly qNaNBuffer: GPUBuffer;
  readonly canaryBuffer: GPUBuffer;
  readonly bytes: number;
}

interface SelectedReadback {
  readonly sha256: string;
  readonly scan: Readonly<Record<string, unknown>>;
  readonly cpu: Readonly<Record<string, unknown>>;
}

interface ExecutionCounts {
  readonly encodedCommandBuffers: 1;
  readonly submissions: 1;
  readonly drains: 1;
  readonly dispatches: 1;
  readonly qNaNPrefillCopies: 1;
  readonly adjacentCanaryRestoreCopies: number;
  readonly queueEmptyIdleTurns: 1;
}

export interface HeartbeatController {
  stop(): Readonly<Record<string, unknown>>;
}

export interface Opt0011ProductionSnakeHeartbeatFailureStop {
  readonly liveness: Readonly<Record<string, unknown>> | null;
  readonly heartbeatStopError: Readonly<Record<string, unknown>> | null;
}

const FLOAT16_BYTES = 2;
const CONTROL_BYTES = 16;
const OUTPUT_GUARD_BYTES = 256;
const OUTPUT_CANARY_BYTES = 256;
const OUTPUT_GUARD_F16 = 0x7e33;
const OUTPUT_CANARY_F16 = 0x7e11;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const SOURCE_PADDING_F16 = 0x7e77;
const QUEUE_EMPTY_IDLE_MILLISECONDS = 1;
const RAW_RESULT_GLOBAL =
  "__ACE_OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_JSON__";
export const OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS =
  100_000;

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

const GRAPH_INPUT_F16_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x0001, 0x8001,
  0x3c00, 0xbc00, 0x3800, 0xb800,
  0x3555, 0xb555, 0x7bff, 0xfbff,
  0x0400, 0x8400, 0x3a00, 0xba00,
]);
const GRAPH_ALPHA_F16_PATTERN = Object.freeze([
  0xcc00, 0xca00, 0xc900, 0xc800,
]);
const GRAPH_BETA_F16_PATTERN = Object.freeze([
  0x4800, 0x4900, 0x4980, 0x4a00,
]);

const ARITHMETIC_F16_TUPLES = Object.freeze([
  Object.freeze([0x0000, 0x8000, 0x0000]),
  Object.freeze([0x8000, 0x0000, 0x0000]),
  Object.freeze([0x0001, 0xcc00, 0x0000]),
  Object.freeze([0x8001, 0xcc00, 0x0000]),
  Object.freeze([0x3c00, 0x0000, 0x7bff]),
  Object.freeze([0xbc00, 0x0000, 0x7bff]),
  Object.freeze([0x3800, 0x4500, 0xc500]),
  Object.freeze([0x3800, 0x4500, 0xca00]),
  Object.freeze([0x7bff, 0xbc00, 0x4000]),
  Object.freeze([0xfbff, 0xbc00, 0x4000]),
  Object.freeze([0x0400, 0xcc00, 0x0000]),
  Object.freeze([0x03ff, 0xcc00, 0x0000]),
  Object.freeze([0x3c01, 0x0000, 0x7bff]),
  Object.freeze([0xbc01, 0x0000, 0x7bff]),
  Object.freeze([0x3555, 0xcc00, 0x0000]),
  Object.freeze([0xb555, 0xcc00, 0x0000]),
  // sin(exp(alpha) * input) rounds to exactly 1.0f. With beta=0 the
  // residual is exactly 1.0f, placing the f32 result at an f16 midpoint.
  Object.freeze([0x6936, 0xc770, 0x0000]),
  Object.freeze([0x6887, 0xc74c, 0x0000]),
] as const);

const B256_GRAPH = planAceVaeDecoder(256);
const B256_QUANTA = planAceVaeDecoderQuanta(B256_GRAPH);

export const OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES = buildGraphCases();
export const OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES =
  buildSelectedGraphFixtures();

export const OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES = Object.freeze([
  Object.freeze({
    id: "snake-arithmetic-channel-reuse-2x17",
    fixtureKind: "arithmetic",
    shape: Object.freeze({ batch: 1, frames: 2, channels: 17 }),
    sourcePatternSalt: 0,
    graphOperationIndex: null,
    graphSnakeOrdinal: null,
    ranges: Object.freeze([
      Object.freeze({ base: 0, count: 34 }),
    ]),
    coverage: Object.freeze([
      "complete-two-frame-channel-modulo-domain",
      "all-curated-alpha-beta-log-scale-tuples",
      "fp32-exp-sin-residual-island",
      "actual-snake-f16-rne-tie-to-even-down",
    ]),
  }),
  Object.freeze({
    id: "snake-arithmetic-odd-tail-257",
    fixtureKind: "arithmetic",
    shape: Object.freeze({ batch: 1, frames: 1, channels: 257 }),
    sourcePatternSalt: 0,
    graphOperationIndex: null,
    graphSnakeOrdinal: null,
    ranges: Object.freeze([
      Object.freeze({ base: 0, count: 256 }),
      Object.freeze({ base: 256, count: 1 }),
    ]),
    coverage: Object.freeze([
      "complete-odd-channel-domain",
      "signed-zero-subnormal-rne-extreme-finite-overflow",
      "odd-tail-and-fp16-binding-padding",
      "actual-snake-f16-rne-tie-to-even-up-at-tail",
    ]),
  }),
] satisfies readonly Opt0011ProductionSnakeFixture[]);

export const OPT_0011_PRODUCTION_SNAKE_FIXTURES = Object.freeze([
  ...OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES,
  ...OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES,
]);

export function parseOpt0011ProductionSnakeRunIdentity(
  parameters: URLSearchParams,
): Opt0011ProductionSnakeRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error(
      "OPT-0011 Snake harnessCommit must be a 40-character lowercase hex commit",
    );
  }
  const coreCommit = requiredIdentity(parameters, "coreCommit");
  if (coreCommit !== OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT) {
    throw new Error("OPT-0011 production Snake coreCommit changed");
  }
  return Object.freeze({ harnessCommit, coreCommit });
}

export function compareOpt0011ProductionSnakeRawBits(
  actual: Uint16Array,
  expected: Uint16Array,
): Opt0011ProductionSnakeRawBitComparison {
  if (actual.constructor !== expected.constructor) {
    throw new Error("OPT-0011 Snake output bit domains differ");
  }
  if (actual.length !== expected.length) {
    throw new Error("OPT-0011 Snake output lengths differ");
  }
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] === expected[index]) continue;
    mismatchCount += 1;
    firstMismatchIndex ??= index;
  }
  return Object.freeze({ mismatchCount, firstMismatchIndex });
}

export function opt0011ProductionSnakeGraphInputBits(
  globalIndex: number,
  salt = 0,
): number {
  requireIndex(globalIndex, 0xffff_ffff, "Snake graph input");
  requireIndex(salt, 0xffff, "Snake graph salt");
  return GRAPH_INPUT_F16_PATTERN[
    (globalIndex + salt * 5) % GRAPH_INPUT_F16_PATTERN.length
  ]!;
}

export function opt0011ProductionSnakeGraphAlphaBits(
  channel: number,
  salt = 0,
): number {
  requireIndex(channel, 0xffff_ffff, "Snake graph alpha channel");
  requireIndex(salt, 0xffff, "Snake graph salt");
  return GRAPH_ALPHA_F16_PATTERN[
    (channel + salt) % GRAPH_ALPHA_F16_PATTERN.length
  ]!;
}

export function opt0011ProductionSnakeGraphBetaBits(
  channel: number,
  salt = 0,
): number {
  requireIndex(channel, 0xffff_ffff, "Snake graph beta channel");
  requireIndex(salt, 0xffff, "Snake graph salt");
  return GRAPH_BETA_F16_PATTERN[
    (channel + salt * 3) % GRAPH_BETA_F16_PATTERN.length
  ]!;
}

export function opt0011ProductionSnakeArithmeticInputBits(
  globalIndex: number,
  channels: number,
): number {
  requireIndex(globalIndex, 0xffff_ffff, "Snake arithmetic input");
  requireIndex(channels, 0xffff_ffff, "Snake arithmetic channels");
  if (channels < 1) throw new RangeError("Snake arithmetic channels are empty");
  const channel = globalIndex % channels;
  return ARITHMETIC_F16_TUPLES[arithmeticTupleIndex(channel)]![0]!;
}

export function opt0011ProductionSnakeArithmeticAlphaBits(
  channel: number,
): number {
  requireIndex(channel, 0xffff_ffff, "Snake arithmetic alpha channel");
  return ARITHMETIC_F16_TUPLES[arithmeticTupleIndex(channel)]![1]!;
}

export function opt0011ProductionSnakeArithmeticBetaBits(
  channel: number,
): number {
  requireIndex(channel, 0xffff_ffff, "Snake arithmetic beta channel");
  return ARITHMETIC_F16_TUPLES[arithmeticTupleIndex(channel)]![2]!;
}

function arithmeticTupleIndex(channel: number): number {
  // The odd-tail fixture gets its own tie-to-even-up tuple instead of
  // silently reusing channel zero's parameters.
  if (channel === 256) return 17;
  return channel % 17;
}

export function opt0011ProductionSnakeCpuBits(
  inputBits: number,
  alphaBits: number,
  betaBits: number,
): number {
  return numberToFloat16Bits(
    snakeCpuArithmetic(inputBits, alphaBits, betaBits).uncontractedResult,
  );
}

export function opt0011ProductionSnakeFinalFusedCpuBits(
  inputBits: number,
  alphaBits: number,
  betaBits: number,
): number {
  return numberToFloat16Bits(
    snakeCpuArithmetic(inputBits, alphaBits, betaBits).finalFusedResult,
  );
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  created = 0;
  destroyed = 0;
  maximumLive = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.maximumLive = Math.max(this.maximumLive, this.live.size);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      created: this.created,
      destroyed: this.destroyed,
      live: this.live.size,
      maximumLive: this.maximumLive,
    });
  }
}

if (typeof document !== "undefined") installBrowserUi();

function installBrowserUi(): void {
  installRawResultChunkRetrieval();
  const start = requireElement<HTMLButtonElement>("#start");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("authenticating frozen Snake source and generated WGSL");
    const heartbeat = startHeartbeat();
    void runBrowser(heartbeat).then(
      (result) => finish("passed", result),
      (error: unknown) => {
        const heartbeatFailure =
          stopOpt0011ProductionSnakeHeartbeatAfterFailure(heartbeat);
        finish("failed", Object.freeze({
          schema: "ace-opt-0011-production-fp16-snake-correctness-v1",
          status: "failed",
          experimentId: "OPT-0011",
          error: errorReceipt(error),
          primaryErrorPreservedAcrossHeartbeatStop: true,
          heartbeatLiveness: heartbeatFailure.liveness,
          heartbeatStopError: heartbeatFailure.heartbeatStopError,
        }));
      },
    );
  }, { once: true });
}

async function runBrowser(
  heartbeat: HeartbeatController,
): Promise<Readonly<Record<string, unknown>>> {
  const identity = parseOpt0011ProductionSnakeRunIdentity(
    new URLSearchParams(window.location.search),
  );
  const sourceAuthority = await authenticateSources(identity);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const rawDevice = await adapter.requestDevice({
    label: "ace-opt-0011-production-fp16-snake-correctness-device",
    requiredFeatures: ["shader-f16"],
    requiredLimits: requiredDeviceLimits(),
  });
  const device = rawDevice;
  const tracker = new BufferTracker();
  const uncapturedErrors: Readonly<Record<string, unknown>>[] = [];
  const runtimeErrors: Readonly<Record<string, unknown>>[] = [];
  let unexpectedDeviceLoss: Readonly<Record<string, unknown>> | null = null;
  let destroyingDevice = false;
  const onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(errorReceipt(event.error));
  };
  const onWindowError = (event: ErrorEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "error",
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    }));
  };
  const onUnhandledRejection = (event: PromiseRejectionEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "unhandledrejection",
      reason: errorReceipt(event.reason),
    }));
  };
  rawDevice.addEventListener("uncapturederror", onUncaptured);
  window.addEventListener("error", onWindowError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  void rawDevice.lost.then((info) => {
    if (!destroyingDevice) {
      unexpectedDeviceLoss = Object.freeze({
        reason: info.reason,
        message: info.message,
      });
    }
  });

  const cases: unknown[] = [];
  let cancellation: unknown = null;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let heartbeatLiveness: Readonly<Record<string, unknown>> | undefined;
  let postCleanupValidationFailure: Error | undefined;
  try {
    for (const [index, fixture] of
      OPT_0011_PRODUCTION_SNAKE_FIXTURES.entries()) {
      updateProgress(
        `Snake fixture ${index + 1}/${OPT_0011_PRODUCTION_SNAKE_FIXTURES.length}: ${fixture.id}`,
      );
      cases.push(await runFixture(device, tracker, fixture));
      await yieldToBrowser();
    }

    updateProgress("post-drain real multi-range Snake cancellation proof");
    cancellation = await runCancellationProof(device, tracker);
    await rawDevice.queue.onSubmittedWorkDone();
    await yieldToBrowser();
  } finally {
    const lifecycleOrder = ["cleanup-started"];
    tracker.destroyAll();
    tracker.destroyAll();
    lifecycleOrder.push("harness-buffers-destroyed");
    const receipt = tracker.receipt();
    const accountingExact =
      receipt.live === 0 && receipt.created === receipt.destroyed;
    destroyingDevice = true;
    rawDevice.destroy();
    lifecycleOrder.push("device-destroy-called");
    const intentionalDeviceLoss = await rawDevice.lost;
    lifecycleOrder.push("device-loss-settled");
    await yieldToBrowser();
    await yieldToBrowser();
    lifecycleOrder.push("post-cleanup-event-turns-completed");
    const finalUncapturedErrors = Object.freeze([...uncapturedErrors]);
    const finalRuntimeErrors = Object.freeze([...runtimeErrors]);
    const finalUnexpectedDeviceLoss = unexpectedDeviceLoss;
    lifecycleOrder.push("final-event-snapshot-captured");
    const cleanupAndEventValidationClean =
      accountingExact && finalUncapturedErrors.length === 0 &&
      finalRuntimeErrors.length === 0 && finalUnexpectedDeviceLoss === null &&
      intentionalDeviceLoss.reason === "destroyed";
    lifecycleOrder.push("cleanup-and-event-validation-completed");
    rawDevice.removeEventListener("uncapturederror", onUncaptured);
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    lifecycleOrder.push("event-listeners-removed");
    const heartbeatStop =
      stopOpt0011ProductionSnakeHeartbeatAfterFailure(heartbeat);
    heartbeatLiveness = heartbeatStop.liveness ?? Object.freeze({
      observed: false,
      stopFailed: true,
    });
    lifecycleOrder.push("heartbeat-stopped");
    const postCleanupValidationClean =
      cleanupAndEventValidationClean &&
      heartbeatStop.heartbeatStopError === null &&
      heartbeatLiveness.observed === true;
    lifecycleOrder.push("post-cleanup-validation-completed");
    cleanup = Object.freeze({
      ...receipt,
      trackedScope: "all-harness-owned-buffers",
      destroyAllCalledTwice: true,
      idempotent: accountingExact,
      deviceDestroyed: true,
      intentionalDeviceLoss: Object.freeze({
        reason: intentionalDeviceLoss.reason,
        message: intentionalDeviceLoss.message,
        exactDestroyedReason: intentionalDeviceLoss.reason === "destroyed",
      }),
      uncapturedErrors: finalUncapturedErrors,
      runtimeErrors: finalRuntimeErrors,
      unexpectedDeviceLoss: finalUnexpectedDeviceLoss,
      heartbeatCoveredCleanup: true,
      heartbeatLiveness,
      heartbeatStopError: heartbeatStop.heartbeatStopError,
      lifecycleOrder: Object.freeze(lifecycleOrder),
    });
    if (!postCleanupValidationClean) {
      postCleanupValidationFailure = new Error(
        "OPT-0011 Snake post-cleanup resource, event, device-loss, or heartbeat validation failed",
      );
    }
  }
  if (postCleanupValidationFailure !== undefined) {
    throw postCleanupValidationFailure;
  }

  return Object.freeze({
    schema: "ace-opt-0011-production-fp16-snake-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification:
      "correctness-only-no-kernel-performance-wall-time-or-thermal-timing",
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter, rawDevice),
    }),
    protocol: Object.freeze({
      requiredFeature: "shader-f16",
      kernel: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
      authority:
        "independent-explicit-f32-island-to-rne-f16-raw-u16-cpu-oracle",
      allThirtySixB256SnakeOperationsBound: true,
      exactB256SnakeGraphRangeCount: 813,
      representativeExactGraphRanges: true,
      completeManageableArithmeticDomains: true,
      fullSelectedRangeRawU16Comparison: true,
      deterministicRerunPerRange: true,
      oneOutstandingCommandBuffer: true,
      drainAndRealQueueEmptyTurnAfterEveryExecution: true,
      qNaNPrefillCanariesGuardsAndBindingPadding: true,
      compilationUploadAndWallTimeReported: false,
      performanceClaim: null,
      thermalClaim: null,
      qualityClaim: null,
      listeningClaim: null,
      productionSelectorClaim: null,
      productionIntegrationClaim: null,
    }),
    sourceAuthority,
    graphCoverage: graphCoverageReceipt(),
    cases: Object.freeze(cases),
    cancellation,
    heartbeatLiveness,
    cleanup,
  });
}

async function authenticateSources(
  identity: Opt0011ProductionSnakeRunIdentity,
): Promise<Readonly<Record<string, unknown>>> {
  const encoder = new TextEncoder();
  const coreSourceSha256 = await sha256Hex(encoder.encode(productionCoreSource));
  if (coreSourceSha256 !== OPT_0011_PRODUCTION_SNAKE_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0011 rejected unauthenticated production Snake source");
  }
  const canonicalGraphTopology = B256_QUANTA.quanta
    .filter((quantum) => quantum.operationKind === "snake")
    .map((quantum) => ({
      index: quantum.index,
      id: quantum.id,
      operationIndex: quantum.operationIndex,
      operationLabel: quantum.operationLabel,
      operationKind: quantum.operationKind,
      logicalOutputBase: quantum.logicalOutputBase,
      logicalOutputCount: quantum.logicalOutputCount,
      estimatedMaximumMultiplyAccumulates:
        quantum.estimatedMaximumMultiplyAccumulates,
      primitives: quantum.primitives,
    }));
  if (canonicalGraphTopology.length !== 813) {
    throw new Error("OPT-0011 Snake canonical graph topology count changed");
  }
  const graphTopologySha256 = await sha256Hex(
    encoder.encode(JSON.stringify(canonicalGraphTopology)),
  );
  if (
    graphTopologySha256 !==
      OPT_0011_PRODUCTION_SNAKE_GRAPH_TOPOLOGY_SHA256
  ) {
    throw new Error("OPT-0011 Snake canonical graph topology SHA-256 changed");
  }
  const fixtureShapes = new Map<string, AceVaePointwiseShape>();
  for (const fixture of OPT_0011_PRODUCTION_SNAKE_FIXTURES) {
    fixtureShapes.set(shapeKey(fixture.shape), fixture.shape);
  }
  const expectedKeys = Object.keys(
    OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256,
  ).sort();
  const actualKeys = [...fixtureShapes.keys()].sort();
  if (JSON.stringify(expectedKeys) !== JSON.stringify(actualKeys)) {
    throw new Error("OPT-0011 Snake generated shader shape set changed");
  }
  const generatedShaders: unknown[] = [];
  for (const key of expectedKeys) {
    const shape = fixtureShapes.get(key)!;
    const source = aceFp16VaeSnakeWgsl(shape);
    const sourceBytes = encoder.encode(source);
    const sha256 = await sha256Hex(sourceBytes);
    if (
      sha256 !== OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256[
        key as keyof typeof OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256
      ]
    ) {
      throw new Error("OPT-0011 Snake generated shader SHA-256 changed");
    }
    generatedShaders.push(Object.freeze({
      shapeKey: key,
      shape,
      sha256,
      byteLength: sourceBytes.byteLength,
    }));
  }
  return Object.freeze({
    ...identity,
    coreSourceSha256,
    graphTopologySha256,
    graphTopologyAuthenticatedBeforeGpuAcquisition: true,
    generatedShaderHashesFrozenAndVerifiedBeforeExecution: true,
    generatedShaders: Object.freeze(generatedShaders),
  });
}

async function runFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: Opt0011ProductionSnakeFixture,
): Promise<Readonly<Record<string, unknown>>> {
  const oracleStability = requireFixtureOracleStable(fixture);
  const prepared = await prepareFixture(device, tracker, fixture);
  try {
    const ranges: unknown[] = [];
    let comparedElementCount = 0;
    for (const [rangeIndex, range] of fixture.ranges.entries()) {
      const first = await executeAndRead(
        device,
        tracker,
        prepared,
        rangeIndex,
        `${fixture.id}-range-${rangeIndex}-first`,
      );
      const rerun = await executeAndRead(
        device,
        tracker,
        prepared,
        rangeIndex,
        `${fixture.id}-range-${rangeIndex}-rerun`,
      );
      if (first.readback.sha256 !== rerun.readback.sha256) {
        throw new Error(
          `${fixture.id} range ${rangeIndex} changed on deterministic rerun`,
        );
      }
      ranges.push(Object.freeze({
        base: range.base,
        count: range.count,
        ...(range.quantumIndex === undefined
          ? {}
          : {
              quantumIndex: range.quantumIndex,
              operationQuantumIndex: range.operationQuantumIndex,
              controlRecordIndex: range.controlRecordIndex,
            }),
        firstExecution: first.execution,
        rerunExecution: rerun.execution,
        firstScan: first.readback.scan,
        rerunScan: rerun.readback.scan,
        firstCpu: first.readback.cpu,
        rerunCpu: rerun.readback.cpu,
        firstSha256: first.readback.sha256,
        rerunSha256: rerun.readback.sha256,
        deterministic: true,
      }));
      comparedElementCount += range.count;
    }
    if (
      fixture.fixtureKind === "arithmetic" &&
      comparedElementCount !== prepared.plan.elements
    ) {
      throw new Error(`${fixture.id} no longer covers its complete domain`);
    }
    return Object.freeze({
      id: fixture.id,
      fixtureKind: fixture.fixtureKind,
      shape: fixture.shape,
      sourcePatternSalt: fixture.sourcePatternSalt,
      graphOperationIndex: fixture.graphOperationIndex,
      graphSnakeOrdinal: fixture.graphSnakeOrdinal,
      coverage: fixture.coverage,
      plan: snakePlanReceipt(prepared.plan),
      sourceIdentity: prepared.sourceIdentity,
      oracleStability,
      ranges: Object.freeze(ranges),
      rangeCount: fixture.ranges.length,
      comparedElementCount,
      completeSelectedRangeRawU16Comparison: true,
      completeManageableDomain:
        fixture.fixtureKind === "arithmetic" &&
        comparedElementCount === prepared.plan.elements,
      exactSelectedB256GraphRanges: fixture.fixtureKind === "graph",
      deterministicRerunHashes: true,
      performanceClaim: null,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

async function prepareFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: Opt0011ProductionSnakeFixture,
): Promise<PreparedFixture> {
  const plan = planAceFp16VaeSnake(fixture.shape);
  const owned: GPUBuffer[] = [];
  const kernel = AceFp16VaeSnakeKernel.create(device);
  let destroyed = false;
  try {
    const input = createInputUpload(
      device,
      tracker,
      `${fixture.id}-input`,
      plan,
      fixture,
    );
    const alpha = createChannelUpload(
      device,
      tracker,
      `${fixture.id}-alpha`,
      plan.alphaStorageBytes,
      plan.alphaBindingBytes,
      fixture,
      "alpha",
    );
    const beta = createChannelUpload(
      device,
      tracker,
      `${fixture.id}-beta`,
      plan.betaStorageBytes,
      plan.betaBindingBytes,
      fixture,
      "beta",
    );
    owned.push(input, alpha, beta);
    const output = createOutputTarget(
      device,
      tracker,
      `${fixture.id}-output`,
      plan,
    );
    owned.push(output.buffer);
    const prefill = createPrefillTemplate(
      device,
      tracker,
      `${fixture.id}-prefill`,
      maximumSelectedCopyBytes(fixture.ranges),
    );
    owned.push(prefill.qNaNBuffer, prefill.canaryBuffer);
    const controls = createRangeControlBuffer(
      device,
      tracker,
      `${fixture.id}-controls`,
      fixture.ranges,
    );
    owned.push(controls.buffer);

    const dispatches: AceFp16VaeSnakeDispatch[] = [];
    for (const [index, range] of fixture.ranges.entries()) {
      const dispatch = await kernel.createDispatch(
        `${fixture.id}-range-${index}`,
        fixture.shape,
        {
          input: binding(input, plan.inputBindingBytes),
          alpha: binding(alpha, plan.alphaBindingBytes),
          beta: binding(beta, plan.betaBindingBytes),
          output: output.binding,
        },
        rangeBinding(range, controls.bindings[index]!),
      );
      assertDispatchRange(dispatch.outputRange, plan, range, fixture.id);
      dispatches.push(dispatch);
    }
    return Object.freeze({
      fixture,
      plan,
      dispatches: Object.freeze(dispatches),
      output,
      prefill,
      sourceIdentity: Object.freeze({
        generator: fixture.fixtureKind === "graph"
          ? "opt-0011-snake-production-graph-f16-patterns-v1"
          : "opt-0011-snake-arithmetic-channel-tuples-v1",
        inputElements: plan.elements,
        channelParameterElements: plan.channels,
        inputPayloadBytes: plan.inputStorageBytes,
        inputBindingBytes: plan.inputBindingBytes,
        alphaPayloadBytes: plan.alphaStorageBytes,
        alphaBindingBytes: plan.alphaBindingBytes,
        betaPayloadBytes: plan.betaStorageBytes,
        betaBindingBytes: plan.betaBindingBytes,
        sourcePatternSalt: fixture.sourcePatternSalt,
        sourceBindingPaddingInitializedToQNaN: true,
        sourcePaddingF16Bits: hex16(SOURCE_PADDING_F16),
        completeDeterministicFormulaUpload: true,
      }),
      destroy(): void {
        kernel.destroy();
        kernel.destroy();
        if (destroyed) return;
        destroyed = true;
        for (const buffer of owned) tracker.destroy(buffer);
      },
    });
  } catch (error) {
    kernel.destroy();
    kernel.destroy();
    for (const buffer of owned) tracker.destroy(buffer);
    throw error;
  }
}

async function executeAndRead(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedFixture,
  rangeIndex: number,
  label: string,
): Promise<{
  readonly execution: ExecutionCounts;
  readonly readback: SelectedReadback;
}> {
  const range = prepared.fixture.ranges[rangeIndex]!;
  const copyBytes = align4(range.count * FLOAT16_BYTES);
  if (copyBytes > prepared.prefill.bytes) {
    throw new RangeError("OPT-0011 Snake prefill template is too small");
  }
  const destinationOffset = OUTPUT_GUARD_BYTES + range.base * FLOAT16_BYTES;
  if (destinationOffset % 4 !== 0) {
    throw new RangeError("OPT-0011 Snake prefill offset is not copy aligned");
  }
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  const beforeBytes = Math.min(
    OUTPUT_CANARY_BYTES,
    range.base * FLOAT16_BYTES,
  );
  const selectedEnd = range.base * FLOAT16_BYTES + copyBytes;
  const afterBytes = Math.min(
    OUTPUT_CANARY_BYTES,
    prepared.plan.outputBindingBytes - selectedEnd,
  );
  if (
    selectedEnd > prepared.plan.outputBindingBytes ||
    beforeBytes % 4 !== 0 || afterBytes % 4 !== 0
  ) {
    throw new RangeError("OPT-0011 Snake canary restore geometry changed");
  }
  let adjacentCanaryRestoreCopies = 0;
  if (beforeBytes > 0) {
    encoder.copyBufferToBuffer(
      prepared.prefill.canaryBuffer,
      0,
      prepared.output.buffer,
      destinationOffset - beforeBytes,
      beforeBytes,
    );
    adjacentCanaryRestoreCopies += 1;
  }
  if (afterBytes > 0) {
    encoder.copyBufferToBuffer(
      prepared.prefill.canaryBuffer,
      0,
      prepared.output.buffer,
      destinationOffset + copyBytes,
      afterBytes,
    );
    adjacentCanaryRestoreCopies += 1;
  }
  encoder.copyBufferToBuffer(
    prepared.prefill.qNaNBuffer,
    0,
    prepared.output.buffer,
    destinationOffset,
    copyBytes,
  );
  const pass = encoder.beginComputePass({ label: `${label}-pass` });
  prepared.dispatches[rangeIndex]!.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await queueEmptyIdleTurn();
  const execution = Object.freeze({
    encodedCommandBuffers: 1,
    submissions: 1,
    drains: 1,
    dispatches: 1,
    qNaNPrefillCopies: 1,
    adjacentCanaryRestoreCopies,
    queueEmptyIdleTurns: 1,
  } as const);
  const readback = await readSelectedOutput(
    device,
    tracker,
    prepared.fixture,
    prepared.plan,
    range,
    prepared.output,
    label,
  );
  return Object.freeze({ execution, readback });
}

async function readSelectedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: Opt0011ProductionSnakeFixture,
  plan: AceFp16VaeSnakePlan,
  selected: Readonly<{ readonly base: number; readonly count: number }>,
  target: OutputTarget,
  label: string,
): Promise<SelectedReadback> {
  const selectedRawBytes = selected.count * FLOAT16_BYTES;
  const selectedCopyBytes = align4(selectedRawBytes);
  const selectedPayloadOffset = selected.base * FLOAT16_BYTES;
  const selectedSourceOffset = OUTPUT_GUARD_BYTES + selectedPayloadOffset;
  if (selectedSourceOffset % 4 !== 0) {
    throw new RangeError("OPT-0011 Snake readback offset is not aligned");
  }
  const selectedEnd = selectedPayloadOffset + selectedCopyBytes;
  if (selectedEnd > plan.outputBindingBytes) {
    throw new RangeError("OPT-0011 Snake readback exceeds output binding");
  }
  const beforeBytes = Math.min(OUTPUT_CANARY_BYTES, selectedPayloadOffset);
  const afterBytes = Math.min(
    OUTPUT_CANARY_BYTES,
    plan.outputBindingBytes - selectedEnd,
  );
  if (beforeBytes % 4 !== 0 || afterBytes % 4 !== 0) {
    throw new RangeError("OPT-0011 Snake canary span is not aligned");
  }
  const prefixOffset = 0;
  const suffixOffset = OUTPUT_GUARD_BYTES;
  const selectedOffset = OUTPUT_GUARD_BYTES * 2;
  const beforeOffset = selectedOffset + selectedCopyBytes;
  const afterOffset = beforeOffset + beforeBytes;
  const totalBytes = afterOffset + afterBytes;
  const readback = tracker.create(device, {
    label: `${label}-readback`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `${label}-readback-encoder`,
    });
    encoder.copyBufferToBuffer(
      target.buffer,
      0,
      readback,
      prefixOffset,
      OUTPUT_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      target.buffer,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
      readback,
      suffixOffset,
      OUTPUT_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      target.buffer,
      selectedSourceOffset,
      readback,
      selectedOffset,
      selectedCopyBytes,
    );
    if (beforeBytes > 0) {
      encoder.copyBufferToBuffer(
        target.buffer,
        selectedSourceOffset - beforeBytes,
        readback,
        beforeOffset,
        beforeBytes,
      );
    }
    if (afterBytes > 0) {
      encoder.copyBufferToBuffer(
        target.buffer,
        OUTPUT_GUARD_BYTES + selectedEnd,
        readback,
        afterOffset,
        afterBytes,
      );
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const prefix = new Uint16Array(
      mappedRange,
      prefixOffset,
      OUTPUT_GUARD_BYTES / FLOAT16_BYTES,
    );
    const suffix = new Uint16Array(
      mappedRange,
      suffixOffset,
      OUTPUT_GUARD_BYTES / FLOAT16_BYTES,
    );
    const before = beforeBytes === 0
      ? new Uint16Array(0)
      : new Uint16Array(
          mappedRange,
          beforeOffset,
          beforeBytes / FLOAT16_BYTES,
        );
    const after = afterBytes === 0
      ? new Uint16Array(0)
      : new Uint16Array(
          mappedRange,
          afterOffset,
          afterBytes / FLOAT16_BYTES,
        );
    const selectedBits = new Uint16Array(
      mappedRange,
      selectedOffset,
      selected.count,
    );
    const paddingBits = selectedCopyBytes === selectedRawBytes
      ? new Uint16Array(0)
      : new Uint16Array(
          mappedRange,
          selectedOffset + selectedRawBytes,
          (selectedCopyBytes - selectedRawBytes) / FLOAT16_BYTES,
        );
    const expectedBits = new Uint16Array(selected.count);
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let expectedFiniteCount = 0;
    let expectedNonFiniteCount = 0;
    let prefillQNaNCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let subnormalCount = 0;
    let infinityCount = 0;
    for (let localIndex = 0; localIndex < selected.count; localIndex += 1) {
      const globalIndex = selected.base + localIndex;
      const expected = fixtureExpectedBits(fixture, globalIndex);
      const actual = selectedBits[localIndex]!;
      expectedBits[localIndex] = expected;
      if (Number.isFinite(float16BitsToNumber(actual))) finiteCount += 1;
      else nonFiniteCount += 1;
      if (Number.isFinite(float16BitsToNumber(expected))) {
        expectedFiniteCount += 1;
      } else {
        expectedNonFiniteCount += 1;
      }
      if (actual === OUTPUT_PREFILL_QNAN_F16) prefillQNaNCount += 1;
      if ((actual & 0x7fff) === 0) {
        if ((actual & 0x8000) === 0) positiveZeroCount += 1;
        else negativeZeroCount += 1;
      }
      if ((actual & 0x7c00) === 0 && (actual & 0x03ff) !== 0) {
        subnormalCount += 1;
      }
      if ((actual & 0x7fff) === 0x7c00) infinityCount += 1;
    }
    const comparison = compareOpt0011ProductionSnakeRawBits(
      selectedBits,
      expectedBits,
    );
    const prefixGuardUntouched = [...prefix].every(
      (bits) => bits === OUTPUT_GUARD_F16,
    );
    const suffixGuardUntouched = [...suffix].every(
      (bits) => bits === OUTPUT_GUARD_F16,
    );
    const adjacentCanariesUntouched = [...before, ...after].every(
      (bits) => bits === OUTPUT_CANARY_F16,
    );
    const qNaNBindingPaddingUntouched = [...paddingBits].every(
      (bits) => bits === OUTPUT_PREFILL_QNAN_F16,
    );
    if (
      !prefixGuardUntouched || !suffixGuardUntouched ||
      !adjacentCanariesUntouched || !qNaNBindingPaddingUntouched ||
      comparison.mismatchCount !== 0 || prefillQNaNCount !== 0 ||
      finiteCount !== expectedFiniteCount ||
      nonFiniteCount !== expectedNonFiniteCount
    ) {
      const first = comparison.firstMismatchIndex;
      throw new Error(
        `${label} raw-U16/CPU/guard validation failed: mismatches=` +
          `${comparison.mismatchCount}@${String(first)}, expected=` +
          `${hex16(first === null ? null : expectedBits[first]!)}, actual=` +
          `${hex16(first === null ? null : selectedBits[first]!)}`,
      );
    }
    const raw = new Uint8Array(
      mappedRange,
      selectedOffset,
      selectedRawBytes,
    ).slice();
    return Object.freeze({
      sha256: await sha256Hex(raw),
      scan: Object.freeze({
        selectedElementCount: selected.count,
        finiteCount,
        nonFiniteCount,
        expectedFiniteCount,
        expectedNonFiniteCount,
        infinityCount,
        prefillQNaNCount,
        positiveZeroCount,
        negativeZeroCount,
        subnormalCount,
        externalPrefixQNaNGuardUntouched: prefixGuardUntouched,
        externalSuffixQNaNGuardUntouched: suffixGuardUntouched,
        adjacentBeforeQNaNCanaryBytes: beforeBytes,
        adjacentAfterQNaNCanaryBytes: afterBytes,
        adjacentQNaNCanariesUntouched: adjacentCanariesUntouched,
        copiedBindingPaddingBytes: selectedCopyBytes - selectedRawBytes,
        qNaNBindingPaddingUntouched,
        completeSelectedRangeReadback: true,
        readbackCommandBuffers: 1,
        readbackSubmissions: 1,
        maps: 1,
        unmaps: 1,
      }),
      cpu: Object.freeze({
        oracle:
          "independent-f16-load-exp-sin-f32-source-order-rne-f16",
        contractionEnvelope: "uncontracted-and-final-fma-produce-same-f16-bits",
        comparedElementCount: selected.count,
        mismatchCount: comparison.mismatchCount,
        firstMismatchIndex: comparison.firstMismatchIndex,
        completeSelectedRangeRawU16Comparison: true,
      }),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function createInputUpload(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceFp16VaeSnakePlan,
  fixture: Opt0011ProductionSnakeFixture,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: plan.inputBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const words = new Uint16Array(buffer.getMappedRange());
    words.fill(SOURCE_PADDING_F16);
    const periodLength = fixture.fixtureKind === "graph"
      ? GRAPH_INPUT_F16_PATTERN.length
      : fixture.shape.channels;
    const period = new Uint16Array(periodLength);
    for (let index = 0; index < period.length; index += 1) {
      period[index] = fixture.fixtureKind === "graph"
        ? opt0011ProductionSnakeGraphInputBits(
            index,
            fixture.sourcePatternSalt,
          )
        : opt0011ProductionSnakeArithmeticInputBits(
            index,
            fixture.shape.channels,
          );
    }
    fillPeriodicPrefix(words, period, plan.elements);
    requireSourcePadding(words, plan.inputStorageBytes, label);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createChannelUpload(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  storageBytes: number,
  bindingBytes: number,
  fixture: Opt0011ProductionSnakeFixture,
  role: "alpha" | "beta",
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const words = new Uint16Array(buffer.getMappedRange());
    words.fill(SOURCE_PADDING_F16);
    for (let channel = 0; channel < fixture.shape.channels; channel += 1) {
      if (fixture.fixtureKind === "graph") {
        words[channel] = role === "alpha"
          ? opt0011ProductionSnakeGraphAlphaBits(
              channel,
              fixture.sourcePatternSalt,
            )
          : opt0011ProductionSnakeGraphBetaBits(
              channel,
              fixture.sourcePatternSalt,
            );
      } else {
        words[channel] = role === "alpha"
          ? opt0011ProductionSnakeArithmeticAlphaBits(channel)
          : opt0011ProductionSnakeArithmeticBetaBits(channel);
      }
    }
    requireSourcePadding(words, storageBytes, label);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function requireSourcePadding(
  words: Uint16Array,
  storageBytes: number,
  label: string,
): void {
  const payloadWords = storageBytes / FLOAT16_BYTES;
  if (!Number.isInteger(payloadWords) || payloadWords > words.length) {
    throw new RangeError(`${label} source padding geometry changed`);
  }
  for (let index = payloadWords; index < words.length; index += 1) {
    if (words[index] !== SOURCE_PADDING_F16) {
      throw new Error(`${label} source binding padding was overwritten`);
    }
  }
}

function fillPeriodicPrefix(
  destination: Uint16Array,
  period: Uint16Array,
  elements: number,
): void {
  if (elements > destination.length || period.length < 1) {
    throw new RangeError("OPT-0011 Snake periodic upload geometry is invalid");
  }
  const initial = Math.min(period.length, elements);
  for (let index = 0; index < initial; index += 1) {
    destination[index] = period[index]!;
  }
  let filled = initial;
  while (filled < elements) {
    const copy = Math.min(filled, elements - filled);
    destination.copyWithin(filled, 0, copy);
    filled += copy;
  }
}

function createOutputTarget(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceFp16VaeSnakePlan,
): OutputTarget {
  const buffer = tracker.create(device, {
    label,
    size: OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  try {
    const bits = new Uint16Array(buffer.getMappedRange());
    bits.fill(OUTPUT_GUARD_F16);
    bits.fill(
      OUTPUT_CANARY_F16,
      OUTPUT_GUARD_BYTES / FLOAT16_BYTES,
      (OUTPUT_GUARD_BYTES + plan.outputBindingBytes) / FLOAT16_BYTES,
    );
    buffer.unmap();
    return Object.freeze({
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OUTPUT_GUARD_BYTES,
        size: plan.outputBindingBytes,
      }),
    });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createPrefillTemplate(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
): PrefillTemplate {
  if (bytes < 4 || bytes % 4 !== 0) {
    throw new RangeError("OPT-0011 Snake prefill bytes changed");
  }
  const qNaNBuffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  try {
    new Uint16Array(qNaNBuffer.getMappedRange()).fill(
      OUTPUT_PREFILL_QNAN_F16,
    );
    qNaNBuffer.unmap();
    const canaryBuffer = tracker.create(device, {
      label: `${label}-adjacent-canary`,
      size: OUTPUT_CANARY_BYTES,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    try {
      new Uint16Array(canaryBuffer.getMappedRange()).fill(OUTPUT_CANARY_F16);
      canaryBuffer.unmap();
      return Object.freeze({ qNaNBuffer, canaryBuffer, bytes });
    } catch (error) {
      if (canaryBuffer.mapState === "mapped") canaryBuffer.unmap();
      tracker.destroy(canaryBuffer);
      throw error;
    }
  } catch (error) {
    if (qNaNBuffer.mapState === "mapped") qNaNBuffer.unmap();
    tracker.destroy(qNaNBuffer);
    throw error;
  }
}

function createRangeControlBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  ranges: readonly Readonly<{ readonly base: number; readonly count: number }>[],
): {
  readonly buffer: GPUBuffer;
  readonly bindings: readonly GPUBufferBinding[];
} {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (
    !Number.isSafeInteger(alignment) || alignment < CONTROL_BYTES ||
    alignment % 4 !== 0 || ranges.length < 1
  ) {
    throw new RangeError("OPT-0011 Snake control alignment changed");
  }
  const buffer = tracker.create(device, {
    label,
    size: alignment * ranges.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  try {
    const words = new Uint32Array(buffer.getMappedRange());
    for (const [index, range] of ranges.entries()) {
      const word = index * alignment / 4;
      words[word] = range.base;
      words[word + 1] = range.count;
      words[word + 2] = 0;
      words[word + 3] = 0;
    }
    buffer.unmap();
    return Object.freeze({
      buffer,
      bindings: Object.freeze(ranges.map((_, index) => Object.freeze({
        buffer,
        offset: index * alignment,
        size: CONTROL_BYTES,
      }))),
    });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

async function runCancellationProof(
  device: GPUDevice,
  tracker: BufferTracker,
): Promise<Readonly<Record<string, unknown>>> {
  const graphCase = OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.find(
    ({ operationIndex }) => operationIndex === 3,
  );
  if (graphCase === undefined || graphCase.ranges.length !== 3) {
    throw new Error("OPT-0011 Snake cancellation graph case changed");
  }
  const fixture = Object.freeze({
    id: "snake-cancellation-real-block-0-res-1-ranges",
    fixtureKind: "graph" as const,
    shape: graphCase.shape,
    sourcePatternSalt: 7,
    graphOperationIndex: graphCase.operationIndex,
    graphSnakeOrdinal: graphCase.snakeOrdinal,
    ranges: Object.freeze(graphCase.ranges.slice(0, 2)),
    coverage: Object.freeze(["real-two-range-cancellation"]),
  } satisfies Opt0011ProductionSnakeFixture);
  const prepared = await prepareFixture(device, tracker, fixture);
  try {
    const controller = new AbortController();
    let encodeCount = 0;
    let submitCount = 0;
    let drainCount = 0;
    let prefillCopyCount = 0;
    let readbackCount = 0;
    let skippedRangeCount = 0;
    let idleTurnDelivered = false;
    for (const [index, dispatch] of prepared.dispatches.entries()) {
      if (controller.signal.aborted) {
        skippedRangeCount += 1;
        continue;
      }
      const range = fixture.ranges[index]!;
      const encoder = device.createCommandEncoder({
        label: `opt-0011-snake-cancel-${index}-encoder`,
      });
      encoder.copyBufferToBuffer(
        prepared.prefill.qNaNBuffer,
        0,
        prepared.output.buffer,
        OUTPUT_GUARD_BYTES + range.base * FLOAT16_BYTES,
        align4(range.count * FLOAT16_BYTES),
      );
      prefillCopyCount += 1;
      const pass = encoder.beginComputePass({
        label: `opt-0011-snake-cancel-${index}-pass`,
      });
      dispatch.encode(pass);
      pass.end();
      encodeCount += 1;
      device.queue.submit([encoder.finish()]);
      submitCount += 1;
      await device.queue.onSubmittedWorkDone();
      drainCount += 1;
      await queueEmptyIdleTurn();
      idleTurnDelivered = true;
      controller.abort("cancel-after-first-drained-real-b256-snake-range");
    }
    if (!controller.signal.aborted) {
      readbackCount += 1;
      await readSelectedOutput(
        device,
        tracker,
        fixture,
        prepared.plan,
        fixture.ranges[0]!,
        prepared.output,
        "opt-0011-snake-cancel-readback",
      );
    }
    if (
      !controller.signal.aborted || !idleTurnDelivered || encodeCount !== 1 ||
      submitCount !== 1 || drainCount !== 1 || prefillCopyCount !== 1 ||
      readbackCount !== 0 || skippedRangeCount !== 1
    ) {
      throw new Error("OPT-0011 Snake cancellation proof changed");
    }
    return Object.freeze({
      fixtureId: fixture.id,
      graphOperationIndex: graphCase.operationIndex,
      plannedRangeCount: fixture.ranges.length,
      plannedRanges: Object.freeze(fixture.ranges.map((range) =>
        Object.freeze({
          quantumIndex: range.quantumIndex,
          operationQuantumIndex: range.operationQuantumIndex,
          controlRecordIndex: range.controlRecordIndex,
          base: range.base,
          count: range.count,
        })
      )),
      cancellationPoint: "after-first-drained-real-b256-snake-range-and-idle",
      encodeCount,
      submitCount,
      drainCount,
      prefillCopyCount,
      readbackCount,
      skippedRangeCount,
      signalAborted: true,
      realQueueEmptyIdleTurnDelivered: true,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
      readbackPrevented: true,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

function buildGraphCases(): readonly Opt0011ProductionSnakeGraphCase[] {
  const operations = B256_GRAPH.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is {
      readonly operation: AceVaeDecoderSnakeOperation;
      readonly operationIndex: number;
    } => entry.operation.kind === "snake");
  if (operations.length !== 36) {
    throw new Error("OPT-0011 expected exactly 36 B-256 Snake operations");
  }
  const shapeFamilies = new Map<string, number>();
  const cases = operations.map(
    ({ operation, operationIndex }, snakeOrdinal) => {
      const key = shapeKey(operation.shape);
      let shapeFamilyIndex = shapeFamilies.get(key);
      if (shapeFamilyIndex === undefined) {
        shapeFamilyIndex = shapeFamilies.size;
        shapeFamilies.set(key, shapeFamilyIndex);
      }
      let operationQuantumIndex = 0;
      const ranges = B256_QUANTA.quanta
        .filter((quantum) => quantum.operationIndex === operationIndex)
        .map((quantum) => graphRange(
          operation,
          quantum,
          operationQuantumIndex++,
        ));
      const elements = operation.shape.batch * operation.shape.frames *
        operation.shape.channels;
      let cursor = 0;
      for (const [index, range] of ranges.entries()) {
        if (range.operationQuantumIndex !== index || range.base !== cursor) {
          throw new Error(`${operation.label} exact range sequence changed`);
        }
        cursor += range.count;
      }
      if (ranges.length < 1 || cursor !== elements) {
        throw new Error(`${operation.label} exact range coverage changed`);
      }
      return Object.freeze({
        id: operation.label,
        operationIndex,
        snakeOrdinal,
        shapeFamilyIndex,
        shape: operation.shape,
        ranges: Object.freeze(ranges),
      });
    },
  );
  const quantumCount = cases.reduce(
    (sum, graphCase) => sum + graphCase.ranges.length,
    0,
  );
  if (shapeFamilies.size !== 6 || quantumCount !== 813) {
    throw new Error("OPT-0011 B-256 Snake topology changed");
  }
  return Object.freeze(cases);
}

function graphRange(
  operation: AceVaeDecoderSnakeOperation,
  quantum: AceVaeDecoderQuantumPlan,
  operationQuantumIndex: number,
): Opt0011ProductionSnakeGraphRange {
  if (
    quantum.operationKind !== "snake" ||
    quantum.operationLabel !== operation.label ||
    quantum.primitives.length !== 1 ||
    quantum.estimatedMaximumMultiplyAccumulates !== 0
  ) {
    throw new Error(`${operation.label} quantum topology changed`);
  }
  const primitive = quantum.primitives[0]!;
  if (
    primitive.firstOutputChannel !== 0 ||
    primitive.outputChannels !== operation.shape.channels ||
    primitive.outputBase !== quantum.logicalOutputBase ||
    primitive.outputCount !== quantum.logicalOutputCount
  ) {
    throw new Error(`${operation.label} primitive topology changed`);
  }
  return Object.freeze({
    quantumIndex: quantum.index,
    operationQuantumIndex,
    controlRecordIndex: primitive.controlRecordIndex,
    base: primitive.outputBase,
    count: primitive.outputCount,
  });
}

function buildSelectedGraphFixtures(): readonly Opt0011ProductionSnakeFixture[] {
  const selections = Object.freeze([
    Object.freeze({ operationIndex: 1, rangeOrdinals: Object.freeze([0]) }),
    Object.freeze({ operationIndex: 3, rangeOrdinals: Object.freeze([0, 2]) }),
    Object.freeze({ operationIndex: 22, rangeOrdinals: Object.freeze([7]) }),
    Object.freeze({ operationIndex: 42, rangeOrdinals: Object.freeze([7]) }),
    Object.freeze({ operationIndex: 69, rangeOrdinals: Object.freeze([29]) }),
    Object.freeze({ operationIndex: 86, rangeOrdinals: Object.freeze([0, 59]) }),
  ]);
  const fixtures = selections.map((selection, selectionIndex) => {
    const graphCase = OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.find(
      ({ operationIndex }) => operationIndex === selection.operationIndex,
    );
    if (graphCase === undefined) {
      throw new Error("OPT-0011 selected Snake graph operation changed");
    }
    const ranges = selection.rangeOrdinals.map((ordinal) => {
      const range = graphCase.ranges[ordinal];
      if (range === undefined) {
        throw new Error(`${graphCase.id} selected range changed`);
      }
      return range;
    });
    return Object.freeze({
      id: `${graphCase.id}-representative-exact-ranges`,
      fixtureKind: "graph" as const,
      shape: graphCase.shape,
      sourcePatternSalt: selectionIndex + 1,
      graphOperationIndex: graphCase.operationIndex,
      graphSnakeOrdinal: graphCase.snakeOrdinal,
      ranges: Object.freeze(ranges),
      coverage: Object.freeze([
        "exact-b256-graph-quantum-ranges",
        `production-shape-family-${graphCase.shapeFamilyIndex}`,
        selectionIndex === 0 ? "first-graph-range" : "interior-or-tail-range",
      ]),
    } satisfies Opt0011ProductionSnakeFixture);
  });
  if (
    fixtures.length !== 6 ||
    fixtures.reduce((sum, fixture) => sum + fixture.ranges.length, 0) !== 8 ||
    new Set(fixtures.map(({ shape }) => shapeKey(shape))).size !== 6
  ) {
    throw new Error("OPT-0011 representative Snake coverage changed");
  }
  return Object.freeze(fixtures);
}

function graphCoverageReceipt(): Readonly<Record<string, unknown>> {
  return Object.freeze({
    decoderInputFrames: B256_GRAPH.inputFrames,
    decoderOutputFrames: B256_GRAPH.outputFrames,
    operationCount: B256_GRAPH.operations.length,
    snakeOperationCount: OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.length,
    exactSnakeQuantumCount: OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.reduce(
      (sum, graphCase) => sum + graphCase.ranges.length,
      0,
    ),
    exactQuantumTopologySha256:
      OPT_0011_PRODUCTION_SNAKE_GRAPH_TOPOLOGY_SHA256,
    topologyHashDomain:
      "canonical-complete-813-snake-quantum-records-authenticated-in-browser",
    completeOperationAndRangeRecordsIncluded: true,
    operations: Object.freeze(OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.map(
      (graphCase) => Object.freeze({
        id: graphCase.id,
        operationIndex: graphCase.operationIndex,
        snakeOrdinal: graphCase.snakeOrdinal,
        shapeFamilyIndex: graphCase.shapeFamilyIndex,
        shape: graphCase.shape,
        rangeCount: graphCase.ranges.length,
        ranges: graphCase.ranges,
      }),
    )),
    selectedFixtureCount:
      OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES.length,
    selectedExactRangeCount:
      OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES.reduce(
        (sum, fixture) => sum + fixture.ranges.length,
        0,
      ),
  });
}

function fixtureExpectedBits(
  fixture: Opt0011ProductionSnakeFixture,
  globalIndex: number,
): number {
  if (
    !Number.isSafeInteger(globalIndex) || globalIndex < 0 ||
    globalIndex >= fixture.shape.batch * fixture.shape.frames *
      fixture.shape.channels
  ) {
    throw new RangeError(`${fixture.id} CPU oracle index is outside its domain`);
  }
  const channel = globalIndex % fixture.shape.channels;
  if (fixture.fixtureKind === "graph") {
    return opt0011ProductionSnakeCpuBits(
      opt0011ProductionSnakeGraphInputBits(
        globalIndex,
        fixture.sourcePatternSalt,
      ),
      opt0011ProductionSnakeGraphAlphaBits(
        channel,
        fixture.sourcePatternSalt,
      ),
      opt0011ProductionSnakeGraphBetaBits(
        channel,
        fixture.sourcePatternSalt,
      ),
    );
  }
  return opt0011ProductionSnakeCpuBits(
    opt0011ProductionSnakeArithmeticInputBits(
      globalIndex,
      fixture.shape.channels,
    ),
    opt0011ProductionSnakeArithmeticAlphaBits(channel),
    opt0011ProductionSnakeArithmeticBetaBits(channel),
  );
}

function requireFixtureOracleStable(
  fixture: Opt0011ProductionSnakeFixture,
): Readonly<Record<string, unknown>> {
  const probeCount = fixture.fixtureKind === "graph"
    ? GRAPH_INPUT_F16_PATTERN.length
    : fixture.shape.channels * Math.min(fixture.shape.frames, 2);
  let finiteCount = 0;
  let nonFiniteCount = 0;
  let infinityCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  let subnormalCount = 0;
  let evenLowerMantissaTieCount = 0;
  let oddLowerMantissaTieCount = 0;
  for (let index = 0; index < probeCount; index += 1) {
    const channel = index % fixture.shape.channels;
    const inputBits = fixture.fixtureKind === "graph"
      ? opt0011ProductionSnakeGraphInputBits(
          index,
          fixture.sourcePatternSalt,
        )
      : opt0011ProductionSnakeArithmeticInputBits(
          index,
          fixture.shape.channels,
        );
    const alphaBits = fixture.fixtureKind === "graph"
      ? opt0011ProductionSnakeGraphAlphaBits(
          channel,
          fixture.sourcePatternSalt,
        )
      : opt0011ProductionSnakeArithmeticAlphaBits(channel);
    const betaBits = fixture.fixtureKind === "graph"
      ? opt0011ProductionSnakeGraphBetaBits(
          channel,
          fixture.sourcePatternSalt,
        )
      : opt0011ProductionSnakeArithmeticBetaBits(channel);
    const arithmetic = snakeCpuArithmetic(inputBits, alphaBits, betaBits);
    const expected = numberToFloat16Bits(arithmetic.uncontractedResult);
    const finalFused = numberToFloat16Bits(arithmetic.finalFusedResult);
    if (expected !== finalFused) {
      throw new Error(
        `${fixture.id} is not stable across the registered contraction envelope`,
      );
    }
    const value = float16BitsToNumber(expected);
    if (Number.isFinite(value)) finiteCount += 1;
    else nonFiniteCount += 1;
    if ((expected & 0x7fff) === 0x7c00) infinityCount += 1;
    if ((expected & 0x7fff) === 0) {
      if ((expected & 0x8000) === 0) positiveZeroCount += 1;
      else negativeZeroCount += 1;
    }
    if ((expected & 0x7c00) === 0 && (expected & 0x03ff) !== 0) {
      subnormalCount += 1;
    }
    const tieParity = f16NormalTieLowerMantissaParity(
      arithmetic.uncontractedResult,
    );
    if (tieParity === 0) evenLowerMantissaTieCount += 1;
    if (tieParity === 1) oddLowerMantissaTieCount += 1;
  }
  if (
    fixture.fixtureKind === "arithmetic" &&
    evenLowerMantissaTieCount + oddLowerMantissaTieCount === 0
  ) {
    throw new Error(`${fixture.id} lost its actual Snake f16 RNE boundary`);
  }
  return Object.freeze({
    probeCount,
    uncontractedAndFinalFusedF16BitsIdentical: true,
    finiteCount,
    nonFiniteCount,
    infinityCount,
    positiveZeroCount,
    negativeZeroCount,
    subnormalCount,
    roundToNearestEvenBoundaryCount:
      evenLowerMantissaTieCount + oddLowerMantissaTieCount,
    evenLowerMantissaTieCount,
    oddLowerMantissaTieCount,
  });
}

function f16NormalTieLowerMantissaParity(value: number): 0 | 1 | null {
  FLOAT32_SCRATCH[0] = Math.fround(value);
  const bits = UINT32_SCRATCH[0]!;
  const exponent = (bits >>> 23) & 0xff;
  const halfExponent = exponent - 127 + 15;
  const mantissa = bits & 0x7f_ffff;
  if (
    halfExponent < 1 || halfExponent > 30 ||
    (mantissa & 0x1fff) !== 0x1000
  ) return null;
  return ((mantissa >>> 13) & 1) as 0 | 1;
}

function snakeCpuArithmetic(
  inputBits: number,
  alphaBits: number,
  betaBits: number,
): Readonly<{
  readonly uncontractedResult: number;
  readonly finalFusedResult: number;
}> {
  for (const [value, label] of [
    [inputBits, "input"],
    [alphaBits, "alpha"],
    [betaBits, "beta"],
  ] as const) requireIndex(value, 0xffff, `Snake CPU ${label}`);
  const input = Math.fround(float16BitsToNumber(inputBits));
  const alphaLogScale = Math.fround(float16BitsToNumber(alphaBits));
  const betaLogScale = Math.fround(float16BitsToNumber(betaBits));
  const alphaValue = Math.fround(Math.exp(alphaLogScale));
  const betaValue = Math.fround(Math.exp(betaLogScale));
  const periodicArgument = Math.fround(alphaValue * input);
  const periodic = Math.fround(Math.sin(periodicArgument));
  const denominator = Math.fround(betaValue + Math.fround(1e-9));
  const reciprocalBeta = Math.fround(1 / denominator);
  const firstProduct = Math.fround(reciprocalBeta * periodic);
  const periodicTerm = Math.fround(firstProduct * periodic);
  const uncontractedResult = Math.fround(input + periodicTerm);
  const finalFusedResult = Math.fround(input + firstProduct * periodic);
  return Object.freeze({ uncontractedResult, finalFusedResult });
}

function rangeBinding(
  range: Readonly<{ readonly base: number; readonly count: number }>,
  control: GPUBufferBinding,
): AceVaeOutputRangeBinding {
  return Object.freeze({ base: range.base, count: range.count, control });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function assertDispatchRange(
  actual: Readonly<{
    readonly base: number;
    readonly count: number;
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>,
  plan: AceFp16VaeSnakePlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  label: string,
): void {
  const expected = planAceFp16VaeSnakeRange(plan, range);
  if (
    actual.base !== expected.base || actual.count !== expected.count ||
    actual.workgroupsX !== expected.workgroupsX ||
    actual.workgroupsY !== expected.workgroupsY
  ) {
    throw new Error(`${label} production Snake dispatch range changed`);
  }
}

function snakePlanReceipt(
  plan: AceFp16VaeSnakePlan,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    batch: plan.batch,
    frames: plan.frames,
    channels: plan.channels,
    elements: plan.elements,
    inputStorageBytes: plan.inputStorageBytes,
    inputBindingBytes: plan.inputBindingBytes,
    alphaStorageBytes: plan.alphaStorageBytes,
    alphaBindingBytes: plan.alphaBindingBytes,
    betaStorageBytes: plan.betaStorageBytes,
    betaBindingBytes: plan.betaBindingBytes,
    outputStorageBytes: plan.outputStorageBytes,
    outputBindingBytes: plan.outputBindingBytes,
    workgroupSize: plan.workgroupSize,
    fullDomainWorkgroupsX: plan.workgroupsX,
    fullDomainWorkgroupsY: plan.workgroupsY,
  });
}

function maximumSelectedCopyBytes(
  ranges: readonly Readonly<{ readonly count: number }>[],
): number {
  const maximum = Math.max(...ranges.map((range) =>
    align4(range.count * FLOAT16_BYTES)
  ));
  if (!Number.isSafeInteger(maximum) || maximum < 4) {
    throw new RangeError("OPT-0011 Snake selected copy bytes changed");
  }
  return maximum;
}

function requiredDeviceLimits(): Record<string, number> {
  let maximumBufferSize = 0;
  let maximumStorageBinding = 0;
  let maximumDispatch = 1;
  for (const fixture of OPT_0011_PRODUCTION_SNAKE_FIXTURES) {
    const plan = planAceFp16VaeSnake(fixture.shape);
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      plan.inputBindingBytes,
      plan.alphaBindingBytes,
      plan.betaBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBufferSize = Math.max(
      maximumBufferSize,
      plan.inputBindingBytes,
      plan.alphaBindingBytes,
      plan.betaBindingBytes,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
      maximumSelectedCopyBytes(fixture.ranges),
    );
    for (const range of fixture.ranges) {
      const planned = planAceFp16VaeSnakeRange(plan, range);
      maximumDispatch = Math.max(
        maximumDispatch,
        planned.workgroupsX,
        planned.workgroupsY,
      );
    }
  }
  return {
    maxBufferSize: maximumBufferSize,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: CONTROL_BYTES,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
    maxBindGroups: 1,
    maxBindingsPerBindGroup: 5,
    maxStorageBuffersPerShaderStage: 4,
    maxUniformBuffersPerShaderStage: 1,
    maxDynamicUniformBuffersPerPipelineLayout: 1,
  };
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("OPT-0011 Snake gate requires adapter shader-f16");
  }
  const required = requiredDeviceLimits();
  for (const [name, minimum] of Object.entries(required)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0011 Snake adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > OUTPUT_GUARD_BYTES) {
    throw new RangeError(
      "OPT-0011 Snake output guard is below storage offset alignment",
    );
  }
}

function adapterReceipt(adapter: GPUAdapter, device: GPUDevice): unknown {
  return Object.freeze({
    info: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    requiredLimits: Object.freeze(requiredDeviceLimits()),
    actualAdapterLimits: gpuLimitReceipt(adapter.limits),
    actualRequestedDeviceLimits: gpuLimitReceipt(device.limits),
  });
}

function gpuLimitReceipt(limits: GPUSupportedLimits): unknown {
  return Object.freeze({
    maxBufferSize: Number(limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize),
    maxUniformBufferBindingSize: Number(limits.maxUniformBufferBindingSize),
    maxComputeInvocationsPerWorkgroup:
      limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupsPerDimension:
      limits.maxComputeWorkgroupsPerDimension,
    maxBindGroups: limits.maxBindGroups,
    maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxUniformBuffersPerShaderStage: limits.maxUniformBuffersPerShaderStage,
    maxDynamicUniformBuffersPerPipelineLayout:
      limits.maxDynamicUniformBuffersPerPipelineLayout,
    minStorageBufferOffsetAlignment: limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
  });
}

export function stopOpt0011ProductionSnakeHeartbeatAfterFailure(
  heartbeat: HeartbeatController,
): Opt0011ProductionSnakeHeartbeatFailureStop {
  try {
    return Object.freeze({
      liveness: heartbeat.stop(),
      heartbeatStopError: null,
    });
  } catch (error) {
    return Object.freeze({
      liveness: null,
      heartbeatStopError: errorReceipt(error),
    });
  }
}

function startHeartbeat(): HeartbeatController {
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let animationFrame = 0;
  const animate = (): void => {
    if (stopped) return;
    animationFrameCount += 1;
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);
  const timer = window.setInterval(() => {
    timerTickCount += 1;
  }, 16);
  let receipt: Readonly<Record<string, unknown>> | undefined;
  return Object.freeze({
    stop(): Readonly<Record<string, unknown>> {
      if (receipt !== undefined) return receipt;
      stopped = true;
      cancelAnimationFrame(animationFrame);
      window.clearInterval(timer);
      receipt = Object.freeze({
        animationFrameCount,
        timerTickCount,
        observed: animationFrameCount > 0 && timerTickCount > 0,
        countsAreCorrectnessSignalsNotTimingMeasurements: true,
      });
      return receipt;
    },
  });
}

async function queueEmptyIdleTurn(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, QUEUE_EMPTY_IDLE_MILLISECONDS);
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function shapeKey(shape: AceVaePointwiseShape): string {
  return `${shape.batch}:${shape.frames}:${shape.channels}`;
}

function requiredIdentity(parameters: URLSearchParams, name: string): string {
  const values = parameters.getAll(name);
  if (values.length !== 1 || values[0] === "") {
    throw new Error(`OPT-0011 Snake requires one ${name}`);
  }
  return values[0]!;
}

function requireIndex(index: number, maximum: number, label: string): void {
  if (!Number.isSafeInteger(index) || index < 0 || index > maximum) {
    throw new RangeError(`${label} index is outside its raw-bit domain`);
  }
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? sign * 0 : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15);
}

export function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = Math.fround(value);
  const bits = UINT32_SCRATCH[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway ||
        (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)
  ) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function hex16(value: number | null): string {
  return value === null ? "not-applicable" :
    `0x${value.toString(16).padStart(4, "0")}`;
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
    });
  }
  return Object.freeze({ name: "UnknownError", message: String(error) });
}

export function parseOpt0011ProductionSnakeRawResultChunkOffset(
  value: string,
): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error(
      "OPT-0011 Snake raw-result offset is not canonical decimal",
    );
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error("OPT-0011 Snake raw-result offset is not a safe integer");
  }
  return offset;
}

export function sliceOpt0011ProductionSnakeRawResultChunk(
  rawResultJson: string,
  offset: number,
): Readonly<{
  readonly chunk: string;
  readonly start: number;
  readonly end: number;
  readonly nextOffset: number;
  readonly totalCodeUnits: number;
  readonly complete: boolean;
}> {
  if (
    !Number.isSafeInteger(offset) || offset < 0 ||
    offset > rawResultJson.length
  ) {
    throw new Error("OPT-0011 Snake raw-result offset is invalid");
  }
  if (
    offset > 0 && offset < rawResultJson.length &&
    isHighSurrogate(rawResultJson.charCodeAt(offset - 1)) &&
    isLowSurrogate(rawResultJson.charCodeAt(offset))
  ) {
    throw new Error("OPT-0011 Snake raw-result offset splits a surrogate pair");
  }
  let end = Math.min(
    offset + OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS,
    rawResultJson.length,
  );
  if (
    end < rawResultJson.length && end > offset &&
    isHighSurrogate(rawResultJson.charCodeAt(end - 1)) &&
    isLowSurrogate(rawResultJson.charCodeAt(end))
  ) {
    end -= 1;
  }
  return Object.freeze({
    chunk: rawResultJson.slice(offset, end),
    start: offset,
    end,
    nextOffset: end,
    totalCodeUnits: rawResultJson.length,
    complete: end === rawResultJson.length,
  });
}

function isHighSurrogate(codeUnit: number): boolean {
  return codeUnit >= 0xd800 && codeUnit <= 0xdbff;
}

function isLowSurrogate(codeUnit: number): boolean {
  return codeUnit >= 0xdc00 && codeUnit <= 0xdfff;
}

function installRawResultChunkRetrieval(): void {
  const offsetInput = requireElement<HTMLInputElement>(
    'input[name="rawResultOffset"]',
  );
  const publish = requireElement<HTMLButtonElement>(
    "#publish-raw-result-chunk",
  );
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  let publicationSequence = 0;
  publish.addEventListener("click", () => {
    output.textContent = "";
    output.dataset.state = "publishing";
    output.dataset.publicationSequence = String(++publicationSequence);
    delete output.dataset.startOffset;
    delete output.dataset.endOffsetExclusive;
    delete output.dataset.chunkCodeUnitLength;
    delete output.dataset.totalCodeUnitLength;
    delete output.dataset.done;
    try {
      const rawResultJson = Reflect.get(globalThis, RAW_RESULT_GLOBAL);
      if (typeof rawResultJson !== "string") {
        throw new Error("OPT-0011 Snake raw result is not available yet");
      }
      const slice = sliceOpt0011ProductionSnakeRawResultChunk(
        rawResultJson,
        parseOpt0011ProductionSnakeRawResultChunkOffset(offsetInput.value),
      );
      output.textContent = slice.chunk;
      output.dataset.startOffset = String(slice.start);
      output.dataset.endOffsetExclusive = String(slice.end);
      output.dataset.chunkCodeUnitLength = String(slice.chunk.length);
      output.dataset.totalCodeUnitLength = String(slice.totalCodeUnits);
      output.dataset.done = String(slice.complete);
      output.dataset.state = "published";
      offsetInput.value = String(slice.nextOffset);
    } catch (error) {
      output.dataset.state = "failed";
      output.textContent = JSON.stringify(errorReceipt(error));
    } finally {
      publish.disabled = false;
    }
  });
}

function resetRawResultChunkRetrieval(): void {
  const offsetInput = requireElement<HTMLInputElement>(
    'input[name="rawResultOffset"]',
  );
  const publish = requireElement<HTMLButtonElement>(
    "#publish-raw-result-chunk",
  );
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  offsetInput.value = "0";
  output.textContent = "";
  output.dataset.state = "ready";
  delete output.dataset.startOffset;
  delete output.dataset.endOffsetExclusive;
  delete output.dataset.chunkCodeUnitLength;
  delete output.dataset.totalCodeUnitLength;
  delete output.dataset.done;
  publish.disabled = false;
}

function publishPageResult(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  const rawResultJson = JSON.stringify(result);
  if (!Reflect.defineProperty(globalThis, RAW_RESULT_GLOBAL, {
    value: rawResultJson,
    configurable: false,
    enumerable: false,
    writable: false,
  })) {
    throw new Error("OPT-0011 Snake could not publish raw receipt");
  }
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    schema: result["schema"] ?? null,
    status,
    experimentId: result["experimentId"] ?? "OPT-0011",
    classification: result["classification"] ?? null,
    rawResultJsonCodeUnitLength: rawResultJson.length,
    rawResultRetrieval: "bounded-restartable-dom-chunks-from-page-start",
    rawResultMainWorldGlobal: RAW_RESULT_GLOBAL,
    rawResultChunkCodeUnitLimit:
      OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS,
    fullReceiptIntentionallyKeptOutOfDom: true,
  }, null, 2);
  resetRawResultChunkRetrieval();
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finish(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  updateProgress(status);
  publishPageResult(status, result);
}

function requireElement<ElementType extends Element>(
  selector: string,
): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) {
    throw new Error(`Missing OPT-0011 Snake element ${selector}`);
  }
  return element;
}
