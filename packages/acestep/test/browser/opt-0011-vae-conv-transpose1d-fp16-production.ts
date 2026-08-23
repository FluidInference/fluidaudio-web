/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import productionCoreSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
  AceFp16VaeConvTranspose1dKernel,
  aceFp16VaeConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dRange,
  type AceFp16VaeConvTranspose1dDispatch,
  type AceFp16VaeConvTranspose1dPlan,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvTransposeOperation,
  type AceVaeDecoderQuantumPlan,
} from "../../src/webgpu/vae-decoder.js";

export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT =
  "d2bf0819d0460f6bd60ebe0457eb091b45e7bf6a" as const;
export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_SOURCE_SHA256 =
  "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2" as const;

export type Opt0011ProductionConvTranspose1dCaseId =
  | "block-0-stride10-left-padding"
  | "block-1-stride6-interior"
  | "block-2-stride4-right-padding"
  | "block-3-stride4-time-tail"
  | "block-4-stride2-longest-tail"
  | "arithmetic-stride6-cin65-cout9";

export interface Opt0011ProductionConvTranspose1dGraphRange {
  readonly quantumIndex: number;
  readonly operationQuantumIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0011ProductionConvTranspose1dGraphCase {
  readonly operationIndex: number;
  readonly operationOrdinal: number;
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly ranges: readonly Opt0011ProductionConvTranspose1dGraphRange[];
}

export interface Opt0011ProductionConvTranspose1dFixture {
  readonly id: Opt0011ProductionConvTranspose1dCaseId;
  readonly graphOperationIndex: number | null;
  readonly graphOperationOrdinal: number | null;
  readonly graphOperationLabel: string | null;
  readonly containingGraphQuantumIndex: number | null;
  readonly containingGraphOperationQuantumIndex: number | null;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly ranges: readonly Readonly<{
    readonly base: number;
    readonly count: number;
  }>[];
  readonly sourcePatternSalt: number;
  readonly coverage: readonly string[];
}

export interface Opt0011ProductionConvTranspose1dRunIdentity {
  readonly harnessCommit: string;
  readonly coreCommit:
    typeof OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT;
}

export interface Opt0011ProductionConvTranspose1dRawBitComparison {
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
}

interface PreparedFixture {
  readonly fixture: Opt0011ProductionConvTranspose1dFixture;
  readonly plan: AceFp16VaeConvTranspose1dPlan;
  readonly dispatches: readonly AceFp16VaeConvTranspose1dDispatch[];
  readonly output: OutputTarget;
  readonly prefill: PrefillTemplate;
  readonly uploadIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface OutputTarget {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

interface PrefillTemplate {
  readonly buffer: GPUBuffer;
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
  readonly queueEmptyIdleTurns: 1;
}

export interface HeartbeatController {
  stop(): Readonly<Record<string, unknown>>;
}

export interface Opt0011ProductionConvTranspose1dHeartbeatFailureStop {
  readonly responsiveness: Readonly<Record<string, unknown>> | null;
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
  "__ACE_OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_JSON__";
export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS =
  100_000;

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

const INPUT_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x0001, 0x8001,
  0x0400, 0x8400, 0x1000, 0x9000,
  0x2000, 0xa000, 0x2800, 0xa800,
  0x3000, 0xb000, 0x3555, 0xb555,
]);
const WEIGHT_PATTERN = Object.freeze([
  0x3c00, 0xbc00, 0x3800, 0xb800,
  0x3400, 0xb400, 0x3000, 0xb000,
  0x2c00, 0xac00, 0x2400, 0xa400,
  0x0000, 0x8000,
]);
const BIAS_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x1000, 0x9000,
  0x2000, 0xa000, 0x2800, 0xa800,
]);

const B256_GRAPH = planAceVaeDecoder(256);
const B256_QUANTA = planAceVaeDecoderQuanta(B256_GRAPH);

export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES =
  buildGraphCases();

export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES = Object.freeze([
  graphFixture(
    "block-0-stride10-left-padding",
    "block-0-conv-t1",
    0,
    1,
    1,
    ["canonical-b256-shape", "stride-10", "left-padding", "single-row"],
  ),
  graphFixture(
    "block-1-stride6-interior",
    "block-1-conv-t1",
    7_680,
    1,
    2,
    ["canonical-b256-shape", "stride-6", "interior-two-tap"],
  ),
  graphFixture(
    "block-2-stride4-right-padding",
    "block-2-conv-t1",
    61_439,
    1,
    3,
    ["canonical-b256-shape", "stride-4", "right-padding", "last-row"],
  ),
  graphFixture(
    "block-3-stride4-time-tail",
    "block-3-conv-t1",
    12_345,
    17,
    4,
    ["canonical-b256-shape", "stride-4", "two-workgroup-time-tail"],
  ),
  graphFixture(
    "block-4-stride2-longest-tail",
    "block-4-conv-t1",
    491_513,
    7,
    5,
    ["canonical-b256-shape", "stride-2", "longest-output", "last-seven-rows"],
  ),
  arithmeticFixture(),
] satisfies readonly Opt0011ProductionConvTranspose1dFixture[]);

/** SHA-256 over each exact generated WGSL string passed to WebGPU. */
export const OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GENERATED_SHADER_SHA256 =
  Object.freeze({
    "block-0-stride10-left-padding":
      "8c481f64f5039196d80c3deb1663704b500f6eb0e8f6cb87e4cbfd8780cebf51",
    "block-1-stride6-interior":
      "3c90a981ad9052e6e4726a338a91a0e52bd82e28143a4d117e8064de3b50743d",
    "block-2-stride4-right-padding":
      "31f30933f458b83ba79791740016fc63478b9d288e45edecae1c694ed2c463c8",
    "block-3-stride4-time-tail":
      "53118381ede4c2ad87909c796cb808c424e88475de2e12edfb2f20676c16f4d0",
    "block-4-stride2-longest-tail":
      "551673f1bb7079660e13781757b29bd3f85760aafeb7162723616b0359afe760",
    "arithmetic-stride6-cin65-cout9":
      "b3ffcebb134987e30f2866e517bb32131c3c74e89b9e56eb4e0986e49da5baa4",
  } satisfies Readonly<Record<
    Opt0011ProductionConvTranspose1dCaseId,
    string
  >>);

export function parseOpt0011ProductionConvTranspose1dRunIdentity(
  parameters: URLSearchParams,
): Opt0011ProductionConvTranspose1dRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error(
      "OPT-0011 ConvTranspose1D harnessCommit must be lowercase 40-hex",
    );
  }
  const coreCommit = requiredIdentity(parameters, "coreCommit");
  if (coreCommit !== OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT) {
    throw new Error("OPT-0011 production ConvTranspose1D coreCommit changed");
  }
  return Object.freeze({ harnessCommit, coreCommit });
}

export function compareOpt0011ProductionConvTranspose1dRawBits(
  actual: Uint16Array,
  expected: Uint16Array,
): Opt0011ProductionConvTranspose1dRawBitComparison {
  if (actual.constructor !== expected.constructor) {
    throw new Error("OPT-0011 ConvTranspose1D output bit domains differ");
  }
  if (actual.length !== expected.length) {
    throw new Error("OPT-0011 ConvTranspose1D output lengths differ");
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

export function opt0011ProductionConvTranspose1dInputBits(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  globalInputIndex: number,
): number {
  const plan = planAceFp16VaeConvTranspose1d(fixture.shape);
  requireIndex(globalInputIndex, plan.inputElements, `${fixture.id} input`);
  return inputBitsUnchecked(fixture, globalInputIndex);
}

export function opt0011ProductionConvTranspose1dWeightBits(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  weightIndex: number,
): number {
  const plan = planAceFp16VaeConvTranspose1d(fixture.shape);
  requireIndex(weightIndex, plan.weightElements, `${fixture.id} weight`);
  return weightBitsUnchecked(fixture, weightIndex);
}

export function opt0011ProductionConvTranspose1dBiasBits(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  outputChannel: number,
): number {
  requireIndex(
    outputChannel,
    fixture.shape.outputChannels,
    `${fixture.id} bias`,
  );
  return biasBitsUnchecked(fixture, outputChannel);
}

export function opt0011ProductionConvTranspose1dCpuBits(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  outputIndex: number,
): number {
  const plan = planAceFp16VaeConvTranspose1d(fixture.shape);
  requireIndex(outputIndex, plan.outputElements, `${fixture.id} output`);
  const outputChannel = outputIndex % plan.outputChannels;
  const row = Math.floor(outputIndex / plan.outputChannels);
  const outputTime = row % plan.outputFrames;
  const batch = Math.floor(row / plan.outputFrames);
  let sum = float16BitsToNumber(
    biasBitsUnchecked(fixture, outputChannel),
  );
  for (let kernel = 0; kernel < plan.kernelSize; kernel += 1) {
    const paddedOutputTime = outputTime + plan.padding;
    const kernelTime = kernel * plan.dilation;
    if (paddedOutputTime < kernelTime) continue;
    const numerator = paddedOutputTime - kernelTime;
    if (numerator % plan.stride !== 0) continue;
    const inputTime = numerator / plan.stride;
    if (inputTime >= plan.inputFrames) continue;
    const inputBase = (batch * plan.inputFrames + inputTime) *
      plan.inputChannels;
    const weightBase = (outputChannel * plan.kernelSize + kernel) *
      plan.inputChannels;
    for (
      let inputChannel = 0;
      inputChannel < plan.inputChannels;
      inputChannel += 1
    ) {
      const inputOperand = float16BitsToNumber(
        inputBitsUnchecked(fixture, inputBase + inputChannel),
      );
      const weightOperand = float16BitsToNumber(
        weightBitsUnchecked(fixture, weightBase + inputChannel),
      );
      sum = Math.fround(sum + inputOperand * weightOperand);
    }
  }
  return numberToFloat16Bits(sum);
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
    updateProgress("authenticating frozen ConvTranspose1D source and WGSL");
    const heartbeat = startHeartbeat();
    void runBrowser(heartbeat).then(
      (result) => finish("passed", result),
      (error: unknown) => {
        const heartbeatFailure =
          stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure(heartbeat);
        finish("failed", Object.freeze({
          schema:
            "ace-opt-0011-production-fp16-conv-transpose1d-correctness-v1",
          status: "failed",
          experimentId: "OPT-0011",
          error: errorReceipt(error),
          primaryErrorPreservedAcrossHeartbeatStop: true,
          responsiveness: heartbeatFailure.responsiveness,
          heartbeatStopError: heartbeatFailure.heartbeatStopError,
        }));
      },
    );
  }, { once: true });
}

async function runBrowser(
  heartbeat: HeartbeatController,
): Promise<Readonly<Record<string, unknown>>> {
  const identity = parseOpt0011ProductionConvTranspose1dRunIdentity(
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
    label: "ace-opt-0011-production-fp16-conv-transpose1d-correctness-device",
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
  let responsiveness: Readonly<Record<string, unknown>> | undefined;
  let postCleanupValidationFailure: Error | undefined;
  try {
    for (
      let index = 0;
      index < OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.length;
      index += 1
    ) {
      const fixture =
        OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES[index]!;
      updateProgress(
        `correctness ${index + 1}/` +
          `${OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.length}: ` +
          fixture.id,
      );
      cases.push(await runFixture(device, tracker, fixture));
      await yieldToBrowser();
    }
    updateProgress("post-drain real two-quantum cancellation proof");
    cancellation = await runCancellationProof(device, tracker);
    await rawDevice.queue.onSubmittedWorkDone();
    await yieldToBrowser();
  } finally {
    const cleanupStartedAtEpochMilliseconds = Date.now();
    tracker.destroyAll();
    tracker.destroyAll();
    const harnessBuffersDestroyedAtEpochMilliseconds = Date.now();
    const receipt = tracker.receipt();
    const accountingExact =
      receipt.live === 0 && receipt.created === receipt.destroyed;
    destroyingDevice = true;
    rawDevice.destroy();
    const deviceDestroyCalledAtEpochMilliseconds = Date.now();
    const intentionalDeviceLoss = await rawDevice.lost;
    const deviceLossSettledAtEpochMilliseconds = Date.now();
    await yieldToBrowser();
    await yieldToBrowser();
    const postCleanupEventTurnsCompletedAtEpochMilliseconds = Date.now();
    const finalUncapturedErrors = Object.freeze([...uncapturedErrors]);
    const finalRuntimeErrors = Object.freeze([...runtimeErrors]);
    const finalUnexpectedDeviceLoss = unexpectedDeviceLoss;
    const finalEventSnapshotAtEpochMilliseconds = Date.now();
    const cleanupAndEventValidationClean =
      accountingExact && finalUncapturedErrors.length === 0 &&
      finalRuntimeErrors.length === 0 && finalUnexpectedDeviceLoss === null &&
      intentionalDeviceLoss.reason === "destroyed";
    const cleanupAndEventValidationAtEpochMilliseconds = Date.now();
    rawDevice.removeEventListener("uncapturederror", onUncaptured);
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    const eventListenersRemovedAtEpochMilliseconds = Date.now();
    const heartbeatStop =
      stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure(heartbeat);
    responsiveness = heartbeatStop.responsiveness ?? Object.freeze({
      observed: false,
      stopFailed: true,
    });
    const heartbeatStoppedAtEpochMilliseconds = Date.now();
    const postCleanupValidationClean =
      cleanupAndEventValidationClean &&
      heartbeatStop.heartbeatStopError === null &&
      responsiveness.observed === true;
    const postCleanupValidationAtEpochMilliseconds = Date.now();
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
      heartbeat: responsiveness,
      heartbeatStopError: heartbeatStop.heartbeatStopError,
      epochs: Object.freeze({
        cleanupStartedAtEpochMilliseconds,
        harnessBuffersDestroyedAtEpochMilliseconds,
        deviceDestroyCalledAtEpochMilliseconds,
        deviceLossSettledAtEpochMilliseconds,
        postCleanupEventTurnsCompletedAtEpochMilliseconds,
        finalEventSnapshotAtEpochMilliseconds,
        cleanupAndEventValidationAtEpochMilliseconds,
        eventListenersRemovedAtEpochMilliseconds,
        heartbeatStoppedAtEpochMilliseconds,
        postCleanupValidationAtEpochMilliseconds,
      }),
    });
    if (!postCleanupValidationClean) {
      postCleanupValidationFailure = new Error(
        "OPT-0011 ConvTranspose1D post-cleanup validation failed",
      );
    }
  }
  if (postCleanupValidationFailure !== undefined) {
    throw postCleanupValidationFailure;
  }
  return Object.freeze({
    schema: "ace-opt-0011-production-fp16-conv-transpose1d-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification: "correctness-only-no-timing-or-thermal-claim",
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter, rawDevice),
    }),
    protocol: Object.freeze({
      requiredFeature: "shader-f16",
      productionKernel:
        ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
      authority:
        "independent-source-order-fp16-bits-to-fp32-cpu-to-rne-fp16",
      allFiveCanonicalB256OperationsAuthenticated: true,
      exactB256GraphQuantumCount: 322,
      representativeCanonicalCompleteRowRanges: true,
      fullSelectedRangeRawU16Comparison: true,
      deterministicRerunPerRange: true,
      onePreparedFixtureResidentAtATime: true,
      oneOutstandingCommandBuffer: true,
      drainAndRealQueueEmptyTurnAfterEveryExecution: true,
      qNaNPrefillCanariesGuardsAndSourcePadding: true,
      compilationUploadAndWallTimeReported: false,
      performanceClaim: null,
      thermalClaim: null,
      qualityClaim: null,
      productionSelectorClaim: null,
    }),
    sourceAuthority,
    graphCoverage: graphCoverageReceipt(),
    cases: Object.freeze(cases),
    cancellation,
    responsiveness,
    cleanup,
  });
}

async function authenticateSources(
  identity: Opt0011ProductionConvTranspose1dRunIdentity,
): Promise<Readonly<Record<string, unknown>>> {
  const encoder = new TextEncoder();
  const coreSourceSha256 = await sha256Hex(
    encoder.encode(productionCoreSource),
  );
  if (
    coreSourceSha256 !==
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_SOURCE_SHA256
  ) {
    throw new Error(
      "OPT-0011 rejected unauthenticated production ConvTranspose1D source",
    );
  }
  const shaders: Record<string, unknown> = {};
  for (const fixture of OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES) {
    const source = aceFp16VaeConvTranspose1dWgsl(fixture.shape);
    const bytes = encoder.encode(source);
    const sha256 = await sha256Hex(bytes);
    if (
      sha256 !==
        OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GENERATED_SHADER_SHA256[
          fixture.id
        ]
    ) {
      throw new Error(`${fixture.id} generated shader SHA-256 changed`);
    }
    shaders[fixture.id] = Object.freeze({ sha256, bytes: bytes.byteLength });
  }
  return Object.freeze({
    ...identity,
    coreSourceSha256,
    generatedShaderHashesFrozenAndVerifiedBeforeExecution: true,
    generatedShaders: Object.freeze(shaders),
  });
}

async function runFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: Opt0011ProductionConvTranspose1dFixture,
): Promise<Readonly<Record<string, unknown>>> {
  const prepared = await prepareFixture(device, tracker, fixture);
  try {
    const ranges: unknown[] = [];
    let comparedElementCount = 0;
    for (const [index, range] of fixture.ranges.entries()) {
      const first = await executeAndRead(
        device,
        tracker,
        prepared,
        index,
        `${fixture.id}-range-${index}-first`,
      );
      const rerun = await executeAndRead(
        device,
        tracker,
        prepared,
        index,
        `${fixture.id}-range-${index}-rerun`,
      );
      if (first.readback.sha256 !== rerun.readback.sha256) {
        throw new Error(
          `${fixture.id} range ${index} changed on deterministic rerun`,
        );
      }
      ranges.push(Object.freeze({
        base: range.base,
        count: range.count,
        rangePlan: rangePlanReceipt(
          planAceFp16VaeConvTranspose1dRange(prepared.plan, range),
        ),
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
    return Object.freeze({
      id: fixture.id,
      graphOperationIndex: fixture.graphOperationIndex,
      graphOperationOrdinal: fixture.graphOperationOrdinal,
      graphOperationLabel: fixture.graphOperationLabel,
      containingGraphQuantumIndex: fixture.containingGraphQuantumIndex,
      containingGraphOperationQuantumIndex:
        fixture.containingGraphOperationQuantumIndex,
      shape: fixture.shape,
      coverage: fixture.coverage,
      sourcePatternSalt: fixture.sourcePatternSalt,
      plan: planReceipt(prepared.plan),
      uploadIdentity: prepared.uploadIdentity,
      ranges: Object.freeze(ranges),
      rangeCount: fixture.ranges.length,
      comparedElementCount,
      completeSelectedRangeRawU16Comparison: true,
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
  fixture: Opt0011ProductionConvTranspose1dFixture,
): Promise<PreparedFixture> {
  const plan = planAceFp16VaeConvTranspose1d(fixture.shape);
  const kernel = AceFp16VaeConvTranspose1dKernel.create(device);
  const owned: GPUBuffer[] = [];
  let destroyed = false;
  try {
    const activeInputRows = selectedInputRows(plan, fixture.ranges);
    const input = createInputUpload(
      device,
      tracker,
      fixture,
      plan,
      activeInputRows,
    );
    owned.push(input);
    const weight = createPeriodicUpload(
      device,
      tracker,
      `${fixture.id}-weight`,
      plan.weightElements,
      plan.weightBindingBytes,
      WEIGHT_PATTERN,
      fixture.sourcePatternSalt * 11,
    );
    owned.push(weight);
    const bias = createPeriodicUpload(
      device,
      tracker,
      `${fixture.id}-bias`,
      plan.outputChannels,
      plan.biasBindingBytes,
      BIAS_PATTERN,
      fixture.sourcePatternSalt * 5,
    );
    owned.push(bias);
    const output = createOutputTarget(
      device,
      tracker,
      `${fixture.id}-output`,
      plan,
      fixture.ranges,
    );
    owned.push(output.buffer);
    const prefill = createPrefillTemplate(
      device,
      tracker,
      `${fixture.id}-prefill`,
      maximumSelectedCopyBytes(fixture.ranges),
    );
    owned.push(prefill.buffer);
    const controls = createRangeControlBuffer(
      device,
      tracker,
      `${fixture.id}-controls`,
      fixture.ranges,
    );
    owned.push(controls.buffer);
    const dispatches: AceFp16VaeConvTranspose1dDispatch[] = [];
    for (const [index, range] of fixture.ranges.entries()) {
      const dispatch = await kernel.createDispatch(
        `${fixture.id}-range-${index}`,
        fixture.shape,
        {
          input: binding(input, plan.inputBindingBytes),
          weight: binding(weight, plan.weightBindingBytes),
          bias: binding(bias, plan.biasBindingBytes),
          output: output.binding,
        },
        rangeBinding(range, controls.bindings[index]!),
      );
      assertDispatchRange(dispatch, plan, range, fixture.id);
      dispatches.push(dispatch);
    }
    return Object.freeze({
      fixture,
      plan,
      dispatches: Object.freeze(dispatches),
      output,
      prefill,
      uploadIdentity: Object.freeze({
        recipe:
          "zero-initialized-input-plus-selected-rows-and-complete-periodic-native-f16-parameters-v1",
        activeInputRowCount: activeInputRows.length,
        activeInputRows: Object.freeze(activeInputRows),
        inputBindingBytes: plan.inputBindingBytes,
        weightBindingBytes: plan.weightBindingBytes,
        biasBindingBytes: plan.biasBindingBytes,
        completeConverterNativeWeightUpload: true,
        sourcePaddingF16: hex16(SOURCE_PADDING_F16),
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
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  encoder.copyBufferToBuffer(
    prepared.prefill.buffer,
    0,
    prepared.output.buffer,
    OUTPUT_GUARD_BYTES + range.base * FLOAT16_BYTES,
    range.count * FLOAT16_BYTES,
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
  fixture: Opt0011ProductionConvTranspose1dFixture,
  plan: AceFp16VaeConvTranspose1dPlan,
  selected: Readonly<{ readonly base: number; readonly count: number }>,
  target: OutputTarget,
  label: string,
): Promise<SelectedReadback> {
  const selectedBytes = selected.count * FLOAT16_BYTES;
  const selectedPayloadOffset = selected.base * FLOAT16_BYTES;
  const selectedSourceOffset = OUTPUT_GUARD_BYTES + selectedPayloadOffset;
  if (selectedSourceOffset % 4 !== 0 || selectedBytes % 4 !== 0) {
    throw new RangeError("OPT-0011 ConvTranspose1D readback is unaligned");
  }
  const beforeBytes = Math.min(OUTPUT_CANARY_BYTES, selectedPayloadOffset);
  const afterBytes = Math.min(
    OUTPUT_CANARY_BYTES,
    plan.outputBindingBytes - selectedPayloadOffset - selectedBytes,
  );
  if (beforeBytes % 4 !== 0 || afterBytes % 4 !== 0) {
    throw new RangeError("OPT-0011 ConvTranspose1D canary is unaligned");
  }
  const prefixOffset = 0;
  const suffixOffset = OUTPUT_GUARD_BYTES;
  const selectedOffset = OUTPUT_GUARD_BYTES * 2;
  const beforeOffset = selectedOffset + selectedBytes;
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
      selectedBytes,
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
        selectedSourceOffset + selectedBytes,
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
      : new Uint16Array(mappedRange, beforeOffset, beforeBytes / 2);
    const after = afterBytes === 0
      ? new Uint16Array(0)
      : new Uint16Array(mappedRange, afterOffset, afterBytes / 2);
    const selectedBits = new Uint16Array(
      mappedRange,
      selectedOffset,
      selected.count,
    );
    const expectedBits = new Uint16Array(selected.count);
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let prefillQNaNCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let subnormalCount = 0;
    for (let index = 0; index < selected.count; index += 1) {
      const actual = selectedBits[index]!;
      expectedBits[index] = opt0011ProductionConvTranspose1dCpuBits(
        fixture,
        selected.base + index,
      );
      const value = float16BitsToNumber(actual);
      if (Number.isFinite(value)) finiteCount += 1;
      else nonFiniteCount += 1;
      if (actual === OUTPUT_PREFILL_QNAN_F16) prefillQNaNCount += 1;
      if ((actual & 0x7fff) === 0) {
        if ((actual & 0x8000) === 0) positiveZeroCount += 1;
        else negativeZeroCount += 1;
      }
      if ((actual & 0x7c00) === 0 && (actual & 0x03ff) !== 0) {
        subnormalCount += 1;
      }
    }
    const comparison = compareOpt0011ProductionConvTranspose1dRawBits(
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
    if (
      !prefixGuardUntouched || !suffixGuardUntouched ||
      !adjacentCanariesUntouched || comparison.mismatchCount !== 0 ||
      nonFiniteCount !== 0 || finiteCount !== selected.count ||
      prefillQNaNCount !== 0
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
      selectedBytes,
    ).slice();
    return Object.freeze({
      sha256: await sha256Hex(raw),
      scan: Object.freeze({
        selectedElementCount: selected.count,
        finiteCount,
        nonFiniteCount,
        prefillQNaNCount,
        positiveZeroCount,
        negativeZeroCount,
        subnormalCount,
        externalPrefixQNaNGuardUntouched: prefixGuardUntouched,
        externalSuffixQNaNGuardUntouched: suffixGuardUntouched,
        adjacentBeforeQNaNCanaryBytes: beforeBytes,
        adjacentAfterQNaNCanaryBytes: afterBytes,
        adjacentQNaNCanariesUntouched: adjacentCanariesUntouched,
        completeSelectedRangeReadback: true,
        readbackCommandBuffers: 1,
        readbackSubmissions: 1,
        maps: 1,
        unmaps: 1,
      }),
      cpu: Object.freeze({
        oracle:
          "independent-kernel-ascending-then-cin-ascending-fp32-cpu-rne-f16",
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
  fixture: Opt0011ProductionConvTranspose1dFixture,
  plan: AceFp16VaeConvTranspose1dPlan,
  activeRows: readonly number[],
): GPUBuffer {
  const buffer = tracker.create(device, {
    label: `${fixture.id}-input`,
    size: plan.inputBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const bits = new Uint16Array(buffer.getMappedRange());
    for (const row of activeRows) {
      const base = row * plan.inputChannels;
      for (let channel = 0; channel < plan.inputChannels; channel += 1) {
        bits[base + channel] = inputBitsUnchecked(fixture, base + channel);
      }
    }
    bits.fill(SOURCE_PADDING_F16, plan.inputElements);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createPeriodicUpload(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  bindingBytes: number,
  pattern: readonly number[],
  shift: number,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const bits = new Uint16Array(buffer.getMappedRange());
    bits.fill(SOURCE_PADDING_F16, elements);
    const period = new Uint16Array(pattern.length);
    for (let index = 0; index < period.length; index += 1) {
      period[index] = pattern[(index + shift) % pattern.length]!;
    }
    fillPeriodicPrefix(bits, period, elements);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function fillPeriodicPrefix(
  destination: Uint16Array,
  period: Uint16Array,
  elements: number,
): void {
  if (elements > destination.length || period.length < 1) {
    throw new RangeError("OPT-0011 ConvTranspose1D upload geometry changed");
  }
  const initial = Math.min(period.length, elements);
  destination.set(period.subarray(0, initial), 0);
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
  plan: AceFp16VaeConvTranspose1dPlan,
  ranges: readonly Readonly<{ readonly base: number; readonly count: number }>[],
): OutputTarget {
  const buffer = tracker.create(device, {
    label,
    size: OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  try {
    const guard = new Uint16Array(OUTPUT_GUARD_BYTES / 2);
    guard.fill(OUTPUT_GUARD_F16);
    device.queue.writeBuffer(buffer, 0, guard);
    device.queue.writeBuffer(
      buffer,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
      guard,
    );
    const canary = new Uint16Array(OUTPUT_CANARY_BYTES / 2);
    canary.fill(OUTPUT_CANARY_F16);
    for (const range of ranges) {
      const start = range.base * FLOAT16_BYTES;
      const end = start + range.count * FLOAT16_BYTES;
      const before = Math.min(OUTPUT_CANARY_BYTES, start);
      const after = Math.min(
        OUTPUT_CANARY_BYTES,
        plan.outputBindingBytes - end,
      );
      if (before > 0) {
        device.queue.writeBuffer(
          buffer,
          OUTPUT_GUARD_BYTES + start - before,
          canary,
          0,
          before / FLOAT16_BYTES,
        );
      }
      if (after > 0) {
        device.queue.writeBuffer(
          buffer,
          OUTPUT_GUARD_BYTES + end,
          canary,
          0,
          after / FLOAT16_BYTES,
        );
      }
    }
    return Object.freeze({
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OUTPUT_GUARD_BYTES,
        size: plan.outputBindingBytes,
      }),
    });
  } catch (error) {
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
    throw new RangeError("OPT-0011 ConvTranspose1D prefill bytes changed");
  }
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  try {
    new Uint16Array(buffer.getMappedRange()).fill(OUTPUT_PREFILL_QNAN_F16);
    buffer.unmap();
    return Object.freeze({ buffer, bytes });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
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
    throw new RangeError("OPT-0011 ConvTranspose1D control alignment changed");
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
  const graphCase = OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES[0]!;
  const graphRanges = graphCase.ranges.slice(0, 2);
  if (graphRanges.length !== 2) {
    throw new Error("OPT-0011 cancellation lost its two real graph ranges");
  }
  const fixture = Object.freeze({
    id: "block-0-stride10-left-padding",
    graphOperationIndex: graphCase.operationIndex,
    graphOperationOrdinal: graphCase.operationOrdinal,
    graphOperationLabel: graphCase.label,
    containingGraphQuantumIndex: graphRanges[0]!.quantumIndex,
    containingGraphOperationQuantumIndex: 0,
    shape: graphCase.shape,
    ranges: Object.freeze(graphRanges.map(({ base, count }) =>
      Object.freeze({ base, count })
    )),
    sourcePatternSalt: 1,
    coverage: Object.freeze(["real-two-range-cancellation"]),
  } satisfies Opt0011ProductionConvTranspose1dFixture);
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
        label: `opt-0011-conv-transpose1d-cancel-${index}-encoder`,
      });
      encoder.copyBufferToBuffer(
        prepared.prefill.buffer,
        0,
        prepared.output.buffer,
        OUTPUT_GUARD_BYTES + range.base * FLOAT16_BYTES,
        range.count * FLOAT16_BYTES,
      );
      prefillCopyCount += 1;
      const pass = encoder.beginComputePass({
        label: `opt-0011-conv-transpose1d-cancel-${index}-pass`,
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
      controller.abort("cancel-after-first-drained-real-b256-transpose-range");
    }
    if (!controller.signal.aborted) {
      readbackCount += 1;
    }
    if (
      encodeCount !== 1 || submitCount !== 1 || drainCount !== 1 ||
      prefillCopyCount !== 1 || skippedRangeCount !== 1 ||
      readbackCount !== 0 || !idleTurnDelivered
    ) {
      throw new Error("OPT-0011 ConvTranspose1D cancellation proof changed");
    }
    return Object.freeze({
      source: "two-real-unchanged-b256-conv-transpose1d-graph-ranges",
      operationLabel: graphCase.label,
      quantumIndices: Object.freeze(graphRanges.map(({ quantumIndex }) =>
        quantumIndex
      )),
      abortReason: controller.signal.reason,
      abortObservedAfterDrain: true,
      encodeCount,
      submitCount,
      drainCount,
      prefillCopyCount,
      skippedRangeCount,
      readbackCount,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
      readbackPrevented: true,
      realQueueEmptyIdleTurnDelivered: idleTurnDelivered,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

function buildGraphCases(): readonly Opt0011ProductionConvTranspose1dGraphCase[] {
  const operations = B256_GRAPH.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is {
      readonly operation: AceVaeDecoderConvTransposeOperation;
      readonly operationIndex: number;
    } => entry.operation.kind === "conv-transpose1d");
  if (operations.length !== 5) {
    throw new Error("OPT-0011 expected five B-256 ConvTranspose1D operations");
  }
  const cases = operations.map(({ operation, operationIndex }, ordinal) => {
    const plan = planAceFp16VaeConvTranspose1d(operation.shape);
    let cursor = 0;
    let operationQuantumIndex = 0;
    const ranges = B256_QUANTA.quanta
      .filter((quantum) => quantum.operationIndex === operationIndex)
      .map((quantum) => {
        const range = graphRange(
          operation,
          quantum,
          operationQuantumIndex++,
        );
        if (range.base !== cursor) {
          throw new Error(`${operation.label} graph range continuity changed`);
        }
        cursor += range.count;
        planAceFp16VaeConvTranspose1dRange(plan, range);
        return range;
      });
    if (ranges.length < 1 || cursor !== plan.outputElements) {
      throw new Error(`${operation.label} graph coverage changed`);
    }
    return Object.freeze({
      operationIndex,
      operationOrdinal: ordinal,
      label: operation.label,
      shape: operation.shape,
      ranges: Object.freeze(ranges),
    });
  });
  if (cases.reduce((sum, entry) => sum + entry.ranges.length, 0) !== 322) {
    throw new Error("OPT-0011 B-256 ConvTranspose1D quantum count changed");
  }
  return Object.freeze(cases);
}

function graphRange(
  operation: AceVaeDecoderConvTransposeOperation,
  quantum: AceVaeDecoderQuantumPlan,
  operationQuantumIndex: number,
): Opt0011ProductionConvTranspose1dGraphRange {
  if (
    quantum.operationKind !== "conv-transpose1d" ||
    quantum.operationLabel !== operation.label ||
    quantum.primitives.length !== 1
  ) {
    throw new Error(`${operation.label} quantum topology changed`);
  }
  const primitive = quantum.primitives[0]!;
  if (
    primitive.physicalPartIndex !== 0 ||
    primitive.firstOutputChannel !== 0 ||
    primitive.outputChannels !== operation.shape.outputChannels ||
    primitive.outputBase !== quantum.logicalOutputBase ||
    primitive.outputCount !== quantum.logicalOutputCount
  ) {
    throw new Error(`${operation.label} primitive topology changed`);
  }
  return Object.freeze({
    quantumIndex: quantum.index,
    operationQuantumIndex,
    base: primitive.outputBase,
    count: primitive.outputCount,
    firstOutputRow: primitive.outputBase / operation.shape.outputChannels,
    outputRowCount: primitive.outputCount / operation.shape.outputChannels,
  });
}

function graphFixture(
  id: Opt0011ProductionConvTranspose1dCaseId,
  operationLabel: string,
  firstOutputTime: number,
  outputRowCount: number,
  sourcePatternSalt: number,
  coverage: readonly string[],
): Opt0011ProductionConvTranspose1dFixture {
  const graphCase = OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.find(
    ({ label }) => label === operationLabel,
  );
  if (graphCase === undefined) {
    throw new Error(`Missing canonical transpose operation ${operationLabel}`);
  }
  const base = firstOutputTime * graphCase.shape.outputChannels;
  const count = outputRowCount * graphCase.shape.outputChannels;
  const containing = graphCase.ranges.find((range) =>
    base >= range.base && base + count <= range.base + range.count
  );
  if (containing === undefined) {
    throw new Error(`${id} representative range crosses a graph quantum`);
  }
  const plan = planAceFp16VaeConvTranspose1d(graphCase.shape);
  planAceFp16VaeConvTranspose1dRange(plan, { base, count });
  return Object.freeze({
    id,
    graphOperationIndex: graphCase.operationIndex,
    graphOperationOrdinal: graphCase.operationOrdinal,
    graphOperationLabel: graphCase.label,
    containingGraphQuantumIndex: containing.quantumIndex,
    containingGraphOperationQuantumIndex: containing.operationQuantumIndex,
    shape: graphCase.shape,
    ranges: Object.freeze([Object.freeze({ base, count })]),
    sourcePatternSalt,
    coverage: Object.freeze([...coverage]),
  });
}

function arithmeticFixture(): Opt0011ProductionConvTranspose1dFixture {
  const shape = Object.freeze({
    batch: 1,
    inputFrames: 17,
    inputChannels: 65,
    outputChannels: 9,
    kernelSize: 12,
    stride: 6,
    dilation: 1,
    padding: 3,
    outputPadding: 0,
  });
  const plan = planAceFp16VaeConvTranspose1d(shape);
  return Object.freeze({
    id: "arithmetic-stride6-cin65-cout9",
    graphOperationIndex: null,
    graphOperationOrdinal: null,
    graphOperationLabel: null,
    containingGraphQuantumIndex: null,
    containingGraphOperationQuantumIndex: null,
    shape,
    ranges: Object.freeze([Object.freeze({
      base: 0,
      count: plan.outputElements,
    })]),
    sourcePatternSalt: 6,
    coverage: Object.freeze([
      "complete-arithmetic-domain",
      "stride-6",
      "left-and-right-padding",
      "cin-65-chunk-tail",
      "cout-9-channel-tail",
      "output-time-tail",
    ]),
  });
}

function selectedInputRows(
  plan: AceFp16VaeConvTranspose1dPlan,
  ranges: readonly Readonly<{ readonly base: number; readonly count: number }>[],
): readonly number[] {
  const rows = new Set<number>();
  for (const range of ranges) {
    const rangePlan = planAceFp16VaeConvTranspose1dRange(plan, range);
    for (
      let offset = 0;
      offset < rangePlan.outputRowCount;
      offset += 1
    ) {
      const outputTime = rangePlan.firstOutputTime + offset;
      for (let kernel = 0; kernel < plan.kernelSize; kernel += 1) {
        const paddedOutputTime = outputTime + plan.padding;
        const kernelTime = kernel * plan.dilation;
        if (paddedOutputTime < kernelTime) continue;
        const numerator = paddedOutputTime - kernelTime;
        if (numerator % plan.stride !== 0) continue;
        const inputTime = numerator / plan.stride;
        if (inputTime >= plan.inputFrames) continue;
        rows.add(rangePlan.batch * plan.inputFrames + inputTime);
      }
    }
  }
  return Object.freeze([...rows].sort((left, right) => left - right));
}

function inputBitsUnchecked(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  globalInputIndex: number,
): number {
  return INPUT_PATTERN[
    (globalInputIndex + fixture.sourcePatternSalt * 7) % INPUT_PATTERN.length
  ]!;
}

function weightBitsUnchecked(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  weightIndex: number,
): number {
  return WEIGHT_PATTERN[
    (weightIndex + fixture.sourcePatternSalt * 11) % WEIGHT_PATTERN.length
  ]!;
}

function biasBitsUnchecked(
  fixture: Opt0011ProductionConvTranspose1dFixture,
  outputChannel: number,
): number {
  return BIAS_PATTERN[
    (outputChannel + fixture.sourcePatternSalt * 5) % BIAS_PATTERN.length
  ]!;
}

function graphCoverageReceipt(): Readonly<Record<string, unknown>> {
  return Object.freeze({
    decoderInputFrames: B256_GRAPH.inputFrames,
    decoderOutputFrames: B256_GRAPH.outputFrames,
    operationCount: OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.length,
    exactQuantumCount:
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.reduce(
        (sum, entry) => sum + entry.ranges.length,
        0,
      ),
    operationRangeCounts: Object.freeze(
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.map((entry) =>
        Object.freeze({ label: entry.label, count: entry.ranges.length })
      ),
    ),
    exactOperationRangeTopology: Object.freeze(
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.map((entry) =>
        Object.freeze({
          operationIndex: entry.operationIndex,
          operationOrdinal: entry.operationOrdinal,
          label: entry.label,
          shape: entry.shape,
          ranges: entry.ranges,
        })
      ),
    ),
    everyPrimitiveCompleteOutputAxis: true,
    everyRangeAuthenticatedWithoutRepartition: true,
  });
}

function assertDispatchRange(
  dispatch: AceFp16VaeConvTranspose1dDispatch,
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  label: string,
): void {
  const expected = planAceFp16VaeConvTranspose1dRange(plan, range);
  if (
    dispatch.outputRange.base !== expected.base ||
    dispatch.outputRange.count !== expected.count ||
    dispatch.outputRange.batch !== expected.batch ||
    dispatch.outputRange.firstOutputTime !== expected.firstOutputTime ||
    dispatch.outputRange.outputRowCount !== expected.outputRowCount ||
    dispatch.outputRange.workgroupsX !== expected.workgroupsX ||
    dispatch.outputRange.workgroupsY !== expected.workgroupsY
  ) {
    throw new Error(`${label} production dispatch range changed`);
  }
}

function rangePlanReceipt(
  range: ReturnType<typeof planAceFp16VaeConvTranspose1dRange>,
): Readonly<Record<string, number>> {
  return Object.freeze({
    base: range.base,
    count: range.count,
    batch: range.batch,
    firstOutputTime: range.firstOutputTime,
    firstOutputRow: range.firstOutputRow,
    outputRowCount: range.outputRowCount,
    workgroupsX: range.workgroupsX,
    workgroupsY: range.workgroupsY,
  });
}

function planReceipt(
  plan: AceFp16VaeConvTranspose1dPlan,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    batch: plan.batch,
    inputFrames: plan.inputFrames,
    outputFrames: plan.outputFrames,
    inputChannels: plan.inputChannels,
    outputChannels: plan.outputChannels,
    kernelSize: plan.kernelSize,
    stride: plan.stride,
    dilation: plan.dilation,
    padding: plan.padding,
    outputPadding: plan.outputPadding,
    inputElements: plan.inputElements,
    weightElements: plan.weightElements,
    outputElements: plan.outputElements,
    inputBindingBytes: plan.inputBindingBytes,
    weightBindingBytes: plan.weightBindingBytes,
    biasBindingBytes: plan.biasBindingBytes,
    outputBindingBytes: plan.outputBindingBytes,
    inputChannelChunkCount: plan.inputChannelChunkCount,
    workgroupStorageBytes: plan.workgroupStorageBytes,
  });
}

function maximumSelectedCopyBytes(
  ranges: readonly Readonly<{ readonly count: number }>[],
): number {
  return Math.max(...ranges.map(({ count }) => count * FLOAT16_BYTES));
}

function requiredDeviceLimits(): Record<string, number> {
  let maximumBuffer = 0;
  let maximumStorageBinding = 0;
  let maximumDispatch = 1;
  for (const fixture of OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES) {
    const plan = planAceFp16VaeConvTranspose1d(fixture.shape);
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    );
    for (const range of fixture.ranges) {
      const planned = planAceFp16VaeConvTranspose1dRange(plan, range);
      maximumDispatch = Math.max(
        maximumDispatch,
        planned.workgroupsX,
        planned.workgroupsY,
      );
    }
  }
  const cancellation = OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES[0]!;
  const cancellationPlan = planAceFp16VaeConvTranspose1d(cancellation.shape);
  for (const range of cancellation.ranges.slice(0, 2)) {
    const planned = planAceFp16VaeConvTranspose1dRange(
      cancellationPlan,
      range,
    );
    maximumDispatch = Math.max(
      maximumDispatch,
      planned.workgroupsX,
      planned.workgroupsY,
    );
  }
  return {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: CONTROL_BYTES,
    maxComputeWorkgroupStorageSize: 3_216,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 16,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error(
      "OPT-0011 ConvTranspose1D gate requires adapter shader-f16",
    );
  }
  for (const [name, minimum] of Object.entries(requiredDeviceLimits())) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0011 ConvTranspose1D adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > OUTPUT_GUARD_BYTES) {
    throw new RangeError(
      "OPT-0011 ConvTranspose1D output guard is below storage alignment",
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
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    minStorageBufferOffsetAlignment: limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
  });
}

export function stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure(
  heartbeat: HeartbeatController,
): Opt0011ProductionConvTranspose1dHeartbeatFailureStop {
  try {
    return Object.freeze({
      responsiveness: heartbeat.stop(),
      heartbeatStopError: null,
    });
  } catch (error) {
    return Object.freeze({
      responsiveness: null,
      heartbeatStopError: errorReceipt(error),
    });
  }
}

function startHeartbeat(): HeartbeatController {
  const startedAtEpochMilliseconds = Date.now();
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let animationFrame = 0;
  const animate = (now: number): void => {
    if (stopped) return;
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrameCount += 1;
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
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
        startedAtEpochMilliseconds,
        stoppedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
        observed: animationFrameCount > 0 && timerTickCount > 0,
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

function rangeBinding(
  range: Readonly<{ readonly base: number; readonly count: number }>,
  control: GPUBufferBinding,
): AceVaeOutputRangeBinding {
  return Object.freeze({ base: range.base, count: range.count, control });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function requiredIdentity(parameters: URLSearchParams, name: string): string {
  const values = parameters.getAll(name);
  if (values.length !== 1 || values[0] === "") {
    throw new Error(`OPT-0011 ConvTranspose1D requires one ${name}`);
  }
  return values[0]!;
}

function requireIndex(index: number, exclusiveEnd: number, label: string): void {
  if (
    !Number.isSafeInteger(index) || index < 0 || index >= exclusiveEnd
  ) {
    throw new RangeError(`${label} index is outside its raw-bit domain`);
  }
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
  return value === null
    ? "not-applicable"
    : `0x${value.toString(16).padStart(4, "0")}`;
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

export function parseOpt0011ProductionConvTranspose1dRawResultChunkOffset(
  value: string,
): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error(
      "OPT-0011 ConvTranspose1D raw-result offset is not canonical decimal",
    );
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error(
      "OPT-0011 ConvTranspose1D raw-result offset is not a safe integer",
    );
  }
  return offset;
}

export function sliceOpt0011ProductionConvTranspose1dRawResultChunk(
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
    throw new Error("OPT-0011 ConvTranspose1D raw-result offset is invalid");
  }
  if (
    offset > 0 && offset < rawResultJson.length &&
    isHighSurrogate(rawResultJson.charCodeAt(offset - 1)) &&
    isLowSurrogate(rawResultJson.charCodeAt(offset))
  ) {
    throw new Error(
      "OPT-0011 ConvTranspose1D raw-result offset splits a surrogate pair",
    );
  }
  let end = Math.min(
    offset +
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS,
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
        throw new Error(
          "OPT-0011 ConvTranspose1D raw result is not available yet",
        );
      }
      const slice = sliceOpt0011ProductionConvTranspose1dRawResultChunk(
        rawResultJson,
        parseOpt0011ProductionConvTranspose1dRawResultChunkOffset(
          offsetInput.value,
        ),
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
    throw new Error("OPT-0011 ConvTranspose1D could not publish raw receipt");
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
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS,
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
    throw new Error(`Missing OPT-0011 ConvTranspose1D element ${selector}`);
  }
  return element;
}
