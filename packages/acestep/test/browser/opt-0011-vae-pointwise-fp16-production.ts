/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import productionCoreSource from
  "../../src/webgpu/kernels/vae-pointwise-fp16.ts?raw";
import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
  AceFp16VaePointwiseKernel,
  aceFp16VaeAddWgsl,
  aceFp16VaeIngressWgsl,
  planAceFp16VaeAdd,
  planAceFp16VaeIngress,
  planAceFp16VaePointwiseRange,
  type AceFp16VaeAddDispatch,
  type AceFp16VaeIngressDispatch,
  type AceFp16VaePointwisePlan,
} from "../../src/webgpu/kernels/vae-pointwise-fp16.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderAddOperation,
  type AceVaeDecoderQuantumPlan,
} from "../../src/webgpu/vae-decoder.js";

export const OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT =
  "dd36a04960f846e53c2fd948d67b9aa9ddced4f2" as const;
export const OPT_0011_PRODUCTION_POINTWISE_CORE_SOURCE_SHA256 =
  "c801eb209132ed2705a3b7e7b742afd2a6b17855d257938b5df515b6285f3eab" as const;

/** SHA-256 over the exact generated WGSL passed to the browser shader modules. */
export const OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256 =
  Object.freeze({
    ingress: "750bdf07e86c2cfd639eb1217f11d35408d444c8dc5460ca067a6c6d656f7d16",
    add: "9998dbcc049a1795a0fb6df16e6d404f541d5cbc5d486515b869b4337a528eb5",
  });

export interface Opt0011ProductionPointwiseRunIdentity {
  readonly harnessCommit: string;
  readonly coreCommit: typeof OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT;
}

export interface Opt0011ProductionPointwiseRawBitComparison {
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
}

export interface Opt0011ProductionPointwiseGraphAddCase {
  readonly id: string;
  readonly operationIndex: number;
  readonly addOrdinal: number;
  readonly shapeFamilyIndex: number;
  readonly shape: AceVaePointwiseShape;
  readonly sourcePatternSalt: number;
  readonly ranges: readonly Readonly<{
    readonly quantumIndex: number;
    readonly operationQuantumIndex: number;
    readonly base: number;
    readonly count: number;
  }>[];
}

interface PointwiseFixture {
  readonly id: string;
  readonly operation: "ingress" | "add";
  readonly shape: AceVaePointwiseShape;
  readonly sourcePatternSalt: number;
  readonly ranges: readonly Readonly<{
    readonly base: number;
    readonly count: number;
  }>[];
  readonly coverage: readonly string[];
}

const FLOAT16_BYTES = 2;
const CONTROL_BYTES = 16;
const OUTPUT_GUARD_BYTES = 256;
const OUTPUT_CANARY_BYTES = 256;
const OUTPUT_GUARD_F16 = 0x7e33;
const OUTPUT_CANARY_F16 = 0x7e11;
const SOURCE_PADDING_F16 = 0x7e55;
const QUEUE_EMPTY_IDLE_MILLISECONDS = 1;
const OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_GLOBAL =
  "__ACE_OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_JSON__";
export const OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS =
  100_000;

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

const INGRESS_F32_PATTERN = Object.freeze([
  0x0000_0000,
  0x8000_0000,
  0x3300_0000,
  0x33c0_0000,
  0x3380_0000,
  0xb380_0000,
  0x3f80_1000,
  0x3f80_3000,
  0xbf80_1000,
  0xbf80_3000,
  0x387f_e000,
  0x477f_e000,
  0x3eaa_a000,
  0xbeaa_a000,
  0x3a80_0000,
  0xba80_0000,
]);

const ADD_LEFT_F16_PATTERN = Object.freeze([
  0x8000, 0x0000, 0x0001, 0x0001,
  0x03ff, 0x3c00, 0x3c01, 0xbc00,
  0xbc01, 0x0400, 0x3555, 0xb555,
  0x2400, 0xa400, 0x3a00, 0xba00,
]);
const ADD_RIGHT_F16_PATTERN = Object.freeze([
  0x8000, 0x8000, 0x0001, 0x8001,
  0x0001, 0x1000, 0x1000, 0x9000,
  0x9000, 0x8001, 0x1555, 0x1555,
  0xa400, 0x2400, 0x1800, 0x9800,
]);

const B256_GRAPH = planAceVaeDecoder(256);
const B256_QUANTA = planAceVaeDecoderQuanta(B256_GRAPH);

export const OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES =
  buildGraphAddCases();

export const OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES = Object.freeze([
  Object.freeze({
    id: "ingress-b256-complete",
    operation: "ingress",
    shape: Object.freeze({ batch: 1, frames: 256, channels: 64 }),
    sourcePatternSalt: 1,
    ranges: Object.freeze([Object.freeze({ base: 0, count: 16_384 })]),
    coverage: Object.freeze([
      "complete-b256-decoder-ingress",
      "fp32-to-fp16-raw-bits",
    ]),
  }),
  Object.freeze({
    id: "ingress-arithmetic-odd-tail-257",
    operation: "ingress",
    shape: Object.freeze({ batch: 1, frames: 1, channels: 257 }),
    sourcePatternSalt: 0,
    ranges: Object.freeze([
      Object.freeze({ base: 0, count: 256 }),
      Object.freeze({ base: 256, count: 1 }),
    ]),
    coverage: Object.freeze([
      "signed-zero",
      "subnormal",
      "rne-even-and-odd-ties",
      "odd-tail-and-binding-padding",
    ]),
  }),
] satisfies readonly PointwiseFixture[]);

export const OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE = Object.freeze({
  id: "add-arithmetic-odd-tail-257",
  operation: "add",
  shape: Object.freeze({ batch: 1, frames: 1, channels: 257 }),
  sourcePatternSalt: 0,
  ranges: Object.freeze([
    Object.freeze({ base: 0, count: 256 }),
    Object.freeze({ base: 256, count: 1 }),
  ]),
  coverage: Object.freeze([
    "fp16-operands-expanded-to-fp32",
    "signed-zero",
    "subnormal",
    "rne-even-and-odd-ties",
    "odd-tail-and-binding-padding",
  ]),
} satisfies PointwiseFixture);

export function parseOpt0011ProductionPointwiseRunIdentity(
  parameters: URLSearchParams,
): Opt0011ProductionPointwiseRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error(
      "OPT-0011 pointwise harnessCommit must be a 40-character lowercase hex commit",
    );
  }
  const coreCommit = requiredIdentity(parameters, "coreCommit");
  if (coreCommit !== OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT) {
    throw new Error("OPT-0011 production pointwise coreCommit changed");
  }
  return Object.freeze({ harnessCommit, coreCommit });
}

export function compareOpt0011ProductionPointwiseRawBits(
  actual: Uint16Array,
  expected: Uint16Array,
): Opt0011ProductionPointwiseRawBitComparison {
  if (actual.constructor !== expected.constructor) {
    throw new Error("OPT-0011 pointwise output bit domains differ");
  }
  if (actual.length !== expected.length) {
    throw new Error("OPT-0011 pointwise output lengths differ");
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

export function opt0011ProductionPointwiseIngressInputBits(
  globalIndex: number,
  salt = 0,
): number {
  requireIndex(globalIndex, 0xffff_ffff, "ingress input");
  return INGRESS_F32_PATTERN[
    (globalIndex + salt * 5) % INGRESS_F32_PATTERN.length
  ]!;
}

export function opt0011ProductionPointwiseIngressCpuBits(
  globalIndex: number,
  salt = 0,
): number {
  return numberToFloat16Bits(float32FromBits(
    opt0011ProductionPointwiseIngressInputBits(globalIndex, salt),
  ));
}

export function opt0011ProductionPointwiseAddLeftBits(
  globalIndex: number,
  salt = 0,
): number {
  requireIndex(globalIndex, 0xffff_ffff, "Add left input");
  return ADD_LEFT_F16_PATTERN[
    (globalIndex + salt * 3) % ADD_LEFT_F16_PATTERN.length
  ]!;
}

export function opt0011ProductionPointwiseAddRightBits(
  globalIndex: number,
  salt = 0,
): number {
  requireIndex(globalIndex, 0xffff_ffff, "Add right input");
  return ADD_RIGHT_F16_PATTERN[
    (globalIndex + salt * 3) % ADD_RIGHT_F16_PATTERN.length
  ]!;
}

export function opt0011ProductionPointwiseAddCpuBits(
  globalIndex: number,
  salt = 0,
): number {
  const left = float16BitsToNumber(
    opt0011ProductionPointwiseAddLeftBits(globalIndex, salt),
  );
  const right = float16BitsToNumber(
    opt0011ProductionPointwiseAddRightBits(globalIndex, salt),
  );
  return numberToFloat16Bits(Math.fround(left + right));
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

export interface Opt0011ProductionPointwiseHeartbeatFailureStop {
  readonly responsiveness: Readonly<Record<string, unknown>> | null;
  readonly heartbeatStopError: Readonly<Record<string, unknown>> | null;
}

interface OutputTarget {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
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

interface PreparedIngressFixture {
  readonly fixture: PointwiseFixture;
  readonly plan: ReturnType<typeof planAceFp16VaeIngress>;
  readonly dispatches: readonly AceFp16VaeIngressDispatch[];
  readonly output: OutputTarget;
  readonly prefill: PrefillTemplate;
  readonly sourceIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface AddDispatchCase {
  readonly fixture: PointwiseFixture | Opt0011ProductionPointwiseGraphAddCase;
  readonly dispatches: readonly AceFp16VaeAddDispatch[];
}

interface PreparedAddFamily {
  readonly plan: ReturnType<typeof planAceFp16VaeAdd>;
  readonly cases: readonly AddDispatchCase[];
  readonly output: OutputTarget;
  readonly prefill: PrefillTemplate;
  readonly sourceIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
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
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing OPT-0011 pointwise start button");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("authenticating frozen production pointwise source");
    const heartbeat = startHeartbeat();
    void runBrowser(heartbeat).then(
      (result) => finish("passed", result),
      (error: unknown) => {
        const heartbeatFailure =
          stopOpt0011ProductionPointwiseHeartbeatAfterFailure(heartbeat);
        finish("failed", Object.freeze({
          schema: "ace-opt-0011-production-fp16-pointwise-correctness-v1",
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
  const identity = parseOpt0011ProductionPointwiseRunIdentity(
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
    label: "ace-opt-0011-production-fp16-pointwise-correctness-device",
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

  const ingressCases: unknown[] = [];
  const addCases: unknown[] = [];
  let cancellation: unknown = null;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let responsiveness: Readonly<Record<string, unknown>> | undefined;
  let postCleanupValidationFailure: Error | undefined;
  try {
    for (
      let index = 0;
      index < OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES.length;
      index += 1
    ) {
      const fixture = OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES[index]!;
      updateProgress(
        `ingress ${index + 1}/${OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES.length}: ${fixture.id}`,
      );
      ingressCases.push(await runIngressFixture(device, tracker, fixture));
      await yieldToBrowser();
    }

    const graphFamilies = graphAddFamilies();
    let completedGraphRanges = 0;
    for (const [familyIndex, family] of graphFamilies.entries()) {
      const prepared = await prepareAddFamily(device, tracker, family);
      try {
        for (const dispatchCase of prepared.cases) {
          addCases.push(await runAddCase(
            device,
            tracker,
            prepared,
            dispatchCase,
            (rangeIndex, rangeCount) => {
              updateProgress(
                `Add graph range ${completedGraphRanges + rangeIndex + 1}/348: ` +
                  `${dispatchCase.fixture.id} ${rangeIndex + 1}/${rangeCount}`,
              );
            },
          ));
          completedGraphRanges += dispatchCase.fixture.ranges.length;
          await yieldToBrowser();
        }
      } finally {
        prepared.destroy();
        prepared.destroy();
      }
      if (familyIndex + 1 !== graphFamilies.length) await yieldToBrowser();
    }
    if (completedGraphRanges !== 348) {
      throw new Error("OPT-0011 pointwise graph range coverage changed");
    }

    const arithmeticPrepared = await prepareAddFamily(
      device,
      tracker,
      [OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE],
    );
    try {
      updateProgress("Add arithmetic odd-tail correctness");
      addCases.push(await runAddCase(
        device,
        tracker,
        arithmeticPrepared,
        arithmeticPrepared.cases[0]!,
      ));
    } finally {
      arithmeticPrepared.destroy();
      arithmeticPrepared.destroy();
    }

    updateProgress("post-drain real multi-range cancellation proof");
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
      stopOpt0011ProductionPointwiseHeartbeatAfterFailure(heartbeat);
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
        "OPT-0011 pointwise post-cleanup resource, event, device-loss, or heartbeat validation failed",
      );
    }
  }
  if (postCleanupValidationFailure !== undefined) {
    throw postCleanupValidationFailure;
  }
  return Object.freeze({
    schema: "ace-opt-0011-production-fp16-pointwise-correctness-v1",
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
      ingressKernel: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
      addKernel: ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
      authority: "independent-raw-bit-cpu-oracle",
      completeB256Ingress: true,
      allFifteenB256AddOperations: true,
      exactB256AddGraphRangeCount: 348,
      fullSelectedRangeRawBitComparison: true,
      deterministicRerunPerRange: true,
      oneOutstandingCommandBuffer: true,
      drainAndRealQueueEmptyTurnAfterEveryExecution: true,
      qNaNPrefillCanariesGuardsAndBindingPadding: true,
      compilationUploadAndWallTimeReported: false,
      performanceClaim: null,
      thermalClaim: null,
      qualityClaim: null,
      productionSelectorClaim: null,
    }),
    sourceAuthority,
    graphCoverage: Object.freeze({
      decoderInputFrames: B256_GRAPH.inputFrames,
      decoderOutputFrames: B256_GRAPH.outputFrames,
      addOperationCount: OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.length,
      exactAddRangeCount: OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.reduce(
        (sum, fixture) => sum + fixture.ranges.length,
        0,
      ),
      distinctAddShapeCount: graphAddFamilies().length,
    }),
    ingressCases: Object.freeze(ingressCases),
    addCases: Object.freeze(addCases),
    cancellation,
    responsiveness,
    cleanup,
  });
}

async function authenticateSources(
  identity: Opt0011ProductionPointwiseRunIdentity,
): Promise<Readonly<Record<string, unknown>>> {
  const ingressSource = aceFp16VaeIngressWgsl();
  const addSource = aceFp16VaeAddWgsl();
  const encoder = new TextEncoder();
  const [coreSourceSha256, ingressShaderSha256, addShaderSha256] =
    await Promise.all([
      sha256Hex(encoder.encode(productionCoreSource)),
      sha256Hex(encoder.encode(ingressSource)),
      sha256Hex(encoder.encode(addSource)),
    ]);
  if (
    coreSourceSha256 !==
      OPT_0011_PRODUCTION_POINTWISE_CORE_SOURCE_SHA256
  ) {
    throw new Error(
      "OPT-0011 rejected unauthenticated production pointwise source",
    );
  }
  if (
    ingressShaderSha256 !==
      OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256.ingress ||
    addShaderSha256 !==
      OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256.add
  ) {
    throw new Error("OPT-0011 pointwise generated shader SHA-256 changed");
  }
  return Object.freeze({
    ...identity,
    coreSourceSha256,
    generatedShaderHashesFrozenAndVerifiedBeforeExecution: true,
    ingressShaderSha256,
    ingressShaderBytes: encoder.encode(ingressSource).byteLength,
    addShaderSha256,
    addShaderBytes: encoder.encode(addSource).byteLength,
  });
}

async function runIngressFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: PointwiseFixture,
): Promise<unknown> {
  const prepared = await prepareIngressFixture(device, tracker, fixture);
  try {
    const ranges: unknown[] = [];
    let comparedElementCount = 0;
    for (const [index, range] of fixture.ranges.entries()) {
      const expected = (globalIndex: number): number =>
        opt0011ProductionPointwiseIngressCpuBits(
          globalIndex,
          fixture.sourcePatternSalt,
        );
      const first = await executeAndRead(
        device,
        tracker,
        prepared.plan,
        range,
        prepared.dispatches[index]!,
        prepared.output,
        prepared.prefill,
        `${fixture.id}-range-${index}-first`,
        expected,
      );
      const rerun = await executeAndRead(
        device,
        tracker,
        prepared.plan,
        range,
        prepared.dispatches[index]!,
        prepared.output,
        prepared.prefill,
        `${fixture.id}-range-${index}-rerun`,
        expected,
      );
      if (first.readback.sha256 !== rerun.readback.sha256) {
        throw new Error(
          `${fixture.id} range ${index} output changed on deterministic rerun`,
        );
      }
      ranges.push(rangeReceipt(range, first, rerun));
      comparedElementCount += range.count;
    }
    if (comparedElementCount !== prepared.plan.elements) {
      throw new Error(`${fixture.id} ranges no longer cover the complete output`);
    }
    return Object.freeze({
      id: fixture.id,
      operation: fixture.operation,
      shape: fixture.shape,
      coverage: fixture.coverage,
      sourcePatternSalt: fixture.sourcePatternSalt,
      sourceIdentity: prepared.sourceIdentity,
      plan: pointwisePlanReceipt(prepared.plan),
      ranges: Object.freeze(ranges),
      rangeCount: fixture.ranges.length,
      comparedElementCount,
      completeSelectedRangeRawBitComparison: true,
      deterministicRerunHashes: true,
      performanceClaim: null,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

async function runAddCase(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedAddFamily,
  dispatchCase: AddDispatchCase,
  onRange?: (rangeIndex: number, rangeCount: number) => void,
): Promise<unknown> {
  const { fixture } = dispatchCase;
  const ranges: unknown[] = [];
  let comparedElementCount = 0;
  for (const [index, declared] of fixture.ranges.entries()) {
    onRange?.(index, fixture.ranges.length);
    const range = Object.freeze({ base: declared.base, count: declared.count });
    const expected = (globalIndex: number): number =>
      opt0011ProductionPointwiseAddCpuBits(
        globalIndex,
        fixture.sourcePatternSalt,
      );
    const first = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      range,
      dispatchCase.dispatches[index]!,
      prepared.output,
      prepared.prefill,
      `${fixture.id}-range-${index}-first`,
      expected,
    );
    const rerun = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      range,
      dispatchCase.dispatches[index]!,
      prepared.output,
      prepared.prefill,
      `${fixture.id}-range-${index}-rerun`,
      expected,
    );
    if (first.readback.sha256 !== rerun.readback.sha256) {
      throw new Error(
        `${fixture.id} range ${index} output changed on deterministic rerun`,
      );
    }
    ranges.push(Object.freeze({
      ...rangeReceipt(range, first, rerun),
      ...(isGraphAddCase(fixture)
        ? {
            quantumIndex: fixture.ranges[index]!.quantumIndex,
            operationQuantumIndex:
              fixture.ranges[index]!.operationQuantumIndex,
          }
        : {}),
    }));
    comparedElementCount += range.count;
  }
  if (comparedElementCount !== prepared.plan.elements) {
    throw new Error(`${fixture.id} ranges no longer cover the complete output`);
  }
  return Object.freeze({
    id: fixture.id,
    operation: "add",
    ...(isGraphAddCase(fixture)
      ? {
          graphOperationIndex: fixture.operationIndex,
          graphAddOrdinal: fixture.addOrdinal,
          shapeFamilyIndex: fixture.shapeFamilyIndex,
          exactB256GraphRanges: true,
        }
      : { coverage: fixture.coverage }),
    shape: fixture.shape,
    sourcePatternSalt: fixture.sourcePatternSalt,
    sourceIdentity: prepared.sourceIdentity,
    plan: pointwisePlanReceipt(prepared.plan),
    ranges: Object.freeze(ranges),
    rangeCount: fixture.ranges.length,
    comparedElementCount,
    completeSelectedRangeRawBitComparison: true,
    deterministicRerunHashes: true,
    performanceClaim: null,
  });
}

async function prepareIngressFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: PointwiseFixture,
): Promise<PreparedIngressFixture> {
  const plan = planAceFp16VaeIngress(fixture.shape);
  const owned: GPUBuffer[] = [];
  const kernel = AceFp16VaePointwiseKernel.create(device);
  let destroyed = false;
  try {
    const input = createPeriodicF32Upload(
      device,
      tracker,
      `${fixture.id}-input`,
      plan.elements,
      plan.sourceBindingBytes,
      fixture.sourcePatternSalt,
    );
    owned.push(input);
    const output = createOutputTarget(
      device,
      tracker,
      `${fixture.id}-output`,
      plan.outputStorageBytes,
      plan.outputBindingBytes,
    );
    owned.push(output.buffer);
    const prefill = createPrefillTemplate(
      device,
      tracker,
      `${fixture.id}-prefill`,
      maximumPrefillBytes(plan, fixture.ranges),
    );
    owned.push(prefill.buffer);
    const controls = createRangeControlBuffer(
      device,
      tracker,
      `${fixture.id}-controls`,
      fixture.ranges,
    );
    owned.push(controls.buffer);
    const dispatches: AceFp16VaeIngressDispatch[] = [];
    for (const [index, range] of fixture.ranges.entries()) {
      const dispatch = await kernel.createIngressDispatch(
        `${fixture.id}-range-${index}`,
        fixture.shape,
        {
          input: binding(input, plan.sourceBindingBytes),
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
        generator: "opt-0011-pointwise-ingress-f32-bits-v1",
        period: INGRESS_F32_PATTERN.length,
        sourcePatternSalt: fixture.sourcePatternSalt,
        elements: plan.elements,
        payloadBytes: plan.sourceStorageBytes,
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

async function prepareAddFamily(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtures: readonly (
    PointwiseFixture | Opt0011ProductionPointwiseGraphAddCase
  )[],
): Promise<PreparedAddFamily> {
  if (fixtures.length < 1) throw new Error("Empty OPT-0011 Add family");
  const template = fixtures[0]!;
  for (const fixture of fixtures) {
    if (
      shapeKey(fixture.shape) !== shapeKey(template.shape) ||
      fixture.sourcePatternSalt !== template.sourcePatternSalt ||
      fixture.ranges.length !== template.ranges.length ||
      fixture.ranges.some((range, index) =>
        range.base !== template.ranges[index]!.base ||
        range.count !== template.ranges[index]!.count
      )
    ) {
      throw new Error("OPT-0011 Add family does not share exact shape/ranges");
    }
  }
  const plan = planAceFp16VaeAdd(template.shape);
  const owned: GPUBuffer[] = [];
  const kernel = AceFp16VaePointwiseKernel.create(device);
  let destroyed = false;
  try {
    const left = createPeriodicF16Upload(
      device,
      tracker,
      `${template.id}-family-left`,
      plan.elements,
      plan.sourceBindingBytes,
      template.sourcePatternSalt,
      ADD_LEFT_F16_PATTERN,
    );
    const right = createPeriodicF16Upload(
      device,
      tracker,
      `${template.id}-family-right`,
      plan.elements,
      plan.sourceBindingBytes,
      template.sourcePatternSalt,
      ADD_RIGHT_F16_PATTERN,
    );
    owned.push(left, right);
    const output = createOutputTarget(
      device,
      tracker,
      `${template.id}-family-output`,
      plan.outputStorageBytes,
      plan.outputBindingBytes,
    );
    owned.push(output.buffer);
    const prefill = createPrefillTemplate(
      device,
      tracker,
      `${template.id}-family-prefill`,
      maximumPrefillBytes(plan, template.ranges),
    );
    owned.push(prefill.buffer);
    const controls = createRangeControlBuffer(
      device,
      tracker,
      `${template.id}-family-controls`,
      template.ranges,
    );
    owned.push(controls.buffer);
    const cases: AddDispatchCase[] = [];
    for (const fixture of fixtures) {
      const dispatches: AceFp16VaeAddDispatch[] = [];
      for (const [index, declared] of fixture.ranges.entries()) {
        const range = Object.freeze({
          base: declared.base,
          count: declared.count,
        });
        const dispatch = await kernel.createAddDispatch(
          `${fixture.id}-range-${index}`,
          fixture.shape,
          {
            left: binding(left, plan.sourceBindingBytes),
            right: binding(right, plan.sourceBindingBytes),
            output: output.binding,
          },
          rangeBinding(range, controls.bindings[index]!),
        );
        assertDispatchRange(dispatch.outputRange, plan, range, fixture.id);
        dispatches.push(dispatch);
      }
      cases.push(Object.freeze({
        fixture,
        dispatches: Object.freeze(dispatches),
      }));
    }
    return Object.freeze({
      plan,
      cases: Object.freeze(cases),
      output,
      prefill,
      sourceIdentity: Object.freeze({
        generator: "opt-0011-pointwise-add-f16-bits-v1",
        leftPeriod: ADD_LEFT_F16_PATTERN.length,
        rightPeriod: ADD_RIGHT_F16_PATTERN.length,
        sourcePatternSalt: template.sourcePatternSalt,
        elements: plan.elements,
        bytesPerSourcePayload: plan.sourceStorageBytes,
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
  plan: AceFp16VaePointwisePlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  dispatch: AceFp16VaeIngressDispatch | AceFp16VaeAddDispatch,
  output: OutputTarget,
  prefill: PrefillTemplate,
  label: string,
  expectedBits: (globalIndex: number) => number,
): Promise<{
  readonly execution: ExecutionCounts;
  readonly readback: SelectedReadback;
}> {
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  encodeQNaNPrefill(encoder, plan, range, output, prefill);
  const pass = encoder.beginComputePass({ label: `${label}-pass` });
  dispatch.encode(pass);
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
    plan,
    range,
    output,
    label,
    expectedBits,
  );
  return Object.freeze({ execution, readback });
}

function encodeQNaNPrefill(
  encoder: GPUCommandEncoder,
  plan: AceFp16VaePointwisePlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  output: OutputTarget,
  prefill: PrefillTemplate,
): void {
  const selectedStart = range.base * FLOAT16_BYTES;
  const selectedEnd = align4(
    selectedStart + range.count * FLOAT16_BYTES,
  );
  const spanStart = Math.max(0, selectedStart - OUTPUT_CANARY_BYTES);
  const spanEnd = Math.min(
    plan.outputBindingBytes,
    selectedEnd + OUTPUT_CANARY_BYTES,
  );
  if (spanStart % 4 !== 0 || spanEnd % 4 !== 0 || spanEnd <= spanStart) {
    throw new RangeError("OPT-0011 pointwise prefill span is not copy aligned");
  }
  const bytes = spanEnd - spanStart;
  if (bytes > prefill.bytes) {
    throw new RangeError("OPT-0011 pointwise prefill template is too small");
  }
  encoder.copyBufferToBuffer(
    prefill.buffer,
    0,
    output.buffer,
    OUTPUT_GUARD_BYTES + spanStart,
    bytes,
  );
}

async function readSelectedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  plan: AceFp16VaePointwisePlan,
  selected: Readonly<{ readonly base: number; readonly count: number }>,
  target: OutputTarget,
  label: string,
  expectedBits: (globalIndex: number) => number,
): Promise<SelectedReadback> {
  const selectedRawBytes = selected.count * FLOAT16_BYTES;
  const selectedCopyBytes = align4(selectedRawBytes);
  const selectedPayloadOffset = selected.base * FLOAT16_BYTES;
  const selectedSourceOffset = OUTPUT_GUARD_BYTES + selectedPayloadOffset;
  if (selectedSourceOffset % 4 !== 0) {
    throw new RangeError("OPT-0011 pointwise selected offset is not copy aligned");
  }
  const selectedEnd = selectedPayloadOffset + selectedCopyBytes;
  const beforeBytes = Math.min(OUTPUT_CANARY_BYTES, selectedPayloadOffset);
  const afterBytes = Math.min(
    OUTPUT_CANARY_BYTES,
    plan.outputBindingBytes - selectedEnd,
  );
  if (beforeBytes % 4 !== 0 || afterBytes % 4 !== 0) {
    throw new RangeError("OPT-0011 pointwise canary span is not copy aligned");
  }
  const prefixOffset = 0;
  const suffixOffset = prefixOffset + OUTPUT_GUARD_BYTES;
  const selectedOffset = suffixOffset + OUTPUT_GUARD_BYTES;
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
    const prefixGuardUntouched = [...prefix].every(
      (bits) => bits === OUTPUT_GUARD_F16,
    );
    const suffixGuardUntouched = [...suffix].every(
      (bits) => bits === OUTPUT_GUARD_F16,
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
    const adjacentCanariesUntouched = [...before, ...after].every(
      (bits) => bits === OUTPUT_CANARY_F16,
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
    const qNaNBindingPaddingUntouched = [...paddingBits].every(
      (bits) => bits === OUTPUT_CANARY_F16,
    );
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    let firstExpectedBits: number | null = null;
    let firstActualBits: number | null = null;
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let outputCanaryCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let subnormalCount = 0;
    for (let localIndex = 0; localIndex < selectedBits.length; localIndex += 1) {
      const actual = selectedBits[localIndex]!;
      const expected = expectedBits(selected.base + localIndex);
      if (actual !== expected) {
        mismatchCount += 1;
        if (firstMismatchIndex === null) {
          firstMismatchIndex = localIndex;
          firstExpectedBits = expected;
          firstActualBits = actual;
        }
      }
      const value = float16BitsToNumber(actual);
      if (Number.isFinite(value)) finiteCount += 1;
      else nonFiniteCount += 1;
      if (actual === OUTPUT_CANARY_F16) outputCanaryCount += 1;
      if ((actual & 0x7fff) === 0) {
        if ((actual & 0x8000) === 0) positiveZeroCount += 1;
        else negativeZeroCount += 1;
      }
      if ((actual & 0x7c00) === 0 && (actual & 0x03ff) !== 0) {
        subnormalCount += 1;
      }
    }
    if (
      !prefixGuardUntouched || !suffixGuardUntouched ||
      !adjacentCanariesUntouched || !qNaNBindingPaddingUntouched ||
      mismatchCount !== 0 || finiteCount !== selected.count ||
      nonFiniteCount !== 0 || outputCanaryCount !== 0
    ) {
      throw new Error(
        `${label} raw-bit/guard validation failed: ` +
          `mismatches=${mismatchCount}@${String(firstMismatchIndex)}, ` +
          `expected=${hex16(firstExpectedBits)}, actual=${hex16(firstActualBits)}`,
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
        outputCanaryCount,
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
        oracle: plan.operation === "ingress"
          ? "exact-f32-bits-to-rne-f16"
          : "f16-operands-to-explicit-f32-add-to-rne-f16",
        comparedElementCount: selected.count,
        mismatchCount,
        firstMismatchIndex,
        completeSelectedRangeRawBitComparison: true,
      }),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function rangeReceipt(
  range: Readonly<{ readonly base: number; readonly count: number }>,
  first: Awaited<ReturnType<typeof executeAndRead>>,
  rerun: Awaited<ReturnType<typeof executeAndRead>>,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    base: range.base,
    count: range.count,
    firstExecution: first.execution,
    rerunExecution: rerun.execution,
    firstScan: first.readback.scan,
    rerunScan: rerun.readback.scan,
    firstCpu: first.readback.cpu,
    rerunCpu: rerun.readback.cpu,
    firstSha256: first.readback.sha256,
    rerunSha256: rerun.readback.sha256,
    deterministic: first.readback.sha256 === rerun.readback.sha256,
  });
}

function createPeriodicF32Upload(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  bindingBytes: number,
  salt: number,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const words = new Uint32Array(buffer.getMappedRange());
    const period = new Uint32Array(INGRESS_F32_PATTERN.length);
    for (let index = 0; index < period.length; index += 1) {
      period[index] = opt0011ProductionPointwiseIngressInputBits(index, salt);
    }
    fillPeriodicPrefix(words, period, elements);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createPeriodicF16Upload(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  bindingBytes: number,
  salt: number,
  pattern: readonly number[],
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
    const period = new Uint16Array(pattern.length);
    for (let index = 0; index < period.length; index += 1) {
      period[index] = pattern[(index + salt * 3) % pattern.length]!;
    }
    fillPeriodicPrefix(words, period, elements);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function fillPeriodicPrefix(
  destination: Uint16Array | Uint32Array,
  period: Uint16Array | Uint32Array,
  elements: number,
): void {
  if (elements > destination.length || period.length < 1) {
    throw new RangeError("OPT-0011 periodic upload geometry is invalid");
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
  outputStorageBytes: number,
  outputBindingBytes: number,
): OutputTarget {
  const buffer = tracker.create(device, {
    label,
    size: OUTPUT_GUARD_BYTES + outputBindingBytes + OUTPUT_GUARD_BYTES,
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
      (OUTPUT_GUARD_BYTES + outputBindingBytes) / FLOAT16_BYTES,
    );
    buffer.unmap();
    return Object.freeze({
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OUTPUT_GUARD_BYTES,
        size: outputBindingBytes,
      }),
      outputStorageBytes,
      outputBindingBytes,
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
    throw new RangeError("OPT-0011 pointwise prefill bytes are invalid");
  }
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  try {
    new Uint16Array(buffer.getMappedRange()).fill(OUTPUT_CANARY_F16);
    buffer.unmap();
    return Object.freeze({ buffer, bytes });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function maximumPrefillBytes(
  plan: AceFp16VaePointwisePlan,
  ranges: readonly Readonly<{
    readonly base: number;
    readonly count: number;
  }>[],
): number {
  let maximum = 0;
  for (const range of ranges) {
    const start = range.base * FLOAT16_BYTES;
    const end = align4(start + range.count * FLOAT16_BYTES);
    const spanStart = Math.max(0, start - OUTPUT_CANARY_BYTES);
    const spanEnd = Math.min(
      plan.outputBindingBytes,
      end + OUTPUT_CANARY_BYTES,
    );
    maximum = Math.max(maximum, spanEnd - spanStart);
  }
  return maximum;
}

function createRangeControlBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  ranges: readonly Readonly<{
    readonly base: number;
    readonly count: number;
  }>[],
): {
  readonly buffer: GPUBuffer;
  readonly bindings: readonly GPUBufferBinding[];
} {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (
    !Number.isSafeInteger(alignment) || alignment < CONTROL_BYTES ||
    alignment % 4 !== 0 || ranges.length < 1
  ) {
    throw new RangeError("OPT-0011 pointwise uniform alignment is invalid");
  }
  const size = alignment * ranges.length;
  const buffer = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  try {
    const payload = new Uint32Array(buffer.getMappedRange());
    for (const [index, range] of ranges.entries()) {
      const word = index * alignment / 4;
      payload[word] = range.base;
      payload[word + 1] = range.count;
      payload[word + 2] = 0;
      payload[word + 3] = 0;
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
): Promise<unknown> {
  const fixture = OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES[0]!;
  const cancellationFixture = Object.freeze({
    ...fixture,
    ranges: Object.freeze(fixture.ranges.slice(0, 2)),
  });
  if (
    cancellationFixture.ranges.length !== 2 ||
    cancellationFixture.ranges.some((range, index) =>
      range.quantumIndex !== fixture.ranges[index]!.quantumIndex
    )
  ) {
    throw new Error(
      "OPT-0011 pointwise cancellation requires two real graph ranges",
    );
  }
  const prepared = await prepareAddFamily(
    device,
    tracker,
    [cancellationFixture],
  );
  try {
    const dispatchCase = prepared.cases[0]!;
    const controller = new AbortController();
    let encodeCount = 0;
    let submitCount = 0;
    let drainCount = 0;
    let prefillCopyCount = 0;
    let readbackCount = 0;
    let skippedRangeCount = 0;
    let idleTurnDelivered = false;
    for (const [index, dispatch] of dispatchCase.dispatches.entries()) {
      if (controller.signal.aborted) {
        skippedRangeCount += 1;
        continue;
      }
      const range = cancellationFixture.ranges[index]!;
      const encoder = device.createCommandEncoder({
        label: `opt-0011-pointwise-cancel-${index}-encoder`,
      });
      encodeQNaNPrefill(
        encoder,
        prepared.plan,
        range,
        prepared.output,
        prepared.prefill,
      );
      prefillCopyCount += 1;
      const pass = encoder.beginComputePass({
        label: `opt-0011-pointwise-cancel-${index}-pass`,
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
      controller.abort("cancel-after-first-drained-real-b256-add-range");
    }
    if (!controller.signal.aborted) {
      readbackCount += 1;
      await readSelectedOutput(
        device,
        tracker,
        prepared.plan,
        cancellationFixture.ranges[0]!,
        prepared.output,
        "opt-0011-pointwise-cancel-readback",
        (globalIndex) => opt0011ProductionPointwiseAddCpuBits(
          globalIndex,
          cancellationFixture.sourcePatternSalt,
        ),
      );
    }
    if (
      !controller.signal.aborted || !idleTurnDelivered || encodeCount !== 1 ||
      submitCount !== 1 || drainCount !== 1 || prefillCopyCount !== 1 ||
      readbackCount !== 0 || skippedRangeCount !== 1
    ) {
      throw new Error(
        "OPT-0011 pointwise cancellation did not stop later graph work",
      );
    }
    return Object.freeze({
      fixtureId: fixture.id,
      graphOperationIndex: fixture.operationIndex,
      plannedRangeCount: cancellationFixture.ranges.length,
      plannedRanges: Object.freeze(cancellationFixture.ranges.map(
        ({ quantumIndex, operationQuantumIndex, base, count }) =>
          Object.freeze({ quantumIndex, operationQuantumIndex, base, count }),
      )),
      cancellationPoint: "after-first-drained-real-b256-add-range-and-idle",
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

function buildGraphAddCases(): readonly Opt0011ProductionPointwiseGraphAddCase[] {
  const operations = B256_GRAPH.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is {
      readonly operation: AceVaeDecoderAddOperation;
      readonly operationIndex: number;
    } => entry.operation.kind === "add");
  if (operations.length !== 15) {
    throw new Error("OPT-0011 pointwise expected exactly 15 B-256 Add operations");
  }
  const shapeFamilies = new Map<string, number>();
  const cases = operations.map(({ operation, operationIndex }, addOrdinal) => {
    const key = shapeKey(operation.shape);
    let shapeFamilyIndex = shapeFamilies.get(key);
    if (shapeFamilyIndex === undefined) {
      shapeFamilyIndex = shapeFamilies.size;
      shapeFamilies.set(key, shapeFamilyIndex);
    }
    let operationQuantumIndex = 0;
    const ranges = B256_QUANTA.quanta
      .filter((quantum) => quantum.operationIndex === operationIndex)
      .map((quantum) => graphAddRange(operation, quantum, operationQuantumIndex++));
    const elements = operation.shape.batch * operation.shape.frames *
      operation.shape.channels;
    const covered = ranges.reduce((sum, range) => sum + range.count, 0);
    if (
      ranges.length < 1 || ranges[0]!.base !== 0 || covered !== elements ||
      ranges.some((range, index) =>
        range.operationQuantumIndex !== index ||
        range.base !== ranges.slice(0, index).reduce(
          (sum, previous) => sum + previous.count,
          0,
        )
      )
    ) {
      throw new Error(`${operation.label} exact graph ranges changed`);
    }
    return Object.freeze({
      id: operation.label,
      operationIndex,
      addOrdinal,
      shapeFamilyIndex,
      shape: operation.shape,
      sourcePatternSalt: shapeFamilyIndex + 1,
      ranges: Object.freeze(ranges),
    });
  });
  if (
    shapeFamilies.size !== 5 ||
    cases.reduce((sum, fixture) => sum + fixture.ranges.length, 0) !== 348
  ) {
    throw new Error("OPT-0011 pointwise B-256 Add topology changed");
  }
  return Object.freeze(cases);
}

function graphAddRange(
  operation: AceVaeDecoderAddOperation,
  quantum: AceVaeDecoderQuantumPlan,
  operationQuantumIndex: number,
): Readonly<{
  readonly quantumIndex: number;
  readonly operationQuantumIndex: number;
  readonly base: number;
  readonly count: number;
}> {
  if (
    quantum.operationKind !== "add" ||
    quantum.operationLabel !== operation.label ||
    quantum.primitives.length !== 1 ||
    quantum.estimatedMaximumMultiplyAccumulates !== 0
  ) {
    throw new Error(`${operation.label} quantum topology changed`);
  }
  const primitive = quantum.primitives[0]!;
  if (
    primitive.outputBase !== quantum.logicalOutputBase ||
    primitive.outputCount !== quantum.logicalOutputCount
  ) {
    throw new Error(`${operation.label} primitive range changed`);
  }
  return Object.freeze({
    quantumIndex: quantum.index,
    operationQuantumIndex,
    base: primitive.outputBase,
    count: primitive.outputCount,
  });
}

function graphAddFamilies(): readonly (
  readonly Opt0011ProductionPointwiseGraphAddCase[]
)[] {
  const families = new Map<number, Opt0011ProductionPointwiseGraphAddCase[]>();
  for (const fixture of OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES) {
    const family = families.get(fixture.shapeFamilyIndex) ?? [];
    family.push(fixture);
    families.set(fixture.shapeFamilyIndex, family);
  }
  const result = [...families.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, family]) => {
      if (family.length !== 3) {
        throw new Error("OPT-0011 pointwise expected three Adds per shape family");
      }
      return Object.freeze(family);
    });
  return Object.freeze(result);
}

function isGraphAddCase(
  fixture: PointwiseFixture | Opt0011ProductionPointwiseGraphAddCase,
): fixture is Opt0011ProductionPointwiseGraphAddCase {
  return "operationIndex" in fixture;
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
  plan: AceFp16VaePointwisePlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  label: string,
): void {
  const expected = planAceFp16VaePointwiseRange(plan, range);
  if (
    actual.base !== expected.base || actual.count !== expected.count ||
    actual.workgroupsX !== expected.workgroupsX ||
    actual.workgroupsY !== expected.workgroupsY
  ) {
    throw new Error(`${label} production dispatch range changed`);
  }
}

function pointwisePlanReceipt(
  plan: AceFp16VaePointwisePlan,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    operation: plan.operation,
    batch: plan.batch,
    frames: plan.frames,
    channels: plan.channels,
    elements: plan.elements,
    sourceStorageBytes: plan.sourceStorageBytes,
    sourceBindingBytes: plan.sourceBindingBytes,
    outputStorageBytes: plan.outputStorageBytes,
    outputBindingBytes: plan.outputBindingBytes,
    workgroupSize: plan.workgroupSize,
  });
}

function requiredDeviceLimits(): Record<string, number> {
  let maximumBufferSize = 0;
  let maximumStorageBinding = 0;
  let maximumDispatch = 1;
  const plans: AceFp16VaePointwisePlan[] = [
    ...OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES.map((fixture) =>
      planAceFp16VaeIngress(fixture.shape)
    ),
    ...OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map((fixture) =>
      planAceFp16VaeAdd(fixture.shape)
    ),
    planAceFp16VaeAdd(
      OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE.shape,
    ),
  ];
  for (const plan of plans) {
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      plan.sourceBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBufferSize = Math.max(
      maximumBufferSize,
      plan.sourceBindingBytes,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    );
    maximumDispatch = Math.max(
      maximumDispatch,
      plan.workgroupsX,
      plan.workgroupsY,
    );
  }
  return {
    maxBufferSize: maximumBufferSize,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
    maxUniformBufferBindingSize: CONTROL_BYTES,
  };
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("OPT-0011 pointwise gate requires adapter shader-f16");
  }
  const required = requiredDeviceLimits();
  for (const [name, minimum] of Object.entries(required)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0011 pointwise adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > OUTPUT_GUARD_BYTES) {
    throw new RangeError(
      "OPT-0011 pointwise output guard is below storage offset alignment",
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
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    minStorageBufferOffsetAlignment: limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
  });
}

export function stopOpt0011ProductionPointwiseHeartbeatAfterFailure(
  heartbeat: HeartbeatController,
): Opt0011ProductionPointwiseHeartbeatFailureStop {
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

function shapeKey(shape: AceVaePointwiseShape): string {
  return `${shape.batch}:${shape.frames}:${shape.channels}`;
}

function requiredIdentity(parameters: URLSearchParams, name: string): string {
  const values = parameters.getAll(name);
  if (values.length !== 1 || values[0] === "") {
    throw new Error(`OPT-0011 pointwise requires one ${name}`);
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

function float32FromBits(bits: number): number {
  UINT32_SCRATCH[0] = bits >>> 0;
  return FLOAT32_SCRATCH[0]!;
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

export function parseOpt0011ProductionPointwiseRawResultChunkOffset(
  value: string,
): number {
  if (!/^(0|[1-9]\d*)$/u.test(value)) {
    throw new Error(
      "OPT-0011 pointwise raw-result chunk offset is not canonical decimal",
    );
  }
  const offset = Number(value);
  if (!Number.isSafeInteger(offset)) {
    throw new Error(
      "OPT-0011 pointwise raw-result chunk offset is not a safe integer",
    );
  }
  return offset;
}

export function sliceOpt0011ProductionPointwiseRawResultChunk(
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
    throw new Error("OPT-0011 pointwise raw-result chunk offset is invalid");
  }
  if (
    offset > 0 && offset < rawResultJson.length &&
    isHighSurrogate(rawResultJson.charCodeAt(offset - 1)) &&
    isLowSurrogate(rawResultJson.charCodeAt(offset))
  ) {
    throw new Error(
      "OPT-0011 pointwise raw-result chunk offset splits a surrogate pair",
    );
  }
  let end = Math.min(
    offset + OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS,
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

function publishPageResult(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  const rawResultJson = JSON.stringify(result);
  if (
    !Reflect.defineProperty(
      globalThis,
      OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_GLOBAL,
      {
        value: rawResultJson,
        configurable: false,
        enumerable: false,
        writable: false,
      },
    )
  ) {
    throw new Error("OPT-0011 pointwise could not publish the raw result receipt");
  }
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    schema: result["schema"] ?? null,
    status,
    experimentId: result["experimentId"] ?? "OPT-0011",
    classification: result["classification"] ?? null,
    rawResultJsonCodeUnitLength: rawResultJson.length,
    rawResultRetrieval: "bounded-dom-chunks",
    rawResultMainWorldGlobal:
      OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_GLOBAL,
    rawResultChunkCodeUnitLimit:
      OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS,
    fullReceiptIntentionallyKeptOutOfDom: true,
  }, null, 2);
  enableRawResultChunkRetrieval();
}

function enableRawResultChunkRetrieval(): void {
  const retrieval = requireElement<HTMLFieldSetElement>(
    "#raw-result-retrieval",
  );
  const offsetInput = requireElement<HTMLInputElement>(
    'input[name="rawResultOffset"]',
  );
  const publish = requireElement<HTMLButtonElement>(
    "#publish-raw-result-chunk",
  );
  const output = requireElement<HTMLElement>("#raw-result-chunk");
  offsetInput.value = "0";
  output.textContent = "";
  output.dataset.state = "empty";
  retrieval.hidden = false;
  retrieval.disabled = false;
  publish.disabled = false;
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
      const rawResultJson = Reflect.get(
        globalThis,
        OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_GLOBAL,
      );
      if (typeof rawResultJson !== "string") {
        throw new Error(
          "OPT-0011 pointwise raw-result main-world receipt is unavailable",
        );
      }
      const slice = sliceOpt0011ProductionPointwiseRawResultChunk(
        rawResultJson,
        parseOpt0011ProductionPointwiseRawResultChunkOffset(offsetInput.value),
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
      publish.disabled = true;
    }
  });
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
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = status;
  publishPageResult(status, result);
}

function requireElement<ElementType extends Element>(
  selector: string,
): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) {
    throw new Error(`Missing OPT-0011 pointwise element ${selector}`);
  }
  return element;
}
