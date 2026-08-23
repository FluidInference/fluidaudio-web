import {
  OPT_0071_ARM_ORDER,
  OPT_0071_FULL_LOGICAL_BYTES,
  OPT_0071_FULL_LOGICAL_RECORDS,
  OPT_0071_FULL_PHYSICAL_BYTES,
  OPT_0071_FULL_UNIQUE_DIGESTS,
  OPT_0071_TIMED_LOGICAL_BYTES,
  OPT_0071_TIMED_LOGICAL_RECORDS,
  OPT_0071_TIMED_PHYSICAL_BYTES,
  OPT_0071_TIMED_UNIQUE_DIGESTS,
  parseOpt0071RunIdentity,
  requireOpt0071ThermalGate,
  requireOpt0071ThermalTrace,
  serializeOpt0071Failure,
  summarizeOpt0071Performance,
  type Opt0071ArmId,
  type Opt0071ArmSample,
  type Opt0071Owner,
  type Opt0071RunIdentity,
  type Opt0071ThermalGate,
  type Opt0071ThermalTrace,
} from "./opt-0071-warm-cache-webcrypto-production-contract.js";

declare global {
  interface Window {
    __ACE_OPT0071_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-arm";
  readonly armId: Opt0071ArmId;
  readonly owner: Opt0071Owner;
  readonly order: number;
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface HeartbeatMessage {
  readonly type: "heartbeat";
  readonly armId: Opt0071ArmId;
  readonly at: number;
}

interface CompleteMessage {
  readonly type: "arm-complete";
  readonly armId: Opt0071ArmId;
  readonly owner: Opt0071Owner;
  readonly order: number;
  readonly disposalCompletedAtEpochMilliseconds: number;
  readonly sample: Opt0071ArmSample;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage | HeartbeatMessage |
  CompleteMessage | FailedMessage;

interface PendingCompletion {
  readonly sample: Opt0071ArmSample;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly workerTerminatedAtEpochMilliseconds: number;
  readonly maximumWorkerArrivalGapMs: number;
  readonly pageHeartbeat: Readonly<Record<string, unknown>>;
}

const prepareButton = element<HTMLButtonElement>("#prepare");
const runButton = element<HTMLButtonElement>("#run-arm");
const cancelButton = element<HTMLButtonElement>("#cancel-arm");
const traceButton = element<HTMLButtonElement>("#record-trace");
const gateInput = element<HTMLTextAreaElement>("#thermal-gate-json");
const traceInput = element<HTMLTextAreaElement>("#thermal-trace-json");
const progressElement = element<HTMLElement>("#progress");
const resultElement = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0071RunIdentity | undefined;
let worker: Worker | undefined;
let armIndex = -1;
let readyAtEpochMilliseconds = 0;
let acceptedGate: Opt0071ThermalGate | undefined;
let pending: PendingCompletion | undefined;
let pageHeartbeat: PageHeartbeat | undefined;
let lastWorkerHeartbeatAt = 0;
let maximumWorkerArrivalGapMs = 0;
let downloadUrl: string | undefined;
let settled = false;
const preparations: Readonly<Record<string, unknown>>[] = [];
const samples: Opt0071ArmSample[] = [];
const armReceipts: Readonly<Record<string, unknown>>[] = [];
const thermalGates: Opt0071ThermalGate[] = [];
const thermalTraces: Opt0071ThermalTrace[] = [];

try {
  identity = parseOpt0071RunIdentity(new URL(location.href).searchParams);
} catch (error) {
  prepareButton.disabled = true;
  publishFailure(error);
}

prepareButton.addEventListener("click", () => {
  if (identity === undefined || worker !== undefined || settled) return;
  prepareButton.disabled = true;
  document.body.dataset.status = "preparing";
  startNextArm();
}, { once: true });

runButton.addEventListener("click", () => {
  const definition = OPT_0071_ARM_ORDER[armIndex];
  if (
    worker === undefined || definition === undefined ||
    readyAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    acceptedGate = requireOpt0071ThermalGate(
      parseJson<Opt0071ThermalGate>(gateInput.value),
      readyAtEpochMilliseconds,
      Date.now(),
    );
    runButton.disabled = true;
    gateInput.disabled = true;
    cancelButton.disabled = false;
    lastWorkerHeartbeatAt = 0;
    maximumWorkerArrivalGapMs = 0;
    pageHeartbeat = startPageHeartbeat();
    document.body.dataset.status = "running";
    progressElement.textContent =
      `${definition.armId}: accepted nominal gate; timing cache-only production initialize through ordinary READY`;
    worker.postMessage({
      type: "run",
      armId: definition.armId,
      thermalGate: acceptedGate,
    });
  } catch (error) {
    displayLocalError(error);
  }
});

cancelButton.addEventListener("click", () => {
  if (worker === undefined || pending !== undefined || settled) return;
  cancelButton.disabled = true;
  worker.postMessage({ type: "cancel" });
  progressElement.textContent =
    "cancellation requested; the active owner stops at its bounded read/digest file boundary";
});

traceButton.addEventListener("click", () => {
  const definition = OPT_0071_ARM_ORDER[armIndex];
  if (
    definition === undefined || pending === undefined ||
    acceptedGate === undefined || settled
  ) return;
  try {
    const thermalTrace = requireOpt0071ThermalTrace(
      parseJson<Opt0071ThermalTrace>(traceInput.value),
      acceptedGate,
      pending.workerTerminatedAtEpochMilliseconds,
      Date.now(),
    );
    const sample: Opt0071ArmSample = Object.freeze({
      ...pending.sample,
      thermalNonNominalObservations: thermalTrace.nonNominalObservationCount,
    });
    samples.push(sample);
    thermalGates.push(acceptedGate);
    thermalTraces.push(thermalTrace);
    armReceipts.push(Object.freeze({
      ...pending.receipt,
      sample,
      pageAndTermination: Object.freeze({
        workerTerminatedAtEpochMilliseconds:
          pending.workerTerminatedAtEpochMilliseconds,
        maximumWorkerArrivalGapMs: pending.maximumWorkerArrivalGapMs,
        pageHeartbeat: pending.pageHeartbeat,
      }),
      thermalTrace,
    }));
    pending = undefined;
    acceptedGate = undefined;
    traceButton.disabled = true;
    traceInput.disabled = true;
    if (samples.length === OPT_0071_ARM_ORDER.length) {
      publishFinal();
    } else {
      startNextArm();
    }
  } catch (error) {
    displayLocalError(error);
  }
});

window.addEventListener("beforeunload", () => {
  worker?.terminate();
  pageHeartbeat?.stop();
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
});

function startNextArm(): void {
  if (identity === undefined || settled) return;
  armIndex += 1;
  const definition = OPT_0071_ARM_ORDER[armIndex];
  if (definition === undefined) {
    publishFailure(new Error("OPT-0071 arm order overflow"));
    return;
  }
  worker?.terminate();
  worker = new Worker(
    new URL("./opt-0071-warm-cache-webcrypto-production-worker.ts", import.meta.url),
    {
      type: "module",
      name: `ace-opt-0071-${definition.armId}-${definition.owner}`,
    },
  );
  readyAtEpochMilliseconds = 0;
  acceptedGate = undefined;
  pending = undefined;
  gateInput.value = "";
  traceInput.value = "";
  gateInput.disabled = true;
  traceInput.disabled = true;
  runButton.disabled = true;
  traceButton.disabled = true;
  cancelButton.disabled = true;
  document.body.dataset.status = "preparing";
  progressElement.textContent =
    `${definition.armId}: fresh worker proving the full 158/156 cache inventory; this preflight is untimed`;
  worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    handleWorkerMessage(event.data);
  });
  worker.addEventListener("error", (event) => {
    publishFailure(event.error ?? event.message);
  });
  worker.postMessage({
    type: "prepare",
    identity,
    armId: definition.armId,
  });
}

function handleWorkerMessage(message: WorkerMessage): void {
  const definition = OPT_0071_ARM_ORDER[armIndex];
  if (definition === undefined) {
    publishFailure(new Error("OPT-0071 worker message has no active arm"));
    return;
  }
  if (message.type === "progress") {
    progressElement.textContent = message.message;
    return;
  }
  if (message.type === "heartbeat") {
    if (message.armId !== definition.armId) {
      publishFailure(new Error("OPT-0071 heartbeat arm changed"));
      return;
    }
    const now = performance.now();
    if (lastWorkerHeartbeatAt !== 0) {
      maximumWorkerArrivalGapMs = Math.max(
        maximumWorkerArrivalGapMs,
        now - lastWorkerHeartbeatAt,
      );
    }
    lastWorkerHeartbeatAt = now;
    return;
  }
  if (message.type === "ready-for-arm") {
    if (
      message.armId !== definition.armId || message.owner !== definition.owner ||
      message.order !== definition.order
    ) {
      publishFailure(new Error("OPT-0071 fresh-worker arm identity changed"));
      return;
    }
    preparations.push(message.preparation);
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    gateInput.disabled = false;
    runButton.disabled = false;
    document.body.dataset.status = "ready";
    resultElement.textContent = JSON.stringify(message.preparation, null, 2);
    progressElement.textContent =
      `${definition.armId} (${definition.owner}) PRE-FLIGHT READY at ` +
      `${readyAtEpochMilliseconds}; start this arm's fresh continuous thermal poll, ` +
      "wait >=30 nominal seconds, paste its gate, then run immediately";
    return;
  }
  if (message.type === "arm-complete") {
    if (
      message.armId !== definition.armId || message.owner !== definition.owner ||
      message.order !== definition.order
    ) {
      publishFailure(new Error("OPT-0071 arm completion identity changed"));
      return;
    }
    const pageMemoryTelemetry = pageHeartbeat?.memoryTelemetry ?? Object.freeze({
      exposed: false,
      minimumUsedJsHeapSize: null,
      maximumUsedJsHeapSize: null,
    });
    const maximumPageHeartbeatGapMs = pageHeartbeat?.maximumGapMs ?? 0;
    pageHeartbeat?.stop();
    pageHeartbeat = undefined;
    cancelButton.disabled = true;
    worker?.terminate();
    worker = undefined;
    const workerTerminatedAtEpochMilliseconds = Date.now();
    pending = Object.freeze({
      sample: message.sample,
      receipt: message.receipt,
      workerTerminatedAtEpochMilliseconds: Math.max(
        message.disposalCompletedAtEpochMilliseconds,
        workerTerminatedAtEpochMilliseconds,
      ),
      maximumWorkerArrivalGapMs,
      pageHeartbeat: Object.freeze({
        maximumPageHeartbeatGapMs,
        memoryTelemetry: pageMemoryTelemetry,
      }),
    });
    traceInput.disabled = false;
    traceButton.disabled = false;
    document.body.dataset.status = "awaiting-trace";
    resultElement.textContent = JSON.stringify({
      sample: message.sample,
      receipt: message.receipt,
      workerTerminatedAtEpochMilliseconds,
    }, null, 2);
    progressElement.textContent =
      `${definition.armId}: ordinary READY, backend/device disposal, and worker termination completed at ` +
      `${pending.workerTerminatedAtEpochMilliseconds}; stop this arm's poll and paste its through-termination trace`;
    return;
  }
  publishFailure(Object.freeze({
    schema: "ace-opt-0071-worker-failure-v1",
    armId: definition.armId,
    error: message.error,
    retainedSamples: Object.freeze([...samples]),
    retainedArmReceipts: Object.freeze([...armReceipts]),
  }));
}

function publishFinal(): void {
  if (identity === undefined) {
    publishFailure(new Error("OPT-0071 final identity is absent"));
    return;
  }
  const performance = summarizeOpt0071Performance(samples);
  publish(Object.freeze({
    schema: "ace-opt-0071-result-v1",
    experimentId: "OPT-0071",
    status: performance.passed ? "passed" : "failed-or-inconclusive",
    identity,
    protocol: Object.freeze({
      fixedBalancedOrder: Object.freeze(OPT_0071_ARM_ORDER.map((item) =>
        `${item.armId}-${item.owner}`
      )),
      productTuple:
        "OPT-0072 revision-7 dual-K4 VAE + OPT-0070 quad attention/C2378",
      freshWorkerBackendDeviceAndThermalTracePerArm: true,
      untimedPreflightBeforeEveryThermalGate: true,
      timedBoundary:
        "immediately before unchanged cache-only backend.initialize through ordinary READY return",
      throughTerminationTrace: true,
      scalarProductionConstantRemainedSelectedDuringGate: true,
    }),
    authoritySplit: Object.freeze({
      fullUntimedTrustProof: Object.freeze({
        logicalRecords: OPT_0071_FULL_LOGICAL_RECORDS,
        uniqueDigests: OPT_0071_FULL_UNIQUE_DIGESTS,
        logicalBytes: OPT_0071_FULL_LOGICAL_BYTES,
        physicalBytes: OPT_0071_FULL_PHYSICAL_BYTES,
      }),
      timedOrdinaryReadyInventory: Object.freeze({
        logicalRecords: OPT_0071_TIMED_LOGICAL_RECORDS,
        uniqueDigests: OPT_0071_TIMED_UNIQUE_DIGESTS,
        logicalBytes: OPT_0071_TIMED_LOGICAL_BYTES,
        physicalBytes: OPT_0071_TIMED_PHYSICAL_BYTES,
      }),
      deferredVaePayloadAuthenticationCountedInsideReady: false,
      deferredVaeExplanation:
        "ordinary initialize authenticates the revision-7 VAE manifest but its seven payload files remain generation-time work",
    }),
    preparations: Object.freeze([...preparations]),
    samples: Object.freeze([...samples]),
    armReceipts: Object.freeze([...armReceipts]),
    thermalGates: Object.freeze([...thermalGates]),
    thermalTraces: Object.freeze([...thermalTraces]),
    performance,
    decisionBoundary: Object.freeze({
      passingGateAuthorizesOnlySequentialWebCryptoWarmCacheVerifier: true,
      skippingAuthenticationAuthorized: false,
      parallelWholeFilePayloadsAuthorized: false,
      packageModelMathGpuSamplerOrOutputChangeAuthorized: false,
      underOneMinuteClaimAuthorized: false,
    }),
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  pageHeartbeat?.stop();
  pageHeartbeat = undefined;
  window.__ACE_OPT0071_RESULT__ = receipt;
  document.body.dataset.status = receipt["status"] === "passed"
    ? "complete"
    : "failed";
  progressElement.textContent = receipt["status"] === "passed"
    ? "PASSED — OPT-0071 bounded WebCrypto production selection gate passed"
    : "FAILED/INCONCLUSIVE — scalar cache authentication remains the production owner";
  const json = JSON.stringify(receipt, null, 2);
  resultElement.textContent = json;
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([json], {
    type: "application/json",
  }));
  download.href = downloadUrl;
  download.hidden = false;
}

function publishFailure(error: unknown): void {
  publish(Object.freeze({
    schema: "ace-opt-0071-page-failure-v1",
    experimentId: "OPT-0071",
    status: "failed",
    error: error !== null && typeof error === "object" && "schema" in error
      ? error
      : serializeOpt0071Failure(error),
    retainedPreparations: Object.freeze([...preparations]),
    retainedSamples: Object.freeze([...samples]),
    retainedArmReceipts: Object.freeze([...armReceipts]),
    retainedThermalGates: Object.freeze([...thermalGates]),
    retainedThermalTraces: Object.freeze([...thermalTraces]),
  }));
}

function displayLocalError(error: unknown): void {
  resultElement.textContent = JSON.stringify(
    serializeOpt0071Failure(error),
    null,
    2,
  );
}

interface PageHeartbeat {
  readonly maximumGapMs: number;
  readonly memoryTelemetry: Readonly<Record<string, unknown>>;
  stop(): void;
}

function startPageHeartbeat(): PageHeartbeat {
  let last = performance.now();
  let maximumGapMs = 0;
  let stopped = false;
  const memory = (performance as Performance & {
    memory?: Readonly<{
      usedJSHeapSize: number;
      totalJSHeapSize: number;
      jsHeapSizeLimit: number;
    }>;
  }).memory;
  let minimumUsedJsHeapSize = Number.POSITIVE_INFINITY;
  let maximumUsedJsHeapSize = 0;
  const sampleMemory = (): void => {
    const used = memory?.usedJSHeapSize;
    if (used === undefined || !Number.isFinite(used) || used < 0) return;
    minimumUsedJsHeapSize = Math.min(minimumUsedJsHeapSize, used);
    maximumUsedJsHeapSize = Math.max(maximumUsedJsHeapSize, used);
  };
  sampleMemory();
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumGapMs = Math.max(maximumGapMs, now - last);
    last = now;
    sampleMemory();
  }, 25);
  return {
    get maximumGapMs(): number {
      if (!stopped) maximumGapMs = Math.max(maximumGapMs, performance.now() - last);
      return maximumGapMs;
    },
    get memoryTelemetry(): Readonly<Record<string, unknown>> {
      sampleMemory();
      return Object.freeze({
        exposed: memory !== undefined,
        minimumUsedJsHeapSize: Number.isFinite(minimumUsedJsHeapSize)
          ? minimumUsedJsHeapSize
          : null,
        maximumUsedJsHeapSize: maximumUsedJsHeapSize === 0
          ? null
          : maximumUsedJsHeapSize,
        totalJSHeapSize: memory?.totalJSHeapSize ?? null,
        jsHeapSizeLimit: memory?.jsHeapSizeLimit ?? null,
        scope:
          "main-page performance.memory; explicit preflight payload accounting is authoritative",
      });
    },
    stop(): void {
      if (stopped) return;
      maximumGapMs = Math.max(maximumGapMs, performance.now() - last);
      window.clearInterval(timer);
      stopped = true;
    },
  };
}

function parseJson<Value>(text: string): Value {
  const value: unknown = JSON.parse(text);
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OPT-0071 expected one JSON object");
  }
  return value as Value;
}

function element<ElementType extends HTMLElement>(
  selector: string,
): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0071 element ${selector}`);
  return value;
}
