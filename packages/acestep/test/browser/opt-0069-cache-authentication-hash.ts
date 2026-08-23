import {
  OPT_0069_ARM_ORDER,
  OPT_0069_COMPLETE_LOGICAL_BYTES,
  OPT_0069_COMPLETE_LOGICAL_RECORDS,
  OPT_0069_COMPLETE_PHYSICAL_BYTES,
  OPT_0069_COMPLETE_UNIQUE_DIGESTS,
  OPT_0069_UPLOAD_SUBSET_BYTES,
  OPT_0069_UPLOAD_SUBSET_FILES,
  OPT_0069_UPLOAD_SUBSET_REPORTED_GB,
  parseOpt0069RunIdentity,
  requireOpt0069ThermalGate,
  requireOpt0069ThermalTrace,
  serializeOpt0069Failure,
  summarizeOpt0069Performance,
  type Opt0069ArmId,
  type Opt0069Owner,
  type Opt0069RunIdentity,
  type Opt0069ThermalGate,
  type Opt0069ThermalTrace,
  type Opt0069TimingSample,
} from "./opt-0069-cache-authentication-hash-contract.js";

declare global {
  interface Window {
    __ACE_OPT0069_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-arm";
  readonly armId: Opt0069ArmId;
  readonly owner: Opt0069Owner;
  readonly order: number;
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface HeartbeatMessage {
  readonly type: "heartbeat";
  readonly armId: Opt0069ArmId;
  readonly at: number;
}

interface CompleteMessage {
  readonly type: "arm-complete";
  readonly armId: Opt0069ArmId;
  readonly owner: Opt0069Owner;
  readonly order: number;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly sample: Opt0069TimingSample;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage | HeartbeatMessage |
  CompleteMessage | FailedMessage;

interface PendingCompletion {
  readonly sample: Opt0069TimingSample;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly maximumWorkerArrivalGapMs: number;
  readonly pageMemoryTelemetry: Readonly<Record<string, unknown>>;
}

const prepareButton = element<HTMLButtonElement>("#prepare");
const runButton = element<HTMLButtonElement>("#run-arm");
const traceButton = element<HTMLButtonElement>("#record-trace");
const cancelButton = element<HTMLButtonElement>("#cancel-arm");
const gateInput = element<HTMLTextAreaElement>("#thermal-gate-json");
const traceInput = element<HTMLTextAreaElement>("#thermal-trace-json");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0069RunIdentity | undefined;
let worker: Worker | undefined;
let armIndex = -1;
let readyAtEpochMilliseconds = 0;
let acceptedGate: Opt0069ThermalGate | undefined;
let pending: PendingCompletion | undefined;
let downloadUrl: string | undefined;
let settled = false;
const preparations: Readonly<Record<string, unknown>>[] = [];
const samples: Opt0069TimingSample[] = [];
const armReceipts: Readonly<Record<string, unknown>>[] = [];
const thermalTraces: Opt0069ThermalTrace[] = [];
let pageHeartbeat: PageHeartbeat | undefined;
let lastWorkerHeartbeatAt = 0;
let maximumWorkerArrivalGapMs = 0;

try {
  identity = parseOpt0069RunIdentity(new URL(location.href).searchParams);
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
  const definition = OPT_0069_ARM_ORDER[armIndex];
  if (
    worker === undefined || definition === undefined ||
    readyAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const parsed = parseJson<Opt0069ThermalGate>(gateInput.value);
    acceptedGate = requireOpt0069ThermalGate(
      parsed,
      readyAtEpochMilliseconds,
      Date.now(),
    );
    runButton.disabled = true;
    gateInput.disabled = true;
    cancelButton.disabled = false;
    lastWorkerHeartbeatAt = 0;
    maximumWorkerArrivalGapMs = 0;
    pageHeartbeat = startPageHeartbeat();
    progress.textContent =
      `${definition.armId}: accepted fresh nominal gate; timing full inventory`;
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
  progress.textContent =
    "cancellation requested; scalar stops by <=4 MiB slice, WebCrypto after its in-flight file";
});

traceButton.addEventListener("click", () => {
  const definition = OPT_0069_ARM_ORDER[armIndex];
  if (
    definition === undefined || pending === undefined || acceptedGate === undefined ||
    settled
  ) return;
  try {
    const thermalTrace = requireOpt0069ThermalTrace(
      parseJson<Opt0069ThermalTrace>(traceInput.value),
      acceptedGate,
      pending.cleanupCompletedAtEpochMilliseconds,
      Date.now(),
    );
    const sample = Object.freeze({
      ...pending.sample,
      thermalNonNominalObservations: thermalTrace.nonNominalObservationCount,
    });
    samples.push(sample);
    thermalTraces.push(thermalTrace);
    armReceipts.push(Object.freeze({
      ...pending.receipt,
      pageHeartbeat: Object.freeze({
        maximumPageHeartbeatGapMs: sample.maximumPageHeartbeatGapMs,
        maximumWorkerArrivalGapMs: pending.maximumWorkerArrivalGapMs,
        memoryTelemetry: pending.pageMemoryTelemetry,
      }),
      thermalTrace,
    }));
    pending = undefined;
    acceptedGate = undefined;
    traceButton.disabled = true;
    traceInput.disabled = true;
    if (samples.length === OPT_0069_ARM_ORDER.length) {
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
  const definition = OPT_0069_ARM_ORDER[armIndex];
  if (definition === undefined) {
    publishFailure(new Error("OPT-0069 arm order overflow"));
    return;
  }
  worker?.terminate();
  worker = new Worker(
    new URL("./opt-0069-cache-authentication-hash-worker.ts", import.meta.url),
    {
      type: "module",
      name: `ace-opt-0069-${definition.armId}-${definition.owner}`,
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
  progress.textContent =
    `${definition.armId}: fresh worker authenticating manifests/cache metadata; no payload timing yet`;
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
  const definition = OPT_0069_ARM_ORDER[armIndex];
  if (definition === undefined) {
    publishFailure(new Error("OPT-0069 worker message has no active arm"));
    return;
  }
  if (message.type === "progress") {
    progress.textContent = message.message;
    return;
  }
  if (message.type === "heartbeat") {
    if (message.armId !== definition.armId) {
      publishFailure(new Error("OPT-0069 heartbeat arm changed"));
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
      publishFailure(new Error("OPT-0069 fresh-worker arm identity changed"));
      return;
    }
    preparations.push(message.preparation);
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    gateInput.disabled = false;
    runButton.disabled = false;
    result.textContent = JSON.stringify(message.preparation, null, 2);
    progress.textContent =
      `${definition.armId} (${definition.owner}) READY at ` +
      `${readyAtEpochMilliseconds}; start this arm's fresh continuous thermal poll, ` +
      "wait for a nominal >=30 s suffix, build/paste its gate, then run immediately";
    return;
  }
  if (message.type === "arm-complete") {
    if (
      message.armId !== definition.armId || message.owner !== definition.owner ||
      message.order !== definition.order
    ) {
      publishFailure(new Error("OPT-0069 arm completion identity changed"));
      return;
    }
    const pageMaximumGapMs = pageHeartbeat?.maximumGapMs ?? 0;
    const pageMemoryTelemetry = pageHeartbeat?.memoryTelemetry ?? Object.freeze({
      exposed: false,
      minimumUsedJsHeapSize: null,
      maximumUsedJsHeapSize: null,
    });
    pageHeartbeat?.stop();
    pageHeartbeat = undefined;
    cancelButton.disabled = true;
    worker?.terminate();
    worker = undefined;
    const terminationCompletedAtEpochMilliseconds = Date.now();
    const sample = Object.freeze({
      ...message.sample,
      maximumPageHeartbeatGapMs: pageMaximumGapMs,
    });
    pending = Object.freeze({
      sample,
      receipt: message.receipt,
      cleanupCompletedAtEpochMilliseconds: Math.max(
        message.cleanupCompletedAtEpochMilliseconds,
        terminationCompletedAtEpochMilliseconds,
      ),
      maximumWorkerArrivalGapMs,
      pageMemoryTelemetry,
    });
    traceInput.disabled = false;
    traceButton.disabled = false;
    result.textContent = JSON.stringify({
      sample,
      receipt: message.receipt,
      workerTerminatedAtEpochMilliseconds:
        terminationCompletedAtEpochMilliseconds,
    }, null, 2);
    progress.textContent =
      `${definition.armId}: payload released and fresh worker terminated at ` +
      `${pending.cleanupCompletedAtEpochMilliseconds}; stop only this arm's poll, ` +
      "build its through-cleanup trace, and paste it";
    return;
  }
  publishFailure(Object.freeze({
    schema: "ace-opt-0069-worker-failure-v1",
    armId: definition.armId,
    error: message.error,
    retainedSamples: Object.freeze([...samples]),
  }));
}

function publishFinal(): void {
  if (identity === undefined) {
    publishFailure(new Error("OPT-0069 final identity is absent"));
    return;
  }
  const performanceSummary = summarizeOpt0069Performance(samples);
  publish(Object.freeze({
    schema: "ace-opt-0069-result-v1",
    experimentId: "OPT-0069",
    status: performanceSummary.passed ? "passed" : "failed-or-inconclusive",
    identity,
    protocol: Object.freeze({
      fixedBalancedOrder: Object.freeze(OPT_0069_ARM_ORDER.map((item) =>
        `${item.armId}-${item.owner}`
      )),
      freshWorkerPerArm: true,
      freshThermalGateAndTracePerArm: true,
      completeInventoryTimed: true,
      armA: "File.stream + AceIncrementalSha256 + <=4MiB slices",
      armB: "one File.arrayBuffer + WebCrypto SHA-256 at a time",
      optionalWasmArmIncluded: false,
      optionalWasmArmDisposition:
        "deferred until a dependency-free true four-file simd128 owner and matched scalar schedule are independently proven",
    }),
    inventory: Object.freeze({
      logicalRecords: OPT_0069_COMPLETE_LOGICAL_RECORDS,
      uniqueDigests: OPT_0069_COMPLETE_UNIQUE_DIGESTS,
      logicalBytes: OPT_0069_COMPLETE_LOGICAL_BYTES,
      physicalBytes: OPT_0069_COMPLETE_PHYSICAL_BYTES,
      uploadSubsetFiles: OPT_0069_UPLOAD_SUBSET_FILES,
      uploadSubsetBytes: OPT_0069_UPLOAD_SUBSET_BYTES,
      retainedReportedSlice: OPT_0069_UPLOAD_SUBSET_REPORTED_GB,
      completePhysicalInventoryIsDecisionAuthority: true,
    }),
    preparations: Object.freeze([...preparations]),
    samples: Object.freeze([...samples]),
    armReceipts: Object.freeze([...armReceipts]),
    thermalTraces: Object.freeze([...thermalTraces]),
    performance: performanceSummary,
    authorityBoundary: Object.freeze({
      isolatedPassAuthorizesOnlyProductionSeamIntegrationUnderOpt0069: true,
      productionIntegrationPerformed: false,
      cacheOrPackageMutationPerformed: false,
      gpuDeviceRequested: false,
      endToEndInitializationSavingClaimed: false,
    }),
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  pageHeartbeat?.stop();
  pageHeartbeat = undefined;
  window.__ACE_OPT0069_RESULT__ = receipt;
  document.body.dataset.status = receipt.status === "passed"
    ? "complete"
    : "failed";
  progress.textContent = receipt.status === "passed"
    ? "PASSED isolated A/B gate — only a production-seam integration is authorized"
    : "FAILED/INCONCLUSIVE — current scalar production verifier remains unchanged";
  const json = JSON.stringify(receipt, null, 2);
  result.textContent = json;
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([json], {
    type: "application/json",
  }));
  download.href = downloadUrl;
  download.hidden = false;
}

function publishFailure(error: unknown): void {
  publish(Object.freeze({
    schema: "ace-opt-0069-page-failure-v1",
    experimentId: "OPT-0069",
    status: "failed",
    error: error !== null && typeof error === "object" &&
        "schema" in error
      ? error
      : serializeOpt0069Failure(error),
    retainedSamples: Object.freeze([...samples]),
    retainedArmReceipts: Object.freeze([...armReceipts]),
  }));
}

function displayLocalError(error: unknown): void {
  result.textContent = JSON.stringify(serializeOpt0069Failure(error), null, 2);
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
        scope: "main-page performance.memory; explicit worker payload accounting is authoritative",
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

function parseJson<T>(text: string): T {
  const value: unknown = JSON.parse(text);
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OPT-0069 expected one JSON object");
  }
  return value as T;
}

function element<T extends HTMLElement>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (value === null) throw new Error(`Missing OPT-0069 element ${selector}`);
  return value;
}
