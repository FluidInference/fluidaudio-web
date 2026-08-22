import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS,
  parseOpt0081RepresentativeThermalCompletion,
  parseOpt0081RepresentativeThermalLaunch,
  type Opt0081RepresentativeHeartbeat,
  type Opt0081RepresentativeThermalLaunch,
} from "./opt-0081-dit-f16-representative-layers-contract.js";
import {
  classifyOpt0081RepresentativeDisposition,
  finalizeOpt0081RepresentativeReceipt,
  type Opt0081RepresentativeReceipt,
  type Opt0081RepresentativeRunEvidence,
} from "./opt-0081-dit-f16-representative-layers-result.js";

declare global {
  interface Window {
    __ACE_OPT0081_REPRESENTATIVE_RESULT__?:
      Opt0081RepresentativeReceipt | Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready";
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface MeasurementCompleteMessage {
  readonly type: "measurement-complete";
  readonly evidence: Opt0081RepresentativeRunEvidence;
}

interface FailureMessage {
  readonly type: "failed";
  readonly phase: string;
  readonly error: Readonly<Record<string, unknown>>;
  readonly evidence?: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage |
  MeasurementCompleteMessage | FailureMessage;

interface MutableHeartbeat {
  readonly startedAtEpochMilliseconds: number;
  lastAtPerformanceMilliseconds: number;
  readonly gapsMilliseconds: number[];
}

const prepareButton = element<HTMLButtonElement>("#prepare");
const runButton = element<HTMLButtonElement>("#run");
const finalizeButton = element<HTMLButtonElement>("#finalize");
const download = element<HTMLAnchorElement>("#download");
const launchFields = element<HTMLFieldSetElement>("#thermal-gate");
const completionFields = element<HTMLFieldSetElement>("#thermal-completion");
const progress = element<HTMLElement>("#progress");
const output = element<HTMLElement>("#result");

let identity: Opt0018RunIdentity | undefined;
let worker: Worker | undefined;
let readyAtEpochMilliseconds = 0;
let timingStarted = false;
let pendingEvidence: Opt0081RepresentativeRunEvidence | undefined;
let heartbeat: MutableHeartbeat | undefined;
let completedHeartbeat: Opt0081RepresentativeHeartbeat | undefined;
let downloadUrl: string | undefined;
let settled = false;

let lastGlobalHeartbeatAt = performance.now();
const heartbeatTimer = window.setInterval(() => {
  const now = performance.now();
  const gap = now - lastGlobalHeartbeatAt;
  lastGlobalHeartbeatAt = now;
  if (heartbeat !== undefined) {
    heartbeat.gapsMilliseconds.push(gap);
    heartbeat.lastAtPerformanceMilliseconds = now;
  }
}, OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS);

try {
  identity = parseOpt0018RunIdentity(new URL(location.href).searchParams);
} catch (error) {
  prepareButton.disabled = true;
  publishFailure(error, "identity");
}

prepareButton.addEventListener("click", () => {
  if (identity === undefined || worker !== undefined || settled) return;
  prepareButton.disabled = true;
  document.body.dataset.status = "preparing";
  progress.textContent =
    "loading the canonical package and running untimed A/A/B/B checkpoints";
  const active = new Worker(
    new URL(
      "./opt-0081-dit-f16-representative-layers-worker.ts",
      import.meta.url,
    ),
    { type: "module", name: "ace-opt-0081-representative-layers" },
  );
  worker = active;
  active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    handleWorkerMessage(event.data);
  });
  active.addEventListener("error", (event) => {
    publishFailure(event.error ?? event.message, "worker-error");
  });
  active.postMessage({ type: "prepare", identity });
}, { once: true });

runButton.addEventListener("click", () => {
  if (
    worker === undefined || readyAtEpochMilliseconds === 0 ||
    timingStarted || settled
  ) return;
  const launchedAtEpochMilliseconds = Date.now();
  try {
    const thermalLaunch = parseOpt0081RepresentativeThermalLaunch(
      fieldParameters("#thermal-gate"),
      readyAtEpochMilliseconds,
      launchedAtEpochMilliseconds,
    );
    timingStarted = true;
    runButton.disabled = true;
    launchFields.disabled = true;
    const now = performance.now();
    lastGlobalHeartbeatAt = now;
    heartbeat = {
      startedAtEpochMilliseconds: launchedAtEpochMilliseconds,
      lastAtPerformanceMilliseconds: now,
      gapsMilliseconds: [],
    };
    document.body.dataset.status = "running";
    progress.textContent =
      "thermal launch accepted — running the single eight-round AB/BA gate";
    worker.postMessage({ type: "run", thermalLaunch });
  } catch (error) {
    output.textContent = JSON.stringify(serializeOpt0018Failure(error), null, 2);
    progress.textContent =
      "thermal launch rejected before timing; begin a new continuous nominal slice";
  }
});

finalizeButton.addEventListener("click", () => {
  if (pendingEvidence === undefined || completedHeartbeat === undefined) return;
  try {
    const completion = parseOpt0081RepresentativeThermalCompletion(
      fieldParameters("#thermal-completion"),
      pendingEvidence.thermalLaunch,
      pendingEvidence.cleanupCompletedAtEpochMilliseconds,
    );
    const receipt = finalizeOpt0081RepresentativeReceipt(
      pendingEvidence,
      completedHeartbeat,
      completion,
    );
    pendingEvidence = undefined;
    finalizeButton.disabled = true;
    completionFields.disabled = true;
    publish(receipt);
  } catch (error) {
    output.textContent = JSON.stringify(serializeOpt0018Failure(error), null, 2);
    progress.textContent =
      "through-cleanup trace rejected; timing will not be repeated or reclassified";
  }
});

window.addEventListener("beforeunload", () => {
  window.clearInterval(heartbeatTimer);
  worker?.terminate();
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
});

function handleWorkerMessage(message: WorkerMessage): void {
  if (message.type === "progress") {
    progress.textContent = message.message;
    return;
  }
  if (message.type === "ready") {
    if (readyAtEpochMilliseconds !== 0 || timingStarted) {
      publishFailure(new Error("OPT-0081 READY repeated"), "protocol");
      return;
    }
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    launchFields.disabled = false;
    runButton.disabled = false;
    document.body.dataset.status = "ready";
    progress.textContent =
      `READY at ${readyAtEpochMilliseconds} — correctness, cancellation, ` +
      "25-command precompute, and one warmup per arm passed; timing has not run";
    output.textContent = JSON.stringify(message.preparation, null, 2);
    return;
  }
  if (message.type === "measurement-complete") {
    if (!timingStarted || heartbeat === undefined || pendingEvidence !== undefined) {
      publishFailure(new Error(
        "OPT-0081 measurement completed outside the accepted launch",
      ), "protocol");
      return;
    }
    const now = performance.now();
    heartbeat.gapsMilliseconds.push(
      now - heartbeat.lastAtPerformanceMilliseconds,
    );
    const gaps = Object.freeze([...heartbeat.gapsMilliseconds]);
    const sorted = [...gaps].sort((left, right) => left - right);
    completedHeartbeat = Object.freeze({
      intervalMilliseconds: OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS,
      startedAtEpochMilliseconds: heartbeat.startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Date.now(),
      gapsMilliseconds: gaps,
      maximumGapMilliseconds: sorted.at(-1)!,
      p99GapMilliseconds:
        sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)]!,
    });
    heartbeat = undefined;
    pendingEvidence = message.evidence;
    completionFields.disabled = false;
    finalizeButton.disabled = false;
    document.body.dataset.status = "awaiting-thermal-completion";
    progress.textContent =
      "measurement and cleanup completed — keep polling through this boundary, then enter the trace receipt";
    output.textContent = JSON.stringify({
      timingSamples: message.evidence.timingSamples,
      cleanup: message.evidence.cleanup,
      heartbeat: completedHeartbeat,
    }, null, 2);
    return;
  }
  if (message.type === "failed") {
    const disposition = classifyOpt0081RepresentativeDisposition(
      message.evidence ?? Object.freeze({}),
    );
    publish(Object.freeze({
      schema: "ace-opt-0081-f16-representative-layers-failure-v1",
      experiment: "OPT-0081",
      status: "failed",
      phase: message.phase,
      error: message.error,
      ...(message.evidence === undefined ? {} : { evidence: message.evidence }),
      decision: Object.freeze({
        disposition,
        passed: false,
        completeEvaluationFollowUpAuthorized: false,
        productionIntegrationAuthorized: false,
      }),
    }));
    return;
  }
  const exhaustive: never = message;
  publishFailure(exhaustive, "protocol");
}

function publish(
  receipt: Opt0081RepresentativeReceipt | Readonly<Record<string, unknown>>,
): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  window.__ACE_OPT0081_REPRESENTATIVE_RESULT__ = receipt;
  const record = receipt as unknown as Readonly<Record<string, unknown>>;
  const decision = record["decision"] as Readonly<Record<string, unknown>> |
    undefined;
  const passed = decision?.["passed"] === true;
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? "PASSED — B may proceed only to an independently cooled complete-evaluation gate"
    : "STOPPED — production remains unchanged and no timing retry is authorized";
  const json = JSON.stringify(receipt, null, 2);
  output.textContent = json;
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([json], {
    type: "application/json",
  }));
  download.href = downloadUrl;
  download.hidden = false;
}

function publishFailure(error: unknown, phase: string): void {
  publish(Object.freeze({
    schema: "ace-opt-0081-f16-representative-layers-page-failure-v1",
    experiment: "OPT-0081",
    status: "failed",
    phase,
    error: serializeOpt0018Failure(error),
    decision: Object.freeze({
      disposition:
        "inconclusive-invalid-correctness-topology-or-lifecycle-evidence",
      passed: false,
      completeEvaluationFollowUpAuthorized: false,
      productionIntegrationAuthorized: false,
    }),
  }));
}

function fieldParameters(selector: string): URLSearchParams {
  const fieldset = element<HTMLFieldSetElement>(selector);
  const parameters = new URLSearchParams();
  for (const input of fieldset.querySelectorAll<HTMLInputElement>("input")) {
    parameters.set(input.name, input.value);
  }
  return parameters;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0081 element ${selector}`);
  return value;
}

export type { Opt0081RepresentativeThermalLaunch };
