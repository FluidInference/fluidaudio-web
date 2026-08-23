import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import type {
  Opt0062Direction,
  Opt0062ThermalGate,
  Opt0062ThermalTrace,
} from "./opt-0062-dit-quad-query-contract.js";

declare global {
  interface Window {
    __ACE_OPT0062_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-direction";
  readonly direction: Opt0062Direction;
  readonly readyAtEpochMilliseconds: number;
  readonly correctness?: Readonly<Record<string, unknown>>;
}

interface DirectionCompleteMessage {
  readonly type: "direction-complete";
  readonly direction: Opt0062Direction;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly samples: readonly Readonly<Record<string, unknown>>[];
}

interface CompleteMessage {
  readonly type: "gate-complete";
  readonly result: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage |
  DirectionCompleteMessage | CompleteMessage | FailedMessage;

const prepare = element<HTMLButtonElement>("#prepare");
const runDirection = element<HTMLButtonElement>("#run-direction");
const recordTrace = element<HTMLButtonElement>("#record-trace");
const gateInput = element<HTMLTextAreaElement>("#thermal-gate-json");
const traceInput = element<HTMLTextAreaElement>("#thermal-trace-json");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0018RunIdentity | undefined;
let worker: Worker | undefined;
let activeDirection: Opt0062Direction | undefined;
let readyAtEpochMilliseconds = 0;
let cleanupCompletedAtEpochMilliseconds = 0;
let downloadUrl: string | undefined;
let settled = false;

try {
  identity = parseOpt0018RunIdentity(new URL(location.href).searchParams);
} catch (error) {
  prepare.disabled = true;
  publishFailure(error);
}

prepare.addEventListener("click", () => {
  if (identity === undefined || worker !== undefined || settled) return;
  prepare.disabled = true;
  document.body.dataset.status = "running";
  progress.textContent =
    "running query8/quad/quad correctness arms; no thermal timing yet";
  const active = new Worker(
    new URL("./opt-0062-dit-quad-query-worker.ts", import.meta.url),
    { type: "module", name: "ace-opt-0062-quad-query-integration-gate" },
  );
  worker = active;
  active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    handleWorkerMessage(event.data);
  });
  active.addEventListener("error", (event) => {
    publishFailure(event.error ?? event.message);
  });
  active.postMessage({ type: "prepare", identity });
}, { once: true });

runDirection.addEventListener("click", () => {
  if (
    worker === undefined || activeDirection === undefined ||
    readyAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const thermalGate = parseJson<Opt0062ThermalGate>(gateInput.value);
    runDirection.disabled = true;
    gateInput.disabled = true;
    progress.textContent =
      `${activeDirection}: running two sequential full M2250 arms`;
    worker.postMessage({
      type: "run-direction",
      direction: activeDirection,
      thermalGate,
    });
  } catch (error) {
    publishFailure(error);
  }
});

recordTrace.addEventListener("click", () => {
  if (
    worker === undefined || activeDirection === undefined ||
    cleanupCompletedAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const thermalTrace = parseJson<Opt0062ThermalTrace>(traceInput.value);
    recordTrace.disabled = true;
    traceInput.disabled = true;
    progress.textContent = `${activeDirection}: authenticating thermal trace`;
    worker.postMessage({
      type: "complete-thermal",
      direction: activeDirection,
      thermalTrace,
    });
  } catch (error) {
    publishFailure(error);
  }
});

window.addEventListener("beforeunload", () => {
  worker?.terminate();
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
});

function handleWorkerMessage(message: WorkerMessage): void {
  if (message.type === "progress") {
    progress.textContent = message.message;
    return;
  }
  if (message.type === "ready-for-direction") {
    activeDirection = message.direction;
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    cleanupCompletedAtEpochMilliseconds = 0;
    gateInput.value = "";
    traceInput.value = "";
    gateInput.disabled = false;
    traceInput.disabled = true;
    runDirection.disabled = false;
    recordTrace.disabled = true;
    progress.textContent =
      `${message.direction} ready at ${message.readyAtEpochMilliseconds}; ` +
      "begin a fresh continuous ≥30 s level-0 notifyutil trace";
    if (message.correctness !== undefined) {
      result.textContent = JSON.stringify(message.correctness, null, 2);
    }
    return;
  }
  if (message.type === "direction-complete") {
    if (message.direction !== activeDirection) {
      publishFailure(new Error("OPT-0062 direction completion order changed"));
      return;
    }
    cleanupCompletedAtEpochMilliseconds =
      message.cleanupCompletedAtEpochMilliseconds;
    traceInput.disabled = false;
    recordTrace.disabled = false;
    progress.textContent =
      `${message.direction} cleanup completed at ` +
      `${cleanupCompletedAtEpochMilliseconds}; stop polling and paste the ` +
      "through-cleanup trace receipt";
    result.textContent = JSON.stringify({
      direction: message.direction,
      samples: message.samples,
    }, null, 2);
    return;
  }
  if (message.type === "gate-complete") {
    publish(message.result);
    return;
  }
  publish(Object.freeze({
    schema: "ace-opt-0062-page-failure-v1",
    experimentId: "OPT-0062",
    status: "failed",
    error: message.error,
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  window.__ACE_OPT0062_RESULT__ = receipt;
  const passed = receipt.status === "passed";
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? "PASSED — correctness, both directional gates, and thermal traces verified"
    : "FAILED/INCONCLUSIVE — query8 remains the default";
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
    schema: "ace-opt-0062-page-failure-v1",
    experimentId: "OPT-0062",
    status: "failed",
    error: serializeOpt0018Failure(error),
  }));
}

function parseJson<Value>(source: string): Value {
  const parsed: unknown = JSON.parse(source);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("OPT-0062 thermal receipt must be a JSON object");
  }
  return parsed as Value;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0062 element ${selector}`);
  return value;
}
