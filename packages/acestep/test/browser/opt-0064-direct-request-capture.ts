import {
  parseOpt0064RunIdentity,
  serializeOpt0064Failure,
  type Opt0064RunIdentity,
  type Opt0064ThermalGate,
  type Opt0064ThermalTrace,
} from "./opt-0064-direct-request-capture-contract.js";

declare global {
  interface Window {
    __ACE_OPT0064_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-thermal-gate";
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface AwaitTraceMessage {
  readonly type: "capture-awaiting-thermal-trace";
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly summary: Readonly<Record<string, unknown>>;
}

interface CompleteMessage {
  readonly type: "capture-complete";
  readonly result: Readonly<Record<string, unknown>>;
}

interface RejectedMessage {
  readonly type: "gate-rejected" | "trace-rejected";
  readonly error: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage | AwaitTraceMessage |
  CompleteMessage | RejectedMessage | FailedMessage;

const prepare = element<HTMLButtonElement>("#prepare");
const run = element<HTMLButtonElement>("#run");
const finalize = element<HTMLButtonElement>("#finalize");
const gateInput = element<HTMLTextAreaElement>("#thermal-gate-json");
const traceInput = element<HTMLTextAreaElement>("#thermal-trace-json");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0064RunIdentity | undefined;
let worker: Worker | undefined;
let readyAtEpochMilliseconds = 0;
let cleanupCompletedAtEpochMilliseconds = 0;
let settled = false;
let downloadUrl: string | undefined;

try {
  identity = parseOpt0064RunIdentity(new URL(location.href).searchParams);
} catch (error) {
  prepare.disabled = true;
  publishFailure(error);
}

prepare.addEventListener("click", () => {
  if (identity === undefined || worker !== undefined || settled) return;
  prepare.disabled = true;
  document.body.dataset.status = "preparing";
  progress.textContent =
    "validating static request/commit authority; no package, device, or GPU work";
  const active = new Worker(
    new URL("./opt-0064-direct-request-capture-worker.ts", import.meta.url),
    { type: "module", name: "ace-opt-0064-direct-request-capture" },
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

run.addEventListener("click", () => {
  if (worker === undefined || readyAtEpochMilliseconds === 0 || settled) return;
  try {
    const thermalGate = parseJson<Opt0064ThermalGate>(gateInput.value);
    run.disabled = true;
    gateInput.disabled = true;
    document.body.dataset.status = "running";
    progress.textContent =
      "validating the nominal gate, then starting all package/device/GPU work";
    worker.postMessage({ type: "run", thermalGate });
  } catch (error) {
    displayLocalError(error);
  }
});

finalize.addEventListener("click", () => {
  if (
    worker === undefined || cleanupCompletedAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const thermalTrace = parseJson<Opt0064ThermalTrace>(traceInput.value);
    finalize.disabled = true;
    traceInput.disabled = true;
    progress.textContent = "authenticating the same trace through cleanup/disposal";
    worker.postMessage({ type: "complete-thermal", thermalTrace });
  } catch (error) {
    displayLocalError(error);
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
  if (message.type === "ready-for-thermal-gate") {
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    gateInput.disabled = false;
    run.disabled = false;
    document.body.dataset.status = "ready";
    progress.textContent =
      `static READY at ${readyAtEpochMilliseconds}; begin a fresh continuous ` +
      "≥30 s level-0 poll, keep it running, then paste its current gate";
    result.textContent = JSON.stringify(message.preparation, null, 2);
    return;
  }
  if (message.type === "capture-awaiting-thermal-trace") {
    cleanupCompletedAtEpochMilliseconds =
      message.cleanupCompletedAtEpochMilliseconds;
    traceInput.disabled = false;
    finalize.disabled = false;
    document.body.dataset.status = "awaiting-trace";
    progress.textContent =
      `cleanup/device disposal completed at ${cleanupCompletedAtEpochMilliseconds}; ` +
      "stop the same poll and paste its complete through-cleanup trace";
    result.textContent = JSON.stringify(message.summary, null, 2);
    return;
  }
  if (message.type === "gate-rejected") {
    gateInput.disabled = false;
    run.disabled = false;
    document.body.dataset.status = "ready";
    progress.textContent =
      "gate rejected before package/device/GPU work; begin a fresh nominal trace";
    result.textContent = JSON.stringify(message.error, null, 2);
    return;
  }
  if (message.type === "trace-rejected") {
    traceInput.disabled = false;
    finalize.disabled = false;
    progress.textContent =
      "through-cleanup trace rejected; the completed request will not be rerun";
    result.textContent = JSON.stringify(message.error, null, 2);
    return;
  }
  if (message.type === "capture-complete") {
    publish(message.result);
    return;
  }
  publish(Object.freeze({
    schema: "ace-opt-0064-page-failure-v1",
    experimentId: "OPT-0064",
    status: "failed",
    error: message.error,
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  window.__ACE_OPT0064_RESULT__ = receipt;
  const passed = receipt["status"] === "passed";
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? "PASSED — capture-only direct request reconciled through cleanup"
    : "FAILED — no OPT-0064 optimization is authorized";
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
    schema: "ace-opt-0064-page-failure-v1",
    experimentId: "OPT-0064",
    status: "failed",
    error: serializeOpt0064Failure(error),
  }));
}

function displayLocalError(error: unknown): void {
  result.textContent = JSON.stringify(serializeOpt0064Failure(error), null, 2);
}

function parseJson<Value>(source: string): Value {
  const value: unknown = JSON.parse(source);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("OPT-0064 thermal receipt must be a JSON object");
  }
  return value as Value;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0064 element ${selector}`);
  return value;
}
