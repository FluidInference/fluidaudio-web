import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import type {
  Opt0067ArmId,
  Opt0067Owner,
  Opt0067ThermalGate,
  Opt0067ThermalTrace,
} from "./opt-0067-dit-quad-query-evaluation-slice-contract.js";

declare global {
  interface Window {
    __ACE_OPT0067_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-arm";
  readonly armId: Opt0067ArmId;
  readonly owner: Opt0067Owner;
  readonly order: number;
  readonly readyAtEpochMilliseconds: number;
  readonly correctness?: Readonly<Record<string, unknown>>;
}

interface ArmCompleteMessage {
  readonly type: "arm-complete";
  readonly armId: Opt0067ArmId;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly sample: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface RejectedMessage {
  readonly type: "gate-rejected" | "trace-rejected";
  readonly armId: Opt0067ArmId;
  readonly readyAtEpochMilliseconds?: number;
  readonly error: Readonly<Record<string, unknown>>;
}

interface CompleteMessage {
  readonly type: "gate-complete";
  readonly result: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | ReadyMessage | ArmCompleteMessage |
  RejectedMessage | CompleteMessage | FailedMessage;

const prepare = element<HTMLButtonElement>("#prepare");
const runArm = element<HTMLButtonElement>("#run-arm");
const recordTrace = element<HTMLButtonElement>("#record-trace");
const gateInput = element<HTMLTextAreaElement>("#thermal-gate-json");
const traceInput = element<HTMLTextAreaElement>("#thermal-trace-json");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const download = element<HTMLAnchorElement>("#download");

let identity: Opt0018RunIdentity | undefined;
let worker: Worker | undefined;
let activeArmId: Opt0067ArmId | undefined;
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
    "running untimed 12-route correctness; accepted timing has not started";
  const active = new Worker(
    new URL(
      "./opt-0067-dit-quad-query-evaluation-slice-worker.ts",
      import.meta.url,
    ),
    { type: "module", name: "ace-opt-0067-evaluation0-thermal-gate" },
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

runArm.addEventListener("click", () => {
  if (
    worker === undefined || activeArmId === undefined ||
    readyAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const thermalGate = parseJson<Opt0067ThermalGate>(gateInput.value);
    runArm.disabled = true;
    gateInput.disabled = true;
    progress.textContent =
      `${activeArmId}: validating nominal gate and starting evaluation 0`;
    worker.postMessage({
      type: "run-arm",
      armId: activeArmId,
      thermalGate,
    });
  } catch (error) {
    displayLocalError(error);
  }
});

recordTrace.addEventListener("click", () => {
  if (
    worker === undefined || activeArmId === undefined ||
    cleanupCompletedAtEpochMilliseconds === 0 || settled
  ) return;
  try {
    const thermalTrace = parseJson<Opt0067ThermalTrace>(traceInput.value);
    recordTrace.disabled = true;
    traceInput.disabled = true;
    progress.textContent = `${activeArmId}: authenticating its distinct trace`;
    worker.postMessage({
      type: "complete-thermal",
      armId: activeArmId,
      thermalTrace,
    });
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
  if (message.type === "ready-for-arm") {
    activeArmId = message.armId;
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    cleanupCompletedAtEpochMilliseconds = 0;
    gateInput.value = "";
    traceInput.value = "";
    gateInput.disabled = false;
    traceInput.disabled = true;
    runArm.disabled = false;
    recordTrace.disabled = true;
    progress.textContent =
      `${message.armId} (${message.owner}) timing-ready at ` +
      `${message.readyAtEpochMilliseconds}; package, inputs, and graph are ` +
      "prepared—begin this arm's fresh continuous ≥30 s level-0 trace";
    if (message.correctness !== undefined) {
      result.textContent = JSON.stringify(message.correctness, null, 2);
    }
    return;
  }
  if (message.type === "arm-complete") {
    if (message.armId !== activeArmId) {
      publishFailure(new Error("OPT-0067 arm completion order changed"));
      return;
    }
    cleanupCompletedAtEpochMilliseconds =
      message.cleanupCompletedAtEpochMilliseconds;
    traceInput.disabled = false;
    recordTrace.disabled = false;
    progress.textContent =
      `${message.armId} cleanup/device disposal completed at ` +
      `${cleanupCompletedAtEpochMilliseconds}; stop only this arm's poll and ` +
      "paste its through-cleanup trace";
    result.textContent = JSON.stringify({
      armId: message.armId,
      sample: message.sample,
      receipt: message.receipt,
    }, null, 2);
    return;
  }
  if (message.type === "gate-rejected") {
    gateInput.disabled = false;
    runArm.disabled = false;
    progress.textContent =
      `${message.armId}: gate rejected before timed GPU work; begin a new trace`;
    result.textContent = JSON.stringify(message.error, null, 2);
    return;
  }
  if (message.type === "trace-rejected") {
    traceInput.disabled = false;
    recordTrace.disabled = false;
    progress.textContent =
      `${message.armId}: trace receipt rejected; no next arm was prepared`;
    result.textContent = JSON.stringify(message.error, null, 2);
    return;
  }
  if (message.type === "gate-complete") {
    publish(message.result);
    return;
  }
  publish(Object.freeze({
    schema: "ace-opt-0067-page-failure-v1",
    experimentId: "OPT-0067",
    status: "failed",
    error: message.error,
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  window.__ACE_OPT0067_RESULT__ = receipt;
  const passed = receipt.status === "passed";
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? "PASSED — exact eval0 ABBA thermal screen verified"
    : "FAILED/INCONCLUSIVE — query8 remains the production default";
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
    schema: "ace-opt-0067-page-failure-v1",
    experimentId: "OPT-0067",
    status: "failed",
    error: serializeOpt0018Failure(error),
  }));
}

function displayLocalError(error: unknown): void {
  result.textContent = JSON.stringify(serializeOpt0018Failure(error), null, 2);
}

function parseJson<Value>(source: string): Value {
  const parsed: unknown = JSON.parse(source);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("OPT-0067 thermal receipt must be a JSON object");
  }
  return parsed as Value;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0067 element ${selector}`);
  return value;
}
