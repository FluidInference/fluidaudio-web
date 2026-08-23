import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS,
  type Opt0080ArmId,
  type Opt0080HeartbeatCapture,
  type Opt0080SchedulingProfile,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
} from "./opt-0080-dit-depth2-completion-epochs-contract.js";

declare global {
  interface Window {
    __ACE_OPT0080_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface ReadyMessage {
  readonly type: "ready-for-arm";
  readonly armId: Opt0080ArmId;
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly order: number;
  readonly readyAtEpochMilliseconds: number;
  readonly preflights?: Readonly<Record<string, unknown>>;
}

interface ArmCompleteMessage {
  readonly type: "arm-complete";
  readonly armId: Opt0080ArmId;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly sample: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface RejectedMessage {
  readonly type: "gate-rejected" | "trace-rejected";
  readonly armId: Opt0080ArmId;
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
let activeArmId: Opt0080ArmId | undefined;
let activeSchedulingProfile: Opt0080SchedulingProfile | undefined;
let readyAtEpochMilliseconds = 0;
let cleanupCompletedAtEpochMilliseconds = 0;
let heartbeat: MutableHeartbeat | undefined;
let completedHeartbeat: Opt0080HeartbeatCapture | undefined;
let downloadUrl: string | undefined;
let settled = false;

interface MutableHeartbeat {
  readonly startedAtEpochMilliseconds: number;
  lastAt: number;
  readonly gapsMilliseconds: number[];
}

let lastGlobalHeartbeatAt = performance.now();
const heartbeatTimer = window.setInterval(() => {
  const now = performance.now();
  const gap = now - lastGlobalHeartbeatAt;
  lastGlobalHeartbeatAt = now;
  if (heartbeat !== undefined) {
    heartbeat.gapsMilliseconds.push(gap);
    heartbeat.lastAt = now;
  }
}, OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS);

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
    "running untimed depth1/depth2 exact repeats and cancellation preflight";
  const activeWorker = new Worker(
    new URL("./opt-0080-dit-depth2-completion-epochs-worker.ts", import.meta.url),
    { type: "module", name: "ace-opt-0080-dit-depth2-completion-epochs" },
  );
  worker = activeWorker;
  activeWorker.addEventListener(
    "message",
    (event: MessageEvent<WorkerMessage>) => handleWorkerMessage(event.data),
  );
  activeWorker.addEventListener("error", (event) => {
    publishFailure(event.error ?? event.message);
  });
  activeWorker.postMessage({ type: "prepare", identity });
}, { once: true });

runArm.addEventListener("click", () => {
  if (
    worker === undefined || activeArmId === undefined ||
    activeSchedulingProfile === undefined || readyAtEpochMilliseconds === 0 ||
    settled
  ) return;
  try {
    const thermalGate = parseJson<Opt0080ThermalGate>(gateInput.value);
    runArm.disabled = true;
    gateInput.disabled = true;
    completedHeartbeat = undefined;
    const now = performance.now();
    lastGlobalHeartbeatAt = now;
    heartbeat = {
      startedAtEpochMilliseconds: Date.now(),
      lastAt: now,
      gapsMilliseconds: [],
    };
    progress.textContent = `${activeArmId}: gate validation then evaluation 0`;
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
    cleanupCompletedAtEpochMilliseconds === 0 || completedHeartbeat === undefined ||
    settled
  ) return;
  try {
    const thermalTrace = parseJson<Opt0080ThermalTrace>(traceInput.value);
    recordTrace.disabled = true;
    traceInput.disabled = true;
    progress.textContent = `${activeArmId}: authenticating trace and heartbeat`;
    worker.postMessage({
      type: "complete-thermal",
      armId: activeArmId,
      thermalTrace,
      heartbeat: completedHeartbeat,
    });
  } catch (error) {
    displayLocalError(error);
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
  if (message.type === "ready-for-arm") {
    activeArmId = message.armId;
    activeSchedulingProfile = message.schedulingProfile;
    readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
    cleanupCompletedAtEpochMilliseconds = 0;
    completedHeartbeat = undefined;
    heartbeat = undefined;
    gateInput.value = "";
    traceInput.value = "";
    gateInput.disabled = false;
    traceInput.disabled = true;
    runArm.disabled = false;
    recordTrace.disabled = true;
    progress.textContent =
      `${message.armId} (${message.schedulingProfile}) ready at ` +
      `${message.readyAtEpochMilliseconds}; start a fresh continuous ≥30 s ` +
      "level-0 trace after this READY boundary";
    if (message.preflights !== undefined) {
      result.textContent = JSON.stringify(message.preflights, null, 2);
    }
    return;
  }
  if (message.type === "arm-complete") {
    if (message.armId !== activeArmId || heartbeat === undefined) {
      publishFailure(new Error("OPT-0080 arm/heartbeat completion order changed"));
      return;
    }
    const now = performance.now();
    heartbeat.gapsMilliseconds.push(now - heartbeat.lastAt);
    const gaps = Object.freeze([...heartbeat.gapsMilliseconds]);
    const sorted = [...gaps].sort((left, right) => left - right);
    completedHeartbeat = Object.freeze({
      intervalMilliseconds: OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS,
      startedAtEpochMilliseconds: heartbeat.startedAtEpochMilliseconds,
      completedAtEpochMilliseconds: Date.now(),
      gapsMilliseconds: gaps,
      maximumGapMilliseconds: sorted.at(-1)!,
      p99GapMilliseconds:
        sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)]!,
    });
    heartbeat = undefined;
    cleanupCompletedAtEpochMilliseconds =
      message.cleanupCompletedAtEpochMilliseconds;
    traceInput.disabled = false;
    recordTrace.disabled = false;
    progress.textContent =
      `${message.armId} cleanup/device disposal completed; stop only this ` +
      "arm's poll and paste its through-cleanup trace";
    result.textContent = JSON.stringify({
      armId: message.armId,
      sample: message.sample,
      heartbeat: completedHeartbeat,
      receipt: message.receipt,
    }, null, 2);
    return;
  }
  if (message.type === "gate-rejected") {
    heartbeat = undefined;
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
      `${message.armId}: trace/heartbeat rejected; no next arm was prepared`;
    result.textContent = JSON.stringify(message.error, null, 2);
    return;
  }
  if (message.type === "gate-complete") {
    publish(message.result);
    return;
  }
  publish(Object.freeze({
    schema: "ace-opt-0080-page-failure-v1",
    experimentId: "OPT-0080",
    status: "failed",
    error: message.error,
  }));
}

function publish(receipt: Readonly<Record<string, unknown>>): void {
  settled = true;
  worker?.terminate();
  worker = undefined;
  window.__ACE_OPT0080_RESULT__ = receipt;
  const passed = receipt.status === "passed";
  document.body.dataset.status = passed ? "complete" : "failed";
  progress.textContent = passed
    ? "PASSED — exact scheduling screen verified"
    : "FAILED/INCONCLUSIVE — depth1 remains the production default";
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
    schema: "ace-opt-0080-page-failure-v1",
    experimentId: "OPT-0080",
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
    throw new Error("OPT-0080 thermal receipt must be a JSON object");
  }
  return parsed as Value;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing OPT-0080 element ${selector}`);
  return value;
}
