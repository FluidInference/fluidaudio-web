import {
  parseOpt0018RunIdentity,
  serializeOpt0018Failure,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import { parseOpt0066ThermalGate } from
  "./opt-0066-vae-dual-k4-quality-c512-contract.js";

declare global {
  interface Window {
    __ACE_OPT0066_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ReadyMessage {
  readonly type: "ready-for-thermal-gate";
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface CompleteMessage {
  readonly type: "comparison-complete";
  readonly result: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ReadyMessage | ProgressMessage | CompleteMessage |
  FailedMessage;

const prepareButton = requireElement<HTMLButtonElement>("#prepare");
const runButton = requireElement<HTMLButtonElement>("#run");
const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
const thermalStarted = requireElement<HTMLInputElement>(
  "input[name=thermalStartedAtEpochMilliseconds]",
);
const progress = requireElement<HTMLElement>("#progress");
const result = requireElement<HTMLElement>("#result");
const download = requireElement<HTMLAnchorElement>("#download");

let worker: Worker | undefined;
let readyAtEpochMilliseconds: number | undefined;
let downloadUrl: string | undefined;
let started = false;

prepareButton.addEventListener("click", () => {
  if (worker !== undefined) return;
  try {
    const identity = parseOpt0018RunIdentity(
      new URLSearchParams(window.location.search),
    );
    prepareButton.disabled = true;
    document.body.dataset.status = "preparing";
    progress.textContent =
      "authenticating both packages and running sequential C512 correctness";
    const active = new Worker(
      new URL(
        "./opt-0066-vae-dual-k4-quality-c512-worker.ts",
        import.meta.url,
      ),
      { type: "module", name: "ace-opt-0066-revision7-c512" },
    );
    worker = active;
    active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") {
        progress.textContent = message.message;
        return;
      }
      if (message.type === "ready-for-thermal-gate") {
        readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
        thermalStarted.value = String(Date.now());
        thermalGate.disabled = false;
        runButton.disabled = false;
        document.body.dataset.status = "ready";
        progress.textContent =
          "READY — wait at least 30 seconds, check notifyutil exactly once, " +
          "enter level 0, then run balanced AB/BA";
        result.textContent = JSON.stringify(message.preparation, null, 2);
        return;
      }
      if (message.type === "comparison-complete") {
        publish(message.result, "complete");
        active.terminate();
        worker = undefined;
        return;
      }
      publish(Object.freeze({
        schema: "ace-opt-0066-page-failure-v1",
        experimentId: "OPT-0066",
        status: "failed",
        error: message.error,
      }), "failed");
      active.terminate();
      worker = undefined;
    });
    active.addEventListener("error", (event) => {
      publishFailure(event.error ?? event.message);
      active.terminate();
      worker = undefined;
    });
    active.postMessage({ type: "initialize", identity });
  } catch (error) {
    publishFailure(error);
  }
});

runButton.addEventListener("click", () => {
  if (
    started || worker === undefined || readyAtEpochMilliseconds === undefined
  ) return;
  try {
    const now = Date.now();
    const gate = parseOpt0066ThermalGate(
      thermalParameters(now),
      readyAtEpochMilliseconds,
      now,
    );
    started = true;
    runButton.disabled = true;
    thermalGate.disabled = true;
    document.body.dataset.status = "running";
    progress.textContent =
      "running sequential rev6 / rev7 / rev7 / rev6 C512 timing";
    worker.postMessage({ type: "run", thermalGate: gate });
  } catch (error) {
    publishFailure(error);
  }
});

window.addEventListener("beforeunload", () => {
  worker?.postMessage({ type: "dispose" });
  worker?.terminate();
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
});

function thermalParameters(now: number): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of thermalGate.querySelectorAll<HTMLInputElement>(
    "input[name]",
  )) {
    parameters.set(input.name, input.value);
  }
  if (parameters.get("thermalCheckedAtEpochMilliseconds") === "") {
    parameters.set("thermalCheckedAtEpochMilliseconds", String(now));
  }
  return parameters;
}

function publish(
  receipt: Readonly<Record<string, unknown>>,
  state: "complete" | "failed",
): void {
  window.__ACE_OPT0066_RESULT__ = receipt;
  document.body.dataset.status = state;
  progress.textContent = state === "complete"
    ? "COMPLETE — authenticated OPT-0066 receipt is ready"
    : "FAILED — inspect the preserved receipt";
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
    schema: "ace-opt-0066-page-failure-v1",
    experimentId: "OPT-0066",
    status: "failed",
    error: serializeOpt0018Failure(error),
  }), "failed");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0066 element ${selector}`);
  return element;
}
