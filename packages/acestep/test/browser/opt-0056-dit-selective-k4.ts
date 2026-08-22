import { serializeOpt0018Failure } from
  "./opt-0018-dit-m2250-production-family-profile.js";

declare global {
  interface Window {
    __ACE_OPT0056_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface ProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface CompleteMessage {
  readonly type: "gate-complete";
  readonly result: Readonly<Record<string, unknown>>;
}

interface FailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

type WorkerMessage = ProgressMessage | CompleteMessage | FailedMessage;

const runButton = requireElement<HTMLButtonElement>("#run");
const progress = requireElement<HTMLElement>("#progress");
const result = requireElement<HTMLElement>("#result");
const download = requireElement<HTMLAnchorElement>("#download");

let worker: Worker | undefined;
let downloadUrl: string | undefined;

runButton.addEventListener("click", () => {
  if (worker !== undefined || runButton.disabled) return;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  progress.textContent =
    "running four-shape primitive identity, then four sequential M2250 arms";
  const active = new Worker(
    new URL("./opt-0056-dit-selective-k4-worker.ts", import.meta.url),
    { type: "module", name: "ace-opt-0056-selective-k4-trajectory-gate" },
  );
  worker = active;
  active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    const message = event.data;
    if (message.type === "progress") {
      progress.textContent = message.message;
      return;
    }
    if (message.type === "gate-complete") {
      const passed = message.result.status === "passed-correctness-gate";
      publish(message.result, passed ? "complete" : "failed");
    } else {
      publish(Object.freeze({
        schema: "ace-opt-0056-page-failure-v1",
        experimentId: "OPT-0056",
        status: "failed",
        error: message.error,
      }), "failed");
    }
    active.terminate();
    worker = undefined;
  });
  active.addEventListener("error", (event) => {
    publishFailure(event.error ?? event.message);
    active.terminate();
    worker = undefined;
  });
  active.postMessage({ type: "run" });
}, { once: true });

window.addEventListener("beforeunload", () => {
  worker?.terminate();
  if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
});

function publish(
  receipt: Readonly<Record<string, unknown>>,
  status: "complete" | "failed",
): void {
  window.__ACE_OPT0056_RESULT__ = receipt;
  document.body.dataset.status = status;
  progress.textContent = status === "complete"
    ? "PASSED — eight-tap selective correctness receipt is ready"
    : "FAILED — inspect the preserved correctness receipt";
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
    schema: "ace-opt-0056-page-failure-v1",
    experimentId: "OPT-0056",
    status: "failed",
    error: serializeOpt0018Failure(error),
  }), "failed");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0056 element ${selector}`);
  return element;
}
