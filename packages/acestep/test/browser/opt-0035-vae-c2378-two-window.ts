/// <reference types="vite/client" />

import type {
  Opt0035QuantaPerCommandBuffer,
  Opt0035WorkerCommand,
} from "./opt-0035-vae-c2378-two-window-worker.js";

const OPT_0035_SCHEMA = "ace-opt-0035-vae-c2378-two-window-abba-v1";

declare global {
  interface Window {
    __ACE_OPT0035_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface WorkerEvent {
  readonly type: "progress" | "prepared" | "result" | "disposed" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
  readonly audio?: Blob;
}

const prepareButton = requireElement<HTMLButtonElement>("prepare");
const runButton = requireElement<HTMLButtonElement>("run");
const disposeButton = requireElement<HTMLButtonElement>("dispose");
const batchSelect = requireElement<HTMLSelectElement>("batch");
const progress = requireElement<HTMLParagraphElement>("progress");
const result = requireElement<HTMLPreElement>("result");
const audio = requireElement<HTMLAudioElement>("audio");
const worker = new Worker(
  new URL("./opt-0035-vae-c2378-two-window-worker.ts", import.meta.url),
  { type: "module" },
);
let audioUrl: string | undefined;

const setStatus = (status: string, message: string): void => {
  document.body.dataset["status"] = status;
  progress.textContent = message;
};

worker.addEventListener("message", (event: MessageEvent<WorkerEvent>) => {
  const data = event.data;
  if (data.type === "progress") {
    setStatus(document.body.dataset["status"] ?? "working",
      data.message ?? "working");
    return;
  }
  if (data.type === "prepared") {
    const receipt = data.receipt ?? Object.freeze({});
    result.textContent = JSON.stringify(receipt, null, 2);
    if (data.audio !== undefined) {
      audioUrl = URL.createObjectURL(data.audio);
      audio.src = audioUrl;
      audio.hidden = false;
    }
    setStatus("ready",
      "READY — correctness passed. Wait for nominal thermal state before timing.");
    runButton.disabled = false;
    disposeButton.disabled = false;
    return;
  }
  if (data.type === "result") {
    const receipt = data.receipt ?? Object.freeze({
      schema: OPT_0035_SCHEMA,
      status: "failed",
      error: "missing OPT-0035 receipt",
    });
    window.__ACE_OPT0035_RESULT__ = receipt;
    result.textContent = JSON.stringify(receipt, null, 2);
    setStatus(receipt["status"] === "passed" ? "passed" : "failed",
      receipt["status"] === "passed"
        ? "OPT-0035 C2378 speed and correctness gates passed"
        : "OPT-0035 did not pass every gate");
    disposeButton.disabled = false;
    return;
  }
  if (data.type === "disposed") {
    revokeAudio();
    setStatus("disposed", "OPT-0035 resources and listening artifact removed");
    disposeButton.disabled = true;
    worker.terminate();
    return;
  }
  const receipt = data.receipt ?? Object.freeze({
    schema: OPT_0035_SCHEMA,
    status: "failed",
    error: data.message ?? "worker failed",
  });
  window.__ACE_OPT0035_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt, null, 2);
  setStatus("failed", data.message ?? "OPT-0035 worker failed");
  runButton.disabled = true;
  disposeButton.disabled = false;
});

worker.addEventListener("error", (event) => {
  setStatus("failed", event.message || "OPT-0035 worker failed");
  runButton.disabled = true;
  disposeButton.disabled = false;
});

prepareButton.addEventListener("click", () => {
  prepareButton.disabled = true;
  batchSelect.disabled = true;
  setStatus("preparing",
    "running sequential untimed C512/C2378 correctness and determinism gates");
  worker.postMessage({
    type: "prepare",
    quantaPerCommandBuffer: Number(batchSelect.value) as
      Opt0035QuantaPerCommandBuffer,
  } satisfies Opt0035WorkerCommand);
}, { once: true });

runButton.addEventListener("click", () => {
  runButton.disabled = true;
  setStatus("timing", "running sequential C512/C2378/C2378/C512 timing");
  worker.postMessage({ type: "run" } satisfies Opt0035WorkerCommand);
}, { once: true });

disposeButton.addEventListener("click", () => {
  disposeButton.disabled = true;
  worker.postMessage({ type: "dispose" } satisfies Opt0035WorkerCommand);
}, { once: true });

window.addEventListener("beforeunload", () => {
  worker.postMessage({ type: "dispose" } satisfies Opt0035WorkerCommand);
  revokeAudio();
}, { once: true });

function revokeAudio(): void {
  if (audioUrl === undefined) return;
  URL.revokeObjectURL(audioUrl);
  audioUrl = undefined;
  audio.removeAttribute("src");
  audio.load();
  audio.hidden = true;
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
}
