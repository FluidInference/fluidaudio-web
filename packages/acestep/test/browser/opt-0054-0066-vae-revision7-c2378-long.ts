/// <reference types="vite/client" />

import {
  OPT_0054_0066_LONG_SCHEMA,
} from "./opt-0054-0066-vae-revision7-c2378-long-contract.js";
import type { Opt00540066WorkerCommand } from
  "./opt-0054-0066-vae-revision7-c2378-long-worker.js";

declare global {
  interface Window {
    __ACE_OPT00540066_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface WorkerEvent {
  readonly type: "progress" | "result" | "cancelled" | "disposed" | "error";
  readonly message?: string;
  readonly receipt?: Readonly<Record<string, unknown>>;
}

const runButton = requireElement<HTMLButtonElement>("run");
const cancelButton = requireElement<HTMLButtonElement>("cancel");
const disposeButton = requireElement<HTMLButtonElement>("dispose");
const progress = requireElement<HTMLParagraphElement>("progress");
const result = requireElement<HTMLPreElement>("result");
const worker = new Worker(
  new URL(
    "./opt-0054-0066-vae-revision7-c2378-long-worker.ts",
    import.meta.url,
  ),
  { type: "module" },
);
let terminal = false;

const setStatus = (status: string, message: string): void => {
  document.body.dataset["status"] = status;
  progress.textContent = message;
};

const publishReceipt = (
  receipt: Readonly<Record<string, unknown>>,
): void => {
  window.__ACE_OPT00540066_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt, null, 2);
};

worker.addEventListener("message", (event: MessageEvent<WorkerEvent>) => {
  const data = event.data;
  if (data.type === "progress") {
    setStatus("running", data.message ?? "working");
    return;
  }
  if (data.type === "disposed") {
    terminal = true;
    setStatus("disposed", "isolated worker disposed; no gate is running");
    cancelButton.disabled = true;
    disposeButton.disabled = true;
    worker.terminate();
    return;
  }
  const receipt = data.receipt ?? Object.freeze({
    schema: OPT_0054_0066_LONG_SCHEMA,
    status: data.type === "cancelled" ? "cancelled" : "failed",
    error: data.message ?? `worker emitted ${data.type} without a receipt`,
    productionDefaultChanged: false,
  });
  publishReceipt(receipt);
  cancelButton.disabled = true;
  disposeButton.disabled = false;
  if (data.type === "result" && receipt["status"] === "passed") {
    setStatus(
      "ready",
      "READY — the isolated revision-7 C2378 long-waveform gate passed",
    );
    return;
  }
  if (data.type === "cancelled") {
    setStatus("cancelled", data.message ?? "gate cancelled after cleanup");
    return;
  }
  setStatus("failed", data.message ?? "long-waveform gate failed");
});

worker.addEventListener("error", (event) => {
  const receipt = Object.freeze({
    schema: OPT_0054_0066_LONG_SCHEMA,
    status: "failed",
    error: event.message || "worker module failed",
    productionDefaultChanged: false,
  });
  publishReceipt(receipt);
  setStatus("failed", String(receipt.error));
  cancelButton.disabled = true;
  disposeButton.disabled = false;
});

runButton.addEventListener("click", () => {
  runButton.disabled = true;
  cancelButton.disabled = false;
  disposeButton.disabled = false;
  setStatus(
    "running",
    "starting authenticated cancellation, control, candidate, and repeat arms",
  );
  worker.postMessage({ type: "run" } satisfies Opt00540066WorkerCommand);
}, { once: true });

cancelButton.addEventListener("click", () => {
  cancelButton.disabled = true;
  setStatus("cancelling", "cancellation requested; waiting for bounded cleanup");
  worker.postMessage({ type: "cancel" } satisfies Opt00540066WorkerCommand);
});

disposeButton.addEventListener("click", () => {
  disposeButton.disabled = true;
  cancelButton.disabled = true;
  runButton.disabled = true;
  setStatus("disposing", "disposing worker resources and OPFS artifacts");
  worker.postMessage({ type: "dispose" } satisfies Opt00540066WorkerCommand);
});

window.addEventListener("beforeunload", () => {
  if (!terminal) {
    worker.postMessage({ type: "dispose" } satisfies Opt00540066WorkerCommand);
  }
}, { once: true });

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
}
