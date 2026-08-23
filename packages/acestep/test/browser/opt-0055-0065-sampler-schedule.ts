type ListeningMode = "direct12" | "vocal30";

interface ArtifactMessage {
  readonly label: "A" | "B" | "C";
  readonly fileName: string;
  readonly wav: Blob;
  readonly byteLength: number;
  readonly sha256: string;
}

type WorkerMessage =
  | Readonly<{ readonly type: "progress"; readonly message: string }>
  | Readonly<{
      readonly type: "complete";
      readonly publicReceipt: Readonly<Record<string, unknown>>;
      readonly artifacts: readonly ArtifactMessage[];
    }>
  | Readonly<{
      readonly type: "mapping";
      readonly mapping: readonly Readonly<Record<string, unknown>>[];
      readonly receipt: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{ readonly type: "released" }>
  | Readonly<{
      readonly type: "failed";
      readonly error: Readonly<Record<string, unknown>>;
    }>;

declare global {
  interface Window {
    __ACE_OPT0055_0065_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

const start = element<HTMLButtonElement>("#start");
const reveal = element<HTMLButtonElement>("#reveal");
const release = element<HTMLButtonElement>("#release");
const mode = element<HTMLSelectElement>("#mode");
const listened = element<HTMLInputElement>("#listened");
const notes = element<HTMLTextAreaElement>("#listening-notes");
const progress = element<HTMLElement>("#progress");
const result = element<HTMLElement>("#result");
const artifacts = element<HTMLElement>("#artifacts");

let worker: Worker | undefined;
let objectUrls: string[] = [];
let complete = false;

const requestedMode = new URL(location.href).searchParams.get("mode");
if (requestedMode === "vocal30" || requestedMode === "direct12") {
  mode.value = requestedMode;
}

start.addEventListener("click", () => {
  if (worker !== undefined) return;
  start.disabled = true;
  mode.disabled = true;
  document.body.dataset.status = "running";
  progress.textContent =
    "starting sequential untimed 8/6/5 generation; keep this page open";
  const active = new Worker(
    new URL("./opt-0055-0065-sampler-schedule-worker.ts", import.meta.url),
    { type: "module", name: "ace-opt-0055-0065-sampler-listening" },
  );
  worker = active;
  active.addEventListener("message", (event: MessageEvent<WorkerMessage>) =>
    handleWorkerMessage(event.data)
  );
  active.addEventListener("error", (event) =>
    fail(event.error ?? new Error(event.message))
  );
  active.postMessage({ type: "start", mode: mode.value as ListeningMode });
}, { once: true });

listened.addEventListener("change", () => {
  reveal.disabled = !complete || !listened.checked || notes.value.trim() === "";
});
notes.addEventListener("input", () => {
  reveal.disabled = !complete || !listened.checked || notes.value.trim() === "";
});

reveal.addEventListener("click", () => {
  if (worker === undefined || !complete || reveal.disabled) return;
  reveal.disabled = true;
  listened.disabled = true;
  notes.disabled = true;
  worker.postMessage({ type: "reveal" });
});

release.addEventListener("click", () => {
  if (worker === undefined) return;
  release.disabled = true;
  progress.textContent = "releasing retained WAV/raw OPFS artifacts";
  worker.postMessage({ type: "release" });
});

window.addEventListener("beforeunload", () => {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  worker?.terminate();
});

function handleWorkerMessage(message: WorkerMessage): void {
  if (message.type === "progress") {
    progress.textContent = message.message;
    return;
  }
  if (message.type === "complete") {
    complete = true;
    document.body.dataset.status = "listening";
    progress.textContent =
      "correctness passed — download A/B/C, listen blind, record notes, then reveal";
    result.textContent = JSON.stringify(message.publicReceipt, null, 2);
    renderArtifacts(message.artifacts);
    listened.disabled = false;
    notes.disabled = false;
    return;
  }
  if (message.type === "mapping") {
    document.body.dataset.status = "revealed";
    progress.textContent =
      "mapping revealed — save the receipt and finish downloads before release";
    const finalReceipt = Object.freeze({
      ...message.receipt,
      blindMapping: message.mapping,
      ownerListeningNotes: notes.value.trim(),
    });
    window.__ACE_OPT0055_0065_RESULT__ = finalReceipt;
    result.textContent = JSON.stringify(finalReceipt, null, 2);
    release.disabled = false;
    return;
  }
  if (message.type === "released") {
    document.body.dataset.status = "released";
    progress.textContent = "OPFS artifacts released; downloaded files remain yours";
    worker?.terminate();
    worker = undefined;
    return;
  }
  fail(message.error);
}

function renderArtifacts(values: readonly ArtifactMessage[]): void {
  artifacts.replaceChildren(...values.map((artifact) => {
    const row = document.createElement("li");
    const link = document.createElement("a");
    const url = URL.createObjectURL(artifact.wav);
    objectUrls.push(url);
    link.href = url;
    link.download = artifact.fileName;
    link.textContent =
      `Download blind ${artifact.label} (${artifact.byteLength} bytes)`;
    row.append(link);
    return row;
  }));
}

function fail(error: unknown): void {
  document.body.dataset.status = "failed";
  progress.textContent = "FAILED — no sampler schedule is approved";
  result.textContent = JSON.stringify(
    typeof error === "object" && error !== null
      ? error
      : { message: String(error) },
    null,
    2,
  );
  worker?.terminate();
  worker = undefined;
}

function element<ElementType extends Element>(selector: string): ElementType {
  const value = document.querySelector<ElementType>(selector);
  if (value === null) throw new Error(`Missing sampler harness element ${selector}`);
  return value;
}

