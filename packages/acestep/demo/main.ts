import {
  DEFAULT_ACE_PLANNER_CONFIGURATION,
  aceSeed,
  isAceFatalGpuErrorCode,
  isAceWorkerMessage,
  releaseAceAudioOutput,
  type AceGenerationRequest,
  type AceGenerationResult,
  type AceWorkerMessage,
} from "../src/index.js";
import { AceIncrementalSha256 } from "../src/model/sha256.js";

const REFERENCE_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
const DIT_MIXED_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f";
const VAE_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7";

const initializeButton = element("initialize", HTMLButtonElement);
const generateButton = element("generate", HTMLButtonElement);
const cancelButton = element("cancel", HTMLButtonElement);
const releaseButton = element("release", HTMLButtonElement);
const promptInput = element("prompt", HTMLTextAreaElement);
const lyricsInput = element("lyrics", HTMLTextAreaElement);
const durationInput = element("duration", HTMLInputElement);
const seedInput = element("seed", HTMLInputElement);
const bpmInput = element("bpm", HTMLInputElement);
const keyScaleInput = element("key-scale", HTMLInputElement);
const timeSignatureInput = element("time-signature", HTMLInputElement);
const vocalLanguageInput = element("vocal-language", HTMLInputElement);
const instrumentalInput = element("instrumental", HTMLInputElement);
const plannerInput = element("planner", HTMLInputElement);
const progress = element("progress", HTMLProgressElement);
const statusText = element("status-text", HTMLOutputElement);
const log = element("log", HTMLPreElement);
const resultCard = element("result-card", HTMLElement);
const audio = element("audio", HTMLAudioElement);
const download = element("download", HTMLAnchorElement);
const resultMetadata = element("result-metadata", HTMLPreElement);

let worker: Worker | undefined;
let ready = false;
let initializing = false;
let initializationRequestId: number | undefined;
let fatalGpuDiagnostic = false;
let activeJobId: number | undefined;
let activeRequest: AceGenerationRequest | undefined;
let activeTrace: Readonly<Record<string, unknown>> = Object.freeze({});
let nextRequestId = 1;
let nextJobId = 1;
let output: { readonly url: string; readonly storageId: string } | undefined;
const pendingReleaseIds = new Set<string>();

initializeButton.addEventListener("click", initialize);
generateButton.addEventListener("click", generate);
cancelButton.addEventListener("click", cancel);
releaseButton.addEventListener("click", () => void releaseOutput());
instrumentalInput.addEventListener("change", () => {
  lyricsInput.disabled = instrumentalInput.checked;
});
lyricsInput.disabled = instrumentalInput.checked;

function initialize(): void {
  if (initializing || ready) return;
  fatalGpuDiagnostic = false;
  initializing = true;
  initializeButton.disabled = true;
  setStatus("Initializing authenticated packed-BF16 reference package…", 0);
  appendLog("Starting a dedicated inference worker.");
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.addEventListener("message", onWorkerMessage);
  worker.addEventListener("error", (event) => {
    worker?.terminate();
    worker = undefined;
    ready = false;
    initializing = false;
    initializationRequestId = undefined;
    initializeButton.disabled = false;
    initializeButton.textContent = "Retry initialization";
    generateButton.disabled = true;
    cancelButton.disabled = true;
    fail(`Worker error: ${event.message}`);
  });
  const requestId = nextRequestId++;
  initializationRequestId = requestId;
  worker.postMessage({
    type: "initialize",
    requestId,
    configuration: {
      manifestUrl: new URL("/model/files-reference/manifest.json", location.href).href,
      manifestSha256: REFERENCE_MANIFEST_SHA256,
      modelProfile: "reference-bf16",
      schedulingProfile: "cooperative",
      ditDensePackage: {
        manifestUrl: new URL(
          "/model/files-fp16-dit-rev7-oracle/manifest.json",
          location.href,
        ).href,
        manifestSha256: DIT_MIXED_MANIFEST_SHA256,
        runtimeProfile: "opt-0009-fp16-fp32-dense-v1",
      },
      ditAttentionRuntimeProfile:
        "opt-0070-fixed32-quad-query32-full-self-production-v1",
      vaePackage: {
        manifestUrl: new URL(
          "/model/files-fp16-vae-revision7-experimental/manifest.json",
          location.href,
        ).href,
        manifestSha256: VAE_MANIFEST_SHA256,
        runtimeProfile:
          "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1",
        windowRuntimeProfile:
          "opt-0070-c2378-overlap64-production-v1",
        maxWindowFrames: 2378,
      },
    },
    modelSource: "cache-or-network",
    reportProgress: true,
    reportDiagnostics: true,
  });
}

function generate(): void {
  if (!ready || worker === undefined || activeJobId !== undefined) return;
  const jobId = nextJobId++;
  activeJobId = jobId;
  generateButton.disabled = true;
  cancelButton.disabled = false;
  setStatus("Preparing generation…", 0);
  appendLog(`Generation ${jobId} started.`);
  try {
    const metadata = {
      ...(bpmInput.value === "" ? {} : { bpm: Number(bpmInput.value) }),
      ...(keyScaleInput.value === "" ? {} : { keyScale: keyScaleInput.value }),
      ...(timeSignatureInput.value === ""
        ? {}
        : { timeSignature: timeSignatureInput.value }),
      ...(vocalLanguageInput.value === ""
        ? {}
        : { vocalLanguage: vocalLanguageInput.value }),
    };
    const request: AceGenerationRequest = {
      generationProfile: "ace-turbo-v1-correctness",
      prompt: promptInput.value,
      lyrics: lyricsInput.value,
      instrumental: instrumentalInput.checked,
      durationSeconds: Number(durationInput.value),
      seed: aceSeed(seedInput.value),
      planner: plannerInput.checked
        ? DEFAULT_ACE_PLANNER_CONFIGURATION
        : { mode: "disabled" },
      ...(Object.keys(metadata).length === 0 ? {} : { metadata }),
    };
    activeRequest = request;
    activeTrace = Object.freeze({});
    worker.postMessage({
      type: "generate",
      jobId,
      request,
      reportProgress: true,
      reportDiagnostics: true,
    });
  } catch (error) {
    activeJobId = undefined;
    activeRequest = undefined;
    activeTrace = Object.freeze({});
    generateButton.disabled = false;
    cancelButton.disabled = true;
    fail(formatError(error));
  }
}

function cancel(): void {
  const jobId = activeJobId;
  if (worker === undefined || jobId === undefined) return;
  cancelButton.disabled = true;
  appendLog(`Cancellation requested for generation ${jobId}.`);
  worker.postMessage({ type: "cancel", jobId });
}

function onWorkerMessage(event: MessageEvent<unknown>): void {
  if (!isAceWorkerMessage(event.data)) {
    fail("Worker emitted a malformed message.");
    return;
  }
  const message = event.data;
  switch (message.type) {
    case "initialization-progress":
      setStatus(message.progress.message ?? message.progress.stage, message.progress.overallFraction);
      break;
    case "ready":
      if (message.requestId !== initializationRequestId) return;
      initializing = false;
      initializationRequestId = undefined;
      ready = true;
      initializeButton.textContent = "Model ready";
      generateButton.disabled = false;
      setStatus(`Ready · ${message.diagnostics.executionProfile.id}`, 1);
      appendLog(`Manifest authenticated: ${message.diagnostics.modelManifestSha256}`);
      break;
    case "initialization-cancelled":
      if (message.requestId !== initializationRequestId) return;
      worker?.terminate();
      worker = undefined;
      initializing = false;
      initializationRequestId = undefined;
      initializeButton.disabled = false;
      initializeButton.textContent = "Retry initialization";
      setStatus("Initialization cancelled", 0);
      break;
    case "generation-progress":
      if (message.jobId !== activeJobId) return;
      setStatus(message.progress.message ?? message.progress.stage, message.progress.overallFraction);
      break;
    case "diagnostic":
      if (
        message.diagnostic.code === "WEBGPU_DEVICE_LOST" ||
        message.diagnostic.code === "WEBGPU_UNCAPTURED_ERROR"
      ) {
        fatalGpuDiagnostic = true;
      }
      appendLog(`[${message.diagnostic.severity}] ${message.diagnostic.code}: ${message.diagnostic.message}`);
      if (
        message.diagnostic.details !== undefined &&
        message.jobId !== undefined &&
        message.jobId === activeJobId &&
        message.diagnostic.code.startsWith("ACE_")
      ) {
        activeTrace = Object.freeze({
          ...activeTrace,
          [message.diagnostic.code]: message.diagnostic.details,
        });
      }
      break;
    case "result":
      if (message.jobId !== activeJobId) return;
      const completedRequest = activeRequest;
      const completedTrace = activeTrace;
      activeJobId = undefined;
      cancelButton.disabled = true;
      activeRequest = undefined;
      activeTrace = Object.freeze({});
      void showResult(
        message.jobId,
        message.result,
        completedRequest,
        completedTrace,
      ).then(() => {
        if (ready && activeJobId === undefined) generateButton.disabled = false;
      }).catch((error: unknown) => {
        if (ready && activeJobId === undefined) generateButton.disabled = false;
        fail(`Could not finalize the result receipt: ${formatError(error)}`);
      });
      break;
    case "cancelled":
      if (message.jobId !== activeJobId) return;
      activeJobId = undefined;
      activeRequest = undefined;
      activeTrace = Object.freeze({});
      cancelButton.disabled = true;
      generateButton.disabled = false;
      setStatus("Generation cancelled and resources released", 0);
      break;
    case "error":
      if (
        message.requestId !== undefined &&
        message.requestId === initializationRequestId
      ) {
        worker?.terminate();
        worker = undefined;
        initializing = false;
        initializationRequestId = undefined;
        initializeButton.disabled = false;
        initializeButton.textContent = "Retry initialization";
      }
      if (message.jobId === undefined || message.jobId === activeJobId) {
        activeJobId = undefined;
        activeRequest = undefined;
        activeTrace = Object.freeze({});
        cancelButton.disabled = true;
        generateButton.disabled = !ready;
      }
      if (
        fatalGpuDiagnostic ||
        isAceFatalGpuErrorCode(message.error.code)
      ) {
        worker?.terminate();
        worker = undefined;
        ready = false;
        initializing = false;
        initializationRequestId = undefined;
        initializeButton.disabled = false;
        initializeButton.textContent = "Reinitialize WebGPU";
        generateButton.disabled = true;
      }
      fail(`${message.error.code}: ${message.error.message}`);
      break;
    case "disposed":
      ready = false;
      break;
  }
}

async function showResult(
  jobId: number,
  result: AceGenerationResult,
  request: AceGenerationRequest | undefined,
  trace: Readonly<Record<string, unknown>>,
): Promise<void> {
  if (request === undefined) {
    throw new Error("The completed generation is missing its submitted request");
  }
  await releaseOutput();
  try {
    const wavSha256 = await sha256Blob(result.audio);
    const receiptBody = Object.freeze({
      schema: "ace-browser-result-receipt-v1",
      jobId,
      submittedRequest: request,
      trace,
      result: Object.freeze({
        durationSeconds: result.durationSeconds,
        seed: result.seed,
        generationProfile: result.generationProfile,
        manifestId: result.modelManifestId,
        manifestSha256: result.modelManifestSha256,
        mimeType: result.mimeType,
        sampleRateHz: result.sampleRateHz,
        channelCount: result.channelCount,
        frameCount: result.frameCount,
        diagnostics: result.diagnostics,
        metrics: result.metrics,
      }),
      wavSha256,
    });
    const receiptSha256 = await sha256Text(JSON.stringify(receiptBody));
    const url = URL.createObjectURL(result.audio);
    output = { url, storageId: result.audioStorageId };
    audio.src = url;
    download.href = url;
    releaseButton.textContent = "Release stored output";
    resultCard.hidden = false;
    resultMetadata.textContent = JSON.stringify({
      ...receiptBody,
      receiptSha256,
    }, null, 2);
    setStatus(`Complete in ${(result.metrics.totalMs / 1000).toFixed(1)} s`, 1);
    appendLog(`WAV ready: ${result.frameCount.toLocaleString()} stereo frames.`);
  } catch (error) {
    if (output?.storageId !== result.audioStorageId) {
      pendingReleaseIds.add(result.audioStorageId);
    }
    await releaseOutput();
    throw error;
  }
}

async function releaseOutput(): Promise<void> {
  const current = output;
  if (current !== undefined) {
    output = undefined;
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    download.removeAttribute("href");
    URL.revokeObjectURL(current.url);
    pendingReleaseIds.add(current.storageId);
  }

  for (const storageId of [...pendingReleaseIds]) {
    try {
      await releaseAceAudioOutput(storageId);
      pendingReleaseIds.delete(storageId);
      appendLog(`Released stored output ${storageId}.`);
    } catch (error) {
      appendLog(`Could not release ${storageId}; retry is available: ${formatError(error)}`);
    }
  }

  if (output === undefined) {
    resultCard.hidden = pendingReleaseIds.size === 0;
    if (pendingReleaseIds.size === 0) resultMetadata.textContent = "";
    if (pendingReleaseIds.size > 0) {
      releaseButton.textContent = "Retry stored-output release";
      resultMetadata.textContent = `${pendingReleaseIds.size} stored output release${pendingReleaseIds.size === 1 ? "" : "s"} pending.`;
    }
  }
}

async function sha256Blob(value: Blob): Promise<string> {
  const hash = new AceIncrementalSha256();
  const reader = value.stream().getReader();
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) return hash.digestHex();
    hash.update(chunk.value);
  }
}

async function sha256Text(value: string): Promise<string> {
  return await sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(value: BufferSource): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", value));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function setStatus(message: string, fraction: number): void {
  statusText.value = message;
  progress.value = fraction;
}

function appendLog(message: string): void {
  const timestamp = new Date().toLocaleTimeString();
  log.textContent += `${log.textContent ? "\n" : ""}[${timestamp}] ${message}`;
  log.scrollTop = log.scrollHeight;
}

function fail(message: string): void {
  setStatus(message, progress.value);
  appendLog(message);
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function element<T extends typeof Element>(id: string, constructor: T): InstanceType<T> {
  const value = document.getElementById(id);
  if (!(value instanceof constructor)) throw new Error(`Missing #${id}`);
  return value as InstanceType<T>;
}

// Preserve committed output until the user explicitly releases it. A page
// unload cannot safely await OPFS deletion and must not invalidate playback.
window.addEventListener("pagehide", () => {
  if (output !== undefined) URL.revokeObjectURL(output.url);
  worker?.terminate();
});

declare global {
  interface DedicatedWorkerGlobalScope {
    postMessage(message: AceWorkerMessage): void;
  }
}
