// Music generation page — ACE-Step 1.5 Turbo on WebGPU. This is the upstream
// demo's UI and state machine (ace-step-1.5.wgsl-demo main.ts, MIT, Hamza
// Qayyum) adapted to fluidaudio-web: model identities come from the
// musicgen-acestep engine config and the inference worker lives with the
// engine. Programmatic consumers should use engines/musicgen-acestep's
// AceStepMusicClient instead of duplicating this wiring.

import {
  aceSeed,
  checkSupport,
  deleteAceModelCache,
  inspectAceModelCache,
  isAceFatalGpuErrorCode,
  isAceWorkerMessage,
  releaseAceAudioOutput,
  requestAceModelStoragePersistence,
  type AceGenerationRequest,
  type AceGenerationResult,
  type AceModelCacheInfo,
  type AceSupportReport,
  type AceWorkerMessage,
} from "ace-step-1.5.wgsl";

import lightModeIcon from "./engines/musicgen-acestep/assets/light-mode.png";
import moonIcon from "./engines/musicgen-acestep/assets/moon.png";

import { aceProductionWorkerConfiguration } from "./engines/musicgen-acestep/config.js";
import { aceInferenceWorkerName } from "./engines/musicgen-acestep/worker-name.js";
import {
  formatDecimalBytes,
  formatModelDownloadAmount,
  INITIAL_MODEL_DOWNLOAD_PROGRESS,
  isModelDownloadComplete,
  MODEL_DOWNLOAD_TOTAL_BYTES,
  shouldShowModelDownloadNote,
  updateModelDownloadProgress,
  type ModelDownloadProgress,
} from "./engines/musicgen-acestep/model-download-progress.js";
import { ACE_MODEL_CACHE_LIFECYCLE_LOCK, ensureCurrentAceDemoModelCache } from "./engines/musicgen-acestep/model-cache-migration.js";
import "./engines/musicgen-acestep/style.css";

type DemoTheme = "light" | "dark";

const THEME_STORAGE_KEY = "ace-step-wgsl-demo-theme";
// Upstream repo is not public yet; the button opens the author's live demo.
const PROJECT_REPOSITORY_URL = "https://acestep.narcotic.sh";

const form = requiredElement<HTMLFormElement>("generation-form");
const githubProjectButton = requiredElement<HTMLButtonElement>("github-project-button");
const githubProjectTooltip = requiredElement<HTMLDivElement>("github-project-tooltip");
const promptInput = requiredElement<HTMLTextAreaElement>("prompt");
const lyricsInput = requiredElement<HTMLTextAreaElement>("lyrics");
const durationInput = requiredElement<HTMLInputElement>("duration");
const seedInput = requiredElement<HTMLInputElement>("seed");
const bpmInput = requiredElement<HTMLInputElement>("bpm");
const keyScaleInput = requiredElement<HTMLInputElement>("key-scale");
const timeSignatureInput = requiredElement<HTMLInputElement>("time-signature");
const vocalLanguageInput = requiredElement<HTMLInputElement>("vocal-language");
const formError = requiredElement<HTMLParagraphElement>("form-error");
const generateButton = requiredElement<HTMLButtonElement>("generate");
const cancelButton = requiredElement<HTMLButtonElement>("cancel");
const supportWarning = requiredElement<HTMLParagraphElement>("support-warning");
const downloadNote = requiredElement<HTMLParagraphElement>("download-note");
const progressPanel = requiredElement<HTMLElement>("progress-panel");
const progressTitle = requiredElement<HTMLHeadingElement>("progress-title");
const progressDetail = requiredElement<HTMLParagraphElement>("progress-detail");
const progressPercent = requiredElement<HTMLSpanElement>("progress-percent");
const progressElement = requiredElement<HTMLProgressElement>("progress");
const summaryDuration = requiredElement<HTMLElement>("summary-duration");
const summaryTime = requiredElement<HTMLElement>("summary-time");
const resultPanel = requiredElement<HTMLElement>("result-panel");
const audioPlayer = requiredElement<HTMLAudioElement>("audio-player");
const download = requiredElement<HTMLAnchorElement>("download");
const settingsToggle = requiredElement<HTMLButtonElement>("settings-toggle");
const settingsDialog = requiredElement<HTMLDialogElement>("settings-dialog");
const settingsTitle = requiredElement<HTMLHeadingElement>("settings-dialog-title");
const settingsClose = requiredElement<HTMLButtonElement>("settings-close");
const cacheStatus = requiredElement<HTMLParagraphElement>("cache-status");
const deleteModelButton = requiredElement<HTMLButtonElement>("delete-model");
const runtimeMetrics = requiredElement<HTMLPreElement>("runtime-metrics");
const themeToggle = requiredElement<HTMLButtonElement>("theme-toggle");
const themeIcon = requiredElement<HTMLImageElement>("theme-icon");

const formControls = Array.from(form.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("input, textarea"));

let worker: Worker | undefined;
let workerReady = false;
let initializationRequestId: number | undefined;
let activeJobId: number | undefined;
let pendingRequest: AceGenerationRequest | undefined;
let nextRequestId = 1;
let nextJobId = 1;
let busy = false;
let supportDetails: AceSupportReport | undefined;
let cacheDetails: AceModelCacheInfo | undefined;
let workerDetails: unknown;
let generationDetails: unknown;
let diagnosticDetails: readonly unknown[] = [];
let modelProgress: ModelDownloadProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
let coldDownload = true;
let fatalGpuDiagnostic = false;
let output: { readonly url: string; readonly storageId: string } | undefined;
let tooltipRenderFrame: number | undefined;
let pendingTooltipPoint: { readonly clientX: number } | undefined;
/** Releases the shared model-cache lifecycle lock held while a worker is alive. */
let releaseRuntimeLock: (() => void) | undefined;
let disposal:
  | {
      readonly requestId: number;
      readonly resolve: () => void;
      readonly reject: (reason: unknown) => void;
    }
  | undefined;

configureTheme();
wireEvents();
void initializePage();

function configureTheme(): void {
  const theme: DemoTheme = document.documentElement.dataset.aceDemoTheme === "dark" ? "dark" : "light";
  applyTheme(theme);
}

function applyTheme(theme: DemoTheme): void {
  document.documentElement.dataset.aceDemoTheme = theme;
  const dark = theme === "dark";
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  themeToggle.setAttribute("aria-pressed", String(dark));
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
  themeIcon.src = dark ? lightModeIcon : moonIcon;
  themeIcon.classList.toggle("is-sun", dark);
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta !== null) meta.content = dark ? "#141517" : "#f5f4ef";
}

function wireEvents(): void {
  githubProjectButton.addEventListener("click", () => {
    window.open(PROJECT_REPOSITORY_URL, "_blank", "noopener,noreferrer");
  });
  githubProjectButton.addEventListener("pointerenter", queueProjectTooltip);
  githubProjectButton.addEventListener("pointermove", queueProjectTooltip);
  githubProjectButton.addEventListener("pointerleave", hideProjectTooltip);
  githubProjectButton.addEventListener("focus", showFocusedProjectTooltip);
  githubProjectButton.addEventListener("blur", hideProjectTooltip);

  themeToggle.addEventListener("click", () => {
    const current: DemoTheme = document.documentElement.dataset.aceDemoTheme === "dark" ? "dark" : "light";
    const next: DemoTheme = current === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Theme selection still applies for the current visit.
    }
    applyTheme(next);
  });

  settingsToggle.addEventListener("click", () => {
    if (settingsDialog.open) return;
    settingsDialog.showModal();
    document.documentElement.classList.add("has-modal-dialog");
    settingsTitle.focus({ preventScroll: true });
    void refreshCacheInfo();
  });
  settingsClose.addEventListener("click", () => settingsDialog.close());
  settingsDialog.addEventListener("click", (event) => {
    if (event.target === settingsDialog) settingsDialog.close();
  });
  settingsDialog.addEventListener("close", () => {
    document.documentElement.classList.remove("has-modal-dialog");
    settingsToggle.focus({ preventScroll: true });
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void beginGeneration();
  });
  cancelButton.addEventListener("click", cancelActiveOperation);
  deleteModelButton.addEventListener("click", () => {
    void deleteDownloadedModel();
  });

  window.addEventListener("pagehide", () => {
    if (output !== undefined) {
      URL.revokeObjectURL(output.url);
      // Best-effort only — the reliable path is the pending-output record
      // reclaimed on the next visit (releaseOrphanedOutputs).
      void releaseAceAudioOutput(output.storageId).then(() => forgetPendingOutput(output?.storageId));
    }
    worker?.terminate();
    releaseRuntimeLock?.();
  });
}

function queueProjectTooltip(event: PointerEvent): void {
  pendingTooltipPoint = { clientX: event.clientX };
  if (tooltipRenderFrame !== undefined) return;
  tooltipRenderFrame = requestAnimationFrame(renderProjectTooltip);
}

function renderProjectTooltip(): void {
  tooltipRenderFrame = undefined;
  const point = pendingTooltipPoint;
  pendingTooltipPoint = undefined;
  if (point === undefined) return;
  showProjectTooltipAt(point.clientX);
}

function showFocusedProjectTooltip(): void {
  const button = githubProjectButton.getBoundingClientRect();
  showProjectTooltipAt(button.left + button.width / 2);
}

function showProjectTooltipAt(clientX: number): void {
  githubProjectTooltip.hidden = false;
  const width = githubProjectTooltip.offsetWidth;
  const height = githubProjectTooltip.offsetHeight;
  const button = githubProjectButton.getBoundingClientRect();
  const margin = 8;
  const gap = 12;
  const left = Math.min(window.innerWidth - width - margin, Math.max(margin, clientX + gap));
  const above = button.top - height - gap;
  const top = above >= margin ? above : Math.min(window.innerHeight - height - margin, button.bottom + gap);
  githubProjectTooltip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function hideProjectTooltip(): void {
  pendingTooltipPoint = undefined;
  if (tooltipRenderFrame !== undefined) {
    cancelAnimationFrame(tooltipRenderFrame);
    tooltipRenderFrame = undefined;
  }
  githubProjectTooltip.hidden = true;
}

const PENDING_OUTPUTS_KEY = "ace-step-pending-output-ids";

function readPendingOutputs(): { id: string; at: number }[] {
  try {
    const raw = localStorage.getItem(PENDING_OUTPUTS_KEY);
    const list = raw === null ? [] : (JSON.parse(raw) as { id: string; at: number }[]);
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

function writePendingOutputs(list: { id: string; at: number }[]): void {
  try {
    localStorage.setItem(PENDING_OUTPUTS_KEY, JSON.stringify(list.slice(-20)));
  } catch {
    // Storage unavailable — the OPFS entries just wait for a later visit.
  }
}

function recordPendingOutput(id: string): void {
  writePendingOutputs([...readPendingOutputs().filter((p) => p.id !== id), { id, at: Date.now() }]);
}

function forgetPendingOutput(id: string | undefined): void {
  if (id === undefined) return;
  writePendingOutputs(readPendingOutputs().filter((p) => p.id !== id));
}

/**
 * Committed WAVs are deliberately excluded from the runtime's own cleanup, so
 * a navigation that skipped releaseCurrentOutput() leaves up to ~92 MB per
 * song in persistent OPFS. Reclaim recorded outputs from previous visits —
 * entries younger than an hour are left alone in case another tab still owns
 * them, and are retried on a later visit once stale.
 */
async function releaseOrphanedOutputs(): Promise<void> {
  const cutoff = Date.now() - 60 * 60 * 1000;
  const keep: { id: string; at: number }[] = [];
  for (const entry of readPendingOutputs()) {
    if (entry.at > cutoff && entry.id !== output?.storageId) {
      keep.push(entry);
      continue;
    }
    if (entry.id === output?.storageId) continue;
    await releaseAceAudioOutput(entry.id).catch(() => undefined);
  }
  writePendingOutputs(keep);
}

async function initializePage(): Promise<void> {
  void releaseOrphanedOutputs();
  try {
    await ensureCurrentAceDemoModelCache();
  } catch (error) {
    supportWarning.textContent = `Could not prepare model storage: ${errorMessage(error)} Reload to retry.`;
    supportWarning.className = "support-warning is-error";
    supportWarning.hidden = false;
    updateActionAvailability();
    updateRuntimeDetails();
    return;
  }

  const [supportResult] = await Promise.allSettled([checkSupport({ modelProfile: "reference-bf16" }), refreshCacheInfo()]);
  if (supportResult.status === "rejected") {
    const message = `Could not inspect WebGPU support: ${errorMessage(supportResult.reason)}`;
    supportWarning.textContent = message;
    supportWarning.className = "support-warning is-error";
    supportWarning.hidden = false;
    updateActionAvailability();
    updateRuntimeDetails();
    return;
  }

  supportDetails = supportResult.value;
  if (!supportDetails.supported) {
    const message = supportDetails.errors.join(" ") || "This browser does not support the required WebGPU features.";
    supportWarning.textContent = message;
    supportWarning.className = "support-warning is-error";
    supportWarning.hidden = false;
  } else {
    const warning = supportDetails.warnings.join(" ");
    supportWarning.textContent = warning;
    supportWarning.className = "support-warning";
    supportWarning.hidden = warning === "";
  }
  updateActionAvailability();
  updateRuntimeDetails();
}

async function beginGeneration(): Promise<void> {
  if (busy || supportDetails?.supported !== true) return;
  let request: AceGenerationRequest;
  try {
    request = readGenerationRequest();
  } catch (error) {
    formError.textContent = errorMessage(error);
    formError.hidden = false;
    return;
  }
  formError.hidden = true;
  formError.textContent = "";
  try {
    await releaseCurrentOutput();
  } catch (error) {
    formError.textContent = `Could not release the previous song: ${errorMessage(error)}`;
    formError.hidden = false;
    return;
  }
  pendingRequest = request;
  generationDetails = undefined;
  diagnosticDetails = [];
  fatalGpuDiagnostic = false;
  modelProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
  coldDownload = !isModelDownloadComplete(cacheDetails);
  setBusy(true);
  resultPanel.hidden = true;

  void requestAceModelStoragePersistence().then(
    (persisted) => {
      workerDetails = { ...recordValue(workerDetails), storagePersisted: persisted };
      updateRuntimeDetails();
    },
    () => {
      // Persistence is advisory; authenticated OPFS caching remains available.
    },
  );

  if (workerReady && worker !== undefined) {
    startPendingGeneration();
    return;
  }
  startWorkerInitialization();
}

function readGenerationRequest(): AceGenerationRequest {
  const prompt = promptInput.value.trim();
  if (prompt.length === 0) throw new Error("Enter a prompt for the song.");

  const minutes = Number(durationInput.value);
  if (!Number.isFinite(minutes)) throw new Error("Enter a duration in minutes.");
  const durationSeconds = Math.round(minutes * 60);
  if (durationSeconds < 10 || durationSeconds > 240) {
    throw new Error("Duration must be between 10 seconds and 4 minutes.");
  }

  const bpmText = bpmInput.value.trim();
  const bpm = bpmText === "" ? undefined : Number(bpmText);
  if (bpm !== undefined && (!Number.isSafeInteger(bpm) || bpm < 30 || bpm > 300)) {
    throw new Error("BPM must be a whole number from 30 through 300.");
  }

  const lyrics = lyricsInput.value;
  const instrumental = lyrics.trim().length === 0;
  const metadata = {
    ...(bpm === undefined ? {} : { bpm }),
    ...optionalText("keyScale", keyScaleInput.value),
    ...optionalText("timeSignature", timeSignatureInput.value),
    ...optionalText("vocalLanguage", vocalLanguageInput.value),
  };
  const seed = seedInput.value.trim() === "" ? randomAceSeed() : aceSeed(seedInput.value.trim());

  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt,
    lyrics,
    instrumental,
    durationSeconds,
    seed,
    planner: { mode: "disabled" },
    ...(Object.keys(metadata).length === 0 ? {} : { metadata }),
  };
}

function acquireRuntimeLock(): void {
  if (releaseRuntimeLock !== undefined || typeof navigator.locks?.request !== "function") return;
  // Shared mode: many tabs may run concurrently; the migration's exclusive
  // request (a future generation bump in a new tab) waits until every tab's
  // runtime has shut down instead of deleting the cache out from under one.
  void navigator.locks.request(
    ACE_MODEL_CACHE_LIFECYCLE_LOCK,
    { mode: "shared" },
    () =>
      new Promise<void>((resolve) => {
        releaseRuntimeLock = resolve;
      }),
  );
}

function startWorkerInitialization(): void {
  acquireRuntimeLock();
  worker?.terminate();
  workerReady = false;
  worker = new Worker(new URL("./engines/musicgen-acestep/worker.ts", import.meta.url), {
    type: "module",
    name: aceInferenceWorkerName(),
  });
  worker.addEventListener("message", onWorkerMessage);
  worker.addEventListener("error", onWorkerError);
  initializationRequestId = nextRequestId++;
  setIndeterminateProgress("Preparing model", "Checking WebGPU and browser storage");
  worker.postMessage({
    type: "initialize",
    requestId: initializationRequestId,
    configuration: aceProductionWorkerConfiguration(),
    modelSource: "cache-or-network",
    reportProgress: true,
    reportDiagnostics: true,
  });
}

function startPendingGeneration(): void {
  const request = pendingRequest;
  if (worker === undefined || !workerReady || request === undefined) return;
  pendingRequest = undefined;
  const jobId = nextJobId++;
  activeJobId = jobId;
  setDeterminateProgress(0, "Generating song", "Preparing inputs", "0%");
  worker.postMessage({
    type: "generate",
    jobId,
    request,
    reportProgress: true,
    reportDiagnostics: true,
  });
}

function cancelActiveOperation(): void {
  if (worker === undefined || !busy) return;
  cancelButton.disabled = true;
  if (initializationRequestId !== undefined) {
    worker.postMessage({
      type: "cancel-initialization",
      requestId: initializationRequestId,
    });
  } else if (activeJobId !== undefined) {
    worker.postMessage({ type: "cancel", jobId: activeJobId });
  }
}

function onWorkerMessage(event: MessageEvent<unknown>): void {
  if (!isAceWorkerMessage(event.data)) {
    failOperation("The inference worker emitted an invalid message.", true);
    return;
  }
  const message = event.data;
  switch (message.type) {
    case "initialization-progress": {
      if (message.requestId !== initializationRequestId) return;
      const updated = updateModelDownloadProgress(modelProgress, message);
      if (updated !== modelProgress) {
        modelProgress = updated;
        renderModelProgress();
      } else {
        setIndeterminateProgress("Preparing model", friendlyProgressMessage(message.progress.message, message.progress.stage));
      }
      return;
    }
    case "ready":
      if (message.requestId !== initializationRequestId) return;
      initializationRequestId = undefined;
      workerReady = true;
      workerDetails = message.diagnostics;
      updateRuntimeDetails();
      void refreshCacheInfo();
      startPendingGeneration();
      return;
    case "initialization-cancelled":
      if (message.requestId !== initializationRequestId) return;
      initializationRequestId = undefined;
      pendingRequest = undefined;
      resetWorker();
      setBusy(false);
      setDeterminateProgress(modelProgress.fraction, "Cancelled", formatModelDownloadAmount(modelProgress), `${modelProgress.percentage.toFixed(1)}%`);
      void refreshCacheInfo();
      return;
    case "generation-progress": {
      if (message.jobId !== activeJobId) return;
      const updated = updateModelDownloadProgress(modelProgress, message);
      if (updated !== modelProgress) {
        modelProgress = updated;
        renderModelProgress();
      } else {
        const fraction = clampFraction(message.progress.overallFraction);
        setDeterminateProgress(
          fraction,
          "Generating song",
          friendlyProgressMessage(message.progress.message, message.progress.stage),
          `${Math.min(99, Math.round(fraction * 100))}%`,
        );
      }
      return;
    }
    case "diagnostic":
      if (message.diagnostic.code === "WEBGPU_DEVICE_LOST" || message.diagnostic.code === "WEBGPU_UNCAPTURED_ERROR") {
        fatalGpuDiagnostic = true;
      }
      diagnosticDetails = [...diagnosticDetails.slice(-19), message.diagnostic];
      updateRuntimeDetails();
      return;
    case "result":
      if (message.jobId !== activeJobId) return;
      activeJobId = undefined;
      void publishResult(message.result);
      return;
    case "cancelled":
      if (message.jobId !== activeJobId) return;
      activeJobId = undefined;
      pendingRequest = undefined;
      setBusy(false);
      setDeterminateProgress(
        progressElement.value,
        "Cancelled",
        "The partial output was removed",
        `${Math.round(clampFraction(progressElement.value) * 100)}%`,
      );
      return;
    case "disposed":
      if (disposal?.requestId !== message.requestId) return;
      disposal.resolve();
      disposal = undefined;
      workerReady = false;
      return;
    case "error": {
      if (disposal !== undefined && message.requestId === disposal.requestId) {
        disposal.reject(new Error(message.error.message));
        disposal = undefined;
        return;
      }
      const fatal = fatalGpuDiagnostic || isAceFatalGpuErrorCode(message.error.code);
      failOperation(`${message.error.code}: ${message.error.message}`, fatal);
      return;
    }
  }
}

function onWorkerError(event: ErrorEvent): void {
  failOperation(`Inference worker error: ${event.message}`, true);
}

async function publishResult(result: AceGenerationResult): Promise<void> {
  try {
    await releaseCurrentOutput();
    const url = URL.createObjectURL(result.audio);
    output = { url, storageId: result.audioStorageId };
    recordPendingOutput(result.audioStorageId);
    audioPlayer.src = url;
    audioPlayer.load();
    download.href = url;
    download.download = `ace-step-${result.seed}.wav`;
    summaryDuration.textContent = formatDuration(result.durationSeconds);
    summaryTime.textContent = formatElapsed(result.metrics.totalMs);
    resultPanel.hidden = false;
    generationDetails = {
      durationSeconds: result.durationSeconds,
      seed: result.seed,
      sampleRateHz: result.sampleRateHz,
      channelCount: result.channelCount,
      frameCount: result.frameCount,
      modelManifestSha256: result.modelManifestSha256,
      metrics: result.metrics,
    };
    modelProgress = updateModelDownloadProgress(modelProgress, {
      stage: "vae-load",
      message: "network: complete 168791552/168791552 bytes",
    });
    setBusy(false);
    progressPanel.hidden = true;
    updateRuntimeDetails();
    await refreshCacheInfo();
  } catch (error) {
    if (output?.storageId !== result.audioStorageId) {
      await releaseAceAudioOutput(result.audioStorageId).catch(() => undefined);
    }
    failOperation(`Could not publish the WAV: ${errorMessage(error)}`, false);
  }
}

async function releaseCurrentOutput(): Promise<void> {
  const current = output;
  if (current === undefined) return;
  output = undefined;
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  download.removeAttribute("href");
  URL.revokeObjectURL(current.url);
  await releaseAceAudioOutput(current.storageId);
  forgetPendingOutput(current.storageId);
}

function failOperation(message: string, reset: boolean): void {
  initializationRequestId = undefined;
  activeJobId = undefined;
  pendingRequest = undefined;
  if (disposal !== undefined) {
    // A pending dispose() would otherwise await forever once the worker dies.
    disposal.reject(new Error(message));
    disposal = undefined;
  }
  if (reset) resetWorker();
  setBusy(false);
  setDeterminateProgress(progressElement.value, "Generation failed", message, `${Math.round(clampFraction(progressElement.value) * 100)}%`);
  void refreshCacheInfo();
}

function resetWorker(): void {
  releaseRuntimeLock?.();
  releaseRuntimeLock = undefined;
  if (disposal !== undefined) {
    disposal.reject(new Error("worker reset while a dispose was pending"));
    disposal = undefined;
  }
  worker?.terminate();
  worker = undefined;
  workerReady = false;
  workerDetails = undefined;
}

async function refreshCacheInfo(): Promise<void> {
  try {
    cacheDetails = await inspectAceModelCache();
    if (!cacheDetails.supported) {
      cacheStatus.textContent = "Model storage is unavailable in this context.";
    } else if (cacheDetails.assetCount === 0 && cacheDetails.partialAssetCount === 0) {
      cacheStatus.textContent = `Not downloaded · ${formatDecimalBytes(MODEL_DOWNLOAD_TOTAL_BYTES)} on first generation`;
    } else {
      const partial = cacheDetails.partialAssetCount === 0 ? "" : ` · ${cacheDetails.partialAssetCount} incomplete`;
      const persistence = cacheDetails.persisted ? "persistent browser storage" : "browser-managed storage";
      cacheStatus.textContent = `${formatDecimalBytes(cacheDetails.sizeBytes)} · ` + `${cacheDetails.assetCount} files${partial} · ${persistence}`;
    }
  } catch (error) {
    cacheDetails = undefined;
    cacheStatus.textContent = `Could not inspect model storage: ${errorMessage(error)}`;
  }
  downloadNote.hidden = !shouldShowModelDownloadNote(cacheDetails);
  updateActionAvailability();
  updateRuntimeDetails();
}

async function deleteDownloadedModel(): Promise<void> {
  if (busy || !cacheCanBeDeleted()) return;
  deleteModelButton.disabled = true;
  cacheStatus.textContent = "Releasing the runtime…";
  try {
    await disposeWorker();
    cacheStatus.textContent = "Deleting downloaded model…";
    await deleteAceModelCache();
    modelProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
    await refreshCacheInfo();
  } catch (error) {
    cacheStatus.textContent = `Could not delete the model: ${errorMessage(error)}`;
  } finally {
    updateActionAvailability();
  }
}

async function disposeWorker(): Promise<void> {
  const current = worker;
  if (current === undefined) return;
  if (!workerReady) {
    resetWorker();
    return;
  }
  const requestId = nextRequestId++;
  await new Promise<void>((resolve, reject) => {
    disposal = { requestId, resolve, reject };
    current.postMessage({ type: "dispose", requestId });
  });
  current.terminate();
  if (worker === current) worker = undefined;
  workerReady = false;
  releaseRuntimeLock?.();
  releaseRuntimeLock = undefined;
}

function renderModelProgress(): void {
  const title = coldDownload ? "Downloading model" : "Preparing model data";
  setDeterminateProgress(modelProgress.fraction, title, formatModelDownloadAmount(modelProgress), `${modelProgress.percentage.toFixed(1)}%`);
}

function setBusy(value: boolean): void {
  busy = value;
  form.setAttribute("aria-busy", String(value));
  for (const control of formControls) control.disabled = value;
  cancelButton.disabled = !value;
  updateActionAvailability();
}

function updateActionAvailability(): void {
  generateButton.disabled = busy || supportDetails?.supported !== true;
  cancelButton.disabled = !busy;
  deleteModelButton.disabled = busy || !cacheCanBeDeleted();
}

function cacheCanBeDeleted(): boolean {
  return cacheDetails?.supported === true && (cacheDetails.assetCount > 0 || cacheDetails.partialAssetCount > 0);
}

function setDeterminateProgress(value: number, title: string, detail: string, percentage: string): void {
  progressPanel.hidden = false;
  progressElement.max = 1;
  progressElement.value = clampFraction(value);
  progressTitle.textContent = title;
  progressDetail.textContent = detail;
  progressPercent.textContent = percentage;
}

function setIndeterminateProgress(title: string, detail: string): void {
  progressPanel.hidden = false;
  progressElement.removeAttribute("value");
  progressTitle.textContent = title;
  progressDetail.textContent = detail;
  progressPercent.textContent = "";
}

function updateRuntimeDetails(): void {
  runtimeMetrics.textContent = JSON.stringify(
    {
      support: supportDetails ?? null,
      modelCache: cacheDetails ?? null,
      runtime: workerDetails ?? null,
      generation: generationDetails ?? null,
      diagnostics: diagnosticDetails,
    },
    null,
    2,
  );
}

function optionalText<Key extends string>(key: Key, value: string): Readonly<Record<Key, string>> | Record<string, never> {
  const text = value.trim();
  return text === "" ? {} : ({ [key]: text } as Record<Key, string>);
}

function randomAceSeed(): ReturnType<typeof aceSeed> {
  const words = crypto.getRandomValues(new Uint32Array(2));
  const value = (BigInt(words[0]!) << 32n) | BigInt(words[1]!);
  return aceSeed(value);
}

function friendlyProgressMessage(message: string | undefined, stage: string): string {
  if (message === undefined || message.trim() === "") {
    return stage.replaceAll("-", " ");
  }
  const withoutFile = message.replace(/^(?:cache|network):\s+.+?(?=\s+[0-9]+\/[0-9]+ bytes$)/u, "Processing model data");
  return withoutFile.length > 120 ? `${withoutFile.slice(0, 117)}…` : withoutFile;
}

function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function formatElapsed(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "—";
  const seconds = milliseconds / 1_000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const whole = Math.round(seconds); // round then split so 119.6s is 2m 0s, not 1m 60s
  return `${Math.floor(whole / 60)}m ${whole % 60}s`;
}

function clampFraction(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
}

declare global {
  interface DedicatedWorkerGlobalScope {
    postMessage(message: AceWorkerMessage): void;
  }
}
