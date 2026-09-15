// Music generation page — ACE-Step 1.5 Turbo on WebGPU. This is the upstream
// demo's UI and state machine (ace-step-1.5.wgsl-demo main.ts, MIT, Hamza
// Qayyum) adapted to fluidaudio-web: model identities come from the
// musicgen-acestep engine config and the inference worker lives with the
// engine. Programmatic consumers should use engines/musicgen-acestep's
// AceStepMusicClient instead of duplicating this wiring.

import {
  aceSeed,
  checkSupport,
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
import { CRASH_BREADCRUMB_KEY, writeProgressBreadcrumb } from "./engines/musicgen-acestep/progress-breadcrumb.js";
import { claimPendingOutput, forgetPendingOutput, reclaimOrphanedOutputs } from "./engines/musicgen-acestep/pending-output-registry.js";
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
import { acquireAceDemoModelCache, deleteAceDemoModelCache } from "./engines/musicgen-acestep/model-cache-migration.js";
import { waitForWorkerDisposal, type PendingWorkerDisposal } from "./engines/musicgen-acestep/worker-disposal.js";
import { pcmToWav } from "./core/audio.js";
import { localWeightDir } from "./engines/registry.js";
import type { DicoseStemEngine } from "./engines/stem-dicose/index.js";
import type { StemAudio } from "./core/types.js";
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
const splitStemsButton = requiredElement<HTMLButtonElement>("split-stems");
const stemsPanel = requiredElement<HTMLDivElement>("stems-panel");
const stemsTime = requiredElement<HTMLSpanElement>("stems-time");
const stemsList = requiredElement<HTMLUListElement>("stems-list");
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
let deletingModel = false;
let cacheAcquisition: AbortController | undefined;
let generationPreparation: AbortController | undefined;
let initializationCancelRequested = false;
const pageLifecycle = new AbortController();
let supportDetails: AceSupportReport | undefined;
let cacheDetails: AceModelCacheInfo | undefined;
let workerDetails: unknown;
let generationDetails: unknown;
let diagnosticDetails: readonly unknown[] = [];
let modelProgress: ModelDownloadProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
let coldDownload = true;
let fatalGpuDiagnostic = false;
let output: { readonly url: string; readonly storageId: string; readonly releaseOwnership: () => Promise<void> } | undefined;
/** The generated song's WAV blob, kept for stem separation (DiCoSe decodes it directly). */
let resultBlob: Blob | undefined;
let resultSeed: string | number | bigint = "song";
/** Lazy DiCoSe engine — 623 MB of weights + a ~625 MB GPU buffer, so it only
 * exists between a "Split stems" click and the next panel reset / pagehide. */
let stemEngine: DicoseStemEngine | undefined;
let stemDisposal: Promise<void> | undefined;
let stemUrls: string[] = [];
let splittingStems = false;
let tooltipRenderFrame: number | undefined;
let pendingTooltipPoint: { readonly clientX: number } | undefined;
/** Releases the shared model-cache lifecycle lock held while a worker is alive. */
let releaseRuntimeLock: (() => Promise<void>) | undefined;
let disposal: PendingWorkerDisposal | undefined;

// Crash breadcrumb: iOS jetsam kills the tab with no error event, so persist
// the last progress stage; after an unclean end the next visit reports where
// the previous attempt died (the only telemetry a killed tab can leave).

function recordBreadcrumb(title: string, detail: string): void {
  // Mirror to the console so a tethered Web Inspector (iPhone debugging)
  // streams the stages — the last line before "Webpage Crashed" is the
  // memory-kill diagnosis.
  console.info(`[ace] ${title}${detail ? ` — ${detail}` : ""}`);
  writeProgressBreadcrumb(title, detail, busy || splittingStems);
}

function closeBreadcrumb(): void {
  try {
    const raw = localStorage.getItem(CRASH_BREADCRUMB_KEY);
    if (raw === null) return;
    const record = JSON.parse(raw) as { open?: boolean };
    localStorage.setItem(CRASH_BREADCRUMB_KEY, JSON.stringify({ ...record, open: false }));
  } catch {
    // Best-effort.
  }
}

function reportCrashBreadcrumb(): void {
  try {
    const raw = localStorage.getItem(CRASH_BREADCRUMB_KEY);
    if (raw === null) return;
    const record = JSON.parse(raw) as { title?: string; detail?: string; at?: number; open?: boolean };
    if (record.open !== true || typeof record.title !== "string") return;
    localStorage.removeItem(CRASH_BREADCRUMB_KEY);
    const when = typeof record.at === "number" ? new Date(record.at).toLocaleTimeString() : "?";
    console.warn(`[ace] previous attempt ended unexpectedly during: ${record.title} — ${record.detail ?? ""}`);
    supportWarning.textContent =
      `The previous attempt ended unexpectedly during: ${record.title}` +
      `${record.detail ? ` — ${record.detail}` : ""} (${when}). ` +
      "If this repeats on a phone, note this stage — it identifies the memory bottleneck.";
    supportWarning.className = "support-warning";
    supportWarning.hidden = false;
  } catch {
    // Best-effort.
  }
}

configureTheme();
wireEvents();
reportCrashBreadcrumb();
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
  splitStemsButton.addEventListener("click", () => {
    void splitStems();
  });

  window.addEventListener("pagehide", () => {
    pageLifecycle.abort();
    generationPreparation?.abort();
    void resetStemSplitter().catch(() => undefined);
    if (output !== undefined) {
      const currentOutput = output;
      output = undefined;
      resultBlob = undefined;
      URL.revokeObjectURL(currentOutput.url);
      // Best-effort only — the reliable path is the pending-output record
      // reclaimed on the next visit (releaseOrphanedOutputs).
      void releaseOwnedOutput(currentOutput).catch(() => undefined);
    }
    resetWorker();
    setBusy(false);
    closeBreadcrumb();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) location.reload();
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

/**
 * Committed WAVs are deliberately excluded from the runtime's own cleanup, so
 * a navigation that skipped releaseCurrentOutput() leaves up to ~92 MB per
 * song in persistent OPFS. Reclaim recorded outputs from previous visits —
 * entries younger than an hour are left alone in case another tab still owns
 * them, and are retried on a later visit once stale.
 */
async function releaseOrphanedOutputs(): Promise<void> {
  await reclaimOrphanedOutputs(output?.storageId, releaseAceAudioOutput);
}

async function initializePage(): Promise<void> {
  void releaseOrphanedOutputs();
  try {
    const release = await acquireAceDemoModelCache(pageLifecycle.signal);
    await release();
  } catch (error) {
    if (pageLifecycle.signal.aborted) return;
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
  if (pageLifecycle.signal.aborted || busy || deletingModel || splittingStems || supportDetails?.supported !== true) return;
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
  const preparation = new AbortController();
  generationPreparation = preparation;
  setBusy(true);
  try {
    await releaseCurrentOutput();
    if (pageLifecycle.signal.aborted) return;
  } catch (error) {
    setBusy(false);
    formError.textContent = `Could not release the previous song: ${errorMessage(error)}`;
    formError.hidden = false;
    return;
  } finally {
    generationPreparation = undefined;
  }
  if (preparation.signal.aborted) {
    setBusy(false);
    setDeterminateProgress(0, "Cancelled", "Song generation cancelled", "");
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
  void startWorkerInitialization();
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

async function startWorkerInitialization(): Promise<void> {
  resetWorker();
  const acquisition = new AbortController();
  cacheAcquisition = acquisition;
  setIndeterminateProgress("Preparing model", "Waiting for model storage");
  try {
    const release = await acquireAceDemoModelCache(acquisition.signal);
    if (cacheAcquisition !== acquisition || acquisition.signal.aborted) {
      await release();
      return;
    }
    cacheAcquisition = undefined;
    releaseRuntimeLock = release;
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
  } catch (error) {
    if (acquisition.signal.aborted) return;
    failOperation(`Could not initialize the model: ${errorMessage(error)}`, true);
  }
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
  if (!busy) return;
  if (generationPreparation !== undefined) {
    generationPreparation.abort();
    cancelButton.disabled = true;
    setIndeterminateProgress("Cancelling", "Releasing the previous song");
    return;
  }
  if (cacheAcquisition !== undefined) {
    resetWorker();
    pendingRequest = undefined;
    setBusy(false);
    setDeterminateProgress(0, "Cancelled", "Model preparation cancelled", "");
    return;
  }
  if (worker === undefined) return;
  cancelButton.disabled = true;
  if (initializationRequestId !== undefined) {
    initializationCancelRequested = true;
    pendingRequest = undefined;
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
      if (initializationCancelRequested) {
        initializationCancelRequested = false;
        setBusy(false);
        setDeterminateProgress(modelProgress.fraction, "Cancelled", "Song generation cancelled", "");
        return;
      }
      startPendingGeneration();
      return;
    case "initialization-cancelled":
      if (message.requestId !== initializationRequestId) return;
      initializationRequestId = undefined;
      initializationCancelRequested = false;
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
      failOperation(`${message.error.code}: ${message.error.message}`, fatal || !workerReady);
      return;
    }
  }
}

function onWorkerError(event: ErrorEvent): void {
  failOperation(`Inference worker error: ${event.message}`, true);
}

async function publishResult(result: AceGenerationResult): Promise<void> {
  let releaseOwnership: (() => Promise<void>) | undefined;
  try {
    await releaseCurrentOutput();
    releaseOwnership = await claimPendingOutput(result.audioStorageId);
    if (pageLifecycle.signal.aborted) throw new DOMException("Page closed", "AbortError");
    const url = URL.createObjectURL(result.audio);
    output = { url, storageId: result.audioStorageId, releaseOwnership };
    releaseOwnership = undefined;
    resultBlob = result.audio;
    resultSeed = result.seed;
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
    closeBreadcrumb();
    setBusy(false);
    progressPanel.hidden = true;
    updateRuntimeDetails();
    await refreshCacheInfo();
  } catch (error) {
    if (output?.storageId !== result.audioStorageId) {
      try {
        await releaseAceAudioOutput(result.audioStorageId);
        await forgetPendingOutput(result.audioStorageId);
      } catch {
        // The pending record remains so a later visit retries cleanup.
      } finally {
        await releaseOwnership?.();
      }
    }
    if (pageLifecycle.signal.aborted) return;
    failOperation(`Could not publish the WAV: ${errorMessage(error)}`, false);
  }
}

async function releaseCurrentOutput(): Promise<void> {
  await resetStemSplitter();
  const current = output;
  if (current === undefined) return;
  output = undefined;
  resultBlob = undefined;
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  download.removeAttribute("href");
  URL.revokeObjectURL(current.url);
  await releaseOwnedOutput(current);
}

async function releaseOwnedOutput(current: { readonly storageId: string; readonly releaseOwnership: () => Promise<void> }): Promise<void> {
  try {
    await releaseAceAudioOutput(current.storageId);
    await forgetPendingOutput(current.storageId);
  } finally {
    await current.releaseOwnership();
  }
}

// ── Split stems (DiCoSe, engines/stem-dicose) ────────────────────────────────
// Suno-style follow-up on a finished song: lazy-load the DiCoSe engine, run
// fast-mode separation on the generated WAV, and render one player + WAV
// download per stem. Progress reuses the generation progress panel.

async function splitStems(): Promise<void> {
  const blob = resultBlob;
  if (splittingStems || busy || deletingModel || blob === undefined) return;
  splittingStems = true;
  updateActionAvailability();
  resetStemsUi();
  try {
    setIndeterminateProgress("Splitting stems", "Loading the DiCoSe separator");
    const { DicoseStemEngine } = await import("./engines/stem-dicose/index.js");
    if (stemEngine === undefined) {
      // Same local-weights-first behavior as the registry entry: a dev-served
      // models-local/dicose export beats the 623 MB HF download.
      const baseUrl = await localWeightDir("dicose", "manifest.json");
      stemEngine = new DicoseStemEngine(baseUrl ? { baseUrl } : {});
    }
    await stemEngine.load((p) => {
      if (p.total > 0) {
        setDeterminateProgress(
          p.fraction,
          "Downloading stem model",
          `${formatDecimalBytes(p.loaded)} of ${formatDecimalBytes(p.total)}`,
          `${(p.fraction * 100).toFixed(1)}%`,
        );
      } else {
        setIndeterminateProgress("Preparing stem model", p.file);
      }
    });
    setIndeterminateProgress("Splitting stems", "Decoding the generated song");
    const input = await stemEngine.decodeFile(await blob.arrayBuffer());
    setIndeterminateProgress("Splitting stems", "Separating drums, bass, other, vocals");
    const t0 = performance.now();
    const stems = await stemEngine.separate(input, {
      onProgress: (p) => {
        setDeterminateProgress(p.fraction, "Splitting stems", "Separating drums, bass, other, vocals", `${Math.min(99, Math.round(p.fraction * 100))}%`);
      },
    });
    renderStems(stems, performance.now() - t0);
    progressPanel.hidden = true;
  } catch (error) {
    setDeterminateProgress(progressElement.value, "Stem split failed", errorMessage(error), "");
  } finally {
    splittingStems = false;
    closeBreadcrumb();
    updateActionAvailability();
  }
}

function renderStems(stems: readonly StemAudio[], elapsedMs: number): void {
  releaseStemUrls();
  stemsList.replaceChildren();
  for (const stem of stems) {
    const url = URL.createObjectURL(pcmToWav(stem.samples, stem.sampleRate, stem.right));
    stemUrls.push(url);
    const item = document.createElement("li");
    item.className = "stem-item";
    const name = document.createElement("span");
    name.className = "stem-item-name";
    name.textContent = stem.name;
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    player.src = url;
    player.setAttribute("aria-label", `${stem.name} stem playback`);
    const link = document.createElement("a");
    link.className = "download-button";
    link.href = url;
    link.download = `ace-step-${resultSeed}-${stem.name}.wav`;
    link.textContent = "Download";
    item.append(name, player, link);
    stemsList.appendChild(item);
  }
  stemsTime.textContent = formatElapsed(elapsedMs);
  stemsPanel.hidden = false;
}

/** Clears rendered stems; keeps the engine (worker + GPU weights) alive. */
function resetStemsUi(): void {
  releaseStemUrls();
  stemsList.replaceChildren();
  stemsTime.textContent = "";
  stemsPanel.hidden = true;
}

/** Full reset for a new generation / page teardown. */
async function resetStemSplitter(): Promise<void> {
  resetStemsUi();
  const current = stemEngine;
  stemEngine = undefined;
  if (current === undefined) {
    await stemDisposal;
    return;
  }
  const pending = current.dispose();
  stemDisposal = pending;
  try {
    await pending;
  } finally {
    if (stemDisposal === pending) stemDisposal = undefined;
  }
}

function releaseStemUrls(): void {
  for (const url of stemUrls) URL.revokeObjectURL(url);
  stemUrls = [];
}

function failOperation(message: string, reset: boolean): void {
  closeBreadcrumb();
  initializationRequestId = undefined;
  initializationCancelRequested = false;
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
  cacheAcquisition?.abort();
  cacheAcquisition = undefined;
  if (disposal !== undefined) {
    disposal.reject(new Error("worker reset while a dispose was pending"));
    disposal = undefined;
  }
  worker?.terminate();
  worker = undefined;
  workerReady = false;
  initializationCancelRequested = false;
  workerDetails = undefined;
  void releaseRuntimeLock?.();
  releaseRuntimeLock = undefined;
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
  if (busy || splittingStems || deletingModel || !cacheCanBeDeleted()) return;
  deletingModel = true;
  updateActionAvailability();
  deleteModelButton.disabled = true;
  cacheStatus.textContent = "Releasing the runtime…";
  try {
    await disposeWorker();
    cacheStatus.textContent = "Deleting downloaded model…";
    await deleteAceDemoModelCache();
    modelProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
    await refreshCacheInfo();
  } catch (error) {
    cacheStatus.textContent = `Could not delete the model: ${errorMessage(error)}`;
  } finally {
    deletingModel = false;
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
  try {
    await waitForWorkerDisposal(requestId, (pending) => {
      disposal = pending;
      current.postMessage({ type: "dispose", requestId });
    });
  } finally {
    if (disposal?.requestId === requestId) disposal = undefined;
    if (worker === current) {
      current.terminate();
      worker = undefined;
      workerReady = false;
      const release = releaseRuntimeLock;
      releaseRuntimeLock = undefined;
      await release?.();
    }
  }
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
  generateButton.disabled = busy || deletingModel || splittingStems || supportDetails?.supported !== true;
  cancelButton.disabled = !busy;
  deleteModelButton.disabled = busy || deletingModel || splittingStems || !cacheCanBeDeleted();
  splitStemsButton.disabled = busy || deletingModel || splittingStems || resultBlob === undefined;
}

function cacheCanBeDeleted(): boolean {
  return cacheDetails?.supported === true && (cacheDetails.assetCount > 0 || cacheDetails.partialAssetCount > 0);
}

function setDeterminateProgress(value: number, title: string, detail: string, percentage: string): void {
  recordBreadcrumb(title, detail);
  progressPanel.hidden = false;
  progressElement.max = 1;
  progressElement.value = clampFraction(value);
  progressTitle.textContent = title;
  progressDetail.textContent = detail;
  progressPercent.textContent = percentage;
}

function setIndeterminateProgress(title: string, detail: string): void {
  recordBreadcrumb(title, detail);
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
