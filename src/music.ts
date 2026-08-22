// Music generation page: ACE-Step 1.5 Turbo via the musicgen-acestep client.
// Engine module (and the 1.3 MB runtime) loads lazily after support probing.

import type {
  AceGenerationRequest,
  AceGenerationResult,
  AceModelCacheInfo,
  AceStepMusicClient,
} from "./engines/musicgen-acestep/index.js";

type MusicEngineModule = typeof import("./engines/musicgen-acestep/index.js");

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
};

const supportWarning = $<HTMLParagraphElement>("support-warning");
const promptInput = $<HTMLTextAreaElement>("prompt");
const lyricsInput = $<HTMLTextAreaElement>("lyrics");
const durationInput = $<HTMLInputElement>("duration");
const seedInput = $<HTMLInputElement>("seed");
const bpmInput = $<HTMLInputElement>("bpm");
const keyScaleInput = $<HTMLInputElement>("key-scale");
const timeSignatureInput = $<HTMLInputElement>("time-signature");
const vocalLanguageInput = $<HTMLInputElement>("vocal-language");
const generateButton = $<HTMLButtonElement>("generate");
const cancelButton = $<HTMLButtonElement>("cancel");
const cacheStatus = $<HTMLSpanElement>("cache-status");
const deleteModelButton = $<HTMLButtonElement>("delete-model");
const formError = $<HTMLParagraphElement>("form-error");
const progressPanel = $<HTMLDivElement>("progress-panel");
const progressTitle = $<HTMLElement>("progress-title");
const progressPercent = $<HTMLSpanElement>("progress-percent");
const progressElement = $<HTMLProgressElement>("progress");
const progressDetail = $<HTMLParagraphElement>("progress-detail");
const resultHeading = $<HTMLHeadingElement>("result-heading");
const resultPanel = $<HTMLDivElement>("result-panel");
const audioPlayer = $<HTMLAudioElement>("audio-player");
const download = $<HTMLAnchorElement>("download");
const summary = $<HTMLElement>("summary");
const status = $<HTMLParagraphElement>("status");
const metrics = $<HTMLPreElement>("metrics");

let engine: MusicEngineModule | undefined;
let client: AceStepMusicClient | undefined;
let cacheDetails: AceModelCacheInfo | undefined;
let supported = false;
let busy = false;
let coldDownload = true;
let output: { url: string; storageId: string } | undefined;
let diagnostics: unknown[] = [];
let lastGeneration: unknown;

void initialize();

async function initialize(): Promise<void> {
  status.textContent = "Loading runtime…";
  try {
    engine = await import("./engines/musicgen-acestep/index.js");
    const support = await engine.checkSupport();
    supported = support.supported;
    if (!supported) {
      supportWarning.textContent =
        support.errors.join(" ") ||
        "This browser does not support the required WebGPU features.";
      supportWarning.hidden = false;
    } else if (support.warnings.length > 0) {
      supportWarning.textContent = support.warnings.join(" ");
      supportWarning.classList.remove("error");
      supportWarning.hidden = false;
    }
    status.textContent = supported ? "Ready." : "WebGPU unsupported.";
    renderMetrics({ support });
    await refreshCacheInfo();
  } catch (error) {
    supportWarning.textContent = `Could not initialize: ${message(error)}`;
    supportWarning.hidden = false;
    status.textContent = "Initialization failed.";
  }
  updateButtons();
}

generateButton.addEventListener("click", () => void generate());
cancelButton.addEventListener("click", () => {
  cancelButton.disabled = true;
  client?.cancel();
});
deleteModelButton.addEventListener("click", () => void deleteModel());
window.addEventListener("pagehide", () => {
  if (output !== undefined) URL.revokeObjectURL(output.url);
  client?.terminate();
});

async function generate(): Promise<void> {
  if (engine === undefined || busy || !supported) return;
  let request: AceGenerationRequest;
  try {
    request = readRequest();
  } catch (error) {
    formError.textContent = message(error);
    formError.hidden = false;
    return;
  }
  formError.hidden = true;
  try {
    await releaseOutput();
  } catch (error) {
    formError.textContent = `Could not release the previous song: ${message(error)}`;
    formError.hidden = false;
    return;
  }

  client ??= new engine.AceStepMusicClient();
  diagnostics = [];
  coldDownload = !engine.isModelDownloadComplete(cacheDetails);
  setBusy(true);
  resultPanel.hidden = true;
  resultHeading.hidden = true;
  setProgress(undefined, "Preparing model", "Checking WebGPU and browser storage");

  void engine.requestAceModelStoragePersistence().catch(() => undefined);

  const startedAt = performance.now();
  try {
    const result = await client.generate(request, {
      onDownloadProgress: (p) => {
        setProgress(
          p.fraction,
          coldDownload ? "Downloading model" : "Preparing model data",
          engine!.formatModelDownloadAmount(p),
          `${p.percentage.toFixed(1)}%`,
        );
      },
      onInitializationProgress: (p) => {
        setProgress(undefined, "Preparing model", friendly(p.message, p.stage));
      },
      onGenerationProgress: (p) => {
        const fraction = clamp(p.overallFraction);
        setProgress(
          fraction,
          "Generating song",
          friendly(p.message, p.stage),
          `${Math.min(99, Math.round(fraction * 100))}%`,
        );
      },
      onDiagnostic: (d) => {
        diagnostics = [...diagnostics.slice(-19), d];
      },
    });
    await publish(result, performance.now() - startedAt);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      setProgress(progressElement.value, "Cancelled", "The partial output was removed");
    } else {
      setProgress(progressElement.value, "Generation failed", message(error));
    }
  } finally {
    setBusy(false);
    await refreshCacheInfo();
  }
}

function readRequest(): AceGenerationRequest {
  if (engine === undefined) throw new Error("Runtime not loaded");
  const prompt = promptInput.value.trim();
  if (prompt.length === 0) throw new Error("Enter a prompt for the song.");

  const durationSeconds = Number(durationInput.value);
  if (
    !Number.isInteger(durationSeconds) ||
    durationSeconds < engine.ACE_MIN_DURATION_SECONDS ||
    durationSeconds > engine.ACE_MAX_DURATION_SECONDS
  ) {
    throw new Error("Duration must be a whole number of seconds from 10 through 240.");
  }

  const bpmText = bpmInput.value.trim();
  const bpm = bpmText === "" ? undefined : Number(bpmText);
  if (bpm !== undefined && (!Number.isSafeInteger(bpm) || bpm < 30 || bpm > 300)) {
    throw new Error("BPM must be a whole number from 30 through 300.");
  }

  const lyrics = lyricsInput.value;
  const metadata = {
    ...(bpm === undefined ? {} : { bpm }),
    ...optional("keyScale", keyScaleInput.value),
    ...optional("timeSignature", timeSignatureInput.value),
    ...optional("vocalLanguage", vocalLanguageInput.value),
  };
  const seed = seedInput.value.trim() === ""
    ? randomSeed()
    : engine.aceSeed(seedInput.value.trim());

  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt,
    lyrics,
    instrumental: lyrics.trim().length === 0,
    durationSeconds,
    seed,
    planner: { mode: "disabled" },
    ...(Object.keys(metadata).length === 0 ? {} : { metadata }),
  };
}

async function publish(result: AceGenerationResult, elapsedMs: number): Promise<void> {
  if (engine === undefined) return;
  await releaseOutput();
  const url = URL.createObjectURL(result.audio);
  output = { url, storageId: result.audioStorageId };
  audioPlayer.src = url;
  audioPlayer.load();
  download.href = url;
  download.download = `ace-step-${result.seed}.wav`;
  summary.textContent =
    `${formatDuration(result.durationSeconds)} song · generated in ` +
    `${formatElapsed(result.metrics.totalMs)} (wall ${formatElapsed(elapsedMs)}) · seed ${result.seed}`;
  resultPanel.hidden = false;
  resultHeading.hidden = false;
  progressPanel.hidden = true;
  lastGeneration = {
    durationSeconds: result.durationSeconds,
    seed: result.seed,
    frameCount: result.frameCount,
    modelManifestSha256: result.modelManifestSha256,
    metrics: result.metrics,
  };
  renderMetrics();
}

async function releaseOutput(): Promise<void> {
  const current = output;
  if (current === undefined || engine === undefined) return;
  output = undefined;
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  download.removeAttribute("href");
  URL.revokeObjectURL(current.url);
  await engine.releaseAceAudioOutput(current.storageId);
}

async function refreshCacheInfo(): Promise<void> {
  if (engine === undefined) return;
  try {
    cacheDetails = await engine.inspectAceModelCache();
    if (!cacheDetails.supported) {
      cacheStatus.textContent = "Model storage is unavailable in this context.";
    } else if (cacheDetails.assetCount === 0 && cacheDetails.partialAssetCount === 0) {
      cacheStatus.textContent =
        `Model not downloaded · ${engine.formatDecimalBytes(engine.MODEL_DOWNLOAD_TOTAL_BYTES)} on first generation`;
    } else {
      const partial = cacheDetails.partialAssetCount === 0
        ? ""
        : ` · ${cacheDetails.partialAssetCount} incomplete`;
      cacheStatus.textContent =
        `${engine.formatDecimalBytes(cacheDetails.sizeBytes)} · ${cacheDetails.assetCount} files${partial}`;
    }
  } catch (error) {
    cacheDetails = undefined;
    cacheStatus.textContent = `Could not inspect model storage: ${message(error)}`;
  }
  updateButtons();
  renderMetrics();
}

async function deleteModel(): Promise<void> {
  if (engine === undefined || busy || !cacheDeletable()) return;
  deleteModelButton.disabled = true;
  cacheStatus.textContent = "Releasing the runtime…";
  try {
    await client?.dispose();
    cacheStatus.textContent = "Deleting downloaded model…";
    await engine.deleteAceModelCache();
    await refreshCacheInfo();
  } catch (error) {
    cacheStatus.textContent = `Could not delete the model: ${message(error)}`;
  } finally {
    updateButtons();
  }
}

function cacheDeletable(): boolean {
  return (
    cacheDetails?.supported === true &&
    (cacheDetails.assetCount > 0 || cacheDetails.partialAssetCount > 0)
  );
}

function setBusy(value: boolean): void {
  busy = value;
  for (const control of document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(
    "input, textarea",
  )) {
    control.disabled = value;
  }
  updateButtons();
}

function updateButtons(): void {
  generateButton.disabled = busy || !supported;
  cancelButton.disabled = !busy;
  deleteModelButton.disabled = busy || !cacheDeletable();
}

function setProgress(
  fraction: number | undefined,
  title: string,
  detail: string,
  percent = "",
): void {
  progressPanel.hidden = false;
  progressTitle.textContent = title;
  progressDetail.textContent = detail;
  progressPercent.textContent = percent;
  if (fraction === undefined) {
    progressElement.removeAttribute("value");
  } else {
    progressElement.value = clamp(fraction);
  }
}

function renderMetrics(extra?: Record<string, unknown>): void {
  metrics.textContent = JSON.stringify(
    {
      ...(extra ?? {}),
      modelCache: cacheDetails ?? null,
      runtime: client?.runtimeDiagnostics ?? null,
      generation: lastGeneration ?? null,
      diagnostics,
    },
    null,
    2,
  );
}

function randomSeed() {
  if (engine === undefined) throw new Error("Runtime not loaded");
  const words = crypto.getRandomValues(new Uint32Array(2));
  const value = (BigInt(words[0]!) << 32n) | BigInt(words[1]!);
  return engine.aceSeed(value);
}

function optional<Key extends string>(
  key: Key,
  value: string,
): Readonly<Record<Key, string>> | Record<string, never> {
  const text = value.trim();
  return text === "" ? {} : ({ [key]: text } as Record<Key, string>);
}

function friendly(text: string | undefined, stage: string): string {
  if (text === undefined || text.trim() === "") return stage.replaceAll("-", " ");
  const cleaned = text.replace(
    /^(?:cache|network):\s+.+?(?=\s+[0-9]+\/[0-9]+ bytes$)/u,
    "Processing model data",
  );
  return cleaned.length > 120 ? `${cleaned.slice(0, 117)}…` : cleaned;
}

function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function formatElapsed(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "—";
  const seconds = milliseconds / 1_000;
  return seconds < 60
    ? `${seconds.toFixed(1)} s`
    : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function clamp(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
