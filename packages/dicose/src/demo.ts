import "./demo.css";

import {
  DICOSE_STEM_NAMES,
  DiCoSeWorkerClient,
  type DiCoSeOutputMode,
  type DiCoSeProgress,
  type DiCoSeSeparation,
  type DiCoSeStemName,
  type StereoPcm,
} from "./index.js";
import { encodeStereoWav } from "./runtime/audio.js";

export interface DiCoSeBrowserRunOptions {
  /** Defaults to the bundled audio fixture served from Vite's public root. */
  readonly audioUrl?: string;
  /** Defaults to `/model/manifest.json`. */
  readonly manifestUrl?: string;
  /** Fixed CD-noise seed used for reproducible browser runs. */
  readonly seed?: number;
  /** Full one-step refinement remains the default. */
  readonly outputMode?: DiCoSeOutputMode;
}

export interface DiCoSeStemSummary {
  readonly sampleRate: number;
  readonly samples: number;
  readonly durationSeconds: number;
  readonly peak: number;
  readonly rms: number;
  readonly finiteSamples: number;
}

export interface DiCoSeBrowserMetrics {
  readonly audioUrl: string;
  readonly inputBytes: number;
  readonly inputDurationSeconds: number;
  readonly outputDurationSeconds: number;
  readonly outputMode: DiCoSeOutputMode;
  readonly elapsedMs: number;
  readonly timing: Readonly<Record<string, number>>;
}

export interface DiCoSeBrowserOutput {
  readonly stems: Readonly<Record<DiCoSeStemName, DiCoSeStemSummary>>;
  /** Decoded input mixture minus the restored vocal estimate. */
  readonly instrumental: DiCoSeStemSummary;
  readonly diagnostics: DiCoSeSeparation["diagnostics"];
}

export interface DiCoSeBrowserBenchmarkMetrics {
  readonly mode: "benchmark";
  readonly warmupRuns: number;
  readonly measuredRuns: number;
  /** End-to-end samples after warmups, in milliseconds. */
  readonly samplesMs: readonly number[];
  readonly aggregate: Readonly<{
    readonly minMs: number;
    readonly maxMs: number;
    readonly meanMs: number;
    readonly medianMs: number;
  }>;
  /** Per-stage model timing from the corresponding measured runs. */
  readonly timingSamples: readonly Readonly<Record<string, number>>[];
}

export type DiCoSeBrowserRunResult =
  | {
    readonly ok: true;
    readonly metrics: DiCoSeBrowserMetrics;
    readonly output: DiCoSeBrowserOutput;
  }
  | {
    readonly ok: false;
    readonly metrics: Readonly<Record<string, never>>;
    readonly error: Readonly<{ name: string; message: string; stack?: string }>;
  };

export type DiCoSeBrowserBenchmarkReport =
  | {
    readonly ok: true;
    readonly metrics: DiCoSeBrowserBenchmarkMetrics;
  }
  | {
    readonly ok: false;
    readonly metrics: Readonly<Record<string, never>>;
    readonly error: Readonly<{ name: string; message: string; stack?: string }>;
  };

export type DiCoSeBrowserReport = DiCoSeBrowserRunResult | DiCoSeBrowserBenchmarkReport;

export interface DiCoSeBrowserHarness {
  /**
   * Run the bundled end-to-end workflow without UI input or file downloads.
   * It always resolves to a serializable success/failure object so automation
   * can inspect `#result` or await this promise directly.
   */
  run(options?: DiCoSeBrowserRunOptions): Promise<DiCoSeBrowserRunResult>;
  /** Filled automatically by `?mode=benchmark`; safe for automation to poll. */
  report?: DiCoSeBrowserReport;
  /** Alias retained for simple browser harness polling. */
  lastReport?: DiCoSeBrowserReport;
}

declare global {
  interface Window {
    __DICOSE_BROWSER__?: DiCoSeBrowserHarness;
    __DICOSE_BROWSER_REPORT__?: DiCoSeBrowserReport;
  }
}

const DEFAULT_AUDIO_URL = "/Mixture_audio_1.wav";
const DEFAULT_MANIFEST_URL = "/model/manifest.json";
const DEMO_OUTPUT_NAMES = [...DICOSE_STEM_NAMES, "instrumental"] as const;
type DemoOutputName = (typeof DEMO_OUTPUT_NAMES)[number];

type DemoConfiguration = "full" | "fast";

interface DemoConfigurationSpec {
  readonly label: string;
  readonly outputMode: DiCoSeOutputMode;
  readonly fileSuffix: string;
}

const DEMO_CONFIGURATIONS: Readonly<Record<DemoConfiguration, DemoConfigurationSpec>> = {
  full: { label: "Full", outputMode: "refined", fileSuffix: "full" },
  fast: { label: "Fast", outputMode: "deterministic", fileSuffix: "fast" },
};

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const fileInput = requiredElement<HTMLInputElement>("#file-input");
const dropZone = requiredElement<HTMLElement>("#drop-zone");
const fileNameNode = requiredElement<HTMLElement>("#file-name");
const fileMetaNode = requiredElement<HTMLElement>("#file-meta");
const workspaceNode = requiredElement<HTMLElement>("#workspace");
const runButton = requiredElement<HTMLButtonElement>("#run-button");
const runButtonLabel = requiredElement<HTMLElement>("#run-button-label");
const progressNode = requiredElement<HTMLElement>("#progress");
const progressFillNode = requiredElement<HTMLElement>("#progress-fill");
const resultsNode = requiredElement<HTMLElement>("#results");
const timingGridNode = requiredElement<HTMLElement>("#timing-grid");
const stemsGridNode = requiredElement<HTMLElement>("#stems-grid");
const errorNode = requiredElement<HTMLElement>("#error");

let client: DiCoSeWorkerClient | undefined;
let clientConfiguration: string | undefined;
let activeRun: Promise<DiCoSeBrowserRunResult> | undefined;
let selectedFile: File | undefined;
let interactiveBusy = false;
let outputUrls: string[] = [];

const harness: DiCoSeBrowserHarness = {
  async run(options: DiCoSeBrowserRunOptions = {}): Promise<DiCoSeBrowserRunResult> {
    return await executeRun(options, true);
  },
};

window.__DICOSE_BROWSER__ = harness;

const query = new URLSearchParams(window.location.search);
const automated = query.get("autorun") === "1" || query.get("mode") === "benchmark";
document.documentElement.dataset.runMode = automated ? "automation" : "interactive";
if (automated) {
  setStatus("Starting automated run…");
  const options = autorunOptions(query);
  if (query.get("mode") === "benchmark") {
    void runConfiguredBenchmark(options, query);
  } else {
    void harness.run(options);
  }
} else {
  initializeInteractiveDemo();
}

async function executeRun(
  options: DiCoSeBrowserRunOptions,
  publish: boolean,
): Promise<DiCoSeBrowserRunResult> {
  if (activeRun !== undefined) return await activeRun;
  const run = performRun(options, publish);
  activeRun = run;
  try {
    return await run;
  } finally {
    activeRun = undefined;
  }
}

async function performRun(
  options: DiCoSeBrowserRunOptions,
  publish: boolean,
): Promise<DiCoSeBrowserRunResult> {
  const started = performance.now();
  try {
    const audioUrl = new URL(options.audioUrl ?? DEFAULT_AUDIO_URL, window.location.href).href;
    setStatus("Fetching input audio…");
    const response = await fetch(audioUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not fetch test audio: HTTP ${response.status}`);
    const blob = await response.blob();
    const separation = await separateBlob(blob, options);
    const elapsedMs = performance.now() - started;
    const result: DiCoSeBrowserRunResult = {
      ok: true,
      metrics: makeMetrics(audioUrl, blob.size, elapsedMs, separation),
      output: {
        stems: summarizeStems(separation),
        instrumental: summarizeStem(separation.instrumental),
        diagnostics: separation.diagnostics,
      },
    };
    setStatus(`Completed in ${result.metrics.elapsedMs.toFixed(1)} ms.`, "success");
    if (publish) publishResult(result);
    return result;
  } catch (error) {
    await discardClient();
    const result: DiCoSeBrowserRunResult = {
      ok: false,
      metrics: {},
      error: serializeError(error),
    };
    setStatus(`Failed: ${result.error.message}`, "danger");
    if (publish) publishResult(result);
    return result;
  }
}

async function separateBlob(
  blob: Blob,
  options: DiCoSeBrowserRunOptions,
): Promise<DiCoSeSeparation> {
  const manifestUrl = new URL(options.manifestUrl ?? DEFAULT_MANIFEST_URL, window.location.href).href;
  const configuration = manifestUrl;
  if (client !== undefined && clientConfiguration !== configuration) {
    setStatus("Switching model configuration…");
    await client.dispose();
    client = undefined;
    clientConfiguration = undefined;
  }
  if (client === undefined) {
    client = new DiCoSeWorkerClient({
      manifestUrl,
      onProgress: reportProgress,
    });
    clientConfiguration = configuration;
  }
  setStatus("Decoding and resampling audio…");
  return await client.separateAudio(blob, {
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.outputMode === undefined ? {} : { outputMode: options.outputMode }),
  });
}

function initializeInteractiveDemo(): void {
  setStatus("Choose a WAV file to begin.");
  setProgress(0);
  updateRunButton();

  fileInput.addEventListener("click", () => {
    fileInput.value = "";
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file !== undefined) selectLocalFile(file);
  });

  let dragDepth = 0;
  dropZone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    dropZone.classList.add("is-dragging");
  });
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer !== null) event.dataTransfer.dropEffect = "copy";
  });
  dropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropZone.classList.remove("is-dragging");
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove("is-dragging");
    const file = event.dataTransfer?.files[0];
    if (file !== undefined) selectLocalFile(file);
  });

  for (const input of document.querySelectorAll<HTMLInputElement>('input[name="configuration"]')) {
    input.addEventListener("change", updateRunButton);
  }
  runButton.addEventListener("click", () => void runInteractiveSeparation());
  window.addEventListener("pagehide", (event) => {
    clearInteractiveResults();
    if (event.persisted) return;
    void discardClient();
  });

  if (navigator.gpu === undefined) {
    showError("WebGPU is unavailable. Open this page in a current WebGPU-capable browser.");
    runButton.disabled = true;
  }
}

function selectLocalFile(file: File): void {
  if (interactiveBusy) return;
  clearError();
  if (!isWavFile(file)) {
    selectedFile = undefined;
    fileInput.value = "";
    fileNameNode.textContent = "No WAV selected";
    fileMetaNode.textContent = "Choose a .wav file or drop it here.";
    dropZone.classList.remove("has-file");
    clearInteractiveResults();
    setProgress(0);
    setStatus("Choose a valid WAV file.", "danger");
    showError("Please choose a WAV file. Other audio containers are not enabled in this demo.");
    updateRunButton();
    return;
  }
  if (file.size === 0) {
    selectedFile = undefined;
    fileInput.value = "";
    dropZone.classList.remove("has-file");
    fileNameNode.textContent = "No WAV selected";
    fileMetaNode.textContent = "Choose a non-empty stereo or mono WAV file.";
    clearInteractiveResults();
    setProgress(0);
    setStatus("Choose a non-empty WAV file.", "danger");
    showError("The selected WAV file is empty.");
    updateRunButton();
    return;
  }
  selectedFile = file;
  dropZone.classList.add("has-file");
  fileNameNode.textContent = file.name;
  fileMetaNode.textContent = `${formatBytes(file.size)} · Local file stays in this browser tab`;
  setStatus("Ready to separate.");
  setProgress(0);
  clearInteractiveResults();
  updateRunButton();
}

async function runInteractiveSeparation(): Promise<void> {
  const file = selectedFile;
  if (file === undefined || interactiveBusy) return;
  const configuration = selectedConfiguration();
  const spec = DEMO_CONFIGURATIONS[configuration];
  interactiveBusy = true;
  clearError();
  clearInteractiveResults();
  setBusy(true);
  setProgress(0.01);
  setStatus(`Preparing ${spec.label.toLowerCase()} separation…`);
  const started = performance.now();
  try {
    const separation = await separateBlob(file, {
      outputMode: spec.outputMode,
    });
    const elapsedMs = performance.now() - started;
    const result: Extract<DiCoSeBrowserRunResult, { readonly ok: true }> = {
      ok: true,
      metrics: makeMetrics(
        `local:${file.name}`,
        file.size,
        elapsedMs,
        separation,
      ),
      output: {
        stems: summarizeStems(separation),
        instrumental: summarizeStem(separation.instrumental),
        diagnostics: separation.diagnostics,
      },
    };
    publishResult(result);
    renderInteractiveResults(file, configuration, separation, elapsedMs);
    setProgress(1);
    setStatus(`${spec.label} separation completed in ${formatElapsed(elapsedMs)}.`, "success");
  } catch (error) {
    clearInteractiveResults();
    await discardClient();
    const failure: DiCoSeBrowserRunResult = {
      ok: false,
      metrics: {},
      error: serializeError(error),
    };
    publishResult(failure);
    setProgress(0);
    setStatus("Separation failed.", "danger");
    showError(friendlyError(error));
  } finally {
    interactiveBusy = false;
    setBusy(false);
  }
}

function renderInteractiveResults(
  file: File,
  configuration: DemoConfiguration,
  separation: DiCoSeSeparation,
  elapsedMs: number,
): void {
  revokeOutputUrls();
  timingGridNode.replaceChildren();
  stemsGridNode.replaceChildren();
  const durationSeconds = separation.stems.vocals.length / separation.stems.vocals.sampleRate;
  const throughput = durationSeconds / (elapsedMs / 1_000);
  const timingCards: readonly [string, string][] = [
    ["Mode", DEMO_CONFIGURATIONS[configuration].label],
    ["Wall time", formatElapsed(elapsedMs)],
    ["Runtime total", formatElapsed(requireTiming(separation, "totalMs"))],
    ["Audio", formatClock(durationSeconds)],
    ["Throughput", `${throughput.toFixed(2)}× realtime`],
    ["Prepare", formatElapsed(requireTiming(separation, "prepareMs"))],
    ["Deterministic", formatElapsed(requireTiming(separation, "deterministicMs"))],
    ["Mapping", formatElapsed(requireTiming(separation, "mappingMs"))],
    ["Refinement", formatElapsed(requireTiming(separation, "refinementMs"))],
    ["ISTFT", formatElapsed(requireTiming(separation, "istftMs"))],
  ];
  for (const [label, value] of timingCards) timingGridNode.append(createMetricCard(label, value));

  const players: HTMLAudioElement[] = [];
  const baseName = safeFileBase(file.name);
  for (const name of DEMO_OUTPUT_NAMES) {
    const pcm = outputPcm(separation, name);
    const wav = encodeStereoWav(pcm);
    const url = URL.createObjectURL(wav);
    outputUrls.push(url);

    const card = document.createElement("article");
    card.className = name === "instrumental"
      ? "stem-card stem-card--instrumental"
      : "stem-card";
    const header = document.createElement("div");
    header.className = "stem-card__header";
    const heading = document.createElement("h3");
    heading.className = "stem-card__name";
    heading.textContent = capitalize(name);
    const stats = summarizeStem(pcm);
    const detail = document.createElement("p");
    detail.className = "stem-card__meta";
    const derivation = name === "instrumental" ? " · mix − vocals" : "";
    detail.textContent = `${formatClock(stats.durationSeconds)} · peak ${stats.peak.toFixed(4)} · RMS ${stats.rms.toFixed(4)}${derivation}`;
    header.append(heading, detail);
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = url;
    audio.setAttribute("aria-label", `Play ${name} output`);
    audio.addEventListener("play", () => {
      for (const player of players) {
        if (player !== audio) player.pause();
      }
    });
    players.push(audio);
    const download = document.createElement("a");
    download.className = "stem-card__download";
    download.href = url;
    download.download = `${baseName}-${name}-${DEMO_CONFIGURATIONS[configuration].fileSuffix}.wav`;
    download.textContent = "Download WAV";
    card.append(header, audio, download);
    stemsGridNode.append(card);
  }
  resultsNode.hidden = false;
  resultsNode.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createMetricCard(label: string, value: string): HTMLElement {
  const card = document.createElement("div");
  card.className = "timing-card";
  const valueNode = document.createElement("strong");
  valueNode.className = "timing-card__value";
  valueNode.textContent = value;
  const labelNode = document.createElement("span");
  labelNode.className = "timing-card__label";
  labelNode.textContent = label;
  card.append(labelNode, valueNode);
  return card;
}

function clearInteractiveResults(): void {
  timingGridNode.replaceChildren();
  stemsGridNode.replaceChildren();
  revokeOutputUrls();
  resultsNode.hidden = true;
}

function revokeOutputUrls(): void {
  for (const url of outputUrls) URL.revokeObjectURL(url);
  outputUrls = [];
}

async function discardClient(): Promise<void> {
  const staleClient = client;
  client = undefined;
  clientConfiguration = undefined;
  await staleClient?.dispose().catch(() => {});
}

function selectedConfiguration(): DemoConfiguration {
  const selected = document.querySelector<HTMLInputElement>('input[name="configuration"]:checked');
  const value = selected?.value;
  if (value === "full" || value === "fast") return value;
  throw new Error("Choose an inference configuration");
}

function setBusy(busy: boolean): void {
  fileInput.disabled = busy;
  workspaceNode.classList.toggle("is-running", busy);
  dropZone.setAttribute("aria-disabled", String(busy));
  dropZone.dataset.busy = String(busy);
  for (const input of document.querySelectorAll<HTMLInputElement>('input[name="configuration"]')) {
    input.disabled = busy;
  }
  updateRunButton();
}

function updateRunButton(): void {
  const hasWebGpu = navigator.gpu !== undefined;
  runButton.disabled = selectedFile === undefined || interactiveBusy || !hasWebGpu;
  const spec = DEMO_CONFIGURATIONS[selectedConfiguration()];
  runButtonLabel.textContent = interactiveBusy ? "Separating…" : `Separate with ${spec.label}`;
}

function reportProgress(progress: DiCoSeProgress): void {
  setStatus(describeProgress(progress.phase, progress.detail));
  const fraction = progressFraction(progress);
  if (fraction !== undefined) setProgress(fraction);
}

function progressFraction(progress: DiCoSeProgress): number | undefined {
  const completion = progress.completed !== undefined && progress.total !== undefined && progress.total > 0
    ? Math.min(1, Math.max(0, progress.completed / progress.total))
    : 0;
  switch (progress.phase) {
    case "initializing": return 0.02;
    case "device": return 0.03 + completion * 0.04;
    case "weights": return 0.07 + completion * 0.23;
    case "separating": return 0.31;
    case "chunk": return 0.32 + completion * 0.67;
    case "stft": return 0.32 + completion * 0.04;
    case "deterministic": return 0.36 + completion * 0.2;
    case "mapping": return 0.57 + completion * 0.02;
    case "refinement": return 0.59 + completion * 0.34;
    case "istft": return 0.94 + completion * 0.05;
    default: return undefined;
  }
}

function setProgress(value: number): void {
  const percent = Math.round(Math.min(1, Math.max(0, value)) * 100);
  progressNode.hidden = percent === 0;
  progressFillNode.style.width = `${percent}%`;
  progressNode.setAttribute("aria-valuenow", String(percent));
}

function showError(message: string): void {
  errorNode.textContent = message;
  errorNode.hidden = false;
}

function clearError(): void {
  errorNode.textContent = "";
  errorNode.hidden = true;
}

function isWavFile(file: File): boolean {
  const mime = file.type.toLowerCase();
  return file.name.toLowerCase().endsWith(".wav") || [
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
  ].includes(mime);
}

function safeFileBase(name: string): string {
  const withoutExtension = name.replace(/\.wav$/i, "");
  const safe = withoutExtension.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "");
  return safe.slice(0, 80) || "dicose-output";
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1_024).toFixed(1)} KB`;
  if (bytes < 1_073_741_824) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
}

function formatElapsed(milliseconds: number): string {
  return milliseconds < 1_000
    ? `${milliseconds.toFixed(0)} ms`
    : `${(milliseconds / 1_000).toFixed(2)} s`;
}

function formatClock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return minutes === 0
    ? `${remainder.toFixed(1)} s`
    : `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function friendlyError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/out of memory|allocation|device lost/i.test(message)) {
    return `${message} The browser GPU could not complete a fixed-size DiCoSe model chunk.`;
  }
  return message;
}

function requireTiming(separation: DiCoSeSeparation, name: string): number {
  const value = separation.timing[name];
  return value !== undefined && Number.isFinite(value) ? value : 0;
}

function makeMetrics(
  audioUrl: string,
  inputBytes: number,
  elapsedMs: number,
  separation: DiCoSeSeparation,
): DiCoSeBrowserMetrics {
  const vocals = separation.stems.vocals;
  return {
    audioUrl,
    inputBytes,
    inputDurationSeconds: vocals.length / vocals.sampleRate,
    outputDurationSeconds: vocals.length / vocals.sampleRate,
    outputMode: separation.outputMode,
    elapsedMs,
    timing: separation.timing,
  };
}

function summarizeStems(separation: DiCoSeSeparation): Readonly<Record<DiCoSeStemName, DiCoSeStemSummary>> {
  const summaries = {} as Record<DiCoSeStemName, DiCoSeStemSummary>;
  for (const name of DICOSE_STEM_NAMES) summaries[name] = summarizeStem(separation.stems[name]);
  return summaries;
}

function outputPcm(separation: DiCoSeSeparation, name: DemoOutputName): StereoPcm {
  return name === "instrumental" ? separation.instrumental : separation.stems[name];
}

function summarizeStem(pcm: StereoPcm): DiCoSeStemSummary {
  let peak = 0;
  let sumSquares = 0;
  let finiteSamples = 0;
  const channels = [pcm.left, pcm.right] as const;
  for (const channel of channels) {
    for (let index = 0; index < channel.length; index += 1) {
      const value = channel[index]!;
      if (!Number.isFinite(value)) continue;
      finiteSamples += 1;
      const magnitude = Math.abs(value);
      if (magnitude > peak) peak = magnitude;
      sumSquares += value * value;
    }
  }
  return {
    sampleRate: pcm.sampleRate,
    samples: pcm.length,
    durationSeconds: pcm.length / pcm.sampleRate,
    peak,
    rms: finiteSamples === 0 ? Number.NaN : Math.sqrt(sumSquares / finiteSamples),
    finiteSamples,
  };
}

function describeProgress(phase: string, detail: string | undefined): string {
  return detail === undefined ? phase : `${phase}: ${detail}`;
}

function setStatus(value: string, tone: "neutral" | "success" | "danger" = "neutral"): void {
  if (statusNode !== null) {
    statusNode.textContent = value;
    statusNode.dataset.tone = tone;
    workspaceNode.dataset.statusTone = tone;
  }
}

function setResult(value: DiCoSeBrowserRunResult | DiCoSeBrowserBenchmarkReport): void {
  if (resultNode !== null) resultNode.textContent = JSON.stringify(value, null, 2);
}

function serializeError(error: unknown): { readonly name: string; readonly message: string; readonly stack?: string } {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      ...(error.stack === undefined ? {} : { stack: error.stack }),
    };
  }
  return { name: "Error", message: String(error) };
}

function autorunSeed(): number | undefined {
  const value = new URLSearchParams(window.location.search).get("seed");
  if (value === null) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    setStatus("Ignoring invalid ?seed value; using the runtime default.");
    return undefined;
  }
  return parsed;
}

function autorunOptions(query: URLSearchParams): DiCoSeBrowserRunOptions {
  const seed = autorunSeed();
  const audioUrl = query.get("source") ?? query.get("audioUrl") ?? undefined;
  const manifestUrl = query.get("manifestUrl") ?? undefined;
  const outputMode = queryChoice(query.get("outputMode"), ["refined", "deterministic"] as const, "outputMode");
  return {
    ...(audioUrl === undefined ? {} : { audioUrl }),
    ...(manifestUrl === undefined ? {} : { manifestUrl }),
    ...(seed === undefined ? {} : { seed }),
    ...(outputMode === undefined ? {} : { outputMode }),
  };
}

function queryChoice<const T extends readonly string[]>(
  value: string | null,
  choices: T,
  name: string,
): T[number] | undefined {
  if (value === null) return undefined;
  if ((choices as readonly string[]).includes(value)) return value as T[number];
  throw new RangeError(`${name} must be one of: ${choices.join(", ")}`);
}

async function runConfiguredBenchmark(
  options: DiCoSeBrowserRunOptions,
  query: URLSearchParams,
): Promise<DiCoSeBrowserBenchmarkReport> {
  try {
    return await runBenchmark(
      options,
      benchmarkCount(query.get("warmupRuns"), 1, 0),
      benchmarkCount(query.get("measuredRuns"), 1, 1),
    );
  } catch (error) {
    const report: DiCoSeBrowserBenchmarkReport = {
      ok: false,
      metrics: {},
      error: serializeError(error),
    };
    publishReport(report);
    setStatus(`Benchmark failed: ${report.error.message}`);
    setResult(report);
    return report;
  }
}

async function runBenchmark(
  options: DiCoSeBrowserRunOptions,
  warmupRuns: number,
  measuredRuns: number,
): Promise<DiCoSeBrowserBenchmarkReport> {
  delete harness.report;
  delete harness.lastReport;
  delete window.__DICOSE_BROWSER_REPORT__;
  if (resultNode !== null) resultNode.textContent = "";
  try {
    for (let index = 0; index < warmupRuns; index += 1) {
      setStatus(`Warmup ${index + 1}/${warmupRuns}…`);
      const result = await executeRun(options, false);
      if (!result.ok) return publishBenchmarkFailure(result);
    }
    const measurements: DiCoSeBrowserMetrics[] = [];
    for (let index = 0; index < measuredRuns; index += 1) {
      setStatus(`Benchmark ${index + 1}/${measuredRuns}…`);
      const result = await executeRun(options, false);
      if (!result.ok) return publishBenchmarkFailure(result);
      measurements.push(result.metrics);
    }
    const samplesMs = measurements.map((sample) => sample.elapsedMs);
    const report: DiCoSeBrowserBenchmarkReport = {
      ok: true,
      metrics: {
        mode: "benchmark",
        warmupRuns,
        measuredRuns,
        samplesMs,
        aggregate: summarizeSamples(samplesMs),
        timingSamples: measurements.map((sample) => sample.timing),
      },
    };
    publishReport(report);
    setStatus(`Benchmark completed: median ${report.metrics.aggregate.medianMs.toFixed(1)} ms.`);
    setResult(report);
    return report;
  } catch (error) {
    const report: DiCoSeBrowserBenchmarkReport = {
      ok: false,
      metrics: {},
      error: serializeError(error),
    };
    publishReport(report);
    setStatus(`Benchmark failed: ${report.error.message}`);
    setResult(report);
    return report;
  }
}

function publishBenchmarkFailure(result: Exclude<DiCoSeBrowserRunResult, { readonly ok: true }>): DiCoSeBrowserBenchmarkReport {
  const report: DiCoSeBrowserBenchmarkReport = {
    ok: false,
    metrics: {},
    error: result.error,
  };
  publishReport(report);
  setStatus(`Benchmark failed: ${result.error.message}`);
  setResult(report);
  return report;
}

function publishResult(result: DiCoSeBrowserRunResult): void {
  publishReport(result);
  setResult(result);
}

function publishReport(report: DiCoSeBrowserReport): void {
  harness.report = report;
  harness.lastReport = report;
  window.__DICOSE_BROWSER_REPORT__ = report;
}

function benchmarkCount(value: string | null, fallback: number, minimum: number): number {
  if (value === null) return fallback;
  const count = Number(value);
  if (!Number.isSafeInteger(count) || count < minimum || count > 20) {
    throw new RangeError(`Benchmark counts must be whole numbers between ${minimum} and 20`);
  }
  return count;
}

function summarizeSamples(samples: readonly number[]): DiCoSeBrowserBenchmarkMetrics["aggregate"] {
  if (samples.length === 0) {
    return { minMs: Number.NaN, maxMs: Number.NaN, meanMs: Number.NaN, medianMs: Number.NaN };
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const medianMs = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  const meanMs = sorted.reduce((sum, sample) => sum + sample, 0) / sorted.length;
  return {
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
    meanMs,
    medianMs,
  };
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Demo page is missing ${selector}`);
  return element;
}
