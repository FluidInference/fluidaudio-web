export const OPT_0023_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0023_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0023_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0023_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0023_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250;
export const OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS = 5_000;
export const OPT_0023_MAXIMUM_RECEIPT_BYTES = 65_536;
const FAILURE_TEXT_LIMIT = 4_096;
const FAILURE_STACK_LIMIT = 8_192;
const FAILURE_FIELD_LIMIT = 12;

export interface Opt0023RunIdentity {
  readonly coreCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0023PreparationClock {
  readonly warmupStartedAtEpochMilliseconds: number;
  readonly warmupCompletedAtEpochMilliseconds: number;
}

export interface Opt0023ThermalGate {
  readonly source: typeof OPT_0023_THERMAL_SOURCE;
  readonly command: typeof OPT_0023_THERMAL_COMMAND;
  readonly traceStartedAtEpochMilliseconds: number;
  readonly traceStartObservationIndex: number;
  readonly traceObservationCountThroughGate: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateStartObservationIndex: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly gateCompletedObservationIndex: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0023_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
}

export interface Opt0023ThermalTransition {
  readonly atEpochMilliseconds: number;
  readonly level: number;
}

export interface Opt0023ThermalCompletion {
  readonly source: typeof OPT_0023_THERMAL_SOURCE;
  readonly command: typeof OPT_0023_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly rawTraceByteLength: number;
  readonly rawTraceSchema:
    "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1";
  readonly traceStartedAtEpochMilliseconds: number;
  readonly traceStartObservationIndex: number;
  readonly completedAtEpochMilliseconds: number;
  readonly completedObservationIndex: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0023_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly missingObservationCount: 0;
  readonly initialLevel: number;
  readonly transitions: readonly Opt0023ThermalTransition[];
  readonly gateStartObservationIndex: number;
  readonly gateObservationCount: number;
  readonly coversWarmupGateRunValidationAndCleanup: true;
  readonly decisionRule: "one-fresh-post-warmup-nominal-30s-gate-then-accept-disclosed-trace";
  readonly unchangedThermalRetryPerformed: false;
}

declare global {
  interface Window {
    __ACE_OPT0023_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface WorkerReadyMessage {
  readonly type: "ready-for-thermal-gate";
  readonly preparation: Readonly<Record<string, unknown>> &
    Opt0023PreparationClock;
}

interface WorkerProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface WorkerCompleteMessage {
  readonly type: "profile-complete";
  readonly result: Readonly<Record<string, unknown>> & {
    readonly lifecycle: Readonly<{
      readonly cleanupCompletedAtEpochMilliseconds: number;
    }>;
  };
}

interface WorkerFailedMessage {
  readonly type: "failed";
  readonly error: Readonly<Record<string, unknown>>;
}

interface WorkerDisposedMessage {
  readonly type: "disposed";
}

type WorkerMessage =
  | WorkerReadyMessage
  | WorkerProgressMessage
  | WorkerCompleteMessage
  | WorkerFailedMessage
  | WorkerDisposedMessage;

export function parseOpt0023RunIdentity(
  parameters: URLSearchParams,
): Opt0023RunIdentity {
  const value: Opt0023RunIdentity = {
    coreCommit: requiredString(parameters, "coreCommit"),
    harnessCommit: requiredString(parameters, "harnessCommit"),
    machineModel: requiredString(parameters, "machineModel"),
    osVersion: requiredString(parameters, "osVersion"),
    osBuild: requiredString(parameters, "osBuild"),
    browserVersion: requiredString(parameters, "browserVersion"),
    gpuCoreCount: requiredPositiveInteger(parameters, "gpuCoreCount"),
    memoryBytes: requiredPositiveInteger(parameters, "memoryBytes"),
  };
  if (!isCommit(value.coreCommit) || !isCommit(value.harnessCommit)) {
    throw new Error("OPT-0023 requires exact 40-hex commit identities");
  }
  return Object.freeze(value);
}

export function parseOpt0023ThermalGate(
  parameters: URLSearchParams,
  preparation: Opt0023PreparationClock,
  nowEpochMilliseconds: number,
): Opt0023ThermalGate {
  const source = parameters.get("thermalSource");
  const command = parameters.get("thermalCommand");
  const traceStartedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalTraceStartedAtEpochMilliseconds",
  );
  const traceStartObservationIndex = requiredNonnegativeInteger(
    parameters,
    "thermalTraceStartObservationIndex",
  );
  const gateStartedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalGateStartedAtEpochMilliseconds",
  );
  const gateStartObservationIndex = requiredNonnegativeInteger(
    parameters,
    "thermalGateStartObservationIndex",
  );
  const gateCompletedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalGateCompletedAtEpochMilliseconds",
  );
  const gateCompletedObservationIndex = requiredNonnegativeInteger(
    parameters,
    "thermalGateCompletedObservationIndex",
  );
  const observationCount = requiredPositiveInteger(
    parameters,
    "thermalGateObservations",
  );
  const pollMilliseconds = requiredNumber(parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredNumber(
    parameters,
    "thermalGateMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredNumber(
    parameters,
    "thermalGateNonNominalObservations",
  );
  const missingObservationCount = requiredNumber(
    parameters,
    "thermalGateMissingObservations",
  );
  const durationMilliseconds =
    gateCompletedAtEpochMilliseconds - gateStartedAtEpochMilliseconds;

  if (
    source !== OPT_0023_THERMAL_SOURCE ||
    command !== OPT_0023_THERMAL_COMMAND
  ) {
    throw new Error("OPT-0023 rejected the external thermal source");
  }
  if (
    !Number.isFinite(preparation.warmupStartedAtEpochMilliseconds) ||
    !Number.isFinite(preparation.warmupCompletedAtEpochMilliseconds) ||
    preparation.warmupCompletedAtEpochMilliseconds <
      preparation.warmupStartedAtEpochMilliseconds ||
    traceStartedAtEpochMilliseconds >
      preparation.warmupStartedAtEpochMilliseconds ||
    gateStartedAtEpochMilliseconds <
      preparation.warmupCompletedAtEpochMilliseconds ||
    gateCompletedAtEpochMilliseconds < gateStartedAtEpochMilliseconds ||
    gateCompletedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error(
      "OPT-0023 requires one trace started before warmup and a fresh gate after warmup",
    );
  }
  if (
    gateStartObservationIndex <= traceStartObservationIndex ||
    gateCompletedObservationIndex < gateStartObservationIndex ||
    observationCount !==
      gateCompletedObservationIndex - gateStartObservationIndex + 1
  ) {
    throw new Error("OPT-0023 thermal gate observation indexes are inconsistent");
  }
  if (
    durationMilliseconds < OPT_0023_MINIMUM_NOMINAL_MILLISECONDS ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0023_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0023 requires one continuous 30-second nominal gate");
  }
  if (
    pollMilliseconds !== OPT_0023_THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > OPT_0023_MAXIMUM_POLL_GAP_MILLISECONDS
  ) {
    throw new Error("OPT-0023 thermal polling cadence is invalid");
  }
  if (
    nonNominalObservationCount !== 0 ||
    missingObservationCount !== 0
  ) {
    throw new Error("OPT-0023 post-warmup thermal gate was not nominal");
  }
  return Object.freeze({
    source,
    command,
    traceStartedAtEpochMilliseconds,
    traceStartObservationIndex,
    traceObservationCountThroughGate:
      gateCompletedObservationIndex - traceStartObservationIndex + 1,
    gateStartedAtEpochMilliseconds,
    gateStartObservationIndex,
    gateCompletedAtEpochMilliseconds,
    gateCompletedObservationIndex,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0023_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
    missingObservationCount: 0,
  });
}

export function parseOpt0023ThermalCompletion(
  parameters: URLSearchParams,
  gate: Opt0023ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0023ThermalCompletion {
  const rawTraceSha256 = requiredString(parameters, "thermalTraceSha256");
  const rawTraceByteLength = requiredPositiveInteger(
    parameters,
    "thermalTraceByteLength",
  );
  const completedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalTraceCompletedAtEpochMilliseconds",
  );
  const completedObservationIndex = requiredNonnegativeInteger(
    parameters,
    "thermalTraceCompletedObservationIndex",
  );
  const observationCount = requiredPositiveInteger(
    parameters,
    "thermalTraceObservations",
  );
  const maximumPollGapMilliseconds = requiredNumber(
    parameters,
    "thermalTraceMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredNonnegativeInteger(
    parameters,
    "thermalTraceNonNominalObservations",
  );
  const missingObservationCount = requiredNonnegativeInteger(
    parameters,
    "thermalTraceMissingObservations",
  );
  const initialLevel = requiredNonnegativeInteger(
    parameters,
    "thermalTraceInitialLevel",
  );
  const transitions = parseTransitions(
    requiredString(parameters, "thermalTraceTransitionsJson"),
    gate.traceStartedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    initialLevel,
  );
  const durationMilliseconds =
    completedAtEpochMilliseconds - gate.traceStartedAtEpochMilliseconds;

  if (!/^[0-9a-f]{64}$/.test(rawTraceSha256)) {
    throw new Error("OPT-0023 thermal trace SHA-256 is invalid");
  }
  if (
    !Number.isFinite(cleanupCompletedAtEpochMilliseconds) ||
    completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error("OPT-0023 thermal trace must cover validation and cleanup");
  }
  if (
    completedObservationIndex < gate.gateCompletedObservationIndex ||
    observationCount !==
      completedObservationIndex - gate.traceStartObservationIndex + 1 ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0023_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0023 full thermal trace observation indexes are incomplete");
  }
  if (
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > OPT_0023_MAXIMUM_POLL_GAP_MILLISECONDS ||
    missingObservationCount !== 0 ||
    initialLevel > 3 ||
    transitions.length > observationCount - 1 ||
    nonNominalObservationCount > observationCount - gate.observationCount
  ) {
    throw new Error("OPT-0023 thermal continuation cadence is malformed");
  }
  const disclosedNonNominalLevel =
    initialLevel !== 0 ||
    transitions.some((transition) => transition.level !== 0);
  let levelAtGateStart = initialLevel;
  for (const transition of transitions) {
    if (transition.atEpochMilliseconds > gate.gateStartedAtEpochMilliseconds) {
      break;
    }
    levelAtGateStart = transition.level;
  }
  const transitionInsideGate = transitions.some((transition) =>
    transition.atEpochMilliseconds > gate.gateStartedAtEpochMilliseconds &&
    transition.atEpochMilliseconds <= gate.gateCompletedAtEpochMilliseconds
  );
  if (
    (nonNominalObservationCount === 0 && disclosedNonNominalLevel) ||
    (nonNominalObservationCount > 0 && !disclosedNonNominalLevel) ||
    levelAtGateStart !== 0 || transitionInsideGate
  ) {
    throw new Error("OPT-0023 thermal transition summary is inconsistent");
  }
  return Object.freeze({
    source: OPT_0023_THERMAL_SOURCE,
    command: OPT_0023_THERMAL_COMMAND,
    rawTraceSha256,
    rawTraceByteLength,
    rawTraceSchema:
      "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
    traceStartedAtEpochMilliseconds: gate.traceStartedAtEpochMilliseconds,
    traceStartObservationIndex: gate.traceStartObservationIndex,
    completedAtEpochMilliseconds,
    completedObservationIndex,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0023_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    missingObservationCount: 0,
    initialLevel,
    transitions,
    gateStartObservationIndex: gate.gateStartObservationIndex,
    gateObservationCount: gate.observationCount,
    coversWarmupGateRunValidationAndCleanup: true,
    decisionRule:
      "one-fresh-post-warmup-nominal-30s-gate-then-accept-disclosed-trace",
    unchangedThermalRetryPerformed: false,
  });
}

export function serializeOpt0023Failure(
  error: unknown,
  cleanupError?: unknown,
): Readonly<Record<string, unknown>> {
  const value = serializeFailureObject(error);
  return Object.freeze({
    ...value,
    ...(cleanupError === undefined
      ? {}
      : { cleanupError: serializeFailureObject(cleanupError) }),
  });
}

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const prepare = element<HTMLButtonElement>("#prepare");
  const run = element<HTMLButtonElement>("#run");
  const finalize = element<HTMLButtonElement>("#finalize");
  const gateFieldset = element<HTMLFieldSetElement>("#thermal-gate");
  const completionFieldset = element<HTMLFieldSetElement>("#thermal-completion");
  const download = element<HTMLAnchorElement>("#download");
  let identity: Opt0023RunIdentity;
  try {
    identity = parseOpt0023RunIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }
  const worker = new Worker(new URL(
    "./opt-0023-vae-c4500-production-family-profile-worker.ts",
    import.meta.url,
  ), { type: "module" });
  let preparation: WorkerReadyMessage["preparation"] | undefined;
  let gate: Opt0023ThermalGate | undefined;
  let pendingResult: WorkerCompleteMessage["result"] | undefined;
  let pendingFailure: unknown;
  let settled = false;

  prepare.addEventListener("click", () => {
    prepare.disabled = true;
    document.body.dataset.status = "preparing";
    updateProgress("authenticating package and preparing the exact VAE backend");
    worker.postMessage({ type: "initialize", identity });
  }, { once: true });

  run.addEventListener("click", () => {
    try {
      if (preparation === undefined) {
        throw new Error("OPT-0023 preparation has not completed");
      }
      gate = parseOpt0023ThermalGate(
        fieldParameters("#thermal-gate"),
        preparation,
        Date.now(),
      );
      const launchDelay = Date.now() - gate.gateCompletedAtEpochMilliseconds;
      if (
        launchDelay < 0 ||
        launchDelay > OPT_0023_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
      ) {
        throw new Error("OPT-0023 launch did not immediately follow the gate");
      }
      run.disabled = true;
      gateFieldset.disabled = true;
      document.body.dataset.status = "running";
      updateProgress("running one complete C4500 stream; no thermal retry");
      worker.postMessage({ type: "run", thermalGate: gate });
    } catch (error) {
      pendingFailure = error;
      worker.postMessage({ type: "dispose" });
      updateProgress("invalid preflight; disposing the prepared VAE backend");
    }
  }, { once: true });

  finalize.addEventListener("click", () => {
    try {
      if (pendingResult === undefined || gate === undefined) {
        throw new Error("OPT-0023 profile result is not ready");
      }
      const thermal = parseOpt0023ThermalCompletion(
        fieldParameters("#thermal-completion"),
        gate,
        pendingResult.lifecycle.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      const receipt = Object.freeze({
        ...pendingResult,
        protocol: Object.freeze({
          ...(pendingResult.protocol as Readonly<Record<string, unknown>>),
          thermal,
          thermalClassification: thermal.nonNominalObservationCount === 0
            ? "complete-trace-nominal"
            : "complete-trace-nonnominal-outside-gate-observed",
          externalThermalArtifactJoined: true,
          finalPageJoinAndSerializationExcludedFromAuthoritativeTiming: true,
          unchangedThermalRetryPerformed: false,
        }),
      });
      const rawJson = JSON.stringify(receipt);
      const rawJsonBytes = new TextEncoder().encode(rawJson).byteLength;
      if (rawJsonBytes > OPT_0023_MAXIMUM_RECEIPT_BYTES) {
        throw new Error(
          `OPT-0023 receipt is ${rawJsonBytes} bytes; bounded limit is ` +
            OPT_0023_MAXIMUM_RECEIPT_BYTES,
        );
      }
      settled = true;
      finalize.disabled = true;
      completionFieldset.disabled = true;
      document.body.dataset.status = "passed";
      updateProgress("passed; the one disclosed thermal trace was accepted");
      window.__ACE_OPT0023_RESULT__ = receipt;
      download.href = URL.createObjectURL(new Blob([rawJson], {
        type: "application/json",
      }));
      download.hidden = false;
      element<HTMLPreElement>("#result").textContent = JSON.stringify(
        compactDomSummary(receipt, rawJsonBytes),
      );
      worker.terminate();
    } catch (error) {
      settled = true;
      finalize.disabled = true;
      completionFieldset.disabled = true;
      worker.terminate();
      finishFailure(error);
    }
  }, { once: true });

  worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    if (settled) return;
    const message = event.data;
    if (message.type === "progress") {
      updateProgress(message.message);
      return;
    }
    if (message.type === "ready-for-thermal-gate") {
      preparation = message.preparation;
      gateFieldset.disabled = false;
      run.disabled = false;
      document.body.dataset.status = "ready";
      updateProgress(
        "ready; enter a fresh 30-second nominal slice from the continuing logger",
      );
      return;
    }
    if (message.type === "disposed") {
      settled = true;
      worker.terminate();
      finishFailure(pendingFailure ?? new Error("OPT-0023 was disposed"));
      return;
    }
    if (message.type === "failed") {
      settled = true;
      worker.terminate();
      finishFailure(
        pendingFailure === undefined
          ? message.error
          : serializeOpt0023Failure(pendingFailure, message.error),
        true,
      );
      return;
    }
    pendingResult = message.result;
    completionFieldset.disabled = false;
    finalize.disabled = false;
    document.body.dataset.status = "awaiting-thermal-trace";
    updateProgress(
      "profile, raw hash, and cleanup complete; stop/hash the external trace",
    );
  });

  worker.addEventListener("error", (event) => {
    if (settled) return;
    settled = true;
    worker.terminate();
    finishFailure(new Error(event.message));
  });
}

function compactDomSummary(
  receipt: Readonly<Record<string, unknown>>,
  rawJsonBytes: number,
): Readonly<Record<string, unknown>> {
  const timing = receipt.timing as Readonly<Record<string, unknown>>;
  const attribution = receipt.attribution as Readonly<Record<string, unknown>>;
  return Object.freeze({
    schema: receipt.schema,
    experimentId: receipt.experimentId,
    status: receipt.status,
    rawJsonBytes,
    rawReceiptAvailableAs:
      "window.__ACE_OPT0023_RESULT__ and the download link",
    fullStreamWallMilliseconds: timing.fullStreamWallMilliseconds,
    summedDecodeWallMilliseconds: timing.summedDecodeWallMilliseconds,
    decoderSubmitThroughDrainMilliseconds:
      attribution.decoderSubmitThroughDrainMilliseconds,
    families: attribution.families,
    output: receipt.output,
    memory: receipt.memory,
    lifecycle: receipt.lifecycle,
    protocol: receipt.protocol,
  });
}

function parseTransitions(
  text: string,
  minimumEpochMilliseconds: number,
  maximumEpochMilliseconds: number,
  initialLevel: number,
): readonly Opt0023ThermalTransition[] {
  const value: unknown = JSON.parse(text);
  if (!Array.isArray(value)) {
    throw new Error("OPT-0023 thermal transitions must be a JSON array");
  }
  let prior = minimumEpochMilliseconds;
  let priorLevel = initialLevel;
  return Object.freeze(value.map((entry) => {
    const at = isRecord(entry) ? entry.atEpochMilliseconds : undefined;
    const level = isRecord(entry) ? entry.level : undefined;
    if (
      typeof at !== "number" || !Number.isFinite(at) || at <= prior ||
      at > maximumEpochMilliseconds || typeof level !== "number" ||
      !Number.isSafeInteger(level) || level < 0 || level > 3 ||
      level === priorLevel
    ) {
      throw new Error("OPT-0023 thermal transition is invalid");
    }
    prior = at;
    priorLevel = level;
    return Object.freeze({ atEpochMilliseconds: at, level });
  }));
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value.trim());
  return parameters;
}

function updateProgress(message: string): void {
  element<HTMLElement>("#progress").textContent = message;
}

function finishFailure(error: unknown, alreadySerialized = false): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  element<HTMLPreElement>("#result").textContent = JSON.stringify({
    error: alreadySerialized ? error : serializeOpt0023Failure(error),
  });
}

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (found === null) throw new Error(`Missing OPT-0023 element ${selector}`);
  return found;
}

function requiredString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0023 field ${name} is missing`);
  }
  return value;
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredString(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0023 field ${name} is invalid`);
  }
  return value;
}

function requiredPositiveInteger(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredNumber(parameters, name);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`OPT-0023 field ${name} must be a positive integer`);
  }
  return value;
}

function requiredNonnegativeInteger(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredNumber(parameters, name);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`OPT-0023 field ${name} must be a non-negative integer`);
  }
  return value;
}

function isCommit(value: string): boolean {
  return /^[0-9a-f]{40}$/.test(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function serializeFailureObject(error: unknown): Readonly<Record<string, unknown>> {
  const record = isRecord(error) ? error : undefined;
  const name = boundedString(
    typeof record?.name === "string" ? record.name : "Error",
    256,
  );
  const message = boundedString(
    typeof record?.message === "string" ? record.message : String(error),
    FAILURE_TEXT_LIMIT,
  );
  const ownFields: Record<string, unknown> = {};
  if (record !== undefined) {
    for (const key of Object.getOwnPropertyNames(record)) {
      if (["name", "message", "stack"].includes(key)) continue;
      if (Object.keys(ownFields).length === FAILURE_FIELD_LIMIT) break;
      const descriptor = Object.getOwnPropertyDescriptor(record, key);
      ownFields[key] = descriptor !== undefined && "value" in descriptor
        ? boundedFailureValue(descriptor.value)
        : "[accessor]";
    }
  }
  const stack = typeof record?.stack === "string"
    ? boundedString(record.stack, FAILURE_STACK_LIMIT)
    : undefined;
  return Object.freeze({
    name,
    message,
    ...(stack === undefined ? {} : { stack }),
    ...(Object.keys(ownFields).length === 0
      ? {}
      : { ownFields: Object.freeze(ownFields) }),
  });
}

function boundedFailureValue(value: unknown): unknown {
  if (
    value === null || typeof value === "number" ||
    typeof value === "boolean"
  ) return value;
  if (typeof value === "string") return boundedString(value, 1_024);
  if (Array.isArray(value)) {
    return Object.freeze(value.slice(0, 8).map((entry) =>
      typeof entry === "string" ? boundedString(entry, 256) : String(entry)
    ));
  }
  if (isRecord(value)) {
    return Object.freeze(Object.fromEntries(
      Object.entries(value).slice(0, 8).map(([key, entry]) => [
        boundedString(key, 128),
        typeof entry === "string" ? boundedString(entry, 256) : String(entry),
      ]),
    ));
  }
  return boundedString(String(value), 1_024);
}

function boundedString(value: string, maximum: number): string {
  return value.length <= maximum
    ? value
    : `${value.slice(0, maximum)}...[truncated ${value.length - maximum} chars]`;
}
