export const OPT_0018_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0018_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0018_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0018_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250;
export const OPT_0018_MAXIMUM_LAUNCH_DELAY_MILLISECONDS = 5_000;
export const OPT_0018_FIXTURE_CONTRACT_SHA256 =
  "63269e63a38b8baeb535d9d7847b7d549a6472d13997459105f722ecbc609e3e" as const;
export const OPT_0018_CANONICAL_REQUEST_JSON =
  '{"generationProfile":"ace-turbo-v1-correctness","prompt":"Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.","lyrics":"","instrumental":true,"durationSeconds":180,"seed":"0000000000c0ffee","planner":{"mode":"disabled"},"metadata":{"bpm":104,"keyScale":"D minor","timeSignature":"4"}}' as const;
export const OPT_0018_CANONICAL_REQUEST_BYTES = 366 as const;
export const OPT_0018_CANONICAL_REQUEST_SHA256 =
  "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f" as const;
const OPT_0018_FAILURE_MESSAGE_LIMIT = 4_096;
const OPT_0018_FAILURE_STACK_LIMIT = 8_192;
const OPT_0018_FAILURE_FIELD_TEXT_LIMIT = 1_024;
const OPT_0018_FAILURE_OWN_FIELD_LIMIT = 12;
const OPT_0018_FAILURE_NESTED_FIELD_LIMIT = 8;

export interface Opt0018RunIdentity {
  readonly coreCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0018ThermalGate {
  readonly source: typeof OPT_0018_THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0018_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

export interface Opt0018ThermalTransition {
  readonly atEpochMilliseconds: number;
  readonly level: number;
}

export interface Opt0018ThermalCompletion {
  readonly source: typeof OPT_0018_THERMAL_SOURCE;
  readonly rawTraceSha256: string;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0018_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly transitions: readonly Opt0018ThermalTransition[];
  readonly coversGateRunAndCleanup: true;
  readonly decisionRule: "one-nominal-30s-start-then-accept-disclosed-trace";
  readonly unchangedThermalRetryPerformed: false;
}

declare global {
  interface Window {
    __ACE_OPT0018_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

interface WorkerReadyMessage {
  readonly type: "ready-for-thermal-gate";
  readonly readyAtEpochMilliseconds: number;
  readonly preparation: Readonly<Record<string, unknown>>;
}

interface WorkerProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface WorkerProfileCompleteMessage {
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

type WorkerMessage =
  | WorkerReadyMessage
  | WorkerProgressMessage
  | WorkerProfileCompleteMessage
  | WorkerFailedMessage;

export function createOpt0018Request(): Readonly<Record<string, unknown>> {
  return Object.freeze(JSON.parse(OPT_0018_CANONICAL_REQUEST_JSON));
}

export function parseOpt0018RunIdentity(
  parameters: URLSearchParams,
): Opt0018RunIdentity {
  return validateOpt0018RunIdentity({
    coreCommit: requiredString(parameters, "coreCommit"),
    harnessCommit: requiredString(parameters, "harnessCommit"),
    machineModel: requiredString(parameters, "machineModel"),
    osVersion: requiredString(parameters, "osVersion"),
    osBuild: requiredString(parameters, "osBuild"),
    browserVersion: requiredString(parameters, "browserVersion"),
    gpuCoreCount: requiredInteger(parameters, "gpuCoreCount"),
    memoryBytes: requiredInteger(parameters, "memoryBytes"),
  });
}

export function validateOpt0018RunIdentity(
  value: Opt0018RunIdentity,
): Opt0018RunIdentity {
  if (
    !isRecord(value) ||
    !isCommit(value.coreCommit) ||
    !isCommit(value.harnessCommit) ||
    !isNonempty(value.machineModel) ||
    !isNonempty(value.osVersion) ||
    !isNonempty(value.osBuild) ||
    !isNonempty(value.browserVersion) ||
    !Number.isSafeInteger(value.gpuCoreCount) ||
    value.gpuCoreCount < 1 ||
    !Number.isSafeInteger(value.memoryBytes) ||
    value.memoryBytes < 1
  ) {
    throw new Error("OPT-0018 requires a complete frozen run identity");
  }
  return Object.freeze({ ...value });
}

export function serializeOpt0018Failure(
  error: unknown,
  cleanupError?: unknown,
): Readonly<Record<string, unknown>> {
  const name = failureName(error);
  const messageValue = readFailureProperty(error, "message");
  const message = typeof messageValue === "string" && messageValue !== ""
    ? boundedText(messageValue, OPT_0018_FAILURE_MESSAGE_LIMIT)
    : isObjectLike(error)
      ? `Non-Error ${name} object was thrown`
      : boundedText(String(error), OPT_0018_FAILURE_MESSAGE_LIMIT);
  const code = readFailureProperty(error, "code");
  const reason = readFailureProperty(error, "reason");
  const stack = readFailureProperty(error, "stack");
  const ownFields = serializeFailureOwnFields(
    error,
    OPT_0018_FAILURE_OWN_FIELD_LIMIT,
  );
  return Object.freeze({
    name,
    message,
    ...(code === undefined
      ? {}
      : { code: serializeFailureValue(code) }),
    ...(reason === undefined
      ? {}
      : { reason: serializeFailureValue(reason) }),
    ...(typeof stack === "string" && stack !== ""
      ? { stack: boundedText(stack, OPT_0018_FAILURE_STACK_LIMIT) }
      : {}),
    ...(Object.keys(ownFields).length === 0 ? {} : { ownFields }),
    ...(cleanupError === undefined
      ? {}
      : { cleanupError: serializeOpt0018Failure(cleanupError) }),
  });
}

export function parseOpt0018ThermalGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0018ThermalGate {
  const source = parameters.get("thermalSource");
  const startedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalCompletedAtEpochMilliseconds",
  );
  const observationCount = requiredNumber(parameters, "thermalObservations");
  const pollMilliseconds = requiredNumber(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredNumber(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredNumber(
    parameters,
    "thermalNonNominalObservations",
  );
  const durationMilliseconds =
    completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  if (source !== OPT_0018_THERMAL_SOURCE) {
    throw new Error("OPT-0018 rejected the external thermal source");
  }
  if (
    !Number.isFinite(readyAtEpochMilliseconds) ||
    !Number.isFinite(nowEpochMilliseconds) ||
    startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    completedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error("OPT-0018 thermal gate must begin after preparation");
  }
  if (
    durationMilliseconds < OPT_0018_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(observationCount) ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0018_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0018 requires one continuous 30-second nominal gate");
  }
  if (
    pollMilliseconds !== OPT_0018_THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > OPT_0018_MAXIMUM_POLL_GAP_MILLISECONDS
  ) {
    throw new Error("OPT-0018 thermal polling cadence is invalid");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0018 pre-run thermal gate was not nominal");
  }
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0018_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

export function parseOpt0018ThermalCompletion(
  parameters: URLSearchParams,
  gate: Opt0018ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0018ThermalCompletion {
  const rawTraceSha256 = requiredString(parameters, "thermalTraceSha256");
  const completedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalTraceCompletedAtEpochMilliseconds",
  );
  const observationCount = requiredNumber(
    parameters,
    "thermalTraceObservations",
  );
  const maximumPollGapMilliseconds = requiredNumber(
    parameters,
    "thermalTraceMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredNumber(
    parameters,
    "thermalTraceNonNominalObservations",
  );
  const transitions = parseTransitions(
    requiredString(parameters, "thermalTraceTransitionsJson"),
    gate.completedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
  );
  const durationMilliseconds =
    completedAtEpochMilliseconds - gate.startedAtEpochMilliseconds;
  if (!/^[0-9a-f]{64}$/.test(rawTraceSha256)) {
    throw new Error("OPT-0018 thermal trace SHA-256 is invalid");
  }
  if (
    !Number.isFinite(cleanupCompletedAtEpochMilliseconds) ||
    completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error("OPT-0018 thermal trace must cover cleanup");
  }
  if (
    !Number.isSafeInteger(observationCount) ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0018_THERMAL_POLL_MILLISECONDS) + 1 ||
    observationCount < gate.observationCount
  ) {
    throw new Error("OPT-0018 thermal trace observation count is incomplete");
  }
  if (
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > OPT_0018_MAXIMUM_POLL_GAP_MILLISECONDS ||
    !Number.isSafeInteger(nonNominalObservationCount) ||
    nonNominalObservationCount < 0
  ) {
    throw new Error("OPT-0018 thermal continuation is malformed");
  }
  if (
    (nonNominalObservationCount === 0 &&
      transitions.some((transition) => transition.level !== 0)) ||
    (nonNominalObservationCount > 0 &&
      transitions.every((transition) => transition.level === 0))
  ) {
    throw new Error("OPT-0018 thermal transition summary is inconsistent");
  }
  return Object.freeze({
    source: OPT_0018_THERMAL_SOURCE,
    rawTraceSha256,
    startedAtEpochMilliseconds: gate.startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0018_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    transitions,
    coversGateRunAndCleanup: true,
    decisionRule: "one-nominal-30s-start-then-accept-disclosed-trace",
    unchangedThermalRetryPerformed: false,
  });
}

if (
  typeof document !== "undefined" &&
  document.querySelector("#finalize") !== null
) {
  initializeBrowserHarness();
}

function initializeBrowserHarness(): void {
  const prepare = element<HTMLButtonElement>("#prepare");
  const run = element<HTMLButtonElement>("#run");
  const finalize = element<HTMLButtonElement>("#finalize");
  const gateFieldset = element<HTMLFieldSetElement>("#thermal-gate");
  const completionFieldset = element<HTMLFieldSetElement>(
    "#thermal-completion",
  );
  const download = element<HTMLAnchorElement>("#download");
  let identity: Opt0018RunIdentity;
  try {
    identity = parseOpt0018RunIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }
  const worker = new Worker(new URL(
    "./opt-0018-dit-m2250-production-family-profile-worker.ts",
    import.meta.url,
  ), { type: "module" });
  let readyAtEpochMilliseconds: number | undefined;
  let gate: Opt0018ThermalGate | undefined;
  let pendingResult: WorkerProfileCompleteMessage["result"] | undefined;
  let settled = false;

  prepare.addEventListener("click", () => {
    prepare.disabled = true;
    document.body.dataset.status = "preparing";
    updateProgress("initializing normal production manifests, packages, and device");
    worker.postMessage({ type: "initialize", identity });
  }, { once: true });

  run.addEventListener("click", () => {
    try {
      if (readyAtEpochMilliseconds === undefined) {
        throw new Error("OPT-0018 preparation has not completed");
      }
      setBlankCompletionToNow("thermalCompletedAtEpochMilliseconds");
      gate = parseOpt0018ThermalGate(
        fieldParameters("#thermal-gate"),
        readyAtEpochMilliseconds,
        Date.now(),
      );
      const launchDelay = Date.now() - gate.completedAtEpochMilliseconds;
      if (
        launchDelay < 0 ||
        launchDelay > OPT_0018_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
      ) {
        throw new Error("OPT-0018 launch did not immediately follow the gate");
      }
      run.disabled = true;
      gateFieldset.disabled = true;
      document.body.dataset.status = "running";
      updateProgress("running the one authorized M2250 DiT profile; no thermal retry");
      worker.postMessage({ type: "run", thermalGate: gate });
    } catch (error) {
      settled = true;
      worker.terminate();
      finishFailure(error);
    }
  }, { once: true });

  finalize.addEventListener("click", () => {
    try {
      if (pendingResult === undefined || gate === undefined) {
        throw new Error("OPT-0018 profile result is not ready");
      }
      setBlankCompletionToNow("thermalTraceCompletedAtEpochMilliseconds");
      const thermal = parseOpt0018ThermalCompletion(
        fieldParameters("#thermal-completion"),
        gate,
        pendingResult.lifecycle.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      settled = true;
      finalize.disabled = true;
      completionFieldset.disabled = true;
      document.body.dataset.status = "passed";
      updateProgress("passed; one disclosed trace accepted without retry");
      const receipt = Object.freeze({
        ...pendingResult,
        protocol: Object.freeze({
          ...(pendingResult.protocol as Readonly<Record<string, unknown>>),
          thermal,
          unchangedThermalRetryPerformed: false,
        }),
      });
      window.__ACE_OPT0018_RESULT__ = receipt;
      const rawJson = JSON.stringify(receipt);
      download.href = URL.createObjectURL(new Blob([rawJson], {
        type: "application/json",
      }));
      download.hidden = false;
      element<HTMLPreElement>("#result").textContent = JSON.stringify(
        compactDomSummary(receipt, new TextEncoder().encode(rawJson).byteLength),
      );
      worker.terminate();
    } catch (error) {
      finishFailure(error);
    }
  });

  worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    if (settled) return;
    const message = event.data;
    if (message.type === "progress") {
      updateProgress(message.message);
      return;
    }
    if (message.type === "ready-for-thermal-gate") {
      readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
      gateFieldset.disabled = false;
      setBlankStartToNow();
      updateProgress("ready; begin one external continuous nominal trace now");
      window.setTimeout(() => {
        if (!settled && pendingResult === undefined) {
          run.disabled = false;
          updateProgress("enter the completed 30-second nominal gate, then run once");
        }
      }, OPT_0018_MINIMUM_NOMINAL_MILLISECONDS);
      return;
    }
    if (message.type === "failed") {
      settled = true;
      worker.terminate();
      finishFailure(message.error, true);
      return;
    }
    pendingResult = message.result;
    completionFieldset.disabled = false;
    finalize.disabled = false;
    document.body.dataset.status = "awaiting-thermal-trace";
    updateProgress(
      "profile complete and cleaned up; stop/hash the external trace, then finalize",
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
  const profiler = receipt.profiler as Readonly<Record<string, unknown>>;
  return Object.freeze({
    schema: receipt.schema,
    experimentId: receipt.experimentId,
    status: receipt.status,
    rawJsonBytes,
    rawReceiptAvailableAs:
      "window.__ACE_OPT0018_RESULT__ and the download link",
    descriptorTable: profiler.descriptorTable,
    graphCommandBufferCount: profiler.graphCommandBufferCount,
    readbackCommandBufferCount: profiler.readbackCommandBufferCount,
    totalCommandBufferCount: profiler.totalCommandBufferCount,
    graphSubmitThroughDrainMs: profiler.graphSubmitThroughDrainMs,
    families: profiler.families,
    correctness: receipt.correctness,
    lifecycle: receipt.lifecycle,
    protocol: receipt.protocol,
    scope: receipt.scope,
  });
}

function parseTransitions(
  text: string,
  minimumEpochMilliseconds: number,
  maximumEpochMilliseconds: number,
): readonly Opt0018ThermalTransition[] {
  const value: unknown = JSON.parse(text);
  if (!Array.isArray(value)) {
    throw new Error("OPT-0018 thermal transitions must be a JSON array");
  }
  let prior = minimumEpochMilliseconds;
  return Object.freeze(value.map((entry) => {
    const atEpochMilliseconds = isRecord(entry)
      ? entry.atEpochMilliseconds
      : undefined;
    const level = isRecord(entry) ? entry.level : undefined;
    if (
      !isRecord(entry) ||
      typeof atEpochMilliseconds !== "number" ||
      !Number.isFinite(atEpochMilliseconds) ||
      atEpochMilliseconds <= prior ||
      atEpochMilliseconds > maximumEpochMilliseconds ||
      typeof level !== "number" ||
      !Number.isSafeInteger(level) ||
      level < 0 ||
      level > 3
    ) {
      throw new Error("OPT-0018 thermal transition is invalid");
    }
    prior = atEpochMilliseconds;
    return Object.freeze({
      atEpochMilliseconds,
      level,
    });
  }));
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value.trim());
  return parameters;
}

function setBlankStartToNow(): void {
  const input = document.querySelector<HTMLInputElement>(
    'input[name="thermalStartedAtEpochMilliseconds"]',
  );
  if (input !== null && input.value.trim() === "") input.value = String(Date.now());
}

function setBlankCompletionToNow(name: string): void {
  const input = document.querySelector<HTMLInputElement>(`input[name="${name}"]`);
  if (input !== null && input.value.trim() === "") input.value = String(Date.now());
}

function updateProgress(message: string): void {
  element<HTMLElement>("#progress").textContent = message;
}

function finishFailure(error: unknown, alreadySerialized = false): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  element<HTMLPreElement>("#result").textContent = JSON.stringify({
    error: alreadySerialized ? error : serializeOpt0018Failure(error),
  });
}

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (found === null) throw new Error(`Missing OPT-0018 element ${selector}`);
  return found;
}

function requiredString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0018 field ${name} is missing`);
  }
  return value;
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredString(parameters, name));
  if (!Number.isFinite(value)) throw new Error(`OPT-0018 field ${name} is invalid`);
  return value;
}

function requiredInteger(parameters: URLSearchParams, name: string): number {
  const value = requiredNumber(parameters, name);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`OPT-0018 field ${name} must be a positive integer`);
  }
  return value;
}

function isCommit(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{40}$/.test(value);
}

function isNonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isObjectLike(value: unknown): value is object {
  return (typeof value === "object" && value !== null) ||
    typeof value === "function";
}

function failureName(error: unknown): string {
  const name = readFailureProperty(error, "name");
  if (typeof name === "string" && name !== "") {
    return boundedText(name, OPT_0018_FAILURE_FIELD_TEXT_LIMIT);
  }
  const constructor = readFailureProperty(error, "constructor");
  const constructorName = readFailureProperty(constructor, "name");
  return typeof constructorName === "string" && constructorName !== ""
    ? boundedText(constructorName, OPT_0018_FAILURE_FIELD_TEXT_LIMIT)
    : "Error";
}

function readFailureProperty(value: unknown, name: string): unknown {
  if (!isObjectLike(value)) return undefined;
  try {
    return (value as Record<string, unknown>)[name];
  } catch {
    return undefined;
  }
}

function serializeFailureOwnFields(
  value: unknown,
  limit: number,
): Readonly<Record<string, unknown>> {
  if (!isObjectLike(value)) return Object.freeze({});
  let names: string[];
  try {
    names = Object.getOwnPropertyNames(value);
  } catch {
    return Object.freeze({ fields: "[unreadable own properties]" });
  }
  const fields: Record<string, unknown> = {};
  for (const name of names) {
    if (["name", "message", "code", "reason", "stack"].includes(name)) {
      continue;
    }
    if (Object.keys(fields).length === limit) break;
    try {
      const descriptor = Object.getOwnPropertyDescriptor(value, name);
      fields[boundedText(name, OPT_0018_FAILURE_FIELD_TEXT_LIMIT)] =
        descriptor !== undefined && "value" in descriptor
          ? serializeFailureValue(descriptor.value)
          : "[accessor]";
    } catch {
      fields[boundedText(name, OPT_0018_FAILURE_FIELD_TEXT_LIMIT)] =
        "[unreadable property]";
    }
  }
  return Object.freeze(fields);
}

function serializeFailureValue(
  value: unknown,
  nested = false,
): unknown {
  if (value === null || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    return boundedText(value, OPT_0018_FAILURE_FIELD_TEXT_LIMIT);
  }
  if (typeof value === "bigint" || typeof value === "symbol") {
    return boundedText(String(value), OPT_0018_FAILURE_FIELD_TEXT_LIMIT);
  }
  if (typeof value === "undefined") return "[undefined]";
  if (!isObjectLike(value)) {
    return boundedText(String(value), OPT_0018_FAILURE_FIELD_TEXT_LIMIT);
  }
  const nestedName = failureName(value);
  if (nested) return `[${nestedName}]`;
  const fields = serializeNestedFailureFields(value);
  return Object.keys(fields).length === 0 ? `[${nestedName}]` : fields;
}

function serializeNestedFailureFields(
  value: object,
): Readonly<Record<string, unknown>> {
  let names: string[];
  try {
    names = Object.getOwnPropertyNames(value);
  } catch {
    return Object.freeze({ fields: "[unreadable own properties]" });
  }
  const fields: Record<string, unknown> = {};
  for (const name of names.slice(0, OPT_0018_FAILURE_NESTED_FIELD_LIMIT)) {
    try {
      const descriptor = Object.getOwnPropertyDescriptor(value, name);
      fields[boundedText(name, OPT_0018_FAILURE_FIELD_TEXT_LIMIT)] =
        descriptor !== undefined && "value" in descriptor
          ? serializeFailureValue(descriptor.value, true)
          : "[accessor]";
    } catch {
      fields[boundedText(name, OPT_0018_FAILURE_FIELD_TEXT_LIMIT)] =
        "[unreadable property]";
    }
  }
  return Object.freeze(fields);
}

function boundedText(value: string, maximumLength: number): string {
  return value.length <= maximumLength
    ? value
    : `${value.slice(0, maximumLength)}...[truncated ${
      value.length - maximumLength
    } chars]`;
}
