import {
  OPT_0084_EXPERIMENT_ID,
  OPT_0084_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0084_RECEIPT_SCHEMA,
  OPT_0084_THERMAL_COMMAND,
  OPT_0084_THERMAL_POLL_MILLISECONDS,
  OPT_0084_THERMAL_SOURCE,
  validateOpt0084RunIdentity,
  type Opt0084RunIdentity,
  type Opt0084ThermalLaunch,
} from "./opt-0084-planner-fused-candidate-radix-sampling-contract.js";

type WorkerMessage =
  | Readonly<{ readonly type: "progress"; readonly message: string }>
  | Readonly<{
      readonly type: "ready";
      readonly readyAtEpochMilliseconds: number;
      readonly preparation: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{
      readonly type: "measurement-complete";
      readonly evidence: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{
      readonly type: "failed";
      readonly phase: string;
      readonly error: Readonly<Record<string, unknown>>;
      readonly cleanup?: Readonly<Record<string, unknown>>;
      readonly cleanupError?: Readonly<Record<string, unknown>>;
    }>;

declare global {
  interface Window {
    __ACE_OPT0084_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

if (typeof document !== "undefined") initializeOpt0084Page();

function initializeOpt0084Page(): void {
  const prepare = element<HTMLButtonElement>("#prepare");
  const cancel = element<HTMLButtonElement>("#cancel");
  const run = element<HTMLButtonElement>("#run");
  const finalize = element<HTMLButtonElement>("#finalize");
  const launchFields = element<HTMLFieldSetElement>("#thermal-gate");
  const completionFields = element<HTMLFieldSetElement>("#thermal-completion");
  const progress = element<HTMLElement>("#progress");
  const result = element<HTMLElement>("#result");
  const download = element<HTMLAnchorElement>("#download");

  let worker: Worker | undefined;
  let readyAtEpochMilliseconds = 0;
  let timingStarted = false;
  let settled = false;
  let pendingEvidence: Readonly<Record<string, unknown>> | undefined;
  let downloadUrl: string | undefined;
  let runIdentity: Opt0084RunIdentity;
  try {
    runIdentity = parseOpt0084RunIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    publishFailure("run-identity", error);
    return;
  }

  prepare.addEventListener("click", () => {
    if (worker !== undefined || settled) return;
    prepare.disabled = true;
    cancel.disabled = false;
    document.body.dataset.status = "preparing";
    progress.textContent =
      "authenticating planner, retaining eight actual BF16 fixture outputs (seven timed), and checking both arms";
    const active = new Worker(new URL(
      "./opt-0084-planner-fused-candidate-radix-sampling-worker.ts",
      import.meta.url,
    ), { type: "module", name: "ace-opt-0084-planner-radix-sampler" });
    worker = active;
    active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") {
        progress.textContent = message.message;
        return;
      }
      if (message.type === "ready") {
        if (readyAtEpochMilliseconds !== 0 || timingStarted) {
          publishFailure("protocol", new Error("OPT-0084 READY repeated"));
          return;
        }
        readyAtEpochMilliseconds = message.readyAtEpochMilliseconds;
        launchFields.disabled = false;
        run.disabled = false;
        document.body.dataset.status = "ready";
        progress.textContent =
          `READY at ${readyAtEpochMilliseconds}; correctness and both-arm warmup are outside timing`;
        result.textContent = JSON.stringify(message.preparation, null, 2);
        return;
      }
      if (message.type === "measurement-complete") {
        if (!timingStarted || pendingEvidence !== undefined) {
          publishFailure("protocol", new Error(
            "OPT-0084 measurement completed outside the accepted launch",
          ));
          return;
        }
        pendingEvidence = message.evidence;
        cancel.disabled = true;
        completionFields.disabled = false;
        finalize.disabled = false;
        document.body.dataset.status = "awaiting-thermal-completion";
        progress.textContent =
          "measurement and cleanup completed; stop the same logger after this boundary, then finalize";
        result.textContent = JSON.stringify(message.evidence, null, 2);
        return;
      }
      publish(Object.freeze({
        schema: "ace-opt-0084-planner-radix-sampler-worker-failure-v1",
        experiment: OPT_0084_EXPERIMENT_ID,
        status: "failed",
        passed: false,
        phase: message.phase,
        error: message.error,
        ...(message.cleanup === undefined ? {} : { cleanup: message.cleanup }),
        ...(message.cleanupError === undefined
          ? {}
          : { cleanupError: message.cleanupError }),
        decision: Object.freeze({
          disposition: "inconclusive-invalid-correctness-or-lifecycle-evidence",
          browserSamplerGatePassed: false,
          productionIntegrationAuthorized: false,
          unchangedTimingRetryAuthorized: false,
        }),
      }));
    });
    active.addEventListener("error", (event) => {
      publishFailure("worker-error", event.error ?? event.message);
    });
    active.postMessage({ type: "prepare", identity: runIdentity });
  }, { once: true });

  cancel.addEventListener("click", () => {
    if (worker === undefined || settled) return;
    cancel.disabled = true;
    progress.textContent =
      "cancellation requested; waiting for the current bounded operation";
    worker.postMessage({ type: "cancel" });
  });

  run.addEventListener("click", () => {
    if (
      worker === undefined || readyAtEpochMilliseconds === 0 ||
      timingStarted || settled
    ) return;
    const launchedAtEpochMilliseconds = Date.now();
    try {
      const thermalLaunch = parseOpt0084ThermalLaunch(
        fieldParameters(launchFields),
        readyAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
      timingStarted = true;
      run.disabled = true;
      launchFields.disabled = true;
      document.body.dataset.status = "running";
      progress.textContent =
        "thermal launch accepted; running the single registered 16-round sampler gate";
      worker.postMessage({ type: "run", thermalLaunch });
    } catch (error) {
      result.textContent = JSON.stringify(errorValue(error), null, 2);
      progress.textContent =
        "thermal launch rejected; enter a fresh continuous nominal slice";
    }
  });

  finalize.addEventListener("click", () => {
    if (pendingEvidence === undefined) return;
    try {
      const cleanupCompletedAtEpochMilliseconds = requiredFiniteRecord(
        pendingEvidence,
        "cleanupCompletedAtEpochMilliseconds",
      );
      const thermalLaunch = pendingEvidence["thermalLaunch"] as
        Opt0084ThermalLaunch;
      const thermalCompletion = parseOpt0084ThermalCompletion(
        fieldParameters(completionFields),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const inPagePassed = pendingEvidence["inPagePassed"] === true;
      const receipt = Object.freeze({
        ...pendingEvidence,
        schema: OPT_0084_RECEIPT_SCHEMA,
        status: inPagePassed ? "passed-browser-sampler-gate" :
          "stopped-browser-sampler-gate",
        passed: inPagePassed,
        thermal: Object.freeze({
          launch: thermalLaunch,
          completion: thermalCompletion,
          passed: true,
        }),
        decision: Object.freeze({
          disposition: inPagePassed
            ? "positive-browser-sampler-gate-integration-gates-still-required"
            : "negative-stop-browser-sampler-gate",
          browserSamplerGatePassed: inPagePassed,
          completeTrajectoryGateRequired: inPagePassed,
          plannerEnabledProductGateRequired: inPagePassed,
          productionIntegrationAuthorized: false,
          unchangedTimingRetryAuthorized: false,
        }),
      });
      publish(receipt);
    } catch (error) {
      result.textContent = JSON.stringify(errorValue(error), null, 2);
      progress.textContent =
        "through-cleanup trace rejected; timing remains stable and will not be repeated";
    }
  });

  window.addEventListener("beforeunload", () => {
    worker?.terminate();
    if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
  });

  function publish(receipt: Readonly<Record<string, unknown>>): void {
    settled = true;
    worker?.terminate();
    worker = undefined;
    cancel.disabled = true;
    run.disabled = true;
    finalize.disabled = true;
    window.__ACE_OPT0084_RESULT__ = receipt;
    const passed = receipt["passed"] === true;
    document.body.dataset.status = passed ? "passed" : "stopped";
    progress.textContent = passed
      ? "PASSED browser sampler gate; trajectory/product integration gates remain and production is unchanged"
      : "STOPPED at the browser sampler gate; production is unchanged";
    const json = JSON.stringify(receipt, null, 2);
    result.textContent = json;
    if (downloadUrl !== undefined) URL.revokeObjectURL(downloadUrl);
    downloadUrl = URL.createObjectURL(new Blob([json, "\n"], {
      type: "application/json",
    }));
    download.href = downloadUrl;
    download.hidden = false;
    download.click();
  }

  function publishFailure(phase: string, error: unknown): void {
    if (settled) return;
    publish(Object.freeze({
      schema: "ace-opt-0084-planner-radix-sampler-page-failure-v1",
      experiment: OPT_0084_EXPERIMENT_ID,
      status: "failed",
      passed: false,
      phase,
      error: errorValue(error),
      decision: Object.freeze({
        disposition: "inconclusive-invalid-page-or-worker-evidence",
        browserSamplerGatePassed: false,
        productionIntegrationAuthorized: false,
        unchangedTimingRetryAuthorized: false,
      }),
    }));
  }
}

export function parseOpt0084RunIdentity(
  parameters: URLSearchParams,
): Opt0084RunIdentity {
  return validateOpt0084RunIdentity({
    harnessCommit: required(parameters, "harnessCommit"),
    machineModel: required(parameters, "machineModel"),
    osVersion: required(parameters, "osVersion"),
    osBuild: required(parameters, "osBuild"),
    browserVersion: required(parameters, "browserVersion"),
    gpuCoreCount: finite(parameters, "gpuCoreCount"),
    memoryBytes: finite(parameters, "memoryBytes"),
  });
}

export function parseOpt0084ThermalLaunch(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0084ThermalLaunch {
  const source = required(parameters, "thermalSource");
  const command = required(parameters, "thermalCommand");
  const traceStartedAtEpochMilliseconds = finite(
    parameters,
    "thermalTraceStartedAtEpochMilliseconds",
  );
  const gateStartedAtEpochMilliseconds = finite(
    parameters,
    "thermalGateStartedAtEpochMilliseconds",
  );
  const gateCompletedAtEpochMilliseconds = finite(
    parameters,
    "thermalGateCompletedAtEpochMilliseconds",
  );
  const observationCount = integer(parameters, "thermalGateObservations");
  const pollMilliseconds = integer(parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = finite(
    parameters,
    "thermalGateMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = integer(
    parameters,
    "thermalGateNonNominalObservations",
  );
  const missingObservationCount = integer(
    parameters,
    "thermalGateMissingObservations",
  );
  const duration = gateCompletedAtEpochMilliseconds -
    gateStartedAtEpochMilliseconds;
  const readyToGateDelayMilliseconds = gateStartedAtEpochMilliseconds -
    readyAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    gateCompletedAtEpochMilliseconds;
  if (
    source !== OPT_0084_THERMAL_SOURCE || command !== OPT_0084_THERMAL_COMMAND ||
    pollMilliseconds !== OPT_0084_THERMAL_POLL_MILLISECONDS ||
    traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    readyToGateDelayMilliseconds < 0 ||
    duration < OPT_0084_MINIMUM_NOMINAL_MILLISECONDS ||
    observationCount < Math.floor(duration /
      OPT_0084_THERMAL_POLL_MILLISECONDS) + 1 ||
    maximumPollGapMilliseconds < 0 || maximumPollGapMilliseconds > 1_250 ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    launchDelayMilliseconds < 0 || launchDelayMilliseconds > 5_000
  ) {
    throw new Error("OPT-0084 fresh continuous nominal thermal launch gate failed");
  }
  return Object.freeze({
    source: OPT_0084_THERMAL_SOURCE,
    command: OPT_0084_THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds,
    gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0084_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
    missingObservationCount: 0,
    readyToGateDelayMilliseconds,
    launchDelayMilliseconds,
  });
}

export function parseOpt0084ThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0084ThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
): Readonly<Record<string, unknown>> {
  const schema = required(parameters, "thermalTraceSchema");
  const sha256 = required(parameters, "thermalTraceSha256").toLowerCase();
  const byteLength = integer(parameters, "thermalTraceByteLength");
  const completedAtEpochMilliseconds = finite(
    parameters,
    "thermalTraceCompletedAtEpochMilliseconds",
  );
  const observationCount = integer(parameters, "thermalTraceObservations");
  const maximumPollGapMilliseconds = finite(
    parameters,
    "thermalTraceMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = integer(
    parameters,
    "thermalTraceNonNominalObservations",
  );
  const missingObservationCount = integer(
    parameters,
    "thermalTraceMissingObservations",
  );
  const initialLevel = integer(parameters, "thermalTraceInitialLevel");
  const finalLevel = integer(parameters, "thermalTraceFinalLevel");
  const transitionsValue: unknown = JSON.parse(required(
    parameters,
    "thermalTraceTransitionsJson",
  ));
  if (!Array.isArray(transitionsValue)) {
    throw new Error("OPT-0084 thermal transitions must be a JSON array");
  }
  let currentLevel = initialLevel;
  let previousTransitionAt = launch.traceStartedAtEpochMilliseconds - 1;
  const transitions = transitionsValue.map((value) => {
    if (typeof value !== "object" || value === null) {
      throw new Error("OPT-0084 thermal transition must be an object");
    }
    const record = value as Record<string, unknown>;
    const atEpochMilliseconds = Number(record["atEpochMilliseconds"]);
    const level = Number(record["level"]);
    if (
      !Number.isSafeInteger(atEpochMilliseconds) ||
      atEpochMilliseconds <= previousTransitionAt ||
      !Number.isSafeInteger(level) || level < 0 || level > 3 ||
      level === currentLevel
    ) {
      throw new Error("OPT-0084 thermal transition is invalid");
    }
    previousTransitionAt = atEpochMilliseconds;
    currentLevel = level;
    return Object.freeze({ atEpochMilliseconds, level });
  });
  const minimumObservations = Math.floor((completedAtEpochMilliseconds -
    launch.traceStartedAtEpochMilliseconds) /
    OPT_0084_THERMAL_POLL_MILLISECONDS) + 1;
  const gateStartedLevel = thermalLevelAt(
    initialLevel,
    transitions,
    launch.gateStartedAtEpochMilliseconds,
  );
  const gateTraceConsistent = gateStartedLevel === 0 && transitions.every(
    (entry) => entry.atEpochMilliseconds < launch.gateStartedAtEpochMilliseconds ||
      entry.atEpochMilliseconds > launch.gateCompletedAtEpochMilliseconds,
  );
  const disclosedNonNominal = initialLevel !== 0 ||
    transitions.some(({ level }) => level !== 0);
  if (
    schema !== "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" ||
    !/^[0-9a-f]{64}$/.test(sha256) || byteLength <= 0 ||
    completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds < launch.gateCompletedAtEpochMilliseconds ||
    observationCount < minimumObservations ||
    observationCount < launch.observationCount ||
    maximumPollGapMilliseconds < 0 || maximumPollGapMilliseconds > 1_250 ||
    maximumPollGapMilliseconds < launch.maximumPollGapMilliseconds ||
    nonNominalObservationCount < 0 ||
    nonNominalObservationCount > observationCount ||
    missingObservationCount !== 0 || initialLevel !== 0 ||
    finalLevel < 0 || finalLevel > 3 || currentLevel !== finalLevel ||
    transitions.length > Math.max(0, observationCount - 1) ||
    !gateTraceConsistent ||
    (nonNominalObservationCount === 0) === disclosedNonNominal ||
    transitions.some((entry) =>
      entry.atEpochMilliseconds < launch.traceStartedAtEpochMilliseconds ||
      entry.atEpochMilliseconds > completedAtEpochMilliseconds)
  ) {
    throw new Error("OPT-0084 complete through-cleanup thermal trace failed");
  }
  return Object.freeze({
    schema,
    sha256,
    byteLength,
    completedAtEpochMilliseconds,
    observationCount,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    missingObservationCount: 0,
    initialLevel,
    finalLevel,
    transitions: Object.freeze(transitions),
    coversCleanup: true,
    laterNonNominalDisclosed: nonNominalObservationCount > 0,
  });
}

function thermalLevelAt(
  initialLevel: number,
  transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[],
  atEpochMilliseconds: number,
): number {
  let level = initialLevel;
  for (const transition of transitions) {
    if (transition.atEpochMilliseconds > atEpochMilliseconds) break;
    level = transition.level;
  }
  return level;
}

function fieldParameters(fieldset: HTMLFieldSetElement): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of fieldset.querySelectorAll<HTMLInputElement>("input")) {
    parameters.set(input.name, input.value);
  }
  return parameters;
}

function required(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (!value) throw new Error(`OPT-0084 field ${name} is missing`);
  return value;
}

function finite(parameters: URLSearchParams, name: string): number {
  const value = Number(required(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0084 field ${name} must be finite`);
  }
  return value;
}

function integer(parameters: URLSearchParams, name: string): number {
  const value = finite(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0084 field ${name} must be an integer`);
  }
  return value;
}

function requiredFiniteRecord(
  record: Readonly<Record<string, unknown>>,
  name: string,
): number {
  const value = Number(record[name]);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0084 evidence ${name} is invalid`);
  }
  return value;
}

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  return error instanceof Error
    ? Object.freeze({
        name: error.name,
        message: error.message,
        stack: error.stack ?? null,
      })
    : Object.freeze({ name: typeof error, message: String(error), stack: null });
}

function element<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (value === null) throw new Error(`Missing OPT-0084 element ${selector}`);
  return value;
}
