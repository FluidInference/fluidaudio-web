import {
  OPT_0087_THERMAL_COMMAND,
  OPT_0087_THERMAL_POLL_MILLISECONDS,
  OPT_0087_THERMAL_SOURCE,
  OPT_0087_THERMAL_TRACE_SCHEMA,
  validateOpt0087RunIdentity,
  validateOpt0087ThermalCompletion,
  validateOpt0087ThermalLaunch,
  type Opt0087RunIdentity,
  type Opt0087ThermalCompletion,
  type Opt0087ThermalLaunch,
} from "./opt-0087-planner-package-native-low-row-gemv-contract.js";

interface HeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

type WorkerMessage =
  | Readonly<{
      readonly type: "ready-for-thermal-gate";
      readonly warmupCompletedAtEpochMilliseconds: number;
      readonly preparation: unknown;
    }>
  | Readonly<{ readonly type: "progress"; readonly message: string }>
  | Readonly<{
      readonly type: "awaiting-through-cleanup-thermal";
      readonly result: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{ readonly type: "failed"; readonly error: unknown }>;

declare global {
  interface Window {
    __ACE_OPT0087_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const prepare = requireElement<HTMLButtonElement>("#prepare");
  const cancel = requireElement<HTMLButtonElement>("#cancel");
  const runTimed = requireElement<HTMLButtonElement>("#run-timed");
  const finalize = requireElement<HTMLButtonElement>("#finalize");
  const launchFields = requireElement<HTMLFieldSetElement>("#pre-gate");
  const completionFields = requireElement<HTMLFieldSetElement>("#final-gate");
  let identity: Opt0087RunIdentity;
  try {
    identity = parseIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }

  let worker: Worker | undefined;
  let heartbeat: ReturnType<typeof startHeartbeat> | undefined;
  let readyAtEpochMilliseconds = 0;
  let timingStarted = false;
  let thermalLaunch: Opt0087ThermalLaunch | undefined;
  let pendingResult: Readonly<Record<string, unknown>> | undefined;
  let settled = false;
  populateInputs(new URL(location.href).searchParams);

  prepare.addEventListener("click", () => {
    if (worker !== undefined || settled) return;
    prepare.disabled = true;
    cancel.disabled = false;
    heartbeat = startHeartbeat();
    document.body.dataset.status = "preparing";
    updateProgress(
      "authenticating one BF16 planner owner and warming paired M1/M2 full-head arms",
    );
    const active = new Worker(new URL(
      "./opt-0087-planner-package-native-low-row-gemv-worker.ts",
      import.meta.url,
    ), {
      type: "module",
      name: "ace-opt-0087-package-native-low-row-gemv",
    });
    worker = active;
    active.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") {
        updateProgress(message.message);
        return;
      }
      if (message.type === "ready-for-thermal-gate") {
        if (readyAtEpochMilliseconds !== 0 || timingStarted) {
          publishFailure("protocol", new Error("OPT-0087 READY repeated"));
          return;
        }
        readyAtEpochMilliseconds =
          message.warmupCompletedAtEpochMilliseconds;
        launchFields.disabled = false;
        runTimed.disabled = false;
        document.body.dataset.status = "ready";
        updateProgress(
          `READY at ${readyAtEpochMilliseconds}; enter a fresh 30-second nominal slice from the already-running logger`,
        );
        requireElement<HTMLElement>("#preparation").textContent =
          JSON.stringify(message.preparation, null, 2);
        return;
      }
      if (message.type === "awaiting-through-cleanup-thermal") {
        if (!timingStarted || pendingResult !== undefined) {
          publishFailure(
            "protocol",
            new Error("OPT-0087 measurement completed outside its launch"),
          );
          return;
        }
        pendingResult = message.result;
        cancel.disabled = true;
        completionFields.disabled = false;
        finalize.disabled = false;
        document.body.dataset.status = "awaiting-final-thermal";
        updateProgress(
          "measurement and cleanup completed; stop the same logger after its next poll, then bind the complete raw trace",
        );
        requireElement<HTMLElement>("#raw-result").textContent =
          JSON.stringify(message.result, null, 2);
        active.terminate();
        worker = undefined;
        return;
      }
      publishFailure("worker", message.error);
    });
    active.addEventListener("error", (event) => {
      publishFailure("worker-error", event.error ?? event.message);
    });
    active.postMessage({ type: "initialize", identity });
  }, { once: true });

  cancel.addEventListener("click", () => {
    if (worker === undefined || settled) return;
    cancel.disabled = true;
    updateProgress(
      "cancellation requested; waiting for the current bounded package operation and explicit owner cleanup",
    );
    worker.postMessage({ type: "cancel" });
  });

  runTimed.addEventListener("click", () => {
    if (
      worker === undefined || readyAtEpochMilliseconds === 0 ||
      timingStarted || settled
    ) return;
    const launchedAtEpochMilliseconds = Date.now();
    try {
      thermalLaunch = parseOpt0087ThermalLaunch(
        fieldParameters(launchFields),
        readyAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
      timingStarted = true;
      runTimed.disabled = true;
      launchFields.disabled = true;
      document.body.dataset.status = "running";
      updateProgress(
        "thermal launch accepted; running 16 balanced same-state A/B pairs",
      );
      worker.postMessage({ type: "run-timed", thermalLaunch });
    } catch (error) {
      requireElement<HTMLElement>("#result").textContent =
        JSON.stringify(errorValue(error), null, 2);
      updateProgress(
        "thermal launch rejected; owner remains READY—enter a fresh continuous nominal slice or cancel",
      );
    }
  });

  finalize.addEventListener("click", () => {
    if (pendingResult === undefined || thermalLaunch === undefined) return;
    try {
      const cleanupCompletedAtEpochMilliseconds = requireNestedNumber(
        pendingResult,
        "lifecycle",
        "cleanupCompletedAtEpochMilliseconds",
      );
      const completion = parseOpt0087ThermalCompletion(
        fieldParameters(completionFields),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const performancePassed = requireNestedBoolean(
        pendingResult,
        "timingGate",
        "passed",
      );
      const finalizedAtEpochMilliseconds = Date.now();
      const receipt = Object.freeze({
        ...pendingResult,
        status: performancePassed
          ? "passed-all-opt-0087-browser-gates"
          : "failed-opt-0087-performance-gate",
        thermal: Object.freeze({
          launch: thermalLaunch,
          completion,
          continuousTraceBoundThroughCleanup: true,
          nominalLaunchIntervalPassed: true,
          laterNonNominalDisclosed: completion.laterNonNominalDisclosed,
        }),
        pageHeartbeat: heartbeat?.stop(),
        finalizedAtEpochMilliseconds,
        finalizedAt: new Date(finalizedAtEpochMilliseconds).toISOString(),
      });
      settled = true;
      finalize.disabled = true;
      completionFields.disabled = true;
      window.__ACE_OPT0087_RESULT__ = receipt;
      document.body.dataset.status = performancePassed ? "passed" : "failed";
      updateProgress(performancePassed
        ? "all OPT-0087 browser correctness, topology, thermal, lifecycle, and performance gates passed"
        : "correctness, topology, thermal, and lifecycle passed; performance gate did not");
      const json = JSON.stringify(receipt, null, 2);
      requireElement<HTMLElement>("#result").textContent = json;
      downloadReceipt(json);
    } catch (error) {
      requireElement<HTMLElement>("#result").textContent =
        JSON.stringify(errorValue(error), null, 2);
      updateProgress(
        "through-cleanup trace rejected; the completed timing result remains stable and is not rerun",
      );
    }
  });

  window.addEventListener("beforeunload", () => {
    worker?.postMessage({ type: "cancel" });
    worker?.terminate();
  });

  function publishFailure(phase: string, error: unknown): void {
    if (settled) return;
    settled = true;
    cancel.disabled = true;
    runTimed.disabled = true;
    finalize.disabled = true;
    worker?.terminate();
    worker = undefined;
    finishFailure(Object.freeze({ phase, error }), heartbeat?.stop());
  }
}

function parseIdentity(parameters: URLSearchParams): Opt0087RunIdentity {
  return validateOpt0087RunIdentity(Object.freeze({
    implementationCommit: requiredParameter(parameters, "implementationCommit"),
    harnessCommit: requiredParameter(parameters, "harnessCommit"),
    machineModel: requiredParameter(parameters, "machineModel"),
    osVersion: requiredParameter(parameters, "osVersion"),
    osBuild: requiredParameter(parameters, "osBuild"),
    browserVersion: requiredParameter(parameters, "browserVersion"),
    gpuCoreCount: positiveIntegerParameter(parameters, "gpuCoreCount"),
    memoryBytes: positiveIntegerParameter(parameters, "memoryBytes"),
  }));
}

export function parseOpt0087ThermalLaunch(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0087ThermalLaunch {
  const traceStartedAtEpochMilliseconds = finiteParameter(
    parameters,
    "thermalTraceStartedAtEpochMilliseconds",
  );
  const gateStartedAtEpochMilliseconds = finiteParameter(
    parameters,
    "thermalGateStartedAtEpochMilliseconds",
  );
  const gateCompletedAtEpochMilliseconds = finiteParameter(
    parameters,
    "thermalGateCompletedAtEpochMilliseconds",
  );
  const launch = Object.freeze({
    source: requiredParameter(
      parameters,
      "thermalSource",
    ) as typeof OPT_0087_THERMAL_SOURCE,
    command: requiredParameter(
      parameters,
      "thermalCommand",
    ) as typeof OPT_0087_THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds,
    gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds,
    observationCount: integerParameter(parameters, "thermalGateObservations"),
    pollMilliseconds: integerParameter(
      parameters,
      "thermalPollMilliseconds",
    ) as typeof OPT_0087_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds: finiteParameter(
      parameters,
      "thermalGateMaximumPollGapMilliseconds",
    ),
    nonNominalObservationCount: integerParameter(
      parameters,
      "thermalGateNonNominalObservations",
    ) as 0,
    missingObservationCount: integerParameter(
      parameters,
      "thermalGateMissingObservations",
    ) as 0,
    readyToGateDelayMilliseconds:
      gateStartedAtEpochMilliseconds - readyAtEpochMilliseconds,
    launchDelayMilliseconds:
      launchedAtEpochMilliseconds - gateCompletedAtEpochMilliseconds,
  });
  validateOpt0087ThermalLaunch(
    launch,
    readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  return launch;
}

export function parseOpt0087ThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0087ThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds = Date.now(),
): Opt0087ThermalCompletion {
  const transitionsValue: unknown = JSON.parse(requiredParameter(
    parameters,
    "thermalTraceTransitionsJson",
  ));
  if (!Array.isArray(transitionsValue)) {
    throw new Error("OPT-0087 thermal transitions must be a JSON array");
  }
  const transitions = Object.freeze(transitionsValue.map((value) => {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new Error("OPT-0087 thermal transition must be an object");
    }
    const record = value as Readonly<Record<string, unknown>>;
    const atEpochMilliseconds = Number(record["atEpochMilliseconds"]);
    if (!Number.isSafeInteger(atEpochMilliseconds)) {
      throw new Error("OPT-0087 thermal transition epoch is invalid");
    }
    return Object.freeze({
      atEpochMilliseconds,
      level: requireThermalLevel(Number(record["level"])),
    });
  }));
  const nonNominalObservationCount = integerParameter(
    parameters,
    "thermalTraceNonNominalObservations",
  );
  const completion = Object.freeze({
    schema: requiredParameter(
      parameters,
      "thermalTraceSchema",
    ) as typeof OPT_0087_THERMAL_TRACE_SCHEMA,
    sha256: requiredParameter(parameters, "thermalTraceSha256").toLowerCase(),
    byteLength: integerParameter(parameters, "thermalTraceByteLength"),
    completedAtEpochMilliseconds: finiteParameter(
      parameters,
      "thermalTraceCompletedAtEpochMilliseconds",
    ),
    observationCount: integerParameter(parameters, "thermalTraceObservations"),
    maximumPollGapMilliseconds: finiteParameter(
      parameters,
      "thermalTraceMaximumPollGapMilliseconds",
    ),
    nonNominalObservationCount,
    missingObservationCount: integerParameter(
      parameters,
      "thermalTraceMissingObservations",
    ) as 0,
    initialLevel: requireThermalLevel(integerParameter(
      parameters,
      "thermalTraceInitialLevel",
    )) as 0,
    finalLevel: requireThermalLevel(integerParameter(
      parameters,
      "thermalTraceFinalLevel",
    )),
    transitions,
    coversCleanup: true as const,
    laterNonNominalDisclosed: nonNominalObservationCount > 0,
  });
  validateOpt0087ThermalCompletion(
    completion,
    launch,
    cleanupCompletedAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
  return completion;
}

function requireThermalLevel(value: number): 0 | 1 | 2 | 3 {
  if (!Number.isSafeInteger(value) || value < 0 || value > 3) {
    throw new Error("OPT-0087 thermal level is invalid");
  }
  return value as 0 | 1 | 2 | 3;
}

function fieldParameters(fieldset: HTMLFieldSetElement): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of fieldset.querySelectorAll<HTMLInputElement>("input")) {
    parameters.set(input.name, input.value);
  }
  return parameters;
}

function populateInputs(parameters: URLSearchParams): void {
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#pre-gate input, #final-gate input",
  )) {
    const value = parameters.get(input.name);
    if (value !== null) input.value = value;
  }
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (!value) throw new Error(`OPT-0087 field ${name} is missing`);
  return value;
}

function finiteParameter(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) throw new Error(`OPT-0087 field ${name} is invalid`);
  return value;
}

function integerParameter(parameters: URLSearchParams, name: string): number {
  const value = finiteParameter(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0087 field ${name} must be an integer`);
  }
  return value;
}

function positiveIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = integerParameter(parameters, name);
  if (value <= 0) throw new Error(`OPT-0087 ${name} must be positive`);
  return value;
}

function requireNestedNumber(
  object: Readonly<Record<string, unknown>>,
  owner: string,
  field: string,
): number {
  const nested = object[owner];
  const value = typeof nested === "object" && nested !== null
    ? (nested as Readonly<Record<string, unknown>>)[field]
    : undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`OPT-0087 raw result omitted ${owner}.${field}`);
  }
  return value;
}

function requireNestedBoolean(
  object: Readonly<Record<string, unknown>>,
  owner: string,
  field: string,
): boolean {
  const nested = object[owner];
  const value = typeof nested === "object" && nested !== null
    ? (nested as Readonly<Record<string, unknown>>)[field]
    : undefined;
  if (typeof value !== "boolean") {
    throw new Error(`OPT-0087 raw result omitted ${owner}.${field}`);
  }
  return value;
}

function startHeartbeat(): { stop(): HeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let stopped = false;
  let animationHandle = 0;
  const animation = (now: number): void => {
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrameCount += 1;
    if (!stopped) animationHandle = requestAnimationFrame(animation);
  };
  animationHandle = requestAnimationFrame(animation);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    timerTickCount += 1;
  }, 10);
  return {
    stop(): HeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(animationHandle);
        clearInterval(timerHandle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

function downloadReceipt(json: string): void {
  const url = URL.createObjectURL(new Blob([json, "\n"], {
    type: "application/json",
  }));
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "opt-0087-planner-package-native-low-row-gemv.json";
    anchor.style.display = "none";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0087 element ${selector}`);
  return element;
}

function updateProgress(message: string): void {
  requireElement<HTMLElement>("#progress").textContent = message;
}

function finishFailure(error: unknown, heartbeat?: HeartbeatSnapshot): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  const receipt = Object.freeze({
    schema: "ace-opt-0087-browser-failure-v1",
    status: "failed",
    error: errorValue(error),
    ...(heartbeat === undefined ? {} : { pageHeartbeat: heartbeat }),
  });
  window.__ACE_OPT0087_RESULT__ = receipt;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
}

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      ...(error.cause === undefined ? {} : { cause: String(error.cause) }),
    });
  }
  if (typeof error === "object" && error !== null) {
    return Object.freeze({ ...error as Readonly<Record<string, unknown>> });
  }
  return Object.freeze({ name: "Error", message: String(error), stack: null });
}
