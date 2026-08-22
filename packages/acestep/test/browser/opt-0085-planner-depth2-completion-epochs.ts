import {
  OPT_0085_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0085_THERMAL_POLL_MILLISECONDS,
  OPT_0085_THERMAL_SOURCE,
  validateOpt0085PreGate,
  validateOpt0085RunIdentity,
  validateOpt0085ThroughCleanupTrace,
  type Opt0085RunIdentity,
  type Opt0085ThermalTraceMetadata,
} from "./opt-0085-planner-depth2-completion-epochs-contract.js";

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

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const prepare = requireElement<HTMLButtonElement>("#prepare");
  const runTimed = requireElement<HTMLButtonElement>("#run-timed");
  const finalize = requireElement<HTMLButtonElement>("#finalize");
  const preGate = requireElement<HTMLFieldSetElement>("#pre-gate");
  const finalGate = requireElement<HTMLFieldSetElement>("#final-gate");
  let identity: Opt0085RunIdentity;
  try {
    identity = parseIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }
  const worker = new Worker(
    new URL(
      "./opt-0085-planner-depth2-completion-epochs-worker.ts",
      import.meta.url,
    ),
    { type: "module" },
  );
  let heartbeat: ReturnType<typeof startHeartbeat> | undefined;
  let warmupCompletedAtEpochMilliseconds: number | undefined;
  let preGateReceipt: Opt0085ThermalTraceMetadata | undefined;
  let rawResult: Readonly<Record<string, unknown>> | undefined;
  let countdown: number | undefined;
  let workerSettled = false;
  let pageSettled = false;

  populateThermalInputs(new URL(location.href).searchParams);
  prepare.addEventListener("click", () => {
    prepare.disabled = true;
    heartbeat = startHeartbeat();
    document.body.dataset.status = "preparing";
    updateProgress(
      "authenticating one BF16 planner owner and warming all four paths",
    );
    worker.postMessage({ type: "initialize", identity });
  }, { once: true });

  runTimed.addEventListener("click", () => {
    try {
      if (warmupCompletedAtEpochMilliseconds === undefined) {
        throw new Error("OPT-0085 warmup is incomplete");
      }
      setBlankCompletionToNow("pre");
      const thermal = parseThermalInputs("pre");
      validateOpt0085PreGate(
        thermal,
        warmupCompletedAtEpochMilliseconds,
      );
      preGateReceipt = thermal;
      runTimed.disabled = true;
      preGate.disabled = true;
      document.body.dataset.status = "running";
      updateProgress(
        "running 16 balanced A/B pairs; keep the external thermal logger running",
      );
      worker.postMessage({ type: "run-timed", thermal });
    } catch (error) {
      finishFailure(error, heartbeat?.stop());
      worker.terminate();
      pageSettled = true;
    }
  }, { once: true });

  finalize.addEventListener("click", () => {
    try {
      if (rawResult === undefined || preGateReceipt === undefined) {
        throw new Error("OPT-0085 raw worker result is unavailable");
      }
      setBlankCompletionToNow("final");
      const throughCleanup = parseThermalInputs("final");
      const cleanupCompletedAtEpochMilliseconds = requireNestedNumber(
        rawResult,
        "lifecycle",
        "cleanupCompletedAtEpochMilliseconds",
      );
      validateOpt0085ThroughCleanupTrace(
        throughCleanup,
        preGateReceipt,
        cleanupCompletedAtEpochMilliseconds,
      );
      finalGate.disabled = true;
      finalize.disabled = true;
      pageSettled = true;
      const pageHeartbeat = heartbeat?.stop();
      const performancePassed = requireNestedBoolean(
        rawResult,
        "timingGate",
        "passed",
      );
      const completedAtEpochMilliseconds = Date.now();
      const receipt = Object.freeze({
        ...rawResult,
        status: performancePassed
          ? "passed-all-opt-0085-gates"
          : "failed-opt-0085-performance-gate",
        thermal: Object.freeze({
          preGate: preGateReceipt,
          throughCleanup,
          continuousNominalCoverageValidated: true,
          cleanupCoveredThroughEpochMilliseconds:
            cleanupCompletedAtEpochMilliseconds,
        }),
        pageHeartbeat,
        finalizedAtEpochMilliseconds: completedAtEpochMilliseconds,
        finalizedAt: new Date(completedAtEpochMilliseconds).toISOString(),
      });
      document.body.dataset.status = performancePassed ? "passed" : "failed";
      updateProgress(
        performancePassed
          ? "all correctness, topology, thermal, and performance gates passed"
          : "correctness/topology/thermal passed; performance gate did not",
      );
      const json = JSON.stringify(receipt, null, 2);
      requireElement<HTMLElement>("#result").textContent = json;
      downloadReceipt(json);
    } catch (error) {
      finishFailure(error, heartbeat?.stop());
      pageSettled = true;
    }
  }, { once: true });

  worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
    if (pageSettled || workerSettled) return;
    const message = event.data;
    if (message.type === "progress") {
      updateProgress(message.message);
      return;
    }
    if (message.type === "ready-for-thermal-gate") {
      warmupCompletedAtEpochMilliseconds =
        message.warmupCompletedAtEpochMilliseconds;
      preGate.disabled = false;
      setBlankStartToNow("pre");
      const eligibleAt = Date.now() + OPT_0085_MINIMUM_NOMINAL_MILLISECONDS;
      updateProgress("warmups passed; begin/reset the nominal trace now");
      countdown = window.setInterval(() => {
        const remaining = Math.max(0, eligibleAt - Date.now());
        if (remaining > 0) {
          updateProgress(
            `external nominal pre-gate: ${(remaining / 1_000).toFixed(1)} s remaining`,
          );
          return;
        }
        if (countdown !== undefined) clearInterval(countdown);
        runTimed.disabled = false;
        updateProgress("enter the external pre-gate trace metadata, then run");
      }, 250);
      requireElement<HTMLElement>("#preparation").textContent =
        JSON.stringify(message.preparation, null, 2);
      return;
    }
    workerSettled = true;
    if (countdown !== undefined) clearInterval(countdown);
    worker.terminate();
    if (message.type === "failed") {
      pageSettled = true;
      finishFailure(message.error, heartbeat?.stop());
      return;
    }
    rawResult = message.result;
    finalGate.disabled = false;
    finalize.disabled = false;
    const preStart = preGateReceipt?.startedAtEpochMilliseconds;
    if (preStart !== undefined) {
      thermalInput("final", "StartedAtEpochMilliseconds").value =
        String(preStart);
    }
    document.body.dataset.status = "awaiting-final-thermal";
    updateProgress(
      "GPU cleanup finished; take the logger's final nominal poll, enter the continuous through-cleanup trace, then finalize",
    );
    requireElement<HTMLElement>("#raw-result").textContent =
      JSON.stringify(rawResult, null, 2);
  });

  worker.addEventListener("error", (event) => {
    if (pageSettled || workerSettled) return;
    workerSettled = true;
    pageSettled = true;
    if (countdown !== undefined) clearInterval(countdown);
    finishFailure(event.error ?? event.message, heartbeat?.stop());
    worker.terminate();
  });
}

function parseIdentity(parameters: URLSearchParams): Opt0085RunIdentity {
  return validateOpt0085RunIdentity(Object.freeze({
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

function parseThermalInputs(
  prefix: "pre" | "final",
): Opt0085ThermalTraceMetadata {
  const startedAtEpochMilliseconds = thermalNumber(
    prefix,
    "StartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = thermalNumber(
    prefix,
    "CompletedAtEpochMilliseconds",
  );
  const source = thermalInput(prefix, "Source").value;
  const pollMilliseconds = thermalNumber(prefix, "PollMilliseconds");
  const observationCount = thermalNumber(prefix, "Observations");
  const maximumPollGapMilliseconds = thermalNumber(
    prefix,
    "MaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = thermalNumber(
    prefix,
    "NonNominalObservations",
  );
  if (
    source !== OPT_0085_THERMAL_SOURCE ||
    pollMilliseconds !== OPT_0085_THERMAL_POLL_MILLISECONDS ||
    nonNominalObservationCount !== 0
  ) throw new Error("OPT-0085 thermal input identity changed");
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds:
      completedAtEpochMilliseconds - startedAtEpochMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0085_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

function populateThermalInputs(parameters: URLSearchParams): void {
  for (const prefix of ["pre", "final"] as const) {
    for (const suffix of [
      "StartedAtEpochMilliseconds",
      "CompletedAtEpochMilliseconds",
      "Observations",
      "MaximumPollGapMilliseconds",
      "NonNominalObservations",
    ]) {
      const value = parameters.get(`${prefix}Thermal${suffix}`);
      if (value !== null) thermalInput(prefix, suffix).value = value;
    }
  }
}

function setBlankStartToNow(prefix: "pre" | "final"): void {
  const input = thermalInput(prefix, "StartedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function setBlankCompletionToNow(prefix: "pre" | "final"): void {
  const input = thermalInput(prefix, "CompletedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function thermalNumber(prefix: "pre" | "final", suffix: string): number {
  const input = thermalInput(prefix, suffix);
  const value = input.value.trim() === "" ? Number.NaN : Number(input.value);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0085 requires ${input.name}`);
  }
  return value;
}

function thermalInput(
  prefix: "pre" | "final",
  suffix: string,
): HTMLInputElement {
  return requireElement<HTMLInputElement>(
    `input[name="${prefix}Thermal${suffix}"]`,
  );
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "" || value !== value.trim()) {
    throw new Error(`OPT-0085 requires URL parameter ${name}`);
  }
  return value;
}

function positiveIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`OPT-0085 ${name} must be a positive integer`);
  }
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
    throw new Error(`OPT-0085 raw result omitted ${owner}.${field}`);
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
    throw new Error(`OPT-0085 raw result omitted ${owner}.${field}`);
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
    anchor.download = "opt-0085-planner-depth2-completion-epochs.json";
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
  if (element === null) throw new Error(`Missing OPT-0085 element ${selector}`);
  return element;
}

function updateProgress(message: string): void {
  requireElement<HTMLElement>("#progress").textContent = message;
}

function finishFailure(error: unknown, heartbeat?: HeartbeatSnapshot): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    error: error instanceof Error
      ? { name: error.name, message: error.message, stack: error.stack }
      : error,
    ...(heartbeat === undefined ? {} : { pageHeartbeat: heartbeat }),
  }, null, 2);
}
