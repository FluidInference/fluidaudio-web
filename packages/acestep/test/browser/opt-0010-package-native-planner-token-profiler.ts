export const OPT_0010_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel";
export const OPT_0010_THERMAL_POLL_MILLISECONDS = 1_000;
export const OPT_0010_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0010_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
export const OPT_0010_PRODUCTION_COMMIT =
  "00dfd4732aa019bbbb238ae40265fe86cb38f27b";

export interface Opt0010RunIdentity {
  readonly harnessCommit: string;
  readonly productionCommit: typeof OPT_0010_PRODUCTION_COMMIT;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0010ThermalGateMetadata {
  readonly source: typeof OPT_0010_THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0010_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

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
      readonly type: "passed";
      readonly result: Readonly<Record<string, unknown>>;
    }>
  | Readonly<{ readonly type: "failed"; readonly error: unknown }>;

export function parseOpt0010RunIdentity(
  parameters: URLSearchParams,
): Opt0010RunIdentity {
  const harnessCommit = requiredIdentityString(parameters, "harnessCommit");
  const productionCommit = requiredIdentityString(
    parameters,
    "productionCommit",
  );
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0010 harnessCommit must be a 40-character hex commit");
  }
  if (productionCommit !== OPT_0010_PRODUCTION_COMMIT) {
    throw new Error("OPT-0010 productionCommit does not match the frozen core");
  }
  return Object.freeze({
    harnessCommit,
    productionCommit: OPT_0010_PRODUCTION_COMMIT,
    machineModel: requiredIdentityString(parameters, "machineModel"),
    osVersion: requiredIdentityString(parameters, "osVersion"),
    osBuild: requiredIdentityString(parameters, "osBuild"),
    browserVersion: requiredIdentityString(parameters, "browserVersion"),
    gpuCoreCount: requiredIdentityInteger(parameters, "gpuCoreCount"),
    memoryBytes: requiredIdentityInteger(parameters, "memoryBytes"),
  });
}

export function parseOpt0010ThermalGateMetadata(
  parameters: URLSearchParams,
  warmupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0010ThermalGateMetadata {
  const source = parameters.get("thermalSource");
  const startedAtEpochMilliseconds = requiredFiniteNumber(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = requiredFiniteNumber(
    parameters,
    "thermalCompletedAtEpochMilliseconds",
  );
  const observationCount = requiredFiniteNumber(
    parameters,
    "thermalObservations",
  );
  const pollMilliseconds = requiredFiniteNumber(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredFiniteNumber(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredFiniteNumber(
    parameters,
    "thermalNonNominalObservations",
  );
  if (source !== OPT_0010_THERMAL_SOURCE) {
    throw new Error("OPT-0010 requires the accepted notifyutil thermal source");
  }
  if (
    startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error(
      "OPT-0010 thermal gate must be a current interval beginning after warmup",
    );
  }
  const durationMilliseconds =
    completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  if (
    durationMilliseconds < OPT_0010_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(observationCount) ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0010_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0010 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== OPT_0010_THERMAL_POLL_MILLISECONDS) {
    throw new Error("OPT-0010 thermal polling must use 1,000 ms intervals");
  }
  if (
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds >
      OPT_0010_THERMAL_POLL_MILLISECONDS +
        OPT_0010_THERMAL_POLL_TOLERANCE_MILLISECONDS
  ) {
    throw new Error("OPT-0010 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0010 thermal gate observed non-nominal pressure");
  }
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0010_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const prepare = requireElement<HTMLButtonElement>("#prepare");
  const runTimed = requireElement<HTMLButtonElement>("#run-timed");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let runIdentity: Opt0010RunIdentity;
  try {
    runIdentity = parseOpt0010RunIdentity(new URL(location.href).searchParams);
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }
  const worker = new Worker(
    new URL(
      "./opt-0010-package-native-planner-token-profiler-worker.ts",
      import.meta.url,
    ),
    { type: "module" },
  );
  let heartbeat: ReturnType<typeof startHeartbeat> | undefined;
  let warmupCompletedAtEpochMilliseconds: number | undefined;
  let enableTimer: number | undefined;
  let settled = false;

  populateThermalInputs(new URL(location.href).searchParams);
  prepare.addEventListener("click", () => {
    prepare.disabled = true;
    document.body.dataset.status = "preparing";
    updateProgress("authenticating and warming six package-native planner cases");
    heartbeat = startHeartbeat();
    worker.postMessage({ type: "initialize", identity: runIdentity });
  }, { once: true });

  runTimed.addEventListener("click", () => {
    try {
      if (warmupCompletedAtEpochMilliseconds === undefined) {
        throw new Error("The OPT-0010 warmup has not completed");
      }
      setBlankThermalCompletionToNow();
      const thermal = parseOpt0010ThermalGateMetadata(
        thermalParameters(),
        warmupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      runTimed.disabled = true;
      thermalGate.disabled = true;
      document.body.dataset.status = "running";
      updateProgress("profiling six unchanged planner token steps");
      worker.postMessage({ type: "run-timed", thermal });
    } catch (error) {
      finishFailure(error, heartbeat?.stop());
      worker.terminate();
      settled = true;
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
      warmupCompletedAtEpochMilliseconds =
        message.warmupCompletedAtEpochMilliseconds;
      thermalGate.disabled = false;
      setBlankThermalStartToNow();
      const eligibleAt = Date.now() + OPT_0010_MINIMUM_NOMINAL_MILLISECONDS;
      updateProgress("warmups passed; begin/reset the external nominal trace now");
      enableTimer = window.setInterval(() => {
        const remaining = Math.max(0, eligibleAt - Date.now());
        if (remaining > 0) {
          updateProgress(
            `external nominal pre-gate: ${(remaining / 1_000).toFixed(1)} s remaining`,
          );
          return;
        }
        if (enableTimer !== undefined) clearInterval(enableTimer);
        runTimed.disabled = false;
        updateProgress("enter the external trace metadata, then run");
      }, 250);
      requireElement<HTMLElement>("#preparation").textContent =
        JSON.stringify(message.preparation, null, 2);
      return;
    }
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    const heartbeatSnapshot = heartbeat?.stop();
    worker.terminate();
    if (message.type === "passed") {
      document.body.dataset.status = "passed";
      updateProgress("passed; keep the external logger running through its final poll");
      requireElement<HTMLElement>("#result").textContent = JSON.stringify({
        ...message.result,
        pageHeartbeat: heartbeatSnapshot,
      }, null, 2);
    } else {
      finishFailure(message.error, heartbeatSnapshot);
    }
  });

  worker.addEventListener("error", (event) => {
    if (settled) return;
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    finishFailure(event.error ?? event.message, heartbeat?.stop());
    worker.terminate();
  });
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

function populateThermalInputs(parameters: URLSearchParams): void {
  for (const name of [
    "thermalStartedAtEpochMilliseconds",
    "thermalCompletedAtEpochMilliseconds",
    "thermalObservations",
    "thermalMaximumPollGapMilliseconds",
    "thermalNonNominalObservations",
  ]) {
    const value = parameters.get(name);
    if (value !== null) thermalInput(name).value = value;
  }
}

function setBlankThermalStartToNow(): void {
  const input = thermalInput("thermalStartedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function setBlankThermalCompletionToNow(): void {
  const input = thermalInput("thermalCompletedAtEpochMilliseconds");
  if (input.value.trim() === "") input.value = String(Date.now());
}

function thermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function thermalInput(name: string): HTMLInputElement {
  return requireElement<HTMLInputElement>(`input[name=\"${name}\"]`);
}

function requiredIdentityString(
  parameters: URLSearchParams,
  name: string,
): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "" || value !== value.trim()) {
    throw new Error(`OPT-0010 requires run identity ${name}`);
  }
  return value;
}

function requiredIdentityInteger(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredIdentityString(parameters, name));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`OPT-0010 run identity ${name} must be positive`);
  }
  return value;
}

function requiredFiniteNumber(
  parameters: URLSearchParams,
  name: string,
): number {
  const raw = parameters.get(name);
  const value = raw === null || raw.trim() === "" ? Number.NaN : Number(raw);
  if (!Number.isFinite(value)) throw new Error(`OPT-0010 requires ${name}`);
  return value;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0010 element ${selector}`);
  return element;
}

function updateProgress(message: string): void {
  requireElement<HTMLElement>("#progress").textContent = message;
}

function finishFailure(
  error: unknown,
  heartbeat?: HeartbeatSnapshot,
): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  requireElement<HTMLElement>("#result").textContent = JSON.stringify({
    error: error instanceof Error
      ? { name: error.name, message: error.message, stack: error.stack }
      : error,
    ...(heartbeat === undefined ? {} : { pageHeartbeat: heartbeat }),
  }, null, 2);
}
