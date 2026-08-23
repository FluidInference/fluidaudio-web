export const OPT_0008_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel";
export const OPT_0008_THERMAL_POLL_MILLISECONDS = 1_000;
export const OPT_0008_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0008_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
export const OPT_0008_PRODUCTION_COMMIT =
  "9dbd6e9cb85da211aa9e8224edfc08a2eef3f706";

export interface Opt0008RunIdentity {
  readonly harnessCommit: string;
  readonly productionCommit: typeof OPT_0008_PRODUCTION_COMMIT;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0008ThermalGateMetadata {
  readonly source: typeof OPT_0008_THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0008_THERMAL_POLL_MILLISECONDS;
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

interface WorkerReadyMessage {
  readonly type: "ready-for-thermal-gate";
  readonly warmupCompletedAtEpochMilliseconds: number;
  readonly preparation: unknown;
}

interface WorkerProgressMessage {
  readonly type: "progress";
  readonly message: string;
}

interface WorkerPassedMessage {
  readonly type: "passed";
  readonly result: Readonly<Record<string, unknown>>;
}

interface WorkerFailedMessage {
  readonly type: "failed";
  readonly error: unknown;
}

type WorkerMessage =
  | WorkerReadyMessage
  | WorkerProgressMessage
  | WorkerPassedMessage
  | WorkerFailedMessage;

export function parseOpt0008RunIdentity(
  parameters: URLSearchParams,
): Opt0008RunIdentity {
  const harnessCommit = requiredIdentityString(parameters, "harnessCommit");
  const productionCommit = requiredIdentityString(
    parameters,
    "productionCommit",
  );
  const machineModel = requiredIdentityString(parameters, "machineModel");
  const osVersion = requiredIdentityString(parameters, "osVersion");
  const osBuild = requiredIdentityString(parameters, "osBuild");
  const browserVersion = requiredIdentityString(parameters, "browserVersion");
  const gpuCoreCount = requiredIdentityInteger(parameters, "gpuCoreCount");
  const memoryBytes = requiredIdentityInteger(parameters, "memoryBytes");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0008 harnessCommit must be a 40-character hex commit");
  }
  if (productionCommit !== OPT_0008_PRODUCTION_COMMIT) {
    throw new Error("OPT-0008 productionCommit does not match the frozen baseline");
  }
  return Object.freeze({
    harnessCommit,
    productionCommit: OPT_0008_PRODUCTION_COMMIT,
    machineModel,
    osVersion,
    osBuild,
    browserVersion,
    gpuCoreCount,
    memoryBytes,
  });
}

export function parseOpt0008ThermalGateMetadata(
  parameters: URLSearchParams,
  warmupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0008ThermalGateMetadata {
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
  if (source !== OPT_0008_THERMAL_SOURCE) {
    throw new Error("OPT-0008 requires the accepted notifyutil thermal source");
  }
  if (
    !Number.isFinite(warmupCompletedAtEpochMilliseconds) ||
    !Number.isFinite(nowEpochMilliseconds) ||
    startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    completedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000
  ) {
    throw new Error(
      "OPT-0008 thermal gate must be a current interval beginning after warmup",
    );
  }
  const durationMilliseconds =
    completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  if (
    durationMilliseconds < OPT_0008_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(observationCount) ||
    observationCount <
      Math.floor(durationMilliseconds / OPT_0008_THERMAL_POLL_MILLISECONDS) + 1
  ) {
    throw new Error("OPT-0008 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== OPT_0008_THERMAL_POLL_MILLISECONDS) {
    throw new Error("OPT-0008 thermal polling must use 1,000 ms intervals");
  }
  if (
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds >
      OPT_0008_THERMAL_POLL_MILLISECONDS +
        OPT_0008_THERMAL_POLL_TOLERANCE_MILLISECONDS
  ) {
    throw new Error("OPT-0008 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0008 thermal gate observed non-nominal pressure");
  }
  return Object.freeze({
    source,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: OPT_0008_THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

if (typeof document !== "undefined") initializeBrowserHarness();

function initializeBrowserHarness(): void {
  const prepare = requireElement<HTMLButtonElement>("#prepare");
  const runTimed = requireElement<HTMLButtonElement>("#run-timed");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  let runIdentity: Opt0008RunIdentity;
  try {
    runIdentity = parseOpt0008RunIdentity(
      new URL(location.href).searchParams,
    );
  } catch (error) {
    prepare.disabled = true;
    finishFailure(error);
    return;
  }
  const worker = new Worker(
    new URL("./opt-0008-package-native-vae-window-profiler-worker.ts", import.meta.url),
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
    updateProgress("starting the package-native VAE worker");
    heartbeat = startHeartbeat();
    worker.postMessage({ type: "initialize", identity: runIdentity });
  }, { once: true });

  runTimed.addEventListener("click", () => {
    try {
      if (warmupCompletedAtEpochMilliseconds === undefined) {
        throw new Error("The OPT-0008 warmup has not completed");
      }
      setBlankThermalCompletionToNow();
      const thermal = parseOpt0008ThermalGateMetadata(
        thermalParameters(),
        warmupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      runTimed.disabled = true;
      thermalGate.disabled = true;
      document.body.dataset.status = "running";
      updateProgress("running one timed production VAE window");
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
      const eligibleAt = Date.now() + OPT_0008_MINIMUM_NOMINAL_MILLISECONDS;
      updateProgress(
        "warmup passed; begin/reset the external nominal thermal trace now",
      );
      enableTimer = window.setInterval(() => {
        const remaining = Math.max(0, eligibleAt - Date.now());
        if (remaining > 0) {
          updateProgress(
            `external nominal pre-gate: ${(remaining / 1_000).toFixed(1)} s remaining`,
          );
          return;
        }
        if (enableTimer !== undefined) clearInterval(enableTimer);
        enableTimer = undefined;
        runTimed.disabled = false;
        updateProgress(
          "enter the completed external thermal observation summary, then run",
        );
      }, 100);
      return;
    }
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    const heartbeatResult = heartbeat?.stop();
    worker.terminate();
    if (message.type === "failed") {
      finishFailure(message.error, heartbeatResult);
      return;
    }
    if (heartbeatResult === undefined) {
      finishFailure(new Error("OPT-0008 page heartbeat never started"));
      return;
    }
    validateHeartbeat(heartbeatResult);
    document.body.dataset.status = "passed";
    updateProgress("passed");
    requireElement<HTMLPreElement>("#result").textContent = JSON.stringify({
      ...message.result,
      mainThreadHeartbeat: heartbeatResult,
    });
  });

  worker.addEventListener("error", (event) => {
    if (settled) return;
    settled = true;
    if (enableTimer !== undefined) clearInterval(enableTimer);
    const heartbeatResult = heartbeat?.stop();
    worker.terminate();
    finishFailure(new Error(event.message), heartbeatResult);
  });
}

function startHeartbeat(): { stop(): HeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  const animationGaps: number[] = [];
  const timerGaps: number[] = [];
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let stopped = false;
  let lastAnimation = performance.now();
  let lastTimer = lastAnimation;
  let frameHandle = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    animationGaps.push(now - lastAnimation);
    lastAnimation = now;
    animationFrameCount += 1;
    frameHandle = requestAnimationFrame(frame);
  };
  frameHandle = requestAnimationFrame(frame);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    timerGaps.push(now - lastTimer);
    lastTimer = now;
    timerTickCount += 1;
  }, 10);
  return {
    stop(): HeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearInterval(timerHandle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds: Math.max(0, ...animationGaps),
        maximumTimerGapMilliseconds: Math.max(0, ...timerGaps),
      });
    },
  };
}

function validateHeartbeat(heartbeat: HeartbeatSnapshot): void {
  if (
    heartbeat.animationFrameCount + heartbeat.timerTickCount === 0 ||
    !Number.isFinite(heartbeat.maximumAnimationFrameGapMilliseconds) ||
    !Number.isFinite(heartbeat.maximumTimerGapMilliseconds) ||
    heartbeat.maximumAnimationFrameGapMilliseconds < 0 ||
    heartbeat.maximumTimerGapMilliseconds < 0
  ) throw new Error("OPT-0008 page heartbeat telemetry is invalid");
}

function thermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  const fields = document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  );
  for (const field of fields) parameters.set(field.name, field.value.trim());
  return parameters;
}

function populateThermalInputs(parameters: URLSearchParams): void {
  for (const field of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) {
    const value = parameters.get(field.name);
    if (value !== null) field.value = value;
  }
}

function setBlankThermalStartToNow(): void {
  const input = requireElement<HTMLInputElement>(
    'input[name="thermalStartedAtEpochMilliseconds"]',
  );
  if (input.value.trim() === "") input.value = String(Date.now());
}

function setBlankThermalCompletionToNow(): void {
  const input = requireElement<HTMLInputElement>(
    'input[name="thermalCompletedAtEpochMilliseconds"]',
  );
  if (input.value.trim() === "") input.value = String(Date.now());
}

function requiredFiniteNumber(
  parameters: URLSearchParams,
  name: string,
): number {
  const raw = parameters.get(name);
  if (raw === null || raw.trim() === "") {
    throw new Error(`OPT-0008 thermal metadata is missing ${name}`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0008 thermal metadata ${name} is not finite`);
  }
  return value;
}

function requiredIdentityString(
  parameters: URLSearchParams,
  name: string,
): string {
  const value = parameters.get(name)?.trim() ?? "";
  if (value === "") {
    throw new Error(`OPT-0008 run identity is missing ${name}`);
  }
  return value;
}

function requiredIdentityInteger(
  parameters: URLSearchParams,
  name: string,
): number {
  const raw = requiredIdentityString(parameters, name);
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(
      `OPT-0008 run identity ${name} must be a positive safe integer`,
    );
  }
  return value;
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function updateProgress(message: string): void {
  requireElement<HTMLElement>("#progress").textContent = message;
}

function finishFailure(error: unknown, heartbeat?: HeartbeatSnapshot): void {
  document.body.dataset.status = "failed";
  updateProgress("failed");
  requireElement<HTMLPreElement>("#result").textContent = JSON.stringify({
    schema: "ace-opt-0008-package-native-vae-window-profiler-v1",
    status: "failed",
    error: errorValue(error),
    ...(heartbeat === undefined ? {} : { mainThreadHeartbeat: heartbeat }),
  });
}

function errorValue(error: unknown): unknown {
  return error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
    : Object.freeze({ error: String(error) });
}
