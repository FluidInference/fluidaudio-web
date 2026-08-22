export interface Opt0034ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly protocol: "wait-30s-then-one-level0-check";
  readonly startedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: 1;
  readonly observedLevel: 0;
  readonly maximumObservationGapMilliseconds: number;
}

export function parseOpt0034ThermalGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0034ThermalGate {
  const source = parameters.get("thermalSource");
  const startedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalCheckedAtEpochMilliseconds",
  );
  const observationCount = requiredNumber(parameters, "thermalObservations");
  const observedLevel = requiredNumber(parameters, "thermalObservedLevel");
  const durationMilliseconds =
    checkedAtEpochMilliseconds - startedAtEpochMilliseconds;
  const gapText = parameters.get(
    "thermalMaximumObservationGapMilliseconds",
  );
  const maximumObservationGapMilliseconds =
    gapText === null || gapText.trim() === ""
      ? durationMilliseconds
      : Number(gapText);
  if (
    source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    !Number.isFinite(nowEpochMilliseconds) ||
    startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    checkedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    checkedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    durationMilliseconds < 30_000 ||
    observationCount !== 1 ||
    observedLevel !== 0 ||
    !Number.isFinite(maximumObservationGapMilliseconds) ||
    maximumObservationGapMilliseconds !== durationMilliseconds
  ) {
    throw new Error(
      "OPT-0034 requires one truthful level-0 notifyutil check after a 30-second wait",
    );
  }
  return Object.freeze({
    source,
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    protocol: "wait-30s-then-one-level0-check",
    startedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount: 1 as const,
    observedLevel: 0 as const,
    maximumObservationGapMilliseconds,
  });
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const text = parameters.get(name);
  const value = text === null || text.trim() === "" ? Number.NaN : Number(text);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0034 requires finite ${name}`);
  }
  return value;
}
