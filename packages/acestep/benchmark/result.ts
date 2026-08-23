import { parseStrictJson } from "../src/model/strict-json.js";
import { aceSha256Hex } from "../src/model/sha256.js";

export const OPTIMIZATION_RESULT_SCHEMA_VERSION = 2 as const;
export const MINIMUM_THERMAL_GATE_SECONDS = 30;
export const DEFAULT_THERMAL_POLL_MILLISECONDS = 1_000;
export const DEFAULT_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;

export const OPTIMIZATION_EVIDENCE_CONCLUSIONS = [
  "positive",
  "negative",
  "inconclusive",
] as const;

export const OPTIMIZATION_DISPOSITION_STATES = [
  "benchmark-only",
  "pending-integration",
  "integrated",
  "superseded",
  "abandoned",
] as const;

export const OPTIMIZATION_RISK_CLASSES = [
  "exact",
  "reordered-rounding",
  "approximate",
] as const;

export const THERMAL_STATES = [
  "nominal",
  "fair",
  "serious",
  "critical",
  "unknown",
] as const;

export type OptimizationEvidenceConclusion =
  (typeof OPTIMIZATION_EVIDENCE_CONCLUSIONS)[number];
export type OptimizationDispositionState =
  (typeof OPTIMIZATION_DISPOSITION_STATES)[number];
export type OptimizationRiskClass = (typeof OPTIMIZATION_RISK_CLASSES)[number];
export type ThermalState = (typeof THERMAL_STATES)[number];

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue =
  | JsonPrimitive
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };
export type JsonObject = { readonly [key: string]: JsonValue };

export interface WebGpuAdapterProvenance {
  readonly info: JsonObject;
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
}

export interface BenchmarkIdentity {
  readonly modelManifestSha256: string;
  readonly fixtureManifestSha256: string;
  readonly browserVersion: string;
  readonly osBuild: string;
  readonly machineModel: string;
  readonly gpuCores: number;
  readonly memoryBytes: number;
  readonly executionProfile: string;
  readonly productionBundleSha256?: string;
  readonly benchmarkHarnessCommit?: string;
  readonly browserCommandLine?: readonly string[];
  readonly webgpuAdapter?: WebGpuAdapterProvenance;
}

export interface BenchmarkProtocol {
  readonly thermalGateSeconds: number;
  readonly thermalPollMilliseconds: number;
  readonly thermalPollToleranceMilliseconds?: number;
  readonly samples: number;
  readonly pairedOrder?: readonly ("A" | "B")[];
  readonly memorySamplingEnabled?: boolean;
}

export interface BenchmarkMetrics {
  readonly baseline: JsonObject;
  readonly candidate: JsonObject;
  readonly delta: JsonObject;
  readonly logicalGpuPeakBytes?: number;
  readonly physicalTreePeakBytes?: number;
}

export interface BenchmarkCorrectness {
  readonly passed: boolean;
  readonly oracleManifestSha256?: string;
  readonly listeningRequired?: boolean;
  readonly listeningDecision?: string | null;
}

export interface OptimizationEvidence {
  readonly conclusion: OptimizationEvidenceConclusion;
  readonly rationale: string;
}

export interface OptimizationDisposition {
  readonly state: OptimizationDispositionState;
  readonly rationale: string;
  readonly revisitWhen: readonly string[];
}

export interface BenchmarkArtifact {
  readonly sha256: string;
  readonly location: string;
}

export interface OptimizationResult {
  readonly schemaVersion: typeof OPTIMIZATION_RESULT_SCHEMA_VERSION;
  readonly experimentId: string;
  readonly hypothesis: string;
  readonly riskClass: OptimizationRiskClass;
  readonly baselineCommit: string;
  readonly candidateCommit: string;
  readonly identity: BenchmarkIdentity;
  readonly protocol: BenchmarkProtocol;
  readonly metrics: BenchmarkMetrics;
  readonly correctness: BenchmarkCorrectness;
  readonly evidence: OptimizationEvidence;
  readonly disposition: OptimizationDisposition;
  readonly artifacts?: readonly BenchmarkArtifact[];
}

export interface BenchmarkSampleSummary extends JsonObject {
  readonly samples: readonly number[];
  readonly sampleCount: number;
  readonly min: number;
  readonly median: number;
  readonly max: number;
  readonly range: number;
}

export interface ThermalObservation {
  /** A monotonic timestamp, normally derived from `performance.now()`. */
  readonly monotonicMilliseconds: number;
  readonly state: ThermalState;
}

export type ThermalGateReason =
  | "passed"
  | "no-observations"
  | "last-state-not-nominal"
  | "poll-gap"
  | "nominal-window-too-short";

export interface ThermalGateEvaluation {
  readonly passed: boolean;
  readonly reason: ThermalGateReason;
  readonly requiredNominalMilliseconds: number;
  readonly maximumPollGapMilliseconds: number;
  readonly observationCount: number;
  readonly nonNominalObservationCount: number;
  readonly lastState: ThermalState | null;
  readonly nominalWindowStartMilliseconds: number | null;
  readonly nominalWindowEndMilliseconds: number | null;
  readonly continuousNominalMilliseconds: number;
  readonly maximumObservedPollGapMilliseconds: number | null;
}

export interface BenchmarkProtocolOptions {
  readonly samples: number;
  readonly thermalGateSeconds?: number;
  readonly thermalPollMilliseconds?: number;
  readonly thermalPollToleranceMilliseconds?: number;
  readonly pairedOrder?: readonly ("A" | "B")[];
  readonly memorySamplingEnabled?: boolean;
}

/**
 * Build the thermal portion of a reportable protocol with the repository's
 * minimum 30-second nominal pre-gate and explicit polling tolerance.
 */
export function createBenchmarkProtocol(
  options: BenchmarkProtocolOptions,
): BenchmarkProtocol {
  const protocol: BenchmarkProtocol = {
    thermalGateSeconds:
      options.thermalGateSeconds ?? MINIMUM_THERMAL_GATE_SECONDS,
    thermalPollMilliseconds:
      options.thermalPollMilliseconds ?? DEFAULT_THERMAL_POLL_MILLISECONDS,
    thermalPollToleranceMilliseconds:
      options.thermalPollToleranceMilliseconds ??
      DEFAULT_THERMAL_POLL_TOLERANCE_MILLISECONDS,
    samples: options.samples,
    ...(options.pairedOrder === undefined
      ? {}
      : { pairedOrder: [...options.pairedOrder] }),
    ...(options.memorySamplingEnabled === undefined
      ? {}
      : { memorySamplingEnabled: options.memorySamplingEnabled }),
  };
  assertBenchmarkProtocol(protocol);
  return protocol;
}

/** Compute the median and full range while retaining every original sample. */
export function summarizeBenchmarkSamples(
  samples: readonly number[],
): BenchmarkSampleSummary {
  if (samples.length === 0) {
    throw new RangeError("benchmark samples must not be empty");
  }
  for (const [index, sample] of samples.entries()) {
    assertFiniteNonNegativeNumber(sample, `samples[${index}]`);
  }

  const sorted = [...samples].sort((left, right) => left - right);
  const min = sorted[0]!;
  const max = sorted.at(-1)!;
  const middle = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 1
      ? sorted[middle]!
      : midpoint(sorted[middle - 1]!, sorted[middle]!);
  return {
    samples: [...samples],
    sampleCount: samples.length,
    min,
    median,
    max,
    range: max - min,
  };
}

/**
 * Evaluate the latest continuous nominal window. A non-nominal observation or
 * an over-tolerance polling gap resets the window, so stale nominal history
 * cannot qualify a later benchmark.
 */
export function evaluateThermalGate(
  observations: readonly ThermalObservation[],
  protocol: Pick<
    BenchmarkProtocol,
    | "thermalGateSeconds"
    | "thermalPollMilliseconds"
    | "thermalPollToleranceMilliseconds"
  >,
): ThermalGateEvaluation {
  assertThermalProtocol(protocol);
  const requiredNominalMilliseconds = protocol.thermalGateSeconds * 1_000;
  const maximumPollGapMilliseconds =
    protocol.thermalPollMilliseconds +
    (protocol.thermalPollToleranceMilliseconds ?? 0);

  let previous: ThermalObservation | undefined;
  let nominalWindowStartMilliseconds: number | null = null;
  let nonNominalObservationCount = 0;
  let maximumObservedPollGapMilliseconds: number | null = null;
  let latestResetWasPollGap = false;

  for (const [index, observation] of observations.entries()) {
    assertFiniteNonNegativeNumber(
      observation.monotonicMilliseconds,
      `observations[${index}].monotonicMilliseconds`,
    );
    assertEnum(
      observation.state,
      THERMAL_STATES,
      `observations[${index}].state`,
    );
    if (
      previous !== undefined &&
      observation.monotonicMilliseconds <= previous.monotonicMilliseconds
    ) {
      throw new RangeError(
        "thermal observation timestamps must be strictly increasing",
      );
    }

    const gap =
      previous === undefined
        ? null
        : observation.monotonicMilliseconds - previous.monotonicMilliseconds;
    if (gap !== null) {
      maximumObservedPollGapMilliseconds = Math.max(
        maximumObservedPollGapMilliseconds ?? 0,
        gap,
      );
    }

    if (observation.state !== "nominal") {
      nonNominalObservationCount += 1;
      nominalWindowStartMilliseconds = null;
      latestResetWasPollGap = false;
    } else if (
      previous === undefined ||
      previous.state !== "nominal" ||
      gap === null
    ) {
      nominalWindowStartMilliseconds = observation.monotonicMilliseconds;
      latestResetWasPollGap = false;
    } else if (gap > maximumPollGapMilliseconds) {
      nominalWindowStartMilliseconds = observation.monotonicMilliseconds;
      latestResetWasPollGap = true;
    }
    previous = observation;
  }

  const last = observations.at(-1);
  const lastIsNominal = last?.state === "nominal";
  const nominalWindowEndMilliseconds = lastIsNominal
    ? last.monotonicMilliseconds
    : null;
  const continuousNominalMilliseconds =
    nominalWindowStartMilliseconds === null ||
    nominalWindowEndMilliseconds === null
      ? 0
      : nominalWindowEndMilliseconds - nominalWindowStartMilliseconds;
  const passed =
    lastIsNominal &&
    continuousNominalMilliseconds >= requiredNominalMilliseconds;

  let reason: ThermalGateReason;
  if (passed) reason = "passed";
  else if (last === undefined) reason = "no-observations";
  else if (!lastIsNominal) reason = "last-state-not-nominal";
  else if (latestResetWasPollGap) reason = "poll-gap";
  else reason = "nominal-window-too-short";

  return {
    passed,
    reason,
    requiredNominalMilliseconds,
    maximumPollGapMilliseconds,
    observationCount: observations.length,
    nonNominalObservationCount,
    lastState: last?.state ?? null,
    nominalWindowStartMilliseconds,
    nominalWindowEndMilliseconds,
    continuousNominalMilliseconds,
    maximumObservedPollGapMilliseconds,
  };
}

/** Validate the committed JSON-schema contract plus cross-field invariants. */
export function assertOptimizationResult(
  value: unknown,
): asserts value is OptimizationResult {
  const root = requireRecord(value, "result");
  assertOnlyKeys(
    root,
    [
      "schemaVersion",
      "experimentId",
      "hypothesis",
      "riskClass",
      "baselineCommit",
      "candidateCommit",
      "identity",
      "protocol",
      "metrics",
      "correctness",
      "evidence",
      "disposition",
      "artifacts",
    ],
    "result",
  );
  assertRequiredKeys(
    root,
    [
      "schemaVersion",
      "experimentId",
      "hypothesis",
      "riskClass",
      "baselineCommit",
      "candidateCommit",
      "identity",
      "protocol",
      "metrics",
      "correctness",
      "evidence",
      "disposition",
    ],
    "result",
  );
  if (root.schemaVersion !== OPTIMIZATION_RESULT_SCHEMA_VERSION) {
    throw new TypeError("result.schemaVersion must be 2");
  }
  assertPattern(root.experimentId, /^OPT-[0-9]{4}$/, "result.experimentId");
  assertNonEmptyString(root.hypothesis, "result.hypothesis");
  assertEnum(root.riskClass, OPTIMIZATION_RISK_CLASSES, "result.riskClass");
  assertGitCommit(root.baselineCommit, "result.baselineCommit");
  assertGitCommit(root.candidateCommit, "result.candidateCommit");
  assertBenchmarkIdentity(root.identity);
  assertBenchmarkProtocol(root.protocol);
  assertMetrics(root.metrics);
  assertCorrectness(root.correctness);
  assertEvidence(root.evidence);
  assertDisposition(root.disposition);

  const disposition = root.disposition as unknown as OptimizationDisposition;
  const correctness = root.correctness as unknown as BenchmarkCorrectness;
  if (disposition.state === "integrated" && !correctness.passed) {
    throw new TypeError("an integrated result must pass its correctness gate");
  }
  if (
    disposition.state === "integrated" &&
    correctness.listeningRequired === true &&
    (correctness.listeningDecision === null ||
      correctness.listeningDecision === undefined ||
      correctness.listeningDecision.length === 0)
  ) {
    throw new TypeError(
      "an integrated listening-gated result must record a listening decision",
    );
  }

  if (root.artifacts !== undefined) {
    if (!Array.isArray(root.artifacts)) {
      throw new TypeError("result.artifacts must be an array");
    }
    for (const [index, artifact] of root.artifacts.entries()) {
      assertArtifact(artifact, `result.artifacts[${index}]`);
    }
  }
}

export function assertBenchmarkIdentity(
  value: unknown,
): asserts value is BenchmarkIdentity {
  const identity = requireRecord(value, "result.identity");
  assertRequiredKeys(
    identity,
    [
      "modelManifestSha256",
      "fixtureManifestSha256",
      "browserVersion",
      "osBuild",
      "machineModel",
      "gpuCores",
      "memoryBytes",
      "executionProfile",
    ],
    "result.identity",
  );
  assertJsonRecord(identity, "result.identity");
  assertSha256(
    identity.modelManifestSha256,
    "result.identity.modelManifestSha256",
  );
  assertSha256(
    identity.fixtureManifestSha256,
    "result.identity.fixtureManifestSha256",
  );
  assertNonEmptyString(identity.browserVersion, "result.identity.browserVersion");
  assertNonEmptyString(identity.osBuild, "result.identity.osBuild");
  assertNonEmptyString(identity.machineModel, "result.identity.machineModel");
  assertPositiveSafeInteger(identity.gpuCores, "result.identity.gpuCores");
  assertPositiveSafeInteger(identity.memoryBytes, "result.identity.memoryBytes");
  assertNonEmptyString(
    identity.executionProfile,
    "result.identity.executionProfile",
  );
  if (identity.productionBundleSha256 !== undefined) {
    assertSha256(
      identity.productionBundleSha256,
      "result.identity.productionBundleSha256",
    );
  }
  if (identity.benchmarkHarnessCommit !== undefined) {
    assertGitCommit(
      identity.benchmarkHarnessCommit,
      "result.identity.benchmarkHarnessCommit",
    );
  }
  if (identity.browserCommandLine !== undefined) {
    assertStringArray(
      identity.browserCommandLine,
      "result.identity.browserCommandLine",
    );
  }
  if (identity.webgpuAdapter !== undefined) {
    assertWebGpuAdapter(identity.webgpuAdapter);
  }
}

/**
 * Bind resume/comparison logic to the complete canonical provenance identity.
 * This intentionally includes adapter limits/features and the isolated browser
 * command line when supplied.
 */
export function benchmarkIdentitySha256(identity: BenchmarkIdentity): string {
  assertBenchmarkIdentity(identity);
  return aceSha256Hex(new TextEncoder().encode(canonicalJson(identity)));
}

export function assertBenchmarkProtocol(
  value: unknown,
): asserts value is BenchmarkProtocol {
  const protocol = requireRecord(value, "result.protocol");
  assertRequiredKeys(
    protocol,
    ["thermalGateSeconds", "thermalPollMilliseconds", "samples"],
    "result.protocol",
  );
  assertJsonRecord(protocol, "result.protocol");
  assertThermalProtocol(protocol);
  assertPositiveSafeInteger(protocol.samples, "result.protocol.samples");
  if (protocol.pairedOrder !== undefined) {
    if (!Array.isArray(protocol.pairedOrder)) {
      throw new TypeError("result.protocol.pairedOrder must be an array");
    }
    for (const [index, entry] of protocol.pairedOrder.entries()) {
      assertEnum(entry, ["A", "B"] as const, `result.protocol.pairedOrder[${index}]`);
    }
  }
  if (
    protocol.memorySamplingEnabled !== undefined &&
    typeof protocol.memorySamplingEnabled !== "boolean"
  ) {
    throw new TypeError(
      "result.protocol.memorySamplingEnabled must be a boolean",
    );
  }
}

/** Parse duplicate-key-safe JSON and validate it against the result contract. */
export function parseOptimizationResultJson(text: string): OptimizationResult {
  const value = parseStrictJson(text);
  assertOptimizationResult(value);
  return value;
}

/** Serialize a validated result as deterministic, whitespace-free JSON. */
export function stringifyOptimizationResult(
  result: OptimizationResult,
): string {
  assertOptimizationResult(result);
  return canonicalJson(result);
}

function assertThermalProtocol(
  protocol: Pick<
    BenchmarkProtocol,
    | "thermalGateSeconds"
    | "thermalPollMilliseconds"
    | "thermalPollToleranceMilliseconds"
  > | Record<string, unknown>,
): void {
  assertFiniteNumber(protocol.thermalGateSeconds, "result.protocol.thermalGateSeconds");
  if (protocol.thermalGateSeconds < MINIMUM_THERMAL_GATE_SECONDS) {
    throw new RangeError(
      `result.protocol.thermalGateSeconds must be at least ${MINIMUM_THERMAL_GATE_SECONDS}`,
    );
  }
  assertPositiveSafeInteger(
    protocol.thermalPollMilliseconds,
    "result.protocol.thermalPollMilliseconds",
  );
  if (protocol.thermalPollMilliseconds < 100) {
    throw new RangeError(
      "result.protocol.thermalPollMilliseconds must be at least 100",
    );
  }
  if (protocol.thermalPollToleranceMilliseconds !== undefined) {
    assertNonNegativeSafeInteger(
      protocol.thermalPollToleranceMilliseconds,
      "result.protocol.thermalPollToleranceMilliseconds",
    );
  }
}

function assertMetrics(value: unknown): void {
  const metrics = requireRecord(value, "result.metrics");
  assertRequiredKeys(metrics, ["baseline", "candidate", "delta"], "result.metrics");
  assertJsonRecord(metrics, "result.metrics");
  requireRecord(metrics.baseline, "result.metrics.baseline");
  requireRecord(metrics.candidate, "result.metrics.candidate");
  requireRecord(metrics.delta, "result.metrics.delta");
  for (const key of ["logicalGpuPeakBytes", "physicalTreePeakBytes"] as const) {
    if (metrics[key] !== undefined) {
      assertNonNegativeSafeInteger(metrics[key], `result.metrics.${key}`);
    }
  }
}

function assertCorrectness(value: unknown): void {
  const correctness = requireRecord(value, "result.correctness");
  assertRequiredKeys(correctness, ["passed"], "result.correctness");
  assertJsonRecord(correctness, "result.correctness");
  if (typeof correctness.passed !== "boolean") {
    throw new TypeError("result.correctness.passed must be a boolean");
  }
  if (correctness.oracleManifestSha256 !== undefined) {
    assertSha256(
      correctness.oracleManifestSha256,
      "result.correctness.oracleManifestSha256",
    );
  }
  if (
    correctness.listeningRequired !== undefined &&
    typeof correctness.listeningRequired !== "boolean"
  ) {
    throw new TypeError("result.correctness.listeningRequired must be a boolean");
  }
  if (
    correctness.listeningDecision !== undefined &&
    correctness.listeningDecision !== null &&
    typeof correctness.listeningDecision !== "string"
  ) {
    throw new TypeError(
      "result.correctness.listeningDecision must be a string or null",
    );
  }
}

function assertEvidence(value: unknown): void {
  const evidence = requireRecord(value, "result.evidence");
  assertOnlyKeys(evidence, ["conclusion", "rationale"], "result.evidence");
  assertRequiredKeys(evidence, ["conclusion", "rationale"], "result.evidence");
  assertJsonRecord(evidence, "result.evidence");
  assertEnum(
    evidence.conclusion,
    OPTIMIZATION_EVIDENCE_CONCLUSIONS,
    "result.evidence.conclusion",
  );
  assertNonEmptyString(evidence.rationale, "result.evidence.rationale");
}

function assertDisposition(value: unknown): void {
  const disposition = requireRecord(value, "result.disposition");
  assertOnlyKeys(
    disposition,
    ["state", "rationale", "revisitWhen"],
    "result.disposition",
  );
  assertRequiredKeys(
    disposition,
    ["state", "rationale", "revisitWhen"],
    "result.disposition",
  );
  assertJsonRecord(disposition, "result.disposition");
  assertEnum(
    disposition.state,
    OPTIMIZATION_DISPOSITION_STATES,
    "result.disposition.state",
  );
  assertNonEmptyString(disposition.rationale, "result.disposition.rationale");
  assertStringArray(disposition.revisitWhen, "result.disposition.revisitWhen");
  for (const [index, condition] of disposition.revisitWhen.entries()) {
    assertNonEmptyString(
      condition,
      `result.disposition.revisitWhen[${index}]`,
    );
  }
}

function assertArtifact(value: unknown, path: string): void {
  const artifact = requireRecord(value, path);
  assertOnlyKeys(artifact, ["sha256", "location"], path);
  assertRequiredKeys(artifact, ["sha256", "location"], path);
  assertSha256(artifact.sha256, `${path}.sha256`);
  assertNonEmptyString(artifact.location, `${path}.location`);
}

function assertWebGpuAdapter(value: unknown): void {
  const adapter = requireRecord(value, "result.identity.webgpuAdapter");
  assertRequiredKeys(
    adapter,
    ["info", "features", "limits"],
    "result.identity.webgpuAdapter",
  );
  assertJsonRecord(adapter, "result.identity.webgpuAdapter");
  requireRecord(adapter.info, "result.identity.webgpuAdapter.info");
  assertStringArray(adapter.features, "result.identity.webgpuAdapter.features");
  const limits = requireRecord(
    adapter.limits,
    "result.identity.webgpuAdapter.limits",
  );
  for (const [name, limit] of Object.entries(limits)) {
    assertFiniteNonNegativeNumber(
      limit,
      `result.identity.webgpuAdapter.limits.${name}`,
    );
  }
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    assertFiniteNumber(value, "JSON number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  const record = requireRecord(value, "JSON value");
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    .join(",")}}`;
}

function assertJsonRecord(value: Record<string, unknown>, path: string): void {
  for (const [key, entry] of Object.entries(value)) {
    assertJsonValue(entry, `${path}.${key}`);
  }
}

function assertJsonValue(value: unknown, path: string): void {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return;
  }
  if (typeof value === "number") {
    assertFiniteNumber(value, path);
    return;
  }
  if (Array.isArray(value)) {
    for (const [index, entry] of value.entries()) {
      assertJsonValue(entry, `${path}[${index}]`);
    }
    return;
  }
  const record = requireRecord(value, path);
  assertJsonRecord(record, path);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  const prototype = Object.getPrototypeOf(value) as object | null;
  if (prototype !== null && prototype !== Object.prototype) {
    throw new TypeError(`${path} must be a plain JSON object`);
  }
  return value as Record<string, unknown>;
}

function assertOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  path: string,
): void {
  const allowedKeys = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedKeys.has(key)) {
      throw new TypeError(`${path} contains unknown field ${JSON.stringify(key)}`);
    }
  }
}

function assertRequiredKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  path: string,
): void {
  for (const key of required) {
    if (!Object.hasOwn(value, key)) {
      throw new TypeError(`${path} is missing required field ${JSON.stringify(key)}`);
    }
  }
}

function assertPattern(value: unknown, pattern: RegExp, path: string): void {
  if (typeof value !== "string" || !pattern.test(value)) {
    throw new TypeError(`${path} has an invalid format`);
  }
}

function assertSha256(value: unknown, path: string): void {
  assertPattern(value, /^[0-9a-f]{64}$/, path);
}

function assertGitCommit(value: unknown, path: string): void {
  assertPattern(value, /^[0-9a-f]{40}$/, path);
}

function assertNonEmptyString(value: unknown, path: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${path} must be a non-empty string`);
  }
}

function assertStringArray(
  value: unknown,
  path: string,
): asserts value is string[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  for (const [index, entry] of value.entries()) {
    if (typeof entry !== "string") {
      throw new TypeError(`${path}[${index}] must be a string`);
    }
  }
}

function assertEnum<T extends string>(
  value: unknown,
  allowed: readonly T[],
  path: string,
): asserts value is T {
  if (typeof value !== "string" || !allowed.includes(value as T)) {
    throw new TypeError(`${path} must be one of ${allowed.join(", ")}`);
  }
}

function assertFiniteNumber(value: unknown, path: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${path} must be a finite number`);
  }
}

function assertFiniteNonNegativeNumber(
  value: unknown,
  path: string,
): asserts value is number {
  assertFiniteNumber(value, path);
  if (value < 0) throw new RangeError(`${path} must be non-negative`);
}

function assertPositiveSafeInteger(
  value: unknown,
  path: string,
): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new RangeError(`${path} must be a positive safe integer`);
  }
}

function assertNonNegativeSafeInteger(
  value: unknown,
  path: string,
): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new RangeError(`${path} must be a non-negative safe integer`);
  }
}

function midpoint(lower: number, upper: number): number {
  return lower + (upper - lower) / 2;
}
