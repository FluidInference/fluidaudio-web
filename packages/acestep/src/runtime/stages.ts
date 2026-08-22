export const ACE_INITIALIZATION_STAGES = [
  "webgpu",
  "storage",
  "manifest",
  "weights",
  "tokenizers",
  "wasm",
  "ready",
] as const;

export type AceInitializationStage =
  (typeof ACE_INITIALIZATION_STAGES)[number];

export const ACE_GENERATION_STAGES = [
  "prepare",
  "semantic-planner",
  "text-encoder",
  "semantic-detokenizer",
  "condition-encoder",
  "release-conditioning",
  "dit-load",
  "dit-denoise",
  "release-dit",
  "vae-load",
  "vae-decode",
  "wav-encode",
  "cleanup",
  "done",
] as const;

export type AceGenerationStage = (typeof ACE_GENERATION_STAGES)[number];

export type AceProgressUnit =
  | "bytes"
  | "items"
  | "tokens"
  | "planner-steps"
  | "transformer-blocks"
  | "denoising-evaluations"
  | "vae-chunks"
  | "audio-frames";

export interface AceProgressBase<Stage extends string> {
  readonly stage: Stage;
  /** Completed work in this stage; never estimated beyond `totalUnits`. */
  readonly completedUnits: number;
  /** Fixed once a stage begins. Zero represents discovery with no known total. */
  readonly totalUnits: number;
  readonly unit: AceProgressUnit;
  /** Monotonic wall-clock progress across the operation. */
  readonly overallFraction: number;
  readonly elapsedMs: number;
  readonly message?: string;
}

export type AceInitializationProgress =
  AceProgressBase<AceInitializationStage>;

export interface AceGenerationProgress
  extends AceProgressBase<AceGenerationStage> {
  /** Index in the request-specific plan returned by `generationStagePlan`. */
  readonly stageIndex: number;
  readonly stageCount: number;
}

export interface AceStageTiming {
  readonly stage: AceGenerationStage;
  readonly wallMs: number;
  /** Submit-to-drain wall time, not timestamp-query GPU-active time. */
  readonly submittedGpuMs?: number;
  readonly cooperativeIdleMs?: number;
}

const PLANNER_DISABLED_STAGES = ACE_GENERATION_STAGES.filter(
  (stage) => stage !== "semantic-planner",
);

/** Stable ordered phase plan used by the worker and every progress consumer. */
export function generationStagePlan(
  plannerEnabled: boolean,
): readonly AceGenerationStage[] {
  return plannerEnabled ? ACE_GENERATION_STAGES : PLANNER_DISABLED_STAGES;
}

export function isAceInitializationStage(
  value: unknown,
): value is AceInitializationStage {
  return (
    typeof value === "string" &&
    (ACE_INITIALIZATION_STAGES as readonly string[]).includes(value)
  );
}

export function isAceGenerationStage(
  value: unknown,
): value is AceGenerationStage {
  return (
    typeof value === "string" &&
    (ACE_GENERATION_STAGES as readonly string[]).includes(value)
  );
}

export function isAceProgressUnit(value: unknown): value is AceProgressUnit {
  return (
    value === "bytes" ||
    value === "items" ||
    value === "tokens" ||
    value === "planner-steps" ||
    value === "transformer-blocks" ||
    value === "denoising-evaluations" ||
    value === "vae-chunks" ||
    value === "audio-frames"
  );
}

/**
 * Reject malformed or regressing generation progress at the worker boundary.
 * This makes UI code safe to treat progress as monotonic state.
 */
export class AceProgressSequence {
  private fraction = 0;
  private stageIndex = -1;
  private stageCompletedUnits = 0;
  private stageTotalUnits: number | undefined;
  private stageUnit: AceProgressUnit | undefined;
  private elapsedMs = 0;

  accept(progress: AceGenerationProgress): void {
    assertProgressNumbers(
      progress.completedUnits,
      progress.totalUnits,
      progress.overallFraction,
      progress.elapsedMs,
    );
    if (!isAceGenerationStage(progress.stage)) {
      throw new TypeError(`Unknown generation stage ${String(progress.stage)}`);
    }
    if (!isAceProgressUnit(progress.unit)) {
      throw new TypeError(`Unknown progress unit ${String(progress.unit)}`);
    }
    if (progress.stageIndex >= progress.stageCount) {
      throw new RangeError("Progress stageIndex must be below stageCount");
    }
    if (progress.stageIndex < this.stageIndex) {
      throw new RangeError("Generation progress cannot return to an earlier stage");
    }
    if (progress.elapsedMs < this.elapsedMs) {
      throw new RangeError("Generation progress elapsedMs cannot decrease");
    }
    if (progress.overallFraction < this.fraction) {
      throw new RangeError("Generation progress cannot decrease");
    }
    if (progress.stageIndex === this.stageIndex) {
      if (progress.completedUnits < this.stageCompletedUnits) {
        throw new RangeError("Generation stage completedUnits cannot decrease");
      }
      if (
        this.stageTotalUnits !== undefined &&
        this.stageTotalUnits !== 0 &&
        progress.totalUnits !== this.stageTotalUnits
      ) {
        throw new RangeError("Generation stage totalUnits cannot change");
      }
      if (
        this.stageTotalUnits === 0 &&
        progress.totalUnits === 0 &&
        progress.completedUnits !== 0
      ) {
        throw new RangeError(
          "Generation discovery progress cannot complete unknown work",
        );
      }
      if (this.stageUnit !== undefined && progress.unit !== this.stageUnit) {
        throw new RangeError("Generation stage unit cannot change");
      }
    } else {
      this.stageCompletedUnits = 0;
      this.stageTotalUnits = undefined;
      this.stageUnit = undefined;
    }
    if (progress.stage === "done" && progress.overallFraction !== 1) {
      throw new RangeError("The done stage must report overallFraction 1");
    }
    this.stageIndex = progress.stageIndex;
    this.stageCompletedUnits = progress.completedUnits;
    this.stageTotalUnits = progress.totalUnits;
    this.stageUnit = progress.unit;
    this.fraction = progress.overallFraction;
    this.elapsedMs = progress.elapsedMs;
  }
}

/** Validate monotonic initialization events from the backend boundary. */
export class AceInitializationProgressSequence {
  private fraction = 0;
  private stageIndex = -1;
  private stageCompletedUnits = 0;
  private stageTotalUnits: number | undefined;
  private stageUnit: AceProgressUnit | undefined;
  private elapsedMs = 0;

  accept(progress: AceInitializationProgress): void {
    if (!isAceInitializationProgress(progress)) {
      throw new TypeError("Initialization progress is malformed");
    }
    const stageIndex = ACE_INITIALIZATION_STAGES.indexOf(progress.stage);
    if (stageIndex < this.stageIndex) {
      throw new RangeError(
        "Initialization progress cannot return to an earlier stage",
      );
    }
    if (progress.elapsedMs < this.elapsedMs) {
      throw new RangeError("Initialization progress elapsedMs cannot decrease");
    }
    if (progress.overallFraction < this.fraction) {
      throw new RangeError("Initialization progress cannot decrease");
    }
    if (stageIndex === this.stageIndex) {
      if (progress.completedUnits < this.stageCompletedUnits) {
        throw new RangeError(
          "Initialization stage completedUnits cannot decrease",
        );
      }
      if (
        this.stageTotalUnits !== undefined &&
        this.stageTotalUnits !== 0 &&
        progress.totalUnits !== this.stageTotalUnits
      ) {
        throw new RangeError("Initialization stage totalUnits cannot change");
      }
      if (
        this.stageTotalUnits === 0 &&
        progress.totalUnits === 0 &&
        progress.completedUnits !== 0
      ) {
        throw new RangeError(
          "Initialization discovery progress cannot complete unknown work",
        );
      }
      if (this.stageUnit !== undefined && progress.unit !== this.stageUnit) {
        throw new RangeError("Initialization stage unit cannot change");
      }
    } else {
      this.stageCompletedUnits = 0;
      this.stageTotalUnits = undefined;
      this.stageUnit = undefined;
    }
    if (progress.stage === "ready" && progress.overallFraction !== 1) {
      throw new RangeError("The ready stage must report overallFraction 1");
    }
    this.stageIndex = stageIndex;
    this.stageCompletedUnits = progress.completedUnits;
    this.stageTotalUnits = progress.totalUnits;
    this.stageUnit = progress.unit;
    this.fraction = progress.overallFraction;
    this.elapsedMs = progress.elapsedMs;
  }
}

export function isAceGenerationProgress(
  value: unknown,
): value is AceGenerationProgress {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      "stage",
      "completedUnits",
      "totalUnits",
      "unit",
      "overallFraction",
      "elapsedMs",
      "message",
      "stageIndex",
      "stageCount",
    ])
  ) {
    return false;
  }
  try {
    assertProgressNumbers(
      value.completedUnits,
      value.totalUnits,
      value.overallFraction,
      value.elapsedMs,
    );
  } catch {
    return false;
  }
  return (
    isAceGenerationStage(value.stage) &&
    isAceProgressUnit(value.unit) &&
    isNonNegativeInteger(value.stageIndex) &&
    isNonNegativeInteger(value.stageCount) &&
    value.stageCount > 0 &&
    value.stageIndex < value.stageCount &&
    (value.message === undefined || typeof value.message === "string")
  );
}

export function isAceInitializationProgress(
  value: unknown,
): value is AceInitializationProgress {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      "stage",
      "completedUnits",
      "totalUnits",
      "unit",
      "overallFraction",
      "elapsedMs",
      "message",
    ])
  ) {
    return false;
  }
  try {
    assertProgressNumbers(
      value.completedUnits,
      value.totalUnits,
      value.overallFraction,
      value.elapsedMs,
    );
  } catch {
    return false;
  }
  return (
    isAceInitializationStage(value.stage) &&
    isAceProgressUnit(value.unit) &&
    (value.message === undefined || typeof value.message === "string")
  );
}

function assertProgressNumbers(
  completedUnits: unknown,
  totalUnits: unknown,
  overallFraction: unknown,
  elapsedMs: unknown,
): void {
  if (
    !isNonNegativeFinite(completedUnits) ||
    !isNonNegativeFinite(totalUnits) ||
    completedUnits > totalUnits ||
    !isUnitFraction(overallFraction) ||
    !isNonNegativeFinite(elapsedMs)
  ) {
    throw new RangeError("Progress contains invalid numeric fields");
  }
}

function isUnitFraction(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isNonNegativeFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
): boolean {
  const keySet = new Set(allowed);
  return Object.keys(value).every((key) => keySet.has(key));
}
