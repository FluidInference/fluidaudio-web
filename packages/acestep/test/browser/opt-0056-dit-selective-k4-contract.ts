import {
  compareOpt0037FinalLatents,
  type Opt0037FinalLatentComparison,
} from "./opt-0037-dit-rev7-vs-rev8-contract.js";

export const OPT_0056_REV7_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
export const OPT_0056_REV7_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0056_REV7_MANIFEST_BYTES = 254_357 as const;
export const OPT_0056_REV7_RUNTIME_PROFILE =
  "opt-0009-fp16-fp32-dense-v1" as const;
export const OPT_0056_REV7_KERNEL_SET_ID =
  "opt-0009-n256-k32-fp16-fp32-v1" as const;

export const OPT_0056_REV8_MANIFEST_PATH =
  "/model/files-fp16-dit-layer-mixed-experimental/manifest.json" as const;
export const OPT_0056_REV8_MANIFEST_SHA256 =
  "a2f70c123fb7c4dbc3b51be68b4b494107c13b575ad2bed68c639791c93574d1" as const;
export const OPT_0056_REV8_MANIFEST_BYTES = 257_789 as const;
export const OPT_0056_ALL_K4_RUNTIME_PROFILE =
  "opt-0037-k4-fp16-partials-v1" as const;
export const OPT_0056_ALL_K4_KERNEL_SET_ID =
  "opt-0037-opt-0032-k4-partials-fixed32-v1" as const;
export const OPT_0056_SELECTIVE_RUNTIME_PROFILE =
  "opt-0056-selective-k4-exact-down-v1" as const;
export const OPT_0056_SELECTIVE_KERNEL_SET_ID =
  "opt-0056-opt0032-k4-plus-exact-down-fixed32-v1" as const;

export const OPT_0056_LATENT_ELEMENTS = 288_000 as const;
export const OPT_0056_LATENT_BYTES = 1_152_000 as const;
export const OPT_0056_EVALUATIONS = 8 as const;

export interface Opt0056TrajectoryComparison {
  readonly evaluationCount: typeof OPT_0056_EVALUATIONS;
  readonly evaluations: readonly Readonly<{
    readonly evaluation: number;
    readonly comparison: Opt0037FinalLatentComparison;
  }>[];
  readonly firstDifferingEvaluation: number | null;
  readonly final: Opt0037FinalLatentComparison;
  readonly passedFinalEnvelope: boolean;
}

export function compareOpt0056Trajectory(
  control: readonly Float32Array[],
  candidate: readonly Float32Array[],
): Opt0056TrajectoryComparison {
  if (
    control.length !== OPT_0056_EVALUATIONS ||
    candidate.length !== OPT_0056_EVALUATIONS
  ) throw new Error("OPT-0056 trajectory must contain exactly eight taps");
  const evaluations = Object.freeze(control.map((latent, evaluation) =>
    Object.freeze({
      evaluation,
      comparison: compareOpt0037FinalLatents(
        latent,
        candidate[evaluation]!,
      ),
    })
  ));
  const firstDifferingEvaluation = evaluations.find(
    ({ comparison }) => comparison.differingU32Count !== 0,
  )?.evaluation ?? null;
  const final = evaluations.at(-1)!.comparison;
  return Object.freeze({
    evaluationCount: OPT_0056_EVALUATIONS,
    evaluations,
    firstDifferingEvaluation,
    final,
    passedFinalEnvelope: final.passed,
  });
}

export function exactOpt0056TrajectoryIdentity(
  left: readonly Float32Array[],
  right: readonly Float32Array[],
): boolean {
  if (left.length !== OPT_0056_EVALUATIONS || right.length !== left.length) {
    return false;
  }
  return left.every((values, evaluation) => {
    const other = right[evaluation];
    if (other === undefined || values.length !== other.length) return false;
    const a = new Uint32Array(values.buffer, values.byteOffset, values.length);
    const b = new Uint32Array(other.buffer, other.byteOffset, other.length);
    return a.every((word, index) => word === b[index]);
  });
}
