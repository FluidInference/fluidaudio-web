export const OPT_0037_CONTROL_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0037_CONTROL_MANIFEST_BYTES = 254_357 as const;
export const OPT_0037_CONTROL_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
export const OPT_0037_CONTROL_RUNTIME_PROFILE =
  "opt-0009-fp16-fp32-dense-v1" as const;
export const OPT_0037_CONTROL_KERNEL_SET_ID =
  "opt-0009-n256-k32-fp16-fp32-v1" as const;
export const OPT_0037_CANDIDATE_MANIFEST_SHA256 =
  "a2f70c123fb7c4dbc3b51be68b4b494107c13b575ad2bed68c639791c93574d1" as const;
export const OPT_0037_CANDIDATE_MANIFEST_BYTES = 257_789 as const;
export const OPT_0037_CANDIDATE_MANIFEST_PATH =
  "/model/files-fp16-dit-layer-mixed-experimental/manifest.json" as const;
export const OPT_0037_CANDIDATE_RUNTIME_PROFILE =
  "opt-0037-k4-fp16-partials-v1" as const;
export const OPT_0037_CANDIDATE_KERNEL_SET_ID =
  "opt-0037-opt-0032-k4-partials-fixed32-v1" as const;
export const OPT_0037_FINAL_LATENT_ELEMENTS = 288_000 as const;
export const OPT_0037_FINAL_LATENT_BYTES = 1_152_000 as const;
export const OPT_0037_FINAL_NRMSE_MAXIMUM = 0.02 as const;
export const OPT_0037_FINAL_SNR_DECIBELS_MINIMUM = 34 as const;
export const OPT_0037_FINAL_PEARSON_MINIMUM = 0.999 as const;
export const OPT_0037_FINAL_MAXIMUM_ABSOLUTE_ERROR = 0.25 as const;

export interface Opt0037FinalLatentComparison {
  readonly count: typeof OPT_0037_FINAL_LATENT_ELEMENTS;
  readonly finitePairCount: number;
  readonly differingU32Count: number;
  readonly signedZeroDifferenceCount: number;
  readonly classChangeCount: number;
  readonly classChanges: Readonly<Record<string, number>>;
  readonly controlNonFiniteCount: number;
  readonly candidateNonFiniteCount: number;
  readonly signedMeanError: number;
  readonly meanAbsoluteError: number;
  readonly rmsError: number;
  readonly relativeRmsError: number;
  readonly nrmse: number;
  readonly snrDecibels: number | "positive-infinity";
  readonly pearsonCorrelation: number;
  readonly maximumAbsoluteControl: number;
  readonly maximumAbsoluteCandidate: number;
  readonly maximumAbsoluteError: number;
  readonly relativeMaximumAbsoluteError: number;
  readonly maximumRelativeError: number;
  readonly firstDifference: Readonly<Record<string, number>> | null;
  readonly worstDifference: Readonly<Record<string, number>> | null;
  readonly thresholds: Readonly<Record<string, number>>;
  readonly passed: boolean;
}

export function compareOpt0037FinalLatents(
  control: Float32Array,
  candidate: Float32Array,
): Opt0037FinalLatentComparison {
  if (
    control.length !== OPT_0037_FINAL_LATENT_ELEMENTS ||
    candidate.length !== OPT_0037_FINAL_LATENT_ELEMENTS ||
    control.byteLength !== OPT_0037_FINAL_LATENT_BYTES ||
    candidate.byteLength !== OPT_0037_FINAL_LATENT_BYTES
  ) throw new Error("OPT-0037 final-latent comparison geometry changed");
  const controlBits = new Uint32Array(
    control.buffer,
    control.byteOffset,
    control.length,
  );
  const candidateBits = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  const classChanges: Record<string, number> = {};
  let differingU32Count = 0;
  let signedZeroDifferenceCount = 0;
  let classChangeCount = 0;
  let controlNonFiniteCount = 0;
  let candidateNonFiniteCount = 0;
  let errorSum = 0;
  let absoluteErrorSum = 0;
  let errorSquareSum = 0;
  let relativeErrorSquareSum = 0;
  let controlSquareSum = 0;
  let maximumAbsoluteControl = 0;
  let maximumAbsoluteCandidate = 0;
  let maximumAbsoluteError = -1;
  let maximumRelativeError = 0;
  let firstDifference: Readonly<Record<string, number>> | null = null;
  let worstDifference: Readonly<Record<string, number>> | null = null;
  let meanControl = 0;
  let meanCandidate = 0;
  let controlM2 = 0;
  let candidateM2 = 0;
  let covariance = 0;
  let finitePairCount = 0;
  for (let index = 0; index < control.length; index += 1) {
    const a = control[index]!;
    const b = candidate[index]!;
    const aFinite = Number.isFinite(a);
    const bFinite = Number.isFinite(b);
    if (!aFinite) controlNonFiniteCount += 1;
    if (!bFinite) candidateNonFiniteCount += 1;
    const aClass = f32Class(controlBits[index]!);
    const bClass = f32Class(candidateBits[index]!);
    if (aClass !== bClass) {
      classChangeCount += 1;
      const transition = `${aClass}->${bClass}`;
      classChanges[transition] = (classChanges[transition] ?? 0) + 1;
    }
    if (
      (controlBits[index] === 0 || controlBits[index] === 0x8000_0000) &&
      (candidateBits[index] === 0 ||
        candidateBits[index] === 0x8000_0000) &&
      controlBits[index] !== candidateBits[index]
    ) signedZeroDifferenceCount += 1;
    if (controlBits[index] !== candidateBits[index]) {
      differingU32Count += 1;
      firstDifference ??= Object.freeze({
        index,
        control: a,
        candidate: b,
        controlU32: controlBits[index]!,
        candidateU32: candidateBits[index]!,
      });
    }
    if (!aFinite || !bFinite) continue;
    finitePairCount += 1;
    const error = b - a;
    const absoluteError = Math.abs(error);
    const relativeError = absoluteError / Math.max(Math.abs(a), 1e-6);
    errorSum += error;
    absoluteErrorSum += absoluteError;
    errorSquareSum += error * error;
    relativeErrorSquareSum += relativeError * relativeError;
    controlSquareSum += a * a;
    maximumAbsoluteControl = Math.max(maximumAbsoluteControl, Math.abs(a));
    maximumAbsoluteCandidate = Math.max(maximumAbsoluteCandidate, Math.abs(b));
    maximumRelativeError = Math.max(maximumRelativeError, relativeError);
    if (absoluteError > maximumAbsoluteError) {
      maximumAbsoluteError = absoluteError;
      worstDifference = Object.freeze({
        index,
        control: a,
        candidate: b,
        error,
        absoluteError,
        relativeError,
      });
    }
    const sampleCount = finitePairCount;
    const controlDelta = a - meanControl;
    meanControl += controlDelta / sampleCount;
    const candidateDelta = b - meanCandidate;
    meanCandidate += candidateDelta / sampleCount;
    controlM2 += controlDelta * (a - meanControl);
    candidateM2 += candidateDelta * (b - meanCandidate);
    covariance += controlDelta * (b - meanCandidate);
  }
  const count = OPT_0037_FINAL_LATENT_ELEMENTS;
  const comparedCount = Math.max(finitePairCount, 1);
  const rmsError = Math.sqrt(errorSquareSum / comparedCount);
  const relativeRmsError = Math.sqrt(
    relativeErrorSquareSum / comparedCount,
  );
  const controlRms = Math.sqrt(controlSquareSum / comparedCount);
  const nrmse = rmsError / Math.max(controlRms, 1e-12);
  const snrDecibels: number | "positive-infinity" = rmsError === 0
    ? "positive-infinity"
    : 20 * Math.log10(controlRms / rmsError);
  const pearsonCorrelation = controlM2 === 0 || candidateM2 === 0
    ? differingU32Count === 0 ? 1 : 0
    : covariance / Math.sqrt(controlM2 * candidateM2);
  maximumAbsoluteError = Math.max(maximumAbsoluteError, 0);
  const relativeMaximumAbsoluteError = maximumAbsoluteError /
    Math.max(maximumAbsoluteControl, 1e-6);
  const passed = controlNonFiniteCount === 0 && candidateNonFiniteCount === 0 &&
    nrmse <= OPT_0037_FINAL_NRMSE_MAXIMUM &&
    (snrDecibels === "positive-infinity" ||
      snrDecibels >= OPT_0037_FINAL_SNR_DECIBELS_MINIMUM) &&
    pearsonCorrelation >= OPT_0037_FINAL_PEARSON_MINIMUM &&
    maximumAbsoluteError <= OPT_0037_FINAL_MAXIMUM_ABSOLUTE_ERROR;
  return Object.freeze({
    count,
    finitePairCount,
    differingU32Count,
    signedZeroDifferenceCount,
    classChangeCount,
    classChanges: Object.freeze(classChanges),
    controlNonFiniteCount,
    candidateNonFiniteCount,
    signedMeanError: errorSum / comparedCount,
    meanAbsoluteError: absoluteErrorSum / comparedCount,
    rmsError,
    relativeRmsError,
    nrmse,
    snrDecibels,
    pearsonCorrelation,
    maximumAbsoluteControl,
    maximumAbsoluteCandidate,
    maximumAbsoluteError,
    relativeMaximumAbsoluteError,
    maximumRelativeError,
    firstDifference,
    worstDifference,
    thresholds: Object.freeze({
      nrmseMaximum: OPT_0037_FINAL_NRMSE_MAXIMUM,
      snrDecibelsMinimum: OPT_0037_FINAL_SNR_DECIBELS_MINIMUM,
      pearsonMinimum: OPT_0037_FINAL_PEARSON_MINIMUM,
      maximumAbsoluteErrorMaximum:
        OPT_0037_FINAL_MAXIMUM_ABSOLUTE_ERROR,
    }),
    passed,
  });
}

function f32Class(bits: number): string {
  const sign = (bits >>> 31) === 0 ? "positive" : "negative";
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7f_ffff;
  if (exponent === 0xff) return fraction === 0 ? `${sign}-infinity` : "nan";
  if (exponent === 0 && fraction === 0) return `${sign}-zero`;
  if (exponent === 0) return `${sign}-subnormal`;
  return `${sign}-finite`;
}
