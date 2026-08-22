export const OPT_0066_REQUIRED_K7_SPEEDUP = 1.50 as const;
export const OPT_0066_REQUIRED_CONV_TRANSPOSE_SPEEDUP = 1.30 as const;

export const OPT_0066_NUMERICAL_ENVELOPE = Object.freeze({
  nrmseMaximum: 0.003,
  snrMinimumDb: 50,
  pearsonMinimum: 0.9999,
  relativeRmsDriftMaximum: 0.005,
  relativeEnergyDriftMaximum: 0.005,
  relativePeakDriftMaximum: 0.01,
  relativeDcOffsetDriftMaximum: 0.001,
  relativeMaximumAbsoluteErrorMaximum: 0.02,
} as const);

export interface Opt0066ThermalGate {
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

export interface Opt0066RawComparison {
  readonly comparedU32WordCount: number;
  readonly u32MismatchCount: number;
  readonly firstU32MismatchIndex: number | null;
  readonly comparedU16WordCount: number;
  readonly u16MismatchCount: number;
  readonly firstU16MismatchIndex: number | null;
  readonly rawU32Exact: boolean;
  readonly rawU16Exact: boolean;
}

export interface Opt0066WaveformMetrics {
  readonly count: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly pearson: number;
  readonly relativeRmsDrift: number;
  readonly relativeEnergyDrift: number;
  readonly relativePeakDrift: number;
  readonly relativeDcOffsetDrift: number;
  readonly relativeMaximumAbsoluteError: number;
  readonly maximumAbsoluteError: number;
  readonly controlRms: number;
  readonly candidateRms: number;
  readonly controlPeak: number;
  readonly candidatePeak: number;
  readonly controlMean: number;
  readonly candidateMean: number;
  readonly finite: boolean;
  readonly passed: boolean;
}

export type Opt0066TimedArm = "rev6-scalar" | "rev7-candidate";

export interface Opt0066TimingSample {
  readonly arm: Opt0066TimedArm;
  readonly k7FamilySubmitThroughDrainMs: number;
  readonly convTransposeFamilySubmitThroughDrainMs: number;
  readonly decoderSubmitThroughDrainMs: number;
  readonly outerWindowWallMs: number;
}

export function parseOpt0066ThermalGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0066ThermalGate {
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
      "OPT-0066 requires one truthful level-0 notifyutil check after a 30-second wait",
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

export function compareOpt0066Raw(
  control: Float32Array,
  candidate: Float32Array,
): Opt0066RawComparison {
  if (
    control.length !== candidate.length ||
    control.byteLength !== candidate.byteLength
  ) {
    throw new RangeError("OPT-0066 output lengths differ");
  }
  const controlU32 = new Uint32Array(
    control.buffer,
    control.byteOffset,
    control.length,
  );
  const candidateU32 = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  const controlU16 = new Uint16Array(
    control.buffer,
    control.byteOffset,
    control.byteLength / Uint16Array.BYTES_PER_ELEMENT,
  );
  const candidateU16 = new Uint16Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.byteLength / Uint16Array.BYTES_PER_ELEMENT,
  );
  let u32MismatchCount = 0;
  let firstU32MismatchIndex: number | null = null;
  for (let index = 0; index < controlU32.length; index += 1) {
    if (controlU32[index] === candidateU32[index]) continue;
    u32MismatchCount += 1;
    if (firstU32MismatchIndex === null) firstU32MismatchIndex = index;
  }
  let u16MismatchCount = 0;
  let firstU16MismatchIndex: number | null = null;
  for (let index = 0; index < controlU16.length; index += 1) {
    if (controlU16[index] === candidateU16[index]) continue;
    u16MismatchCount += 1;
    if (firstU16MismatchIndex === null) firstU16MismatchIndex = index;
  }
  return Object.freeze({
    comparedU32WordCount: controlU32.length,
    u32MismatchCount,
    firstU32MismatchIndex,
    comparedU16WordCount: controlU16.length,
    u16MismatchCount,
    firstU16MismatchIndex,
    rawU32Exact: u32MismatchCount === 0,
    rawU16Exact: u16MismatchCount === 0,
  });
}

/** OPT-0044's unchanged OPT-0024 C512 numerical envelope. */
export function compareOpt0066Waveforms(
  control: Float32Array,
  candidate: Float32Array,
  stride = 1,
  offset = 0,
): Opt0066WaveformMetrics {
  if (
    control.length !== candidate.length ||
    !Number.isSafeInteger(stride) || stride < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 || offset >= stride
  ) {
    throw new RangeError("OPT-0066 waveform geometry is invalid");
  }
  let count = 0;
  let sumControl = 0;
  let sumCandidate = 0;
  let sumControlSquared = 0;
  let sumCandidateSquared = 0;
  let sumProduct = 0;
  let sumSquaredError = 0;
  let maximumAbsoluteError = 0;
  let controlPeak = 0;
  let candidatePeak = 0;
  let finite = true;
  for (let index = offset; index < control.length; index += stride) {
    const a = control[index]!;
    const b = candidate[index]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) finite = false;
    const error = b - a;
    maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(error));
    controlPeak = Math.max(controlPeak, Math.abs(a));
    candidatePeak = Math.max(candidatePeak, Math.abs(b));
    sumControl += a;
    sumCandidate += b;
    sumControlSquared += a * a;
    sumCandidateSquared += b * b;
    sumProduct += a * b;
    sumSquaredError += error * error;
    count += 1;
  }
  if (count === 0) throw new RangeError("OPT-0066 waveform comparison is empty");
  const controlMean = sumControl / count;
  const candidateMean = sumCandidate / count;
  const meanControlSquared = sumControlSquared / count;
  const meanCandidateSquared = sumCandidateSquared / count;
  const controlRms = Math.sqrt(meanControlSquared);
  const candidateRms = Math.sqrt(meanCandidateSquared);
  const rmse = Math.sqrt(sumSquaredError / count);
  const nrmse = rmse / Math.max(controlRms, 1e-30);
  const snrDb = rmse === 0
    ? Number.POSITIVE_INFINITY
    : 20 * Math.log10(Math.max(controlRms, 1e-30) / rmse);
  const controlVariance = Math.max(
    0,
    meanControlSquared - controlMean * controlMean,
  );
  const candidateVariance = Math.max(
    0,
    meanCandidateSquared - candidateMean * candidateMean,
  );
  const covariance = sumProduct / count - controlMean * candidateMean;
  const denominator = Math.sqrt(controlVariance * candidateVariance);
  const pearson = denominator === 0
    ? (rmse === 0 ? 1 : 0)
    : covariance / denominator;
  const relativeRmsDrift = Math.abs(candidateRms - controlRms) /
    Math.max(controlRms, 1e-30);
  const relativeEnergyDrift = Math.abs(
    meanCandidateSquared - meanControlSquared,
  ) / Math.max(meanControlSquared, 1e-30);
  const relativePeakDrift = Math.abs(candidatePeak - controlPeak) /
    Math.max(controlPeak, 1e-30);
  const relativeDcOffsetDrift = Math.abs(candidateMean - controlMean) /
    Math.max(controlRms, 1e-6);
  const relativeMaximumAbsoluteError = maximumAbsoluteError /
    Math.max(controlPeak, 1e-6);
  const limits = OPT_0066_NUMERICAL_ENVELOPE;
  const passed = finite && nrmse <= limits.nrmseMaximum &&
    snrDb >= limits.snrMinimumDb && pearson >= limits.pearsonMinimum &&
    relativeRmsDrift <= limits.relativeRmsDriftMaximum &&
    relativeEnergyDrift <= limits.relativeEnergyDriftMaximum &&
    relativePeakDrift <= limits.relativePeakDriftMaximum &&
    relativeDcOffsetDrift <= limits.relativeDcOffsetDriftMaximum &&
    relativeMaximumAbsoluteError <=
      limits.relativeMaximumAbsoluteErrorMaximum;
  return Object.freeze({
    count,
    nrmse,
    snrDb,
    pearson,
    relativeRmsDrift,
    relativeEnergyDrift,
    relativePeakDrift,
    relativeDcOffsetDrift,
    relativeMaximumAbsoluteError,
    maximumAbsoluteError,
    controlRms,
    candidateRms,
    controlPeak,
    candidatePeak,
    controlMean,
    candidateMean,
    finite,
    passed,
  });
}

export function evaluateOpt0066BalancedTiming(
  samples: readonly Opt0066TimingSample[],
): Readonly<Record<string, unknown>> & { readonly passed: boolean } {
  const expected = [
    "rev6-scalar",
    "rev7-candidate",
    "rev7-candidate",
    "rev6-scalar",
  ] as const;
  if (
    samples.length !== expected.length ||
    samples.some((sample, index) =>
      sample.arm !== expected[index] ||
      [
        sample.k7FamilySubmitThroughDrainMs,
        sample.convTransposeFamilySubmitThroughDrainMs,
        sample.decoderSubmitThroughDrainMs,
        sample.outerWindowWallMs,
      ].some((value) => !Number.isFinite(value) || value <= 0)
    )
  ) {
    throw new Error("OPT-0066 timing samples are not balanced finite AB/BA");
  }
  const forward = evaluatePair(samples[0]!, samples[1]!, "AB");
  const reverse = evaluatePair(samples[3]!, samples[2]!, "BA");
  const scalar = [samples[0]!, samples[3]!];
  const candidate = [samples[1]!, samples[2]!];
  const medianK7Speedup = median(scalar.map((sample) =>
    sample.k7FamilySubmitThroughDrainMs
  )) / median(candidate.map((sample) =>
    sample.k7FamilySubmitThroughDrainMs
  ));
  const medianK7Passed = medianK7Speedup >= OPT_0066_REQUIRED_K7_SPEEDUP;
  const medianConvTransposeSpeedup = median(scalar.map((sample) =>
    sample.convTransposeFamilySubmitThroughDrainMs
  )) / median(candidate.map((sample) =>
    sample.convTransposeFamilySubmitThroughDrainMs
  ));
  const medianConvTransposePassed = medianConvTransposeSpeedup >=
    OPT_0066_REQUIRED_CONV_TRANSPOSE_SPEEDUP;
  const passed = forward.k7Improved && reverse.k7Improved &&
    forward.convTransposeImproved && reverse.convTransposeImproved &&
    forward.decoderNoRegression && reverse.decoderNoRegression &&
    forward.outerWindowNoRegression && reverse.outerWindowNoRegression &&
    medianK7Passed && medianConvTransposePassed;
  return Object.freeze({
    forward,
    reverse,
    aggregate: Object.freeze({
      medianK7Speedup,
      requiredMedianK7Speedup: OPT_0066_REQUIRED_K7_SPEEDUP,
      medianK7Passed,
      medianConvTransposeSpeedup,
      requiredMedianConvTransposeSpeedup:
        OPT_0066_REQUIRED_CONV_TRANSPOSE_SPEEDUP,
      medianConvTransposePassed,
      medianDecoderSpeedup:
        median(scalar.map((sample) => sample.decoderSubmitThroughDrainMs)) /
          median(candidate.map((sample) => sample.decoderSubmitThroughDrainMs)),
      medianOuterWindowSpeedup:
        median(scalar.map((sample) => sample.outerWindowWallMs)) /
          median(candidate.map((sample) => sample.outerWindowWallMs)),
    }),
    outerWindowWallGatingNoRegression: true,
    passed,
  });
}

function evaluatePair(
  scalar: Opt0066TimingSample,
  candidate: Opt0066TimingSample,
  order: "AB" | "BA",
): Readonly<{
  order: "AB" | "BA";
  k7Speedup: number;
  convTransposeSpeedup: number;
  decoderSpeedup: number;
  outerWindowSpeedup: number;
  k7Improved: boolean;
  convTransposeImproved: boolean;
  decoderNoRegression: boolean;
  outerWindowNoRegression: boolean;
}> {
  const k7Speedup = scalar.k7FamilySubmitThroughDrainMs /
    candidate.k7FamilySubmitThroughDrainMs;
  const convTransposeSpeedup =
    scalar.convTransposeFamilySubmitThroughDrainMs /
      candidate.convTransposeFamilySubmitThroughDrainMs;
  const decoderSpeedup = scalar.decoderSubmitThroughDrainMs /
    candidate.decoderSubmitThroughDrainMs;
  const outerWindowSpeedup = scalar.outerWindowWallMs /
    candidate.outerWindowWallMs;
  return Object.freeze({
    order,
    k7Speedup,
    convTransposeSpeedup,
    decoderSpeedup,
    outerWindowSpeedup,
    k7Improved: k7Speedup > 1,
    convTransposeImproved: convTransposeSpeedup > 1,
    decoderNoRegression: decoderSpeedup >= 1,
    outerWindowNoRegression: outerWindowSpeedup >= 1,
  });
}

function median(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError("OPT-0066 median requires finite values");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const text = parameters.get(name);
  const value = text === null || text.trim() === "" ? Number.NaN : Number(text);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0066 requires finite ${name}`);
  }
  return value;
}
