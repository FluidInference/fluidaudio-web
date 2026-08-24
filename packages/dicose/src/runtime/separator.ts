import { loadGpuWeightPackage, type LoadModelProgress } from "../model/package.js";
import { requestDiCoSeDevice } from "../webgpu/capabilities.js";
import {
  DICOSE_SAMPLE_RATE,
  DICOSE_STFT_HOP_LENGTH,
  DICOSE_STFT_N_FFT,
  SeededGaussian,
  centeredHannIstft,
  centeredHannStft,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  resampleStereoSinc,
  resampleStereoSincToLength,
  type CenteredHannStft,
  type GaussianSeed,
  type StereoPcm,
} from "./audio.js";
import {
  DiCoSeBsrRoFormer,
  type CdTrace,
  type DeterministicTrace,
  type DiCoSeMappingContexts,
} from "./bs-roformer.js";
import {
  addDiCoSeCoordinateNoise,
  DICOSE_FAST_CHUNK_GEOMETRY,
  DICOSE_FULL_CHUNK_GEOMETRY,
  DICOSE_SINGLE_PASS_SAMPLES,
  makeDiCoSeChunkPlan,
  makeDiCoSeChunkWindow,
  materializeDiCoSeChunk,
  normalizeDiCoSeOverlapAdd,
  overlapAddDiCoSeChunk,
  type StereoPcmAccumulator,
} from "./chunking.js";

export const DICOSE_STEMS = ["drums", "bass", "other", "vocals"] as const;
export type DiCoSeStem = typeof DICOSE_STEMS[number];

const SIGMA_MAX = 0.003_934;
const SIGMA_MIN = 0.000_1;
const SIGMA_DATA = 0.06;
const DEFAULT_NOISE_SEED = 0xd1c05e;
const SPECTRAL_COMPONENTS = 4;

export interface SeparatorProgress {
  readonly phase: "device" | "weights" | "chunk" | "stft" | "deterministic" | "mapping" | "refinement" | "istft";
  readonly completed: number;
  readonly total: number;
  readonly detail?: string;
}

export interface DiCoSeSeparatorOptions {
  readonly manifestUrl?: string | URL;
  /** Test seam for exact-q64 versus blockwise-Flash quality comparisons. */
  readonly attentionKernel?: "q64" | "flash";
  readonly onProgress?: (progress: SeparatorProgress) => void;
}

export interface SeparatePcmOptions {
  /** Fixed by default so identical input produces identical CD noise. */
  readonly seed?: GaussianSeed;
  /** Return the released deterministic separator directly, or apply one-step CD refinement. */
  readonly outputMode?: "refined" | "deterministic";
  /** Reference-audit seam; omitted in normal inference, so it has zero runtime cost. */
  readonly traceSamples?: number;
}

export interface SeparatorTiming {
  readonly prepareMs: number;
  readonly deterministicMs: number;
  readonly mappingMs: number;
  readonly refinementMs: number;
  readonly istftMs: number;
  readonly totalMs: number;
}

export interface StemSignalStats {
  readonly peak: number;
  readonly rms: number;
}

export interface SeparatorResult {
  readonly outputMode: "refined" | "deterministic";
  readonly stems: Readonly<Record<DiCoSeStem, StereoPcm>>;
  readonly timing: SeparatorTiming;
  readonly trace?: DeterministicTrace;
  readonly cdTrace?: CdTrace;
  /** Compact execution checkpoints for unattended correctness monitoring. */
  readonly diagnostics: Readonly<{
    readonly deterministic: Readonly<Record<DiCoSeStem, StemSignalStats>>;
    readonly cdModelOutput?: Readonly<Record<DiCoSeStem, StemSignalStats>>;
  }>;
}

/**
 * Raw-WebGPU DiCoSe inference owner. The BS-RoFormer execution, including
 * all dense layers, attention, FiLM, masks, and feature adapters, stays on
 * the GPU. The CPU only performs the published centered STFT/ISTFT boundary
 * and the one-step CM waveform affine.
 */
export class DiCoSeSeparator {
  private disposed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly model: DiCoSeBsrRoFormer,
    private readonly weights: Awaited<ReturnType<typeof loadGpuWeightPackage>>,
    private readonly onProgress?: (progress: SeparatorProgress) => void,
  ) {}

  static async create(options: DiCoSeSeparatorOptions = {}): Promise<DiCoSeSeparator> {
    options.onProgress?.({ phase: "device", completed: 0, total: 1, detail: "Requesting WebGPU device" });
    const device = await requestDiCoSeDevice();
    options.onProgress?.({ phase: "device", completed: 1, total: 1, detail: "WebGPU device ready" });
    let weights: Awaited<ReturnType<typeof loadGpuWeightPackage>> | undefined;
    try {
      weights = await loadGpuWeightPackage(
        device,
        options.manifestUrl ?? new URL("/model/manifest.json", globalThis.location.href),
        (event) => reportWeightProgress(options.onProgress, event),
      );
      return new DiCoSeSeparator(
        device,
        new DiCoSeBsrRoFormer(
          device,
          weights,
          options.attentionKernel ?? "flash",
        ),
        weights,
        options.onProgress,
      );
    } catch (error) {
      weights?.destroy();
      device.destroy();
      throw error;
    }
  }

  async separatePcm(
    input: StereoPcm,
    options: SeparatePcmOptions = {},
  ): Promise<SeparatorResult> {
    this.requireLive();
    const started = performance.now();
    const resampleStarted = performance.now();
    const source = input.sampleRate === DICOSE_SAMPLE_RATE
      ? input
      : resampleStereoSinc(input, DICOSE_SAMPLE_RATE);
    const resampleMs = performance.now() - resampleStarted;
    if (source.length <= DICOSE_STFT_N_FFT / 2) {
      throw new RangeError(`DiCoSe requires more than ${DICOSE_STFT_N_FFT / 2} samples`);
    }

    const outputMode = requireOutputMode(options.outputMode);
    if (source.length <= DICOSE_SINGLE_PASS_SAMPLES) {
      const result = await this.withGpuErrorScopes(
        "single-pass inference",
        () => this.separateSinglePcm(source, { ...options, outputMode }),
      );
      const withInputTiming = {
        ...result,
        timing: {
          ...result.timing,
          prepareMs: result.timing.prepareMs + resampleMs,
          totalMs: performance.now() - started,
        },
      };
      return restoreInputTimeline(withInputTiming, input, started);
    }
    if (options.traceSamples !== undefined) {
      throw new RangeError("Deterministic tracing is only supported for a single DiCoSe model chunk");
    }
    const result = await this.separateChunkedPcm(source, { ...options, outputMode }, started, resampleMs);
    return restoreInputTimeline(result, input, started);
  }

  private async separateSinglePcm(
    source: StereoPcm,
    options: SeparatePcmOptions,
    shared: {
      readonly mappings?: DiCoSeMappingContexts;
      readonly random?: SeededGaussian;
      readonly noise?: (source: StereoPcm, stem: number) => StereoPcm;
      readonly reportStages?: boolean;
    } = {},
  ): Promise<SeparatorResult> {
    const started = performance.now();
    const report = shared.reportStages === false
      ? (_progress: SeparatorProgress): void => {}
      : (progress: SeparatorProgress): void => this.report(progress);

    report({ phase: "stft", completed: 0, total: 1, detail: "Computing mixture STFT" });
    const prepareStarted = performance.now();
    const mixture = stereoStft(source);
    const mixturePacked = packModelSpectrum(mixture.left, mixture.right);
    const prepareMs = performance.now() - prepareStarted;
    report({ phase: "stft", completed: 1, total: 1, detail: `${mixture.left.frameCount} STFT frames` });

    const outputMode = requireOutputMode(options.outputMode);
    if (outputMode === "deterministic") {
      report({ phase: "deterministic", completed: 0, total: 1, detail: "Running deterministic BS-RoFormer" });
      const deterministic = await this.model.runDeterministic(
        mixturePacked,
        mixture.left.frameCount,
        {
          captureConditions: false,
          ...(options.traceSamples === undefined ? {} : { traceSamples: options.traceSamples }),
        },
      );
      const deterministicMs = deterministic.elapsedMs;
      report({ phase: "deterministic", completed: 1, total: 1, detail: "Deterministic stems ready" });

      const istftStarted = performance.now();
      const deterministicPcm = deterministic.spectra.map((spectrum) => istftStereo(
        modelSpectrumToStereo(spectrum, mixture.left.frameCount, source.length),
      ));
      const stems = namedStems(deterministicPcm);
      const deterministicStats = {} as Record<DiCoSeStem, StemSignalStats>;
      for (const name of DICOSE_STEMS) deterministicStats[name] = signalStats(stems[name]);
      const istftMs = performance.now() - istftStarted;
      const totalMs = performance.now() - started;
      report({ phase: "istft", completed: 1, total: 1, detail: "Final deterministic stem waveforms ready" });
      return {
        outputMode,
        stems,
        timing: {
          prepareMs,
          deterministicMs,
          mappingMs: 0,
          refinementMs: 0,
          istftMs,
          totalMs,
        },
        diagnostics: { deterministic: deterministicStats },
        ...(deterministic.trace === undefined ? {} : { trace: deterministic.trace }),
      };
    }

    report({ phase: "deterministic", completed: 0, total: 1, detail: "Running deterministic BS-RoFormer" });
    const deterministic = await this.model.runDeterministic(
      mixturePacked,
      mixture.left.frameCount,
      options.traceSamples === undefined ? {} : { traceSamples: options.traceSamples },
    );
    const deterministicMs = deterministic.elapsedMs;
    report({ phase: "deterministic", completed: 1, total: 1, detail: "Deterministic stems ready" });

    // `runDeterministic` owns a large condition arena. Establish cleanup as
    // soon as it has succeeded, so a later mapping/ISTFT failure cannot strand
    // GPU memory in the reusable browser worker.
    const conditions = deterministic.conditions;
    let ownedMappings: DiCoSeMappingContexts | undefined;
    try {
      const mappingStarted = performance.now();
      if (shared.mappings === undefined) {
        report({ phase: "mapping", completed: 0, total: 1, detail: "Preparing CD FiLM mappings" });
        ownedMappings = await this.model.createMappings(
          SIGMA_MAX,
          options.traceSamples === undefined ? {} : { traceSamples: options.traceSamples },
        );
      }
      const mappings = shared.mappings ?? ownedMappings;
      if (mappings === undefined) throw new Error("DiCoSe CD mappings were not created");
      const mappingMs = shared.mappings === undefined ? performance.now() - mappingStarted : 0;
      if (shared.mappings === undefined) {
        report({ phase: "mapping", completed: 1, total: 1, detail: "CD FiLM mappings ready" });
      }

      const istftStarted = performance.now();
      const initialStems = deterministic.spectra.map((spectrum) =>
        modelSpectrumToStereo(spectrum, mixture.left.frameCount, source.length),
      );
      const deterministicPcm = initialStems.map((spectrum) => istftStereo(spectrum));
      const deterministicStats = {} as Record<DiCoSeStem, StemSignalStats>;
      for (let stem = 0; stem < DICOSE_STEMS.length; stem += 1) {
        deterministicStats[DICOSE_STEMS[stem]!] = signalStats(deterministicPcm[stem]!);
      }
      const istftAfterDeterministicMs = performance.now() - istftStarted;

      const random = shared.noise === undefined
        ? shared.random ?? new SeededGaussian(options.seed ?? DEFAULT_NOISE_SEED)
        : undefined;
      const scales = consistencyScales(SIGMA_MAX);
      const refined = {} as Record<DiCoSeStem, StereoPcm>;
      const cdModelOutputStats = {} as Record<DiCoSeStem, StemSignalStats>;
      const cdTraces: CdTrace[] = [];
      for (const name of [
        "cd.stftAdapter",
        "cd.bandConditionInput",
        "cd.bandConditionLinear",
        "cd.bandConditionGelu",
        "cd.bandCondition",
      ] as const) {
        const tensor = deterministic.trace?.[name];
        if (tensor !== undefined) cdTraces.push(singleCallCdTrace(name, tensor));
      }
      if (mappings.trace !== undefined) cdTraces.push(mappings.trace);
      let refinementMs = 0;
      let outputIstftMs = 0;

      for (let stem = 0; stem < DICOSE_STEMS.length; stem += 1) {
        const stemName = DICOSE_STEMS[stem]!;
        report({
          phase: "refinement",
          completed: stem,
          total: DICOSE_STEMS.length,
          detail: `Refining ${stemName}`,
        });
        const noisy = shared.noise?.(deterministicPcm[stem]!, stem) ?? addNoise(
          deterministicPcm[stem]!,
          requireRandom(random),
          SIGMA_MAX,
        );
        const cdInput = scaleStereo(noisy, scales.cIn);
        const cdStft = stereoStft(cdInput);
        const refinementStarted = performance.now();
        const cdPass = await this.model.runCdStem(
          packModelSpectrum(cdStft.left, cdStft.right),
          stem,
          conditions,
          mappings,
          options.traceSamples === undefined ? {} : { traceSamples: options.traceSamples },
        );
        if (cdPass.trace !== undefined) cdTraces.push(cdPass.trace);
        refinementMs += performance.now() - refinementStarted;
        const outputStarted = performance.now();
        const outputSpectra = modelSpectrumToStereo(cdPass.spectrum, mixture.left.frameCount, source.length);
        const modelPcm = istftStereo(outputSpectra);
        cdModelOutputStats[stemName] = signalStats(modelPcm);
        refined[stemName] = combineConsistency(modelPcm, noisy, scales.cOut, scales.cSkip);
        outputIstftMs += performance.now() - outputStarted;
        report({
          phase: "refinement",
          completed: stem + 1,
          total: DICOSE_STEMS.length,
          detail: `Refined ${stemName}`,
        });
      }

      const totalMs = performance.now() - started;
      report({ phase: "istft", completed: 1, total: 1, detail: "Final stem waveforms ready" });
      return {
        outputMode,
        stems: refined,
        timing: {
          prepareMs,
          deterministicMs,
          mappingMs,
          refinementMs,
          istftMs: istftAfterDeterministicMs + outputIstftMs,
          totalMs,
        },
        diagnostics: {
          deterministic: deterministicStats,
          cdModelOutput: cdModelOutputStats,
        },
        ...(deterministic.trace === undefined ? {} : { trace: deterministic.trace }),
        ...(cdTraces.length === 0 ? {} : { cdTrace: mergeCdTraces(cdTraces) }),
      };
    } finally {
      conditions.destroy();
      ownedMappings?.destroy();
    }
  }

  private async separateChunkedPcm(
    source: StereoPcm,
    options: SeparatePcmOptions,
    started: number,
    resampleMs: number,
  ): Promise<SeparatorResult> {
    const outputMode = requireOutputMode(options.outputMode);
    const geometry = outputMode === "deterministic"
      ? DICOSE_FAST_CHUNK_GEOMETRY
      : DICOSE_FULL_CHUNK_GEOMETRY;
    const plan = makeDiCoSeChunkPlan(source.length, geometry);
    const window = makeDiCoSeChunkWindow(undefined, geometry.fadeSamples);
    const denominator = new Float32Array(source.length);
    const accumulators = DICOSE_STEMS.map<StereoPcmAccumulator>(() => ({
      left: new Float32Array(source.length),
      right: new Float32Array(source.length),
    }));
    const timing = {
      prepareMs: resampleMs,
      deterministicMs: 0,
      mappingMs: 0,
      refinementMs: 0,
      istftMs: 0,
    };
    const deterministicStats = makeStatsAccumulators();
    const cdModelOutputStats = outputMode === "refined" ? makeStatsAccumulators() : undefined;
    const noiseSeed = options.seed ?? DEFAULT_NOISE_SEED;
    let mappings: DiCoSeMappingContexts | undefined;

    try {
      if (outputMode === "refined") {
        this.report({ phase: "mapping", completed: 0, total: 1, detail: "Preparing shared CD FiLM mappings" });
        const mappingStarted = performance.now();
        mappings = await this.withGpuErrorScopes(
          "shared CD mapping",
          () => this.model.createMappings(SIGMA_MAX),
        );
        timing.mappingMs = performance.now() - mappingStarted;
        this.report({ phase: "mapping", completed: 1, total: 1, detail: "Shared CD FiLM mappings ready" });
      }

      for (let index = 0; index < plan.spans.length; index += 1) {
        const span = plan.spans[index]!;
        this.report({
          phase: "chunk",
          completed: index,
          total: plan.spans.length,
          detail: `Running model chunk ${index + 1} of ${plan.spans.length}`,
        });
        const chunk = materializeDiCoSeChunk(source, plan, span);
        const result = await this.withGpuErrorScopes(
          `model chunk ${index + 1}`,
          () => this.separateSinglePcm(
            chunk,
            options,
            {
              reportStages: false,
              ...(mappings === undefined ? {} : { mappings }),
              ...(outputMode === "deterministic" ? {} : {
                noise: (deterministic: StereoPcm, stem: number) =>
                  addDiCoSeCoordinateNoise(deterministic, span, stem, noiseSeed, SIGMA_MAX),
              }),
            },
          ),
        );
        overlapAddDiCoSeChunk(
          accumulators,
          denominator,
          DICOSE_STEMS.map((name) => result.stems[name]),
          span,
          window,
        );
        timing.prepareMs += result.timing.prepareMs;
        timing.deterministicMs += result.timing.deterministicMs;
        timing.mappingMs += result.timing.mappingMs;
        timing.refinementMs += result.timing.refinementMs;
        timing.istftMs += result.timing.istftMs;
        accumulateStats(deterministicStats, result.diagnostics.deterministic, span.outputSamples);
        if (cdModelOutputStats !== undefined && result.diagnostics.cdModelOutput !== undefined) {
          accumulateStats(cdModelOutputStats, result.diagnostics.cdModelOutput, span.outputSamples);
        }
      }

      normalizeDiCoSeOverlapAdd(accumulators, denominator);
      const stems = {} as Record<DiCoSeStem, StereoPcm>;
      for (let stem = 0; stem < DICOSE_STEMS.length; stem += 1) {
        const name = DICOSE_STEMS[stem]!;
        const accumulator = accumulators[stem]!;
        stems[name] = stereoPcm(accumulator.left, accumulator.right, DICOSE_SAMPLE_RATE);
      }
      const diagnostics = outputMode === "deterministic"
        ? { deterministic: stemSignalStats(stems) }
        : {
          deterministic: finishStats(deterministicStats),
          cdModelOutput: finishStats(requireStats(cdModelOutputStats)),
        };
      this.report({
        phase: "chunk",
        completed: plan.spans.length,
        total: plan.spans.length,
        detail: `Reassembled ${plan.spans.length} model chunks`,
      });
      return {
        outputMode,
        stems,
        timing: {
          ...timing,
          totalMs: performance.now() - started,
        },
        diagnostics,
      };
    } finally {
      mappings?.destroy();
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.model.destroy();
    this.weights.destroy();
    await this.device.queue.onSubmittedWorkDone();
    this.device.destroy();
  }

  private report(progress: SeparatorProgress): void {
    this.onProgress?.(progress);
  }

  private async withGpuErrorScopes<T>(label: string, operation: () => Promise<T>): Promise<T> {
    const filters = ["validation", "out-of-memory", "internal"] as const;
    for (const filter of filters) this.device.pushErrorScope(filter);
    let outcome: { readonly value: T } | undefined;
    let failed = false;
    let failure: unknown;
    try {
      outcome = { value: await operation() };
    } catch (error) {
      failed = true;
      failure = error;
    }

    const gpuErrors: string[] = [];
    for (let index = filters.length - 1; index >= 0; index -= 1) {
      try {
        const error = await this.device.popErrorScope();
        if (error !== null) gpuErrors.push(error.message);
      } catch (error) {
        gpuErrors.push(error instanceof Error ? error.message : String(error));
      }
    }
    if (failed) throw failure;
    if (gpuErrors.length > 0) {
      throw new Error(`DiCoSe ${label} failed WebGPU validation: ${gpuErrors.join("; ")}`);
    }
    if (outcome === undefined) throw new Error(`DiCoSe ${label} returned no result`);
    return outcome.value;
  }

  private requireLive(): void {
    if (this.disposed) throw new Error("DiCoSe separator was disposed");
  }
}

interface StereoSpectrum {
  readonly left: CenteredHannStft;
  readonly right: CenteredHannStft;
}

function stereoStft(source: StereoPcm): StereoSpectrum {
  return { left: centeredHannStft(source.left), right: centeredHannStft(source.right) };
}

function istftStereo(spectrum: StereoSpectrum): StereoPcm {
  const left = centeredHannIstft(spectrum.left, { length: spectrum.left.sourceLength });
  const right = centeredHannIstft(spectrum.right, { length: spectrum.right.sourceLength });
  return stereoPcm(left, right, DICOSE_SAMPLE_RATE);
}

/** `[time][frequency][left re, left im, right re, right im]` for BS-RoFormer. */
function packModelSpectrum(left: CenteredHannStft, right: CenteredHannStft): Uint16Array {
  if (left.frameCount !== right.frameCount || left.binCount !== right.binCount) {
    throw new Error("Stereo STFT dimensions differ");
  }
  const packed = new Uint16Array(left.frameCount * left.binCount * SPECTRAL_COMPONENTS);
  for (let index = 0; index < left.real.length; index += 1) {
    const target = index * SPECTRAL_COMPONENTS;
    packed[target] = float32ToFloat16Bits(left.real[index]!);
    packed[target + 1] = float32ToFloat16Bits(left.imag[index]!);
    packed[target + 2] = float32ToFloat16Bits(right.real[index]!);
    packed[target + 3] = float32ToFloat16Bits(right.imag[index]!);
  }
  return packed;
}

function modelSpectrumToStereo(
  packed: Uint16Array,
  frameCount: number,
  sourceLength: number,
): StereoSpectrum {
  const binCount = DICOSE_STFT_N_FFT / 2 + 1;
  const values = frameCount * binCount;
  if (packed.length !== values * SPECTRAL_COMPONENTS) {
    throw new RangeError("DiCoSe model returned a spectrum with an unexpected shape");
  }
  const leftReal = new Float32Array(values);
  const leftImag = new Float32Array(values);
  const rightReal = new Float32Array(values);
  const rightImag = new Float32Array(values);
  for (let index = 0; index < values; index += 1) {
    const source = index * SPECTRAL_COMPONENTS;
    // The released Python implementation explicitly zeroes DC before ISTFT.
    if (index % binCount === 0) continue;
    leftReal[index] = float16BitsToFloat32(packed[source]!);
    leftImag[index] = float16BitsToFloat32(packed[source + 1]!);
    rightReal[index] = float16BitsToFloat32(packed[source + 2]!);
    rightImag[index] = float16BitsToFloat32(packed[source + 3]!);
  }
  return {
    left: spectrum(leftReal, leftImag, frameCount, sourceLength),
    right: spectrum(rightReal, rightImag, frameCount, sourceLength),
  };
}

function spectrum(
  real: Float32Array,
  imag: Float32Array,
  frameCount: number,
  sourceLength: number,
): CenteredHannStft {
  return {
    layout: "frame-frequency",
    window: "hann-periodic",
    center: true,
    nFft: DICOSE_STFT_N_FFT,
    hopLength: DICOSE_STFT_HOP_LENGTH,
    binCount: DICOSE_STFT_N_FFT / 2 + 1,
    frameCount,
    sourceLength,
    real,
    imag,
  };
}

function addNoise(source: StereoPcm, random: SeededGaussian, sigma: number): StereoPcm {
  const left = new Float32Array(source.length);
  const right = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) left[index] = Math.fround(source.left[index]! + sigma * random.next());
  for (let index = 0; index < source.length; index += 1) right[index] = Math.fround(source.right[index]! + sigma * random.next());
  return stereoPcm(left, right, source.sampleRate);
}

function scaleStereo(source: StereoPcm, scale: number): StereoPcm {
  const left = new Float32Array(source.length);
  const right = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    left[index] = Math.fround(source.left[index]! * scale);
    right[index] = Math.fround(source.right[index]! * scale);
  }
  return stereoPcm(left, right, source.sampleRate);
}

function combineConsistency(
  modelOutput: StereoPcm,
  noisyInput: StereoPcm,
  cOut: number,
  cSkip: number,
): StereoPcm {
  const left = new Float32Array(modelOutput.length);
  const right = new Float32Array(modelOutput.length);
  for (let index = 0; index < modelOutput.length; index += 1) {
    left[index] = clampUnit(cOut * modelOutput.left[index]! + cSkip * noisyInput.left[index]!);
    right[index] = clampUnit(cOut * modelOutput.right[index]! + cSkip * noisyInput.right[index]!);
  }
  return stereoPcm(left, right, modelOutput.sampleRate);
}

function consistencyScales(sigma: number): { readonly cIn: number; readonly cOut: number; readonly cSkip: number } {
  const denominator = Math.sqrt(sigma * sigma + SIGMA_DATA * SIGMA_DATA);
  return {
    cIn: 1 / denominator,
    cSkip: (SIGMA_DATA * SIGMA_DATA) / ((sigma - SIGMA_MIN) ** 2 + SIGMA_DATA * SIGMA_DATA),
    cOut: ((sigma - SIGMA_MIN) * SIGMA_DATA) / denominator,
  };
}

function namedStems(stems: readonly StereoPcm[]): Readonly<Record<DiCoSeStem, StereoPcm>> {
  if (stems.length !== DICOSE_STEMS.length) {
    throw new Error(`DiCoSe deterministic graph returned ${stems.length} stems`);
  }
  const output = {} as Record<DiCoSeStem, StereoPcm>;
  for (let index = 0; index < DICOSE_STEMS.length; index += 1) {
    output[DICOSE_STEMS[index]!] = stems[index]!;
  }
  return output;
}

function stereoPcm(left: Float32Array, right: Float32Array, sampleRate: number): StereoPcm {
  return { sampleRate, length: left.length, left, right, channels: [left, right] };
}

function restoreInputTimeline(
  result: SeparatorResult,
  input: StereoPcm,
  started: number,
): SeparatorResult {
  const firstStem = result.stems[DICOSE_STEMS[0]];
  if (firstStem.sampleRate === input.sampleRate && firstStem.length === input.length) return result;
  const restoreStarted = performance.now();
  const stems = {} as Record<DiCoSeStem, StereoPcm>;
  for (const name of DICOSE_STEMS) {
    stems[name] = resampleStereoSincToLength(
      result.stems[name],
      input.sampleRate,
      input.length,
    );
  }
  const restoreMs = performance.now() - restoreStarted;
  return {
    ...result,
    stems,
    timing: {
      ...result.timing,
      istftMs: result.timing.istftMs + restoreMs,
      totalMs: performance.now() - started,
    },
  };
}

function clampUnit(value: number): number {
  return Math.fround(Math.min(1, Math.max(-1, value)));
}

function signalStats(source: StereoPcm): StemSignalStats {
  let peak = 0;
  let sumSquares = 0;
  const count = source.length * 2;
  for (const channel of source.channels) {
    for (let index = 0; index < channel.length; index += 1) {
      const sample = channel[index]!;
      peak = Math.max(peak, Math.abs(sample));
      sumSquares += sample * sample;
    }
  }
  return { peak, rms: Math.sqrt(sumSquares / count) };
}

interface SignalStatsAccumulator {
  peak: number;
  sumSquares: number;
  samples: number;
}

function makeStatsAccumulators(): Record<DiCoSeStem, SignalStatsAccumulator> {
  const output = {} as Record<DiCoSeStem, SignalStatsAccumulator>;
  for (const name of DICOSE_STEMS) output[name] = { peak: 0, sumSquares: 0, samples: 0 };
  return output;
}

function accumulateStats(
  accumulators: Record<DiCoSeStem, SignalStatsAccumulator>,
  stats: Readonly<Record<DiCoSeStem, StemSignalStats>>,
  outputSamples: number,
): void {
  const scalarSamples = outputSamples * 2;
  for (const name of DICOSE_STEMS) {
    const accumulator = accumulators[name];
    const chunk = stats[name];
    accumulator.peak = Math.max(accumulator.peak, chunk.peak);
    accumulator.sumSquares += chunk.rms * chunk.rms * scalarSamples;
    accumulator.samples += scalarSamples;
  }
}

function finishStats(
  accumulators: Record<DiCoSeStem, SignalStatsAccumulator>,
): Readonly<Record<DiCoSeStem, StemSignalStats>> {
  const output = {} as Record<DiCoSeStem, StemSignalStats>;
  for (const name of DICOSE_STEMS) {
    const accumulator = accumulators[name];
    if (accumulator.samples <= 0) throw new Error(`DiCoSe omitted ${name} diagnostics`);
    output[name] = {
      peak: accumulator.peak,
      rms: Math.sqrt(accumulator.sumSquares / accumulator.samples),
    };
  }
  return output;
}

function stemSignalStats(
  stems: Readonly<Record<DiCoSeStem, StereoPcm>>,
): Readonly<Record<DiCoSeStem, StemSignalStats>> {
  const output = {} as Record<DiCoSeStem, StemSignalStats>;
  for (const name of DICOSE_STEMS) output[name] = signalStats(stems[name]);
  return output;
}

function requireStats(
  value: Record<DiCoSeStem, SignalStatsAccumulator> | undefined,
): Record<DiCoSeStem, SignalStatsAccumulator> {
  if (value === undefined) throw new Error("DiCoSe CD diagnostics accumulator is missing");
  return value;
}

function requireRandom(value: SeededGaussian | undefined): SeededGaussian {
  if (value === undefined) throw new Error("DiCoSe CD noise generator is missing");
  return value;
}

function singleCallCdTrace(
  name: string,
  tensor: { readonly elements: number; readonly values: Uint16Array },
): CdTrace {
  return Object.freeze({
    [name]: Object.freeze({
      elementsPerCall: Object.freeze([tensor.elements]),
      values: tensor.values,
    }),
  });
}

function mergeCdTraces(traces: readonly CdTrace[]): CdTrace {
  const groups = new Map<string, { elements: number[]; values: Uint16Array[] }>();
  for (const trace of traces) {
    for (const [name, tensor] of Object.entries(trace)) {
      let group = groups.get(name);
      if (group === undefined) {
        group = { elements: [], values: [] };
        groups.set(name, group);
      }
      group.elements.push(...tensor.elementsPerCall);
      group.values.push(tensor.values);
    }
  }
  const merged: Record<string, CdTrace[string]> = {};
  for (const [name, group] of groups) {
    const length = group.values.reduce((total, values) => total + values.length, 0);
    const values = new Uint16Array(length);
    let offset = 0;
    for (const call of group.values) {
      values.set(call, offset);
      offset += call.length;
    }
    merged[name] = Object.freeze({
      elementsPerCall: Object.freeze(group.elements.slice()),
      values,
    });
  }
  return Object.freeze(merged);
}

function requireOutputMode(value: SeparatePcmOptions["outputMode"]): "refined" | "deterministic" {
  const outputMode = value ?? "refined";
  if (outputMode !== "refined" && outputMode !== "deterministic") {
    throw new RangeError(`Unsupported DiCoSe output mode: ${String(outputMode)}`);
  }
  return outputMode;
}

function reportWeightProgress(
  progress: ((progress: SeparatorProgress) => void) | undefined,
  event: LoadModelProgress,
): void {
  progress?.({
    phase: "weights",
    completed: event.loadedBytes,
    total: event.totalBytes,
    detail: event.phase === "manifest" ? "Model manifest ready" : "Streaming f16 weights to GPU",
  });
}
