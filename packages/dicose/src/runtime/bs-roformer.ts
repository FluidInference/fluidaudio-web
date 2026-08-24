import type { GpuWeightPackage, GpuWeightTensor } from "../model/package.js";
import { GpuOps, type AttentionKernel } from "../webgpu/ops.js";
import {
  createF16Tensor,
  destroyTensors,
  readF16Tensor,
  writeF16Tensor,
  type GpuTensor,
} from "../webgpu/tensor.js";
import { packFloat16 } from "./audio.js";

export const DICOSE_BANDS = Object.freeze([
  ...Array<number>(24).fill(2),
  ...Array<number>(12).fill(4),
  ...Array<number>(8).fill(12),
  ...Array<number>(8).fill(24),
  ...Array<number>(8).fill(48),
  128,
  129,
]);

const DIM = 384;
const BANDS = 62;
const SPECTRAL_WIDTH = 4_100;
const FREQUENCIES = 1_025;
const STEMS = 4;

type WorkspaceName = "det" | "cd";

interface DeterministicPassBase {
  readonly spectra: readonly Uint16Array[];
  readonly elapsedMs: number;
  /** Evenly sampled f16 activations, populated only by the reference audit harness. */
  readonly trace?: DeterministicTrace;
}

export interface DeterministicTraceTensor {
  readonly elements: number;
  readonly values: Uint16Array;
}

export type DeterministicTrace = Readonly<Record<string, DeterministicTraceTensor>>;

export interface CdTraceTensor {
  readonly elementsPerCall: readonly number[];
  readonly values: Uint16Array;
}

export type CdTrace = Readonly<Record<string, CdTraceTensor>>;

export interface CdStemPass {
  readonly spectrum: Uint16Array;
  readonly trace?: CdTrace;
}

export interface ConditionedDeterministicPass extends DeterministicPassBase {
  readonly conditions: DiCoSeConditions;
}

export interface StandaloneDeterministicPass extends DeterministicPassBase {
  readonly conditions?: undefined;
}

/** GPU-resident mixture conditioning, valid only for one input length. */
export class DiCoSeConditions {
  private destroyed = false;

  constructor(
    readonly frames: number,
    readonly stft: GpuTensor,
    readonly band: GpuTensor,
    readonly time: readonly GpuTensor[],
    readonly frequency: readonly GpuTensor[],
  ) {}

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    destroyTensors([this.stft, this.band, ...this.time, ...this.frequency]);
  }
}

export class DiCoSeMappingContexts {
  private destroyed = false;

  constructor(
    readonly perStem: readonly StemMappingContext[],
    readonly trace?: CdTrace,
  ) {}

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const context of this.perStem) destroyTensors(context.scaleShifts);
  }
}

interface StemMappingContext {
  readonly scaleShifts: readonly GpuTensor[];
}

interface Workspace {
  readonly frames: number;
  readonly rows: number;
  readonly x: GpuTensor;
  readonly alternate: GpuTensor;
  readonly norm: GpuTensor;
  readonly attention: GpuTensor;
  readonly wide: GpuTensor;
  readonly gates: GpuTensor;
  readonly adapter: GpuTensor;
  readonly bandInput: GpuTensor;
  readonly bandNorm: GpuTensor;
  readonly bandFeature: GpuTensor;
  readonly maskMid: GpuTensor;
  readonly maskWide: GpuTensor;
  readonly bandMask: GpuTensor;
  readonly globalMask: GpuTensor;
  readonly masked: GpuTensor;
}

interface PendingTraceTensor {
  readonly name: string;
  readonly elements: number;
  readonly samples: number;
  readonly tensor: GpuTensor;
  readonly repeatPeriod?: number;
}

/**
 * Exact graph ordering for the released BS-RoFormer separator and one-step
 * consistency-distilled refiner. DSP and sampling remain outside this class;
 * this owner only consumes and produces centered-STFT f16 tensors.
 */
export class DiCoSeBsrRoFormer {
  private readonly ops: GpuOps;
  private readonly workspaces = new Map<string, Workspace>();
  /** The worker may serve many input durations; retain at most one CD arena. */
  private cdWorkspaceFrames: number | undefined;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly weights: GpuWeightPackage,
    attentionKernel: AttentionKernel = "flash",
  ) {
    this.ops = new GpuOps(device, attentionKernel);
    const bands = weights.manifest.config.freqsPerBands;
    if (bands.length !== BANDS || bands.some((band, index) => band !== DICOSE_BANDS[index])) {
      throw new Error("This runtime only supports the released 62-band DiCoSe BS-RoFormer profile");
    }
  }

  /** Run the deterministic separator and construct CD conditioning once. */
  async runDeterministic(
    spectral: Uint16Array,
    frames: number,
    options: { readonly captureConditions: false; readonly traceSamples?: number },
  ): Promise<StandaloneDeterministicPass>;
  async runDeterministic(
    spectral: Uint16Array,
    frames: number,
    options?: { readonly captureConditions?: true; readonly traceSamples?: number },
  ): Promise<ConditionedDeterministicPass>;
  async runDeterministic(
    spectral: Uint16Array,
    frames: number,
    options: { readonly captureConditions?: boolean; readonly traceSamples?: number } = {},
  ): Promise<ConditionedDeterministicPass | StandaloneDeterministicPass> {
    this.requireAlive();
    requireSpectral(spectral, frames);
    const captureConditions = options.captureConditions ?? true;
    const started = performance.now();
    let input: GpuTensor | undefined;
    let conditions: DiCoSeConditions | undefined;
    let convIntermediates: readonly [GpuTensor, GpuTensor, GpuTensor] | undefined;
    const outputs: GpuTensor[] = [];
    const traceOutputs: PendingTraceTensor[] = [];
    const traceSamples = options.traceSamples;
    if (traceSamples !== undefined && (!Number.isSafeInteger(traceSamples) || traceSamples <= 0)) {
      throw new RangeError("Deterministic trace sample count must be a positive integer");
    }
    const captureTrace = (
      pass: GPUComputePassEncoder,
      name: string,
      source: GpuTensor,
      elements: number,
    ): void => {
      if (traceSamples === undefined) return;
      const samples = Math.min(traceSamples, elements);
      const tensor = createF16Tensor(this.device, samples, `dicose-trace-${name}`);
      traceOutputs.push({ name, elements, samples, tensor });
      this.ops.sampleEven(pass, source, tensor, elements, samples);
    };
    try {
      input = createF16Tensor(this.device, frames * SPECTRAL_WIDTH, "dicose-mixture-stft");
      writeF16Tensor(this.device, input, spectral);
      const workspace = this.workspace("det", frames);
      if (captureConditions) {
        conditions = this.createConditions(frames);
        convIntermediates = this.createStftAdapterIntermediates(frames);
      }
      for (let stem = 0; stem < STEMS; stem += 1) {
        outputs.push(createF16Tensor(this.device, frames * SPECTRAL_WIDTH, `dicose-det-output-${stem}`));
      }
      this.ops.beginGraph();
      const encoder = this.device.createCommandEncoder({ label: "dicose-deterministic-graph" });
      const pass = encoder.beginComputePass({ label: "dicose-deterministic" });
      if (conditions !== undefined && convIntermediates !== undefined) {
        this.runStftAdapter(pass, input, conditions.stft, frames, convIntermediates);
        captureTrace(pass, "cd.stftAdapter", conditions.stft, frames * SPECTRAL_WIDTH);
      }
      captureTrace(pass, "spectrum.input", input, frames * SPECTRAL_WIDTH);
      this.bandSplit(pass, "det", input, workspace);
      if (conditions !== undefined) {
        captureTrace(pass, "cd.bandConditionInput", workspace.x, workspace.rows * DIM);
        this.runAdapter(
          pass,
          "cd.band_split_feature_adapter",
          workspace.x,
          conditions.band,
          workspace.rows,
          (stage, source) => captureTrace(
            pass,
            `cd.bandCondition${stage}`,
            source,
            workspace.rows * DIM,
          ),
        );
        captureTrace(pass, "cd.bandCondition", conditions.band, workspace.rows * DIM);
      }
      captureTrace(pass, "band", workspace.x, workspace.rows * DIM);
      for (let layer = 0; layer < 8; layer += 1) {
        this.transformer(
          pass,
          "det",
          layer,
          0,
          workspace.x,
          workspace.alternate,
          BANDS,
          workspace.frames,
          undefined,
          0,
          layer === 0
            ? (name, source, elements) => captureTrace(
              pass,
              `layer0.time.${name}`,
              source,
              elements,
            )
            : undefined,
        );
        captureTrace(pass, `layer${layer}.time`, workspace.x, workspace.rows * DIM);
        if (conditions !== undefined) {
          this.runAdapter(
            pass,
            `cd.transformer_feature_adapters.${layer * 2}`,
            workspace.x,
            conditions.time[layer]!,
            workspace.rows,
          );
        }
        this.transformer(
          pass,
          "det",
          layer,
          1,
          workspace.x,
          workspace.alternate,
          workspace.frames,
          BANDS,
        );
        captureTrace(pass, `layer${layer}.frequency`, workspace.x, workspace.rows * DIM);
        if (conditions !== undefined) {
          this.runAdapter(
            pass,
            `cd.transformer_feature_adapters.${layer * 2 + 1}`,
            workspace.x,
            conditions.frequency[layer]!,
            workspace.rows,
          );
        }
      }
      this.ops.rmsNorm(
        pass,
        workspace.x,
        this.weight("det.final_norm.gamma"),
        workspace.norm,
        workspace.rows,
        DIM,
      );
      captureTrace(pass, "finalNorm", workspace.norm, workspace.frames * BANDS * DIM);
      for (let stem = 0; stem < STEMS; stem += 1) {
        this.maskEstimator(pass, "det", stem, input, workspace.norm, workspace);
        captureTrace(pass, `mask.${stem}`, workspace.globalMask, frames * SPECTRAL_WIDTH);
        captureTrace(pass, `spectrum.${stem}`, workspace.masked, frames * SPECTRAL_WIDTH);
        this.ops.copy(pass, workspace.masked, outputs[stem]!, frames * SPECTRAL_WIDTH);
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const spectra = await Promise.all(outputs.map(async (output) => await readF16Tensor(this.device, output)));
      const traceEntries = await Promise.all(traceOutputs.map(async ({ name, elements, tensor }) => [
        name,
        { elements, values: await readF16Tensor(this.device, tensor) },
      ] as const));
      const trace = traceEntries.length === 0
        ? undefined
        : Object.freeze(Object.fromEntries(traceEntries)) as DeterministicTrace;
      const elapsedMs = performance.now() - started;
      if (conditions === undefined) return { spectra, elapsedMs, ...(trace === undefined ? {} : { trace }) };
      const resultConditions = conditions;
      conditions = undefined;
      return {
        spectra,
        conditions: resultConditions,
        elapsedMs,
        ...(trace === undefined ? {} : { trace }),
      };
    } finally {
      input?.buffer.destroy();
      destroyTensors(convIntermediates ?? []);
      destroyTensors(outputs);
      destroyTensors(traceOutputs.map(({ tensor }) => tensor));
      conditions?.destroy();
      // Conditioning survives a successful return, but this transient arena
      // must not outlive a failed deterministic graph.
      this.releaseWorkspace("det", frames);
    }
  }

  /** Precompute every stem/layer FiLM vector for a CD sampler sigma. */
  async createMappings(
    sigma: number,
    options: { readonly traceSamples?: number } = {},
  ): Promise<DiCoSeMappingContexts> {
    this.requireAlive();
    if (!Number.isFinite(sigma) || sigma <= 0) throw new RangeError("CD sigma must be positive");
    const temporaries: GpuTensor[] = [];
    const traceOutputs: PendingTraceTensor[] = [];
    let contexts: StemMappingContext[] = [];
    const traceSamples = options.traceSamples;
    requireTraceSamples(traceSamples, "CD mapping");
    const temporary = (elements: number, label: string): GpuTensor => {
      const tensor = createF16Tensor(this.device, elements, label);
      temporaries.push(tensor);
      return tensor;
    };
    const captureTrace = (
      pass: GPUComputePassEncoder,
      name: string,
      source: GpuTensor,
      elements: number,
    ): void => {
      if (traceSamples === undefined) return;
      const samples = Math.min(traceSamples, elements);
      const tensor = createF16Tensor(this.device, samples, `dicose-trace-${name}-${traceOutputs.length}`);
      traceOutputs.push({ name, elements, samples, tensor });
      this.ops.sampleEven(pass, source, tensor, elements, samples);
    };
    try {
      const timeInput = temporary(DIM, "dicose-cd-time-input");
      // KarrasDenoiser first transports sigma as 250*log(sigma), but the CD
      // wrapper unscales it and passes log(sigma)/4 into the BS-RoFormer.
      writeF16Tensor(this.device, timeInput, positionalEmbedding(Math.log(sigma) / 4, DIM));
      for (let stem = 0; stem < STEMS; stem += 1) contexts.push(this.createMappingContext());
      const embedded = Array.from({ length: STEMS }, (_, stem) =>
        temporary(1_536, `dicose-cd-stem-embedding-${stem}`),
      );
      const time = Array.from({ length: STEMS }, (_, stem) =>
        temporary(1_536, `dicose-cd-time-${stem}`),
      );
      const mappingA = Array.from({ length: STEMS }, (_, stem) =>
        temporary(1_536, `dicose-cd-mapping-a-${stem}`),
      );
      const mappingB = Array.from({ length: STEMS }, (_, stem) =>
        temporary(1_536, `dicose-cd-mapping-b-${stem}`),
      );
      const mappingGelu = Array.from({ length: STEMS }, (_, stem) =>
        temporary(1_536, `dicose-cd-mapping-gelu-${stem}`),
      );
      this.ops.beginGraph();
      const encoder = this.device.createCommandEncoder({ label: "dicose-cd-mapping" });
      const embedding = this.weight("cd.stem_embedding.weight");
      for (let stem = 0; stem < STEMS; stem += 1) {
        encoder.copyBufferToBuffer(
          embedding.buffer,
          embedding.offset + stem * 1_536 * 2,
          embedded[stem]!.buffer,
          0,
          1_536 * 2,
        );
      }
      const pass = encoder.beginComputePass({ label: "dicose-cd-mapping-compute" });
      for (let stem = 0; stem < STEMS; stem += 1) {
        this.ops.linear(pass, timeInput, this.weight("cd.to_time.0.1.weight"), this.weight("cd.to_time.0.1.bias"), time[stem]!, {
          rows: 1, inner: DIM, columns: 1_536, activation: "gelu",
        });
        if (stem === 0) captureTrace(pass, "cd.timeEmbedding", time[stem]!, 1_536);
        this.ops.add(pass, embedded[stem]!, time[stem]!, 1_536);
        captureTrace(pass, "cd.mappingInput", time[stem]!, 1_536);
        this.ops.linear(pass, time[stem]!, this.weight("cd.to_mapping.0.weight"), this.weight("cd.to_mapping.0.bias"), mappingA[stem]!, {
          rows: 1, inner: 1_536, columns: 1_536, activation: "gelu",
        });
        this.ops.linear(pass, mappingA[stem]!, this.weight("cd.to_mapping.2.weight"), this.weight("cd.to_mapping.2.bias"), mappingB[stem]!, {
          rows: 1, inner: 1_536, columns: 1_536, activation: "gelu",
        });
        captureTrace(pass, "cd.mappingOutput", mappingB[stem]!, 1_536);
        // MappingToScaleShift is nn.Sequential(GELU, Linear), so each FiLM
        // projection receives one additional GELU after the mapping network.
        this.ops.copy(pass, mappingB[stem]!, mappingGelu[stem]!, 1_536);
        this.ops.geluInPlace(pass, mappingGelu[stem]!, 1_536);
        let mappingIndex = 0;
        for (let layer = 0; layer < 8; layer += 1) {
          for (const axis of [0, 1] as const) {
            for (const block of [0, 1] as const) {
              const base = `cd.layers.${layer}.${axis}.layers.0.${block}.to_scale_shift.to_scale_shift.1`;
              this.ops.linear(pass, mappingGelu[stem]!, this.weight(`${base}.weight`), this.weight(`${base}.bias`), contexts[stem]!.scaleShifts[mappingIndex]!, {
                rows: 1, inner: 1_536, columns: 768,
              });
              if (mappingIndex === 0) {
                if (traceSamples !== undefined) {
                  const semanticElements = BANDS * 768;
                  const samples = Math.min(traceSamples, semanticElements);
                  const tensor = createF16Tensor(
                    this.device,
                    768,
                    `dicose-trace-cd-film-layer0-time-attention-${stem}`,
                  );
                  traceOutputs.push({
                    name: "cd.film.layer0.time.attention",
                    elements: semanticElements,
                    samples,
                    tensor,
                    repeatPeriod: 768,
                  });
                  this.ops.copy(
                    pass,
                    contexts[stem]!.scaleShifts[mappingIndex]!,
                    tensor,
                    768,
                  );
                }
              }
              mappingIndex += 1;
            }
          }
        }
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const trace = await readCdTrace(this.device, traceOutputs);
      const result = new DiCoSeMappingContexts(contexts, trace);
      contexts = [];
      return result;
    } finally {
      destroyTensors(temporaries);
      destroyTensors(traceOutputs.map(({ tensor }) => tensor));
      for (const context of contexts) destroyTensors(context.scaleShifts);
    }
  }

  /** One CD network evaluation for one stem, returning masked STFT f16. */
  async runCdStem(
    noisySpectral: Uint16Array,
    stem: number,
    conditions: DiCoSeConditions,
    mappings: DiCoSeMappingContexts,
    options: { readonly traceSamples?: number } = {},
  ): Promise<CdStemPass> {
    this.requireAlive();
    if (!Number.isInteger(stem) || stem < 0 || stem >= STEMS) throw new RangeError("Invalid stem index");
    requireSpectral(noisySpectral, conditions.frames);
    const context = mappings.perStem[stem];
    if (context === undefined) throw new Error("CD mapping context is missing a stem");
    const traceOutputs: PendingTraceTensor[] = [];
    const traceSamples = options.traceSamples;
    requireTraceSamples(traceSamples, "CD stem");
    const captureTrace = (
      pass: GPUComputePassEncoder,
      name: string,
      source: GpuTensor,
      elements: number,
    ): void => {
      if (traceSamples === undefined) return;
      const samples = Math.min(traceSamples, elements);
      const tensor = createF16Tensor(this.device, samples, `dicose-trace-${name}-${stem}`);
      traceOutputs.push({ name, elements, samples, tensor });
      this.ops.sampleEven(pass, source, tensor, elements, samples);
    };
    let input: GpuTensor | undefined;
    try {
      input = createF16Tensor(this.device, noisySpectral.length, `dicose-cd-input-${stem}`);
      writeF16Tensor(this.device, input, noisySpectral);
      const workspace = this.cdWorkspace(conditions.frames);
      this.ops.beginGraph();
      const encoder = this.device.createCommandEncoder({ label: `dicose-cd-stem-${stem}` });
      const pass = encoder.beginComputePass({ label: `dicose-cd-stem-${stem}-compute` });
      this.ops.add(pass, conditions.stft, input, noisySpectral.length);
      captureTrace(pass, "cd.stftCombined", input, noisySpectral.length);
      this.bandSplit(pass, "cd", input, workspace);
      captureTrace(pass, "cd.bandRaw", workspace.x, workspace.rows * DIM);
      this.ops.add(pass, conditions.band, workspace.x, workspace.rows * DIM);
      captureTrace(pass, "cd.bandConditioned", workspace.x, workspace.rows * DIM);
      for (let layer = 0; layer < 8; layer += 1) {
        this.transformer(
          pass,
          "cd",
          layer,
          0,
          workspace.x,
          workspace.alternate,
          BANDS,
          workspace.frames,
          context,
          layer * 4,
        );
        if (layer === 0) {
          captureTrace(pass, "cd.layer0.time", workspace.x, workspace.rows * DIM);
        }
        this.ops.add(pass, conditions.time[layer]!, workspace.x, workspace.rows * DIM);
        this.transformer(
          pass,
          "cd",
          layer,
          1,
          workspace.x,
          workspace.alternate,
          workspace.frames,
          BANDS,
          context,
          layer * 4 + 2,
        );
        if (layer === 0) {
          captureTrace(pass, "cd.layer0.frequency", workspace.x, workspace.rows * DIM);
        } else if (layer === 7) {
          captureTrace(pass, "cd.layer7.frequency", workspace.x, workspace.rows * DIM);
        }
        this.ops.add(pass, conditions.frequency[layer]!, workspace.x, workspace.rows * DIM);
      }
      this.ops.rmsNorm(
        pass,
        workspace.x,
        this.weight("cd.final_norm.gamma"),
        workspace.norm,
        workspace.rows,
        DIM,
      );
      captureTrace(pass, "cd.finalNorm", workspace.norm, workspace.frames * BANDS * DIM);
      this.maskEstimator(pass, "cd", stem, input, workspace.norm, workspace);
      captureTrace(pass, "cd.mask", workspace.globalMask, conditions.frames * SPECTRAL_WIDTH);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const [spectrum, trace] = await Promise.all([
        readF16Tensor(this.device, workspace.masked),
        readCdTrace(this.device, traceOutputs),
      ]);
      return { spectrum, ...(trace === undefined ? {} : { trace }) };
    } finally {
      input?.buffer.destroy();
      destroyTensors(traceOutputs.map(({ tensor }) => tensor));
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const workspace of this.workspaces.values()) destroyWorkspace(workspace);
    this.workspaces.clear();
    this.ops.destroy();
  }

  private runStftAdapter(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    output: GpuTensor,
    frames: number,
    intermediates: readonly [GpuTensor, GpuTensor, GpuTensor],
  ): void {
    const [pixels, first, second] = intermediates;
    this.ops.spectralToPixels(pass, source, pixels, frames, FREQUENCIES, 4);
    this.ops.conv2d(pass, pixels, this.weight("cd.stft_feature_adapter.1.weight"), this.weight("cd.stft_feature_adapter.1.bias"), first, FREQUENCIES, frames, 4, 128, 3);
    this.ops.geluInPlace(pass, first, FREQUENCIES * frames * 128);
    this.ops.conv2d(pass, first, this.weight("cd.stft_feature_adapter.3.weight"), this.weight("cd.stft_feature_adapter.3.bias"), second, FREQUENCIES, frames, 128, 128, 1);
    this.ops.geluInPlace(pass, second, FREQUENCIES * frames * 128);
    this.ops.conv2d(pass, second, this.weight("cd.stft_feature_adapter.5.weight"), this.weight("cd.stft_feature_adapter.5.bias"), first, FREQUENCIES, frames, 128, 128, 1);
    this.ops.geluInPlace(pass, first, FREQUENCIES * frames * 128);
    this.ops.conv2d(pass, first, this.weight("cd.stft_feature_adapter.7.weight"), this.weight("cd.stft_feature_adapter.7.bias"), pixels, FREQUENCIES, frames, 128, 4, 3);
    this.ops.pixelsToSpectral(pass, pixels, output, frames, FREQUENCIES, 4);
  }

  private bandSplit(
    pass: GPUComputePassEncoder,
    prefix: "det" | "cd",
    spectral: GpuTensor,
    workspace: Workspace,
  ): void {
    const frames = workspace.frames;
    let offset = 0;
    for (let band = 0; band < BANDS; band += 1) {
      const width = DICOSE_BANDS[band]! * 4;
      this.ops.gatherSlice(pass, spectral, workspace.bandInput, frames, SPECTRAL_WIDTH, offset, width);
      this.ops.rmsNorm(pass, workspace.bandInput, this.weight(`${prefix}.band_split.to_features.${band}.0.gamma`), workspace.bandNorm, frames, width);
      this.ops.linear(pass, workspace.bandNorm, this.weight(`${prefix}.band_split.to_features.${band}.1.weight`), this.weight(`${prefix}.band_split.to_features.${band}.1.bias`), workspace.bandFeature, {
        rows: frames, inner: width, columns: DIM,
      });
      this.ops.scatterSlice(pass, workspace.bandFeature, workspace.x, frames, BANDS * DIM, band * DIM, DIM);
      offset += width;
    }
  }

  private transformer(
    pass: GPUComputePassEncoder,
    prefix: "det" | "cd",
    layer: number,
    axis: 0 | 1,
    input: GpuTensor,
    alternate: GpuTensor,
    sequences: number,
    tokens: number,
    mapping?: StemMappingContext,
    mappingIndex = 0,
    trace?: (name: string, source: GpuTensor, elements: number) => void,
  ): void {
    const rows = sequences * tokens;
    const base = `${prefix}.layers.${layer}.${axis}.layers.0`;
    const attentionMapping = mapping?.scaleShifts[mappingIndex];
    const ffMapping = mapping?.scaleShifts[mappingIndex + 1];
    const attentionGeometry = {
      sequences,
      tokens,
      strided: axis === 0,
    } as const;
    this.ops.rmsNorm(pass, input, this.weight(`${base}.0.norm.gamma`), this.workspaceFor(input).norm, rows, DIM, attentionMapping);
    const workspace = this.workspaceFor(input);
    trace?.("norm", workspace.norm, rows * DIM);
    this.ops.linear(pass, workspace.norm, this.weight(`${base}.0.to_qkv.weight`), undefined, workspace.wide, {
      rows,
      inner: DIM,
      columns: DIM * 4,
      rotaryKeys: attentionGeometry,
    });
    trace?.("qkv", workspace.wide, rows * DIM * 4);
    this.ops.linear(pass, workspace.norm, this.weight(`${base}.0.to_gates.weight`), this.weight(`${base}.0.to_gates.bias`), workspace.gates, {
      rows, inner: DIM, columns: 8,
    });
    trace?.("gates", workspace.gates, rows * 8);
    this.ops.attention(
      pass,
      workspace.wide,
      workspace.attention,
      { ...attentionGeometry, gates: workspace.gates, rotatedKeys: true },
    );
    this.ops.linear(pass, workspace.attention, this.weight(`${base}.0.to_out.0.weight`), undefined, alternate, {
      rows, inner: 512, columns: DIM, residual: input,
    });
    trace?.("postAttentionResidual", alternate, rows * DIM);
    this.ops.rmsNorm(pass, alternate, this.weight(`${base}.1.net.0.gamma`), workspace.norm, rows, DIM, ffMapping);
    trace?.("feedForwardNorm", workspace.norm, rows * DIM);
    this.ops.linear(pass, workspace.norm, this.weight(`${base}.1.net.1.weight`), this.weight(`${base}.1.net.1.bias`), workspace.wide, {
      rows, inner: DIM, columns: 1_536, activation: "gelu",
    });
    trace?.("feedForwardGelu", workspace.wide, rows * 1_536);
    this.ops.linear(pass, workspace.wide, this.weight(`${base}.1.net.4.weight`), this.weight(`${base}.1.net.4.bias`), input, {
      rows, inner: 1_536, columns: DIM, residual: alternate,
    });
    trace?.("output", input, rows * DIM);
  }

  private maskEstimator(
    pass: GPUComputePassEncoder,
    prefix: "det" | "cd",
    stem: number,
    spectral: GpuTensor,
    features: GpuTensor,
    workspace: Workspace,
  ): void {
    let maskOffset = 0;
    const frames = workspace.frames;
    for (let band = 0; band < BANDS; band += 1) {
      const width = DICOSE_BANDS[band]! * 4;
      // MaskEstimator.to_freqs[band] is Sequential(MLP(...), GLU), and the
      // MLP itself is Sequential(Linear, Tanh, Linear):
      // `to_freqs.{band}.0.0` / `.0.2` in the checkpoint.
      const base = `${prefix}.mask_estimators.${stem}.to_freqs.${band}.0`;
      this.ops.gatherSlice(pass, features, workspace.bandFeature, frames, BANDS * DIM, band * DIM, DIM);
      this.ops.linear(pass, workspace.bandFeature, this.weight(`${base}.0.weight`), this.weight(`${base}.0.bias`), workspace.maskMid, {
        rows: frames, inner: DIM, columns: 768, activation: "tanh",
      });
      this.ops.linear(pass, workspace.maskMid, this.weight(`${base}.2.weight`), this.weight(`${base}.2.bias`), workspace.maskWide, {
        rows: frames, inner: 768, columns: width * 2,
      });
      this.ops.gluInPlace(pass, workspace.maskWide, workspace.bandMask, frames, width);
      this.ops.scatterSlice(pass, workspace.bandMask, workspace.globalMask, frames, SPECTRAL_WIDTH, maskOffset, width);
      maskOffset += width;
    }
    this.ops.complexMultiply(pass, spectral, workspace.globalMask, workspace.masked, frames * SPECTRAL_WIDTH / 2);
  }

  private runAdapter(
    pass: GPUComputePassEncoder,
    base: string,
    input: GpuTensor,
    output: GpuTensor,
    rows: number,
    trace?: (stage: "Linear" | "Gelu", source: GpuTensor) => void,
  ): void {
    const workspace = this.workspaceFor(input);
    this.ops.linear(pass, input, this.weight(`${base}.0.weight`), this.weight(`${base}.0.bias`), workspace.adapter, {
      rows, inner: DIM, columns: DIM,
    });
    trace?.("Linear", workspace.adapter);
    this.ops.geluInPlace(pass, workspace.adapter, rows * DIM);
    trace?.("Gelu", workspace.adapter);
    this.ops.linear(pass, workspace.adapter, this.weight(`${base}.2.weight`), this.weight(`${base}.2.bias`), output, {
      rows, inner: DIM, columns: DIM,
    });
  }

  private createConditions(frames: number): DiCoSeConditions {
    const rows = frames * BANDS;
    const tensors: GpuTensor[] = [];
    const temporary = (elements: number, label: string): GpuTensor => {
      const tensor = createF16Tensor(this.device, elements, label);
      tensors.push(tensor);
      return tensor;
    };
    try {
      const stft = temporary(frames * SPECTRAL_WIDTH, "dicose-condition-stft");
      const band = temporary(rows * DIM, "dicose-condition-band");
      const time = Array.from({ length: 8 }, (_, index) =>
        temporary(rows * DIM, `dicose-condition-time-${index}`),
      );
      const frequency = Array.from({ length: 8 }, (_, index) =>
        temporary(rows * DIM, `dicose-condition-frequency-${index}`),
      );
      return new DiCoSeConditions(frames, stft, band, time, frequency);
    } catch (error) {
      destroyTensors(tensors);
      throw error;
    }
  }

  private createStftAdapterIntermediates(frames: number): readonly [GpuTensor, GpuTensor, GpuTensor] {
    const pixels = FREQUENCIES * frames;
    const tensors: GpuTensor[] = [];
    const temporary = (elements: number, label: string): GpuTensor => {
      const tensor = createF16Tensor(this.device, elements, label);
      tensors.push(tensor);
      return tensor;
    };
    try {
      return [
        temporary(pixels * 4, "dicose-stft-adapter-pixels"),
        temporary(pixels * 128, "dicose-stft-adapter-first"),
        temporary(pixels * 128, "dicose-stft-adapter-second"),
      ];
    } catch (error) {
      destroyTensors(tensors);
      throw error;
    }
  }

  private createMappingContext(): StemMappingContext {
    const scaleShifts: GpuTensor[] = [];
    try {
      for (let index = 0; index < 32; index += 1) {
        scaleShifts.push(createF16Tensor(this.device, 768, `dicose-cd-scale-shift-${index}`));
      }
      return { scaleShifts };
    } catch (error) {
      destroyTensors(scaleShifts);
      throw error;
    }
  }

  private workspace(name: WorkspaceName, frames: number): Workspace {
    const key = `${name}:${frames}`;
    const existing = this.workspaces.get(key);
    if (existing !== undefined) return existing;
    const rows = frames * BANDS;
    const tensors: GpuTensor[] = [];
    const temporary = (elements: number, label: string): GpuTensor => {
      const tensor = createF16Tensor(this.device, elements, label);
      tensors.push(tensor);
      return tensor;
    };
    try {
      const workspace: Workspace = {
        frames,
        rows,
        x: temporary(rows * DIM, `${key}:x`),
        alternate: temporary(rows * DIM, `${key}:alternate`),
        norm: temporary(rows * DIM, `${key}:norm`),
        // Multi-head attention has 8 × 64 = 512 inner features. It is projected
        // back to the model's 384 features only by `to_out`.
        attention: temporary(rows * 512, `${key}:attention`),
        wide: temporary(rows * 1_536, `${key}:wide`),
        gates: temporary(rows * 8, `${key}:gates`),
        adapter: temporary(rows * DIM, `${key}:adapter`),
        bandInput: temporary(frames * 516, `${key}:band-input`),
        bandNorm: temporary(frames * 516, `${key}:band-norm`),
        bandFeature: temporary(frames * DIM, `${key}:band-feature`),
        maskMid: temporary(frames * 768, `${key}:mask-mid`),
        maskWide: temporary(frames * 1_032, `${key}:mask-wide`),
        bandMask: temporary(frames * 516, `${key}:band-mask`),
        globalMask: temporary(frames * SPECTRAL_WIDTH, `${key}:global-mask`),
        masked: temporary(frames * SPECTRAL_WIDTH, `${key}:masked`),
      };
      this.workspaces.set(key, workspace);
      return workspace;
    } catch (error) {
      destroyTensors(tensors);
      throw error;
    }
  }

  /**
   * A CD arena is deliberately reused across stems and same-size requests, but
   * a long-lived browser worker must not retain one ~hundreds-of-MiB arena for
   * every historical input duration.
   */
  private cdWorkspace(frames: number): Workspace {
    if (this.cdWorkspaceFrames !== undefined && this.cdWorkspaceFrames !== frames) {
      this.releaseWorkspace("cd", this.cdWorkspaceFrames);
    }
    const workspace = this.workspace("cd", frames);
    this.cdWorkspaceFrames = frames;
    return workspace;
  }

  private releaseWorkspace(name: WorkspaceName, frames: number): void {
    const key = `${name}:${frames}`;
    const workspace = this.workspaces.get(key);
    if (workspace === undefined) return;
    destroyWorkspace(workspace);
    this.workspaces.delete(key);
    if (name === "cd" && this.cdWorkspaceFrames === frames) this.cdWorkspaceFrames = undefined;
  }

  private workspaceFor(tensor: GpuTensor): Workspace {
    for (const workspace of this.workspaces.values()) {
      if (workspace.x === tensor || workspace.alternate === tensor) return workspace;
    }
    throw new Error("DiCoSe transformer tensor is not owned by a workspace");
  }

  private weight(name: string): GpuWeightTensor {
    return this.weights.tensor(name);
  }

  private requireAlive(): void {
    if (this.destroyed) throw new Error("DiCoSe separator was destroyed");
  }
}

async function readCdTrace(
  device: GPUDevice,
  pending: readonly PendingTraceTensor[],
): Promise<CdTrace | undefined> {
  if (pending.length === 0) return undefined;
  const captured = await Promise.all(pending.map(async (entry) => {
    const physical = await readF16Tensor(device, entry.tensor);
    if (entry.repeatPeriod === undefined) {
      if (physical.length !== entry.samples) {
        throw new Error(`CD trace ${entry.name} returned the wrong sample count`);
      }
      return { name: entry.name, elements: entry.elements, values: physical };
    }
    if (physical.length !== entry.repeatPeriod) {
      throw new Error(`CD trace ${entry.name} returned the wrong repeat period`);
    }
    const values = new Uint16Array(entry.samples);
    for (let index = 0; index < values.length; index += 1) {
      const semanticIndex = Math.floor(index * entry.elements / entry.samples);
      values[index] = physical[semanticIndex % entry.repeatPeriod]!;
    }
    return { name: entry.name, elements: entry.elements, values };
  }));
  const grouped = new Map<string, { elements: number[]; values: Uint16Array[] }>();
  for (const entry of captured) {
    let group = grouped.get(entry.name);
    if (group === undefined) {
      group = { elements: [], values: [] };
      grouped.set(entry.name, group);
    }
    group.elements.push(entry.elements);
    group.values.push(entry.values);
  }
  const trace: Record<string, CdTraceTensor> = {};
  for (const [name, group] of grouped) {
    const length = group.values.reduce((total, values) => total + values.length, 0);
    const values = new Uint16Array(length);
    let offset = 0;
    for (const call of group.values) {
      values.set(call, offset);
      offset += call.length;
    }
    trace[name] = Object.freeze({
      elementsPerCall: Object.freeze(group.elements.slice()),
      values,
    });
  }
  return Object.freeze(trace);
}

function requireTraceSamples(value: number | undefined, label: string): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError(`${label} trace sample count must be a positive integer`);
  }
}

function destroyWorkspace(workspace: Workspace): void {
  destroyTensors(Object.values(workspace).filter((value): value is GpuTensor =>
    typeof value === "object" && value !== null && "buffer" in value,
  ));
}

function positionalEmbedding(time: number, dim: number): Uint16Array {
  const values = new Float32Array(dim);
  const half = dim / 2;
  for (let index = 0; index < half; index += 1) {
    const frequency = Math.pow(1 / 10_000, index / (half - 1));
    // PositionalEmbedding emits [cos(all), sin(all)], then reshapes and
    // flips its two halves, so the final representation is [sin(all),
    // cos(all)] rather than interleaved pairs.
    values[index] = Math.sin(time * frequency);
    values[half + index] = Math.cos(time * frequency);
  }
  return packFloat16(values);
}

function requireSpectral(spectral: Uint16Array, frames: number): void {
  if (spectral.length !== frames * SPECTRAL_WIDTH) {
    throw new RangeError(`Expected ${frames * SPECTRAL_WIDTH} f16 STFT values, received ${spectral.length}`);
  }
}
