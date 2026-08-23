import {
  resolveAceLogicalTensor,
  type AcePackageFileRecord,
  type AcePackageManifest,
  type AcePackageTensorRecord,
  type AceTensorPhase,
} from "./manifest.js";
import {
  uploadAcePackageFileToGpu,
  type AceGpuUploadProgress,
  type AceGpuUploadTrace,
} from "./gpu-upload.js";

export interface AceGpuTensorPart {
  readonly tensorName: string;
  readonly tensor: AcePackageTensorRecord;
  readonly binding: GPUBufferBinding;
}

export interface AceGpuLogicalTensor {
  readonly logicalTensor: string;
  readonly logicalShape: readonly number[];
  readonly parts: readonly AceGpuTensorPart[];
}

export interface AceGpuTensorPhaseProgress {
  readonly phaseFileIndex: number;
  readonly phaseFileCount: number;
  readonly loadedPhaseBytes: number;
  readonly totalPhaseBytes: number;
  readonly upload: AceGpuUploadProgress;
}

export interface AceGpuTensorPhaseUploadTrace extends AceGpuUploadTrace {
  readonly phases: readonly AceTensorPhase[];
  readonly phaseFileIndex: number;
  readonly phaseFileCount: number;
  readonly totalPhaseBytes: number;
}

export interface AceLoadGpuTensorPhaseOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceGpuTensorPhaseProgress) => void;
  /** @internal Capture-only upload attribution; normal phase loads omit it. */
  readonly onUploadTrace?: (trace: AceGpuTensorPhaseUploadTrace) => void;
  /** @internal Deterministic capture clock. */
  readonly now?: () => number;
  /** @internal Deterministic uploader injection for contract tests. */
  readonly upload?: AceGpuUploader;
}

type AceGpuUploader = typeof uploadAcePackageFileToGpu;

const ACE_TENSOR_PHASES = new Set<AceTensorPhase>([
  "planner",
  "text",
  "conditioner",
  "semantic",
  "dit",
  "vae",
  "constants",
]);

/**
 * Resident, authenticated GPU shards for one explicit phase-lifetime set.
 *
 * The package remains content-addressed on disk. Loading a phase streams and
 * re-authenticates only the physical shards referenced by its tensors, then
 * publishes bindings only after every upload succeeds. Callers must destroy
 * the phase after the queue has drained before loading the next heavyweight
 * lifetime.
 */
export class AceGpuTensorPhase {
  readonly phases: readonly AceTensorPhase[];
  readonly residentBytes: number;
  /** Exact parsed manifest object that authenticated this resident phase. */
  readonly packageManifest: AcePackageManifest;

  private destroyed = false;

  private constructor(
    private readonly manifest: AcePackageManifest,
    phases: readonly AceTensorPhase[],
    private readonly buffers: ReadonlyMap<string, GPUBuffer>,
    residentBytes: number,
  ) {
    this.packageManifest = manifest;
    this.phases = Object.freeze([...phases]);
    this.residentBytes = residentBytes;
  }

  static async load(
    device: GPUDevice,
    manifest: AcePackageManifest,
    acquiredFiles: ReadonlyMap<string, File>,
    phases: readonly AceTensorPhase[],
    options: AceLoadGpuTensorPhaseOptions = {},
  ): Promise<AceGpuTensorPhase> {
    const loaded = await loadAceGpuTensorPhaseWithUploader(
      device,
      manifest,
      acquiredFiles,
      phases,
      options,
      options.upload ?? uploadAcePackageFileToGpu,
    );
    return new AceGpuTensorPhase(
      manifest,
      loaded.phases,
      loaded.buffers,
      loaded.residentBytes,
    );
  }

  /** Resolve all canonical axis-0 parts of one resident logical tensor. */
  logicalTensor(logicalTensor: string): AceGpuLogicalTensor {
    this.requireLive();
    const resolved = resolveAceLogicalTensor(this.manifest, logicalTensor);
    const parts = resolved.parts.map(({ tensorName, tensor }) => {
      if (!this.phases.includes(tensor.phase)) {
        throw new Error(
          `ACE logical tensor ${logicalTensor} belongs to phase ${tensor.phase}, ` +
            `not resident phases ${this.phases.join(",")}`,
        );
      }
      const buffer = this.buffers.get(tensor.shard);
      if (buffer === undefined) {
        throw new Error(
          `ACE logical tensor ${logicalTensor} is not resident in phases ` +
            this.phases.join(","),
        );
      }
      return Object.freeze({
        tensorName,
        tensor,
        binding: Object.freeze({
          buffer,
          offset: tensor.byteOffset,
          size: tensor.byteLength,
        }),
      });
    });
    return Object.freeze({
      logicalTensor,
      logicalShape: resolved.logicalShape,
      parts: Object.freeze(parts),
    });
  }

  /** Resolve an unsharded tensor, failing closed when a graph forgot sharding. */
  binding(logicalTensor: string): GPUBufferBinding {
    const resolved = this.logicalTensor(logicalTensor);
    if (resolved.parts.length !== 1) {
      throw new Error(
        `ACE logical tensor ${logicalTensor} has ${resolved.parts.length} parts; ` +
          "the graph must dispatch its explicit row shards",
      );
    }
    return resolved.parts[0]!.binding;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.buffers.values()) buffer.destroy();
  }

  private requireLive(): void {
    if (this.destroyed) throw new Error("ACE GPU tensor phase is destroyed");
  }
}

async function loadAceGpuTensorPhaseWithUploader(
  device: GPUDevice,
  manifest: AcePackageManifest,
  acquiredFiles: ReadonlyMap<string, File>,
  phases: readonly AceTensorPhase[],
  options: AceLoadGpuTensorPhaseOptions,
  upload: AceGpuUploader,
): Promise<Readonly<{
  phases: readonly AceTensorPhase[];
  buffers: ReadonlyMap<string, GPUBuffer>;
  residentBytes: number;
}>> {
  const selectedPhases = validatePhases(phases);
  const selectedFiles = selectPhaseFiles(manifest, selectedPhases);
  const phaseSources = new Map<string, File>();
  for (const record of selectedFiles) {
    const source = acquiredFiles.get(record.name);
    if (source === undefined) {
      throw new Error(`ACE acquired package is missing ${record.name}`);
    }
    if (!(source instanceof File) || source.size !== record.byteLength) {
      throw new Error(`ACE acquired package has the wrong source for ${record.name}`);
    }
    phaseSources.set(record.name, source);
  }
  const totalPhaseBytes = checkedSum(
    selectedFiles.map((file) => file.byteLength),
    "ACE phase byte total",
  );
  const buffers = new Map<string, GPUBuffer>();
  let loadedPhaseBytes = 0;
  try {
    for (let index = 0; index < selectedFiles.length; index += 1) {
      options.signal?.throwIfAborted();
      const record = selectedFiles[index]!;
      const source = phaseSources.get(record.name)!;
      const completedBeforeFile = loadedPhaseBytes;
      const buffer = await upload(device, record, source, {
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        label: `ace-${selectedPhases.join("+")}-${record.name}`,
        onProgress: (event) => {
          options.onProgress?.({
            phaseFileIndex: index,
            phaseFileCount: selectedFiles.length,
            loadedPhaseBytes: completedBeforeFile + event.uploadedBytes,
            totalPhaseBytes,
            upload: event,
          });
        },
        ...(options.onUploadTrace === undefined
          ? {}
          : {
              onTrace: (trace: AceGpuUploadTrace) => {
                options.onUploadTrace?.(Object.freeze({
                  ...trace,
                  phases: selectedPhases,
                  phaseFileIndex: index,
                  phaseFileCount: selectedFiles.length,
                  totalPhaseBytes,
                }));
              },
            }),
        ...(options.now === undefined ? {} : { now: options.now }),
      });
      buffers.set(record.name, buffer);
      loadedPhaseBytes = checkedAdd(
        loadedPhaseBytes,
        record.byteLength,
        "ACE loaded phase bytes",
      );
    }
    options.signal?.throwIfAborted();
    return Object.freeze({
      phases: selectedPhases,
      buffers,
      residentBytes: totalPhaseBytes,
    });
  } catch (error) {
    for (const buffer of buffers.values()) buffer.destroy();
    throw error;
  }
}

function validatePhases(phases: readonly AceTensorPhase[]): readonly AceTensorPhase[] {
  if (phases.length === 0) throw new RangeError("ACE GPU phase set is empty");
  const unique = new Set<AceTensorPhase>();
  for (const phase of phases) {
    if (!ACE_TENSOR_PHASES.has(phase)) {
      throw new RangeError(`ACE GPU phase set contains unknown phase ${String(phase)}`);
    }
    if (unique.has(phase)) {
      throw new RangeError(`ACE GPU phase set repeats ${phase}`);
    }
    unique.add(phase);
  }
  return Object.freeze([...unique]);
}

function selectPhaseFiles(
  manifest: AcePackageManifest,
  phases: readonly AceTensorPhase[],
): readonly AcePackageFileRecord[] {
  const phaseSet = new Set(phases);
  const names = new Set<string>();
  for (const tensor of Object.values(manifest.tensors)) {
    if (phaseSet.has(tensor.phase)) names.add(tensor.shard);
  }
  if (names.size === 0) {
    throw new Error(`ACE package has no tensors for phases ${phases.join(",")}`);
  }
  const records = manifest.files.filter((file) => names.has(file.name));
  if (records.length !== names.size) {
    throw new Error("ACE phase tensor inventory references an absent physical shard");
  }
  for (const record of records) {
    if (record.kind !== "weights" && record.kind !== "constant") {
      throw new Error(`ACE phase shard ${record.name} has invalid kind ${record.kind}`);
    }
  }
  return Object.freeze(records);
}

function checkedSum(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) total = checkedAdd(total, value, label);
  return total;
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new RangeError(`${label} exceeds safe integer arithmetic`);
  }
  return result;
}
