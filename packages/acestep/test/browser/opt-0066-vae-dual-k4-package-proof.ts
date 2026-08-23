import {
  ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  type AcePackageManifest,
  type AcePackageTensorRecord,
} from "../../src/model/manifest.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import {
  packAceOpt0051VaeK7WeightU16,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  packAceOpt0048VaeConvTranspose1dK4WeightU16,
  planAceOpt0048VaeConvTranspose1dK4Weight,
} from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";

const FLOAT16_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const SELECTED_LAYOUT_COUNT =
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.length +
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.length;

export interface Opt0066AuthenticatedPackageView {
  readonly manifest: AcePackageManifest;
  readonly files: ReadonlyMap<string, File>;
}

export interface Opt0066U16Comparison {
  readonly comparedWordCount: number;
  readonly mismatchCount: number;
  readonly firstMismatch: Readonly<{
    readonly index: number;
    readonly expected: number;
    readonly actual: number;
  }> | null;
}

export interface Opt0066DerivedTransposeWeight {
  readonly operationLabel: string;
  readonly tensor: string;
  readonly reuseAxis: "channel" | "row";
  readonly words: Uint16Array<ArrayBuffer>;
}

export interface Opt0066PackageLayoutProof {
  readonly schema: "ace-opt-0066-package-layout-proof-v1";
  readonly revision6LogicalTensorCount: number;
  readonly revision7LogicalTensorCount: number;
  readonly selectedLayoutCount: number;
  readonly selectedK7Count: number;
  readonly selectedConvTransposeCount: number;
  readonly comparedU16WordCount: number;
  readonly mismatchCount: 0;
  readonly entries: readonly Readonly<Record<string, unknown>>[];
}

export interface Opt0066PreparedPackageLayoutProof {
  readonly proof: Opt0066PackageLayoutProof;
  readonly derivedTransposeWeights:
    readonly Opt0066DerivedTransposeWeight[];
}

/**
 * Pre-device proof over authenticated File slices. Revision-6 logical bytes
 * are packed with the independent TypeScript indexers and compared with the
 * exact revision-7 physical spans. Only four bounded ConvTranspose arrays are
 * retained because they are the same-arithmetic oracle's GPU source.
 */
export async function prepareOpt0066PackageLayoutProof(
  revision6: Opt0066AuthenticatedPackageView,
  revision7: Opt0066AuthenticatedPackageView,
): Promise<Opt0066PreparedPackageLayoutProof> {
  requireLogicalInventory(revision6.manifest, 6);
  requireLogicalInventory(revision7.manifest, 7);
  const entries: Readonly<Record<string, unknown>>[] = [];
  const derivedTransposeWeights: Opt0066DerivedTransposeWeight[] = [];
  let comparedU16WordCount = 0;

  for (const contract of ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS) {
    const nativeRecord = requireRecord(
      revision6.manifest,
      contract.tensor,
      ACE_VAE_CONV1D_FP16_LAYOUT,
      [contract.channels, 7, contract.channels],
      [contract.channels, 7, contract.channels],
    );
    const packedRecord = requireRecord(
      revision7.manifest,
      contract.tensor,
      ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
      [contract.channels, 7, contract.channels],
      [7, contract.channels / 4, contract.channels / 64, 32, 2, 4],
    );
    const native = await readTensorWords(revision6, nativeRecord);
    const candidate = await readTensorWords(revision7, packedRecord);
    const derived = packAceOpt0051VaeK7WeightU16(
      native,
      contract.channels,
      contract.channels,
    );
    const comparison = compareOpt0066U16(derived, candidate);
    requireExactComparison(contract.tensor, comparison);
    comparedU16WordCount += comparison.comparedWordCount;
    entries.push(Object.freeze({
      family: "k7-conv1d",
      operationLabel: contract.operationLabel,
      tensor: contract.tensor,
      sourceLayout: nativeRecord.layout,
      candidateLayout: packedRecord.layout,
      sourceStorageShape: nativeRecord.storageShape,
      candidateStorageShape: packedRecord.storageShape,
      sourceShard: nativeRecord.shard,
      sourceByteOffset: nativeRecord.byteOffset,
      candidateShard: packedRecord.shard,
      candidateByteOffset: packedRecord.byteOffset,
      byteLength: nativeRecord.byteLength,
      comparedU16WordCount: comparison.comparedWordCount,
      mismatchCount: comparison.mismatchCount,
      firstMismatch: comparison.firstMismatch,
      sourceSha256: sha256Words(native),
      derivedPackedSha256: sha256Words(derived),
      candidatePackedSha256: sha256Words(candidate),
    }));
  }

  for (const contract of ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS) {
    const logicalShape = [
      contract.outputChannels,
      contract.stride * 2,
      contract.inputChannels,
    ] as const;
    const nativeStorageShape = [
      contract.stride,
      2,
      contract.inputChannels,
      contract.outputChannels,
    ] as const;
    const outputsPerLane = contract.reuseAxis === "channel" ? 8 : 4;
    const packedStorageShape = [
      contract.stride,
      2,
      contract.inputChannels / 4,
      contract.outputChannels / (32 * outputsPerLane),
      32,
      outputsPerLane,
      4,
    ] as const;
    const nativeRecord = requireRecord(
      revision6.manifest,
      contract.tensor,
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
      logicalShape,
      nativeStorageShape,
    );
    const packedRecord = requireRecord(
      revision7.manifest,
      contract.tensor,
      ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
      logicalShape,
      packedStorageShape,
    );
    const native = await readTensorWords(revision6, nativeRecord);
    const candidate = await readTensorWords(revision7, packedRecord);
    const plan = planAceOpt0048VaeConvTranspose1dK4Weight({
      kernelSize: contract.stride * 2,
      stride: contract.stride,
      dilation: 1,
      outputPadding: 0,
      inputChannels: contract.inputChannels,
      outputChannels: contract.outputChannels,
    }, contract.reuseAxis);
    if (
      plan.reuseAxis !== contract.reuseAxis ||
      !sameShape(plan.packedWeightStorageShape, packedStorageShape)
    ) {
      throw new Error(
        `OPT-0066 ${contract.operationLabel} transpose pack plan changed`,
      );
    }
    const derived = packAceOpt0048VaeConvTranspose1dK4WeightU16(
      native,
      plan,
    ) as Uint16Array<ArrayBuffer>;
    const comparison = compareOpt0066U16(derived, candidate);
    requireExactComparison(contract.tensor, comparison);
    comparedU16WordCount += comparison.comparedWordCount;
    entries.push(Object.freeze({
      family: "conv-transpose1d",
      operationLabel: contract.operationLabel,
      tensor: contract.tensor,
      reuseAxis: contract.reuseAxis,
      sourceLayout: nativeRecord.layout,
      candidateLayout: packedRecord.layout,
      sourceStorageShape: nativeRecord.storageShape,
      candidateStorageShape: packedRecord.storageShape,
      sourceShard: nativeRecord.shard,
      sourceByteOffset: nativeRecord.byteOffset,
      candidateShard: packedRecord.shard,
      candidateByteOffset: packedRecord.byteOffset,
      byteLength: nativeRecord.byteLength,
      comparedU16WordCount: comparison.comparedWordCount,
      mismatchCount: comparison.mismatchCount,
      firstMismatch: comparison.firstMismatch,
      sourceSha256: sha256Words(native),
      derivedPackedSha256: sha256Words(derived),
      candidatePackedSha256: sha256Words(candidate),
      retainedAsRevision6OracleSource: true,
    }));
    derivedTransposeWeights.push(Object.freeze({
      operationLabel: contract.operationLabel,
      tensor: contract.tensor,
      reuseAxis: contract.reuseAxis,
      words: derived,
    }));
  }

  if (
    entries.length !== SELECTED_LAYOUT_COUNT ||
    derivedTransposeWeights.length !==
      ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.length ||
    derivedTransposeWeights.reduce((sum, entry) =>
      sum + entry.words.byteLength, 0) !== 15_335_424
  ) {
    throw new Error("OPT-0066 selected package-layout inventory changed");
  }
  return Object.freeze({
    proof: Object.freeze({
      schema: "ace-opt-0066-package-layout-proof-v1",
      revision6LogicalTensorCount: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
      revision7LogicalTensorCount: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
      selectedLayoutCount: entries.length,
      selectedK7Count: ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.length,
      selectedConvTransposeCount:
        ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.length,
      comparedU16WordCount,
      mismatchCount: 0,
      entries: Object.freeze(entries),
    }),
    derivedTransposeWeights: Object.freeze(derivedTransposeWeights),
  });
}

export function compareOpt0066U16(
  expected: Uint16Array,
  actual: Uint16Array,
): Opt0066U16Comparison {
  if (expected.length !== actual.length) {
    throw new RangeError("OPT-0066 U16 comparison length changed");
  }
  let mismatchCount = 0;
  let firstMismatch: Opt0066U16Comparison["firstMismatch"] = null;
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      mismatchCount += 1;
      firstMismatch ??= Object.freeze({
        index,
        expected: expected[index]!,
        actual: actual[index]!,
      });
    }
  }
  return Object.freeze({
    comparedWordCount: expected.length,
    mismatchCount,
    firstMismatch,
  });
}

function requireLogicalInventory(
  manifest: AcePackageManifest,
  revision: 6 | 7,
): void {
  const vae = Object.values(manifest.tensors).filter((record) =>
    record.phase === "vae"
  );
  const names = new Set(vae.map((record) => record.logicalTensor));
  if (
    vae.length !== ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT ||
    names.size !== ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT
  ) {
    throw new Error(`OPT-0066 revision-${revision} VAE inventory changed`);
  }
}

function requireRecord(
  manifest: AcePackageManifest,
  tensor: string,
  layout: AcePackageTensorRecord["layout"],
  logicalShape: readonly number[],
  storageShape: readonly number[],
): AcePackageTensorRecord {
  const record = manifest.tensors[tensor];
  if (
    record === undefined || record.logicalTensor !== tensor ||
    record.dtype !== "float16" || record.phase !== "vae" ||
    record.lifetime !== "vae" || record.layout !== layout ||
    !sameShape(record.logicalShape, logicalShape) ||
    !sameShape(record.storageShape, storageShape) ||
    record.byteLength !== product(storageShape) * FLOAT16_BYTES ||
    record.partAxis !== 0 || record.partStart !== 0 ||
    record.partEnd !== logicalShape[0]
  ) {
    throw new Error(`OPT-0066 tensor ${tensor} changed its package contract`);
  }
  return record;
}

async function readTensorWords(
  pkg: Opt0066AuthenticatedPackageView,
  record: AcePackageTensorRecord,
): Promise<Uint16Array<ArrayBuffer>> {
  const file = pkg.files.get(record.shard);
  if (file === undefined) {
    throw new Error(`OPT-0066 tensor ${record.logicalTensor} has no exact File span`);
  }
  return await readOpt0066ExactFileSliceU16(
    file,
    record.byteOffset,
    record.byteLength,
    record.logicalTensor,
  );
}

/** Exact bounded File slicing seam, exported for canary/offset unit proof. */
export async function readOpt0066ExactFileSliceU16(
  file: Blob,
  byteOffset: number,
  byteLength: number,
  label = "tensor",
): Promise<Uint16Array<ArrayBuffer>> {
  if (
    !Number.isSafeInteger(byteOffset) || byteOffset < 0 ||
    !Number.isSafeInteger(byteLength) || byteLength < FLOAT16_BYTES ||
    byteLength % FLOAT16_BYTES !== 0 ||
    byteOffset + byteLength > file.size
  ) {
    throw new Error(`OPT-0066 ${label} has no exact even File span`);
  }
  const bytes = await file.slice(
    byteOffset,
    byteOffset + byteLength,
  ).arrayBuffer();
  if (bytes.byteLength !== byteLength) {
    throw new Error(`OPT-0066 ${label} produced a short read`);
  }
  return new Uint16Array(bytes);
}

function requireExactComparison(
  tensor: string,
  comparison: Opt0066U16Comparison,
): void {
  if (comparison.mismatchCount !== 0 || comparison.firstMismatch !== null) {
    throw new Error(
      `OPT-0066 tensor ${tensor} layout proof failed: ` +
        JSON.stringify(comparison),
    );
  }
}

function sha256Words(words: Uint16Array): string {
  return aceSha256Hex(new Uint8Array(
    words.buffer,
    words.byteOffset,
    words.byteLength,
  ));
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function product(values: readonly number[]): number {
  return values.reduce((result, value) => result * value, 1);
}
