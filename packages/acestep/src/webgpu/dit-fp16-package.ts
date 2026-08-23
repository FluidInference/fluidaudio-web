import {
  ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
  type AcePackageManifest,
} from "../model/manifest.js";
import type { AceLoadedPackageManifest } from "../model/package.js";

export const ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES = 254_357 as const;
export const ACE_OPT_0009_DIT_DENSE_CONVERTER_REVISION = 7 as const;
export const ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE =
  "opt-0009-fp16-fp32-dense-v1" as const;
export const ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID =
  "opt-0009-n256-k32-fp16-fp32-v1" as const;
export const ACE_OPT_0009_DIT_MIXED_LAYER_BYTES = 3_020_808_192 as const;
export const ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES = 130_109_696 as const;
export const ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES =
  3_150_917_888 as const;
export const ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES = Object.freeze(
  Array.from({ length: 24 }, (_, layer) => [
    `weights/dit/layer-${String(layer).padStart(2, "0")}-00.bin`,
    `weights/dit/layer-${String(layer).padStart(2, "0")}-01.bin`,
  ]).flat(),
);

export const ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 =
  "a2f70c123fb7c4dbc3b51be68b4b494107c13b575ad2bed68c639791c93574d1" as const;
export const ACE_OPT_0037_DIT_K4_MANIFEST_BYTES = 257_789 as const;
export const ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE =
  "opt-0037-k4-fp16-partials-v1" as const;
export const ACE_OPT_0037_DIT_K4_KERNEL_SET_ID =
  "opt-0037-opt-0032-k4-partials-fixed32-v1" as const;
export const ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE =
  "opt-0056-selective-k4-exact-down-v1" as const;
export const ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID =
  "opt-0056-opt0032-k4-plus-exact-down-fixed32-v1" as const;
export const ACE_OPT_0037_DIT_K4_LAYER_BYTES = 3_020_808_192 as const;
export const ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES =
  3_150_917_888 as const;
export const ACE_OPT_0037_DIT_K4_WEIGHT_FILES = Object.freeze(
  [...ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES],
);

/**
 * OPT-0088: kernel-set identity for serving the unchanged OPT-0009 rev7
 * dense package (`ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE`, tile layout
 * unchanged) through the portable workgroup-memory dense kernel on devices
 * without WebGPU subgroups. This is a kernel identity, not a package
 * identity: the manifest, layout, and runtime profile stay OPT-0009's.
 */
export const ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID =
  "opt-0088-dense-fp16-fp32-portable-v1" as const;

export type AceDitDenseRuntimeProfile =
  | typeof ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE
  | typeof ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE
  | typeof ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;

const REFERENCE_DIT_LAYER_WEIGHT_FILE =
  /^weights\/dit\/layer-(?:0[0-9]|1[0-9]|2[0-3])-0[01]\.bin$/;

export function isAceReferenceDitLayerWeightFile(name: string): boolean {
  return REFERENCE_DIT_LAYER_WEIGHT_FILE.test(name);
}

/** Derived view of an authenticated reference manifest for its two shared shards. */
export function createAceReferenceDitSharedManifestView(
  manifest: AcePackageManifest,
): AcePackageManifest {
  if (manifest.profile !== "reference") {
    throw new Error("ACE mixed DiT shared view requires the reference package");
  }
  const tensors = Object.freeze(Object.fromEntries(
    Object.entries(manifest.tensors).filter(([name, tensor]) =>
      tensor.phase === "dit" && !name.startsWith("ace.decoder.layers.")
    ),
  ));
  const shardNames = new Set(Object.values(tensors).map((tensor) => tensor.shard));
  const files = Object.freeze(
    manifest.files.filter((file) => shardNames.has(file.name)),
  );
  if (
    Object.keys(tensors).length !== 20 ||
    files.length !== 2 ||
    files.some((file) => file.kind !== "weights") ||
    files.reduce((sum, file) => sum + file.byteLength, 0) !==
      ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES
  ) {
    throw new Error("ACE reference DiT shared physical inventory changed");
  }
  return Object.freeze({ ...manifest, files, tensors });
}

export function requireAceOpt0009DitDensePackageIdentity(
  loaded: AceLoadedPackageManifest,
): void {
  const manifest = loaded.manifest;
  const weightFiles = manifest.files.filter((file) => file.kind === "weights");
  if (
    loaded.manifestSha256 !== ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES ||
    manifest.profile !== "fp16-dit-dense-experimental" ||
    manifest.provenance.converterRevision !==
      ACE_OPT_0009_DIT_DENSE_CONVERTER_REVISION ||
    weightFiles.length !== ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES.length ||
    ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES.some((name, index) =>
      weightFiles[index]?.name !== name ||
      weightFiles[index]?.byteLength !==
        (index % 2 === 0 ? 121_668_608 : 4_198_400)
    ) ||
    weightFiles.reduce((sum, file) => sum + file.byteLength, 0) !==
      ACE_OPT_0009_DIT_MIXED_LAYER_BYTES ||
    ACE_OPT_0009_DIT_MIXED_LAYER_BYTES !==
      ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES
  ) {
    throw new Error("OPT-0009 mixed DiT layer package identity changed");
  }
}

export function requireAceOpt0037DitK4PackageIdentity(
  loaded: AceLoadedPackageManifest,
): void {
  const manifest = loaded.manifest;
  const weightFiles = manifest.files.filter((file) => file.kind === "weights");
  if (
    loaded.manifestSha256 !== ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 ||
    loaded.manifestByteLength !== ACE_OPT_0037_DIT_K4_MANIFEST_BYTES ||
    manifest.profile !== "fp16-dit-dense-experimental" ||
    manifest.provenance.converterRevision !==
      ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION ||
    weightFiles.length !== ACE_OPT_0037_DIT_K4_WEIGHT_FILES.length ||
    ACE_OPT_0037_DIT_K4_WEIGHT_FILES.some((name, index) =>
      weightFiles[index]?.name !== name ||
      weightFiles[index]?.byteLength !==
        (index % 2 === 0 ? 121_668_608 : 4_198_400)
    ) ||
    weightFiles.reduce((sum, file) => sum + file.byteLength, 0) !==
      ACE_OPT_0037_DIT_K4_LAYER_BYTES ||
    ACE_OPT_0037_DIT_K4_LAYER_BYTES !==
      ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES
  ) {
    throw new Error(
      "OPT-0037 rev8 K4 DiT package failed authenticated identity validation",
    );
  }
}
