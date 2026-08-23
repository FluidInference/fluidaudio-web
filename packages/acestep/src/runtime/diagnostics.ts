import type {
  AceExecutionProfile,
  AceSchedulingProfile,
  AceWebGpuCapabilityReport,
} from "../webgpu/capabilities.js";
import type { AceGenerationStage } from "./stages.js";

export const ACE_SOURCE_REVISION =
  "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0" as const;
export const ACE_PLANNER_SOURCE_REVISION =
  "148d8ea0225bdab342ee1ae3a354275ccd60ca80" as const;
export const PARAKEET_REFERENCE_REVISION =
  "7ee112738262a6f5a0efd2f150748a4087432fbb" as const;

export type AceDiagnosticSeverity = "info" | "warning" | "error";

export type AceDiagnosticValue =
  | string
  | number
  | boolean
  | null
  | readonly (string | number | boolean | null)[];

export interface AceDiagnostic {
  readonly severity: AceDiagnosticSeverity;
  readonly code: string;
  readonly message: string;
  readonly elapsedMs: number;
  readonly stage?: AceGenerationStage;
  readonly details?: Readonly<Record<string, AceDiagnosticValue>>;
}

/** Captured with each result and actionable worker error. */
export interface AceRuntimeDiagnostics {
  readonly backend: "custom-webgpu-wgsl-and-wasm";
  readonly modelManifestId: string;
  readonly modelManifestUrl: string;
  /** Digest pinned by the caller and verified over the received manifest bytes. */
  readonly modelManifestSha256: string;
  readonly ditDenseManifestId: string;
  readonly ditDenseManifestUrl: string;
  readonly ditDenseManifestSha256: string;
  readonly ditDenseManifestByteLength: 254_357 | 257_789;
  readonly ditDenseRuntimeProfile:
    | "opt-0009-fp16-fp32-dense-v1"
    | "opt-0037-k4-fp16-partials-v1"
    | "opt-0056-selective-k4-exact-down-v1";
  readonly ditDenseKernelSetId:
    | "opt-0009-n256-k32-fp16-fp32-v1"
    | "opt-0037-opt-0032-k4-partials-fixed32-v1"
    | "opt-0056-opt0032-k4-plus-exact-down-fixed32-v1";
  /** Present for the explicit diagnostic or exact production quad profile. */
  readonly ditAttentionRuntimeProfile?:
    | "opt-0062-fixed32-quad-query32-full-self-v1"
    | "opt-0070-fixed32-quad-query32-full-self-production-v1";
  readonly ditAttentionKernelSetId?:
    | "opt-0062-query8-plus-quad-query32-full-self-v1"
    | "opt-0070-opt0062-query8-plus-quad-query32-full-self-production-v1";
  readonly ditDenseLayerBytes: 3_020_808_192;
  readonly ditResidentWeightBytes: 3_150_917_888;
  readonly vaeManifestId: string;
  readonly vaeManifestUrl: string;
  readonly vaeManifestSha256: string;
  readonly vaeManifestByteLength: 715_301 | 716_185;
  readonly vaeRuntimeProfile:
    | "opt-0028-mixed-fp16-fixed32-exact-packed-v1"
    | "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1"
    | "opt-0054-mixed-fp16-fixed32-revision7-v1"
    | "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1";
  readonly vaeKernelSetId:
    | "opt-0028-vae-fp16-fixed32-exact-packed-kernel-set-v1"
    | "opt-0040-vae-fp16-fixed32-exact-packed-shape-selected-kernel-set-v1"
    | "opt-0054-vae-fp16-fixed32-revision7-k4-shape-selected-kernel-set-v1"
    | "opt-0066-vae-fp16-fixed32-dual-k4-quality-kernel-set-v1";
  readonly vaePrecisionMapSha256: string;
  readonly vaeWindowRuntimeProfile:
    | "ace-vae-c512-overlap64-v1"
    | "opt-0070-c2378-overlap64-production-v1";
  readonly vaeMaxWindowFrames: 512 | 2_378;
  readonly executionProfile: AceExecutionProfile;
  readonly schedulingProfile: AceSchedulingProfile;
  readonly capabilities: AceWebGpuCapabilityReport;
  readonly aceSourceRevision: typeof ACE_SOURCE_REVISION;
  readonly plannerSourceRevision: typeof ACE_PLANNER_SOURCE_REVISION;
  readonly parakeetReferenceRevision: typeof PARAKEET_REFERENCE_REVISION;
}

export interface AceFailureContext {
  readonly stage?: AceGenerationStage;
  readonly operation?: string;
  readonly resourceLabel?: string;
  readonly requestedBytes?: number;
  readonly modelManifestId?: string;
  readonly executionProfileId?: string;
}
