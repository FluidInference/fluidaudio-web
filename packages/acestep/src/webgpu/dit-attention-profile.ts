export const ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE =
  "ace-dit-fixed32-query8-v1" as const;
export const ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE =
  "opt-0070-fixed32-quad-query32-full-self-production-v1" as const;

export const ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID =
  "fixed32-subgroup-query8" as const;
export const ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID =
  "opt-0062-query8-plus-quad-query32-full-self-v1" as const;
export const ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID =
  "opt-0070-opt0062-query8-plus-quad-query32-full-self-production-v1" as const;

/**
 * Public OPT-0070 shape policy. The measured OPT-0062 physical owner remains
 * exact-M2250-only; every other valid product full-self shape stays on the
 * existing exact query8 owner.
 */
export const ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY = Object.freeze({
  schema: "ace-opt-0070-attention-shape-policy-v1" as const,
  exactQuadQueryTokens: 2_250 as const,
  exactQuadKeyValueTokens: 2_250 as const,
  exactM2250FullSelfOwner: "opt-0062-fixed32-quad-query32" as const,
  otherFullSelfOwner: "fixed32-subgroup-query8" as const,
  slidingSelfAttentionOwner: "fixed32-subgroup-query8" as const,
  crossAttentionOwner: "fixed32-subgroup-query8" as const,
});

export type AceDitAttentionRuntimeProfile =
  | typeof ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE
  | typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
  | typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;

export interface AceDitAttentionRuntimeProfileContract {
  readonly id: AceDitAttentionRuntimeProfile;
  readonly kernelSetId:
    | typeof ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID
    | typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID
    | typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID;
  readonly fullSelfAttentionOwner:
    | "fixed32-subgroup-query8"
    | "opt-0062-fixed32-quad-query32";
  readonly slidingSelfAttentionOwner: "fixed32-subgroup-query8";
  readonly crossAttentionOwner: "fixed32-subgroup-query8";
}

export const ACE_DIT_QUERY8_ATTENTION_PROFILE_CONTRACT:
  Readonly<AceDitAttentionRuntimeProfileContract> = Object.freeze({
    id: ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
    kernelSetId: ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID,
    fullSelfAttentionOwner: "fixed32-subgroup-query8",
    slidingSelfAttentionOwner: "fixed32-subgroup-query8",
    crossAttentionOwner: "fixed32-subgroup-query8",
  });

export const ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT:
  Readonly<AceDitAttentionRuntimeProfileContract> = Object.freeze({
    id: ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    kernelSetId: ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    fullSelfAttentionOwner: "opt-0062-fixed32-quad-query32",
    slidingSelfAttentionOwner: "fixed32-subgroup-query8",
    crossAttentionOwner: "fixed32-subgroup-query8",
  });

/**
 * Public production identity. `fullSelfAttentionOwner` names the exact-M2250
 * owner; `ACE_OPT_0070_DIT_ATTENTION_SHAPE_POLICY` freezes the non-M2250
 * query8 fallback without changing the diagnostic OPT-0062 contract.
 */
export const ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT:
  Readonly<AceDitAttentionRuntimeProfileContract> = Object.freeze({
    id: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    fullSelfAttentionOwner: "opt-0062-fixed32-quad-query32",
    slidingSelfAttentionOwner: "fixed32-subgroup-query8",
    crossAttentionOwner: "fixed32-subgroup-query8",
  });

/** Runtime authentication for the opt-in attention-only profile boundary. */
export function requireAceDitAttentionRuntimeProfile(
  profile: AceDitAttentionRuntimeProfile =
    ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
): Readonly<AceDitAttentionRuntimeProfileContract> {
  if (profile === ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE) {
    return ACE_DIT_QUERY8_ATTENTION_PROFILE_CONTRACT;
  }
  if (profile === ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) {
    return ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT;
  }
  if (profile === ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) {
    return ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT;
  }
  throw new Error("ACE DiT attention runtime profile is not authenticated");
}
