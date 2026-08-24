import { describe, expect, it } from "vitest";

import {
  ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID,
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
  resolveAceDitAttentionKernelSetId,
} from "../src/webgpu/dit-attention-profile.js";

describe("OPT-0088 portable attention kernel-set identity", () => {
  it("pins the portable oracle kernel-set id", () => {
    expect(ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID).toBe(
      "opt-0088-portable-attention-oracle-v1",
    );
  });

  it("returns the portable oracle for every authenticated profile", () => {
    for (const profileId of [
      ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    ] as const) {
      expect(resolveAceDitAttentionKernelSetId(profileId, "portable")).toBe(
        ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
      );
    }
  });

  it("keeps every subgroup profile on its own contract kernel set", () => {
    expect(resolveAceDitAttentionKernelSetId(
      ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      "subgroups",
    )).toBe(ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID);
    expect(resolveAceDitAttentionKernelSetId(
      ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      "subgroups",
    )).toBe(ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID);
    expect(resolveAceDitAttentionKernelSetId(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      "subgroups",
    )).toBe(ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID);
  });

  it("fails closed on unauthenticated profiles or backends", () => {
    expect(() => resolveAceDitAttentionKernelSetId(
      "future-profile" as never,
      "portable",
    )).toThrow(/not authenticated/);
    expect(() => resolveAceDitAttentionKernelSetId(
      "future-profile" as never,
      "subgroups",
    )).toThrow(/not authenticated/);
    expect(() => resolveAceDitAttentionKernelSetId(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      "future-backend" as never,
    )).toThrow(/kernel backend is not authenticated/);
  });

  it("does not mutate the existing OPT-0070 production contract", () => {
    expect(ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT).toEqual({
      id: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      fullSelfAttentionOwner: "opt-0062-fixed32-quad-query32",
      slidingSelfAttentionOwner: "fixed32-subgroup-query8",
      crossAttentionOwner: "fixed32-subgroup-query8",
    });
    expect(Object.isFrozen(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_PROFILE_CONTRACT,
    )).toBe(true);
  });
});
