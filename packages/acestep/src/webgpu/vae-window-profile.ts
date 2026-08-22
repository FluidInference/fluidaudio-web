export const ACE_VAE_C512_WINDOW_RUNTIME_PROFILE =
  "ace-vae-c512-overlap64-v1" as const;
export const ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE =
  "opt-0070-c2378-overlap64-production-v1" as const;
export const ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES = 2_378;
export const ACE_OPT_0070_VAE_C2378_REQUIRED_WORKSPACE_BYTES = 1_168_834_560;

const ACE_VAE_C512_REQUIRED_WORKSPACE_BYTES = 251_658_240;

export type AceVaeWindowRuntimeProfile =
  | typeof ACE_VAE_C512_WINDOW_RUNTIME_PROFILE
  | typeof ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE;

export interface AceVaeWindowRuntimeProfileContract {
  readonly id: AceVaeWindowRuntimeProfile;
  readonly maximumWindowFrames: 512 | 2_378;
  readonly overlapFrames: 64;
  readonly requiredWorkspaceBytes: number;
}

const ACE_VAE_C512_WINDOW_RUNTIME_PROFILE_CONTRACT = Object.freeze({
  id: ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
  maximumWindowFrames: 512,
  overlapFrames: 64,
  requiredWorkspaceBytes: ACE_VAE_C512_REQUIRED_WORKSPACE_BYTES,
} as const satisfies AceVaeWindowRuntimeProfileContract);

const ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE_CONTRACT = Object.freeze({
  id: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  maximumWindowFrames: ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
  overlapFrames: 64,
  requiredWorkspaceBytes: ACE_OPT_0070_VAE_C2378_REQUIRED_WORKSPACE_BYTES,
} as const satisfies AceVaeWindowRuntimeProfileContract);

/** Authenticate the only two product window geometries; never accept a size alone. */
export function requireAceVaeWindowRuntimeProfile(
  profile: AceVaeWindowRuntimeProfile | undefined,
  maximumWindowFrames: number,
): Readonly<AceVaeWindowRuntimeProfileContract> {
  if (
    (profile === undefined || profile === ACE_VAE_C512_WINDOW_RUNTIME_PROFILE) &&
    maximumWindowFrames === 512
  ) {
    return ACE_VAE_C512_WINDOW_RUNTIME_PROFILE_CONTRACT;
  }
  if (
    profile === ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE &&
    maximumWindowFrames ===
      ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES
  ) {
    return ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE_CONTRACT;
  }
  throw new Error("ACE VAE window runtime profile is not authenticated");
}
