export const ACE_VAE_C512_WINDOW_RUNTIME_PROFILE =
  "ace-vae-c512-overlap64-v1" as const;
export const ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE =
  "opt-0070-c2378-overlap64-production-v1" as const;
export const ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES = 2_378;
export const ACE_OPT_0070_VAE_C2378_REQUIRED_WORKSPACE_BYTES = 1_168_834_560;

/**
 * Capped production geometry for adapters whose maxBufferSize and
 * maxStorageBufferBindingSize stop at one GiB (every iOS WebGPU adapter).
 * Same kernels, precision maps, and 64-frame overlap-discard seams as the
 * C2378 profile; only the maximum window length shrinks so one workspace
 * buffer (windowFrames x 491,520 bytes) stays under 2^30. OPT-0035's
 * correctness authority proved the chunked decode byte-identical across
 * window geometries at this overlap.
 */
export const ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE =
  "ace-vae-c2176-overlap64-capped-v1" as const;
export const ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES = 2_176;
export const ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES = 1_069_547_520;

/** One latent frame of the widest FP16 activation: 128ch x 1920 x 2 bytes. */
export const ACE_VAE_FP16_WORKSPACE_BYTES_PER_LATENT_FRAME = 491_520;

/**
 * Every iOS WebGPU adapter reports exactly 2^30 bytes for both maxBufferSize
 * and maxStorageBufferBindingSize. An adapter at or below this line is an
 * iPhone-class device whose process-level memory budget is far smaller than
 * desktop; the binding constraint there is not single-buffer bindability but
 * total resident workspace bytes (three whole-window workspaces, so the
 * C2176 geometry holds ~3.2 GB where C512 holds ~755 MB).
 */
export const ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES = 1_073_741_824;

const ACE_VAE_C512_REQUIRED_WORKSPACE_BYTES = 251_658_240;

export type AceVaeWindowRuntimeProfile =
  | typeof ACE_VAE_C512_WINDOW_RUNTIME_PROFILE
  | typeof ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE
  | typeof ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE;

export interface AceVaeWindowRuntimeProfileContract {
  readonly id: AceVaeWindowRuntimeProfile;
  readonly maximumWindowFrames: 512 | 2_176 | 2_378;
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

const ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE_CONTRACT = Object.freeze({
  id: ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
  maximumWindowFrames: ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
  overlapFrames: 64,
  requiredWorkspaceBytes: ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
} as const satisfies AceVaeWindowRuntimeProfileContract);

/** Authenticate the only three product window geometries; never accept a size alone. */
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
  if (
    profile === ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE &&
    maximumWindowFrames ===
      ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES
  ) {
    return ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE_CONTRACT;
  }
  throw new Error("ACE VAE window runtime profile is not authenticated");
}

/**
 * Resolve the window geometry an adapter can actually hold.
 *
 * The configured contract wins whenever the adapter can bind its workspace.
 * A C2378 configuration downgrades on shortfall to an authenticated smaller
 * geometry, chosen by what the adapter's limits betray about the device:
 *
 * - At or below the one-GiB cap (the iOS adapter signature, limits-derived,
 *   never UA-sniffed) the upstream C512 baseline geometry is selected. The
 *   capped C2176 contract would bind (1,069,547,520 <= 2^30) but its three
 *   whole-window workspaces total ~3.2 GB, which iOS jetsam kills; C512's
 *   total ~755 MB. OPT-0035's correctness authority proved the overlap-64
 *   discard decode byte-identical across window geometries, so the smaller
 *   authenticated window is exact by construction.
 * - Above the one-GiB cap but below the C2378 workspace, the capped C2176
 *   contract is kept (desktop-class memory, only bindability is short).
 *
 * Any other shortfall returns the configured contract unchanged so the
 * device request fails closed with its true deficits, and a configured C512
 * contract never changes geometry.
 */
export function selectAceVaeWindowRuntimeProfileForLimits(
  configured: Readonly<AceVaeWindowRuntimeProfileContract>,
  limits: Readonly<{
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
  }>,
): Readonly<AceVaeWindowRuntimeProfileContract> {
  const available = Math.min(
    requireLimitValue(limits.maxBufferSize, "maxBufferSize"),
    requireLimitValue(
      limits.maxStorageBufferBindingSize,
      "maxStorageBufferBindingSize",
    ),
  );
  if (available >= configured.requiredWorkspaceBytes) return configured;
  if (configured.id === ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE) {
    if (
      available <= ACE_VAE_ONE_GIB_CAPPED_ADAPTER_LIMIT_BYTES &&
      available >= ACE_VAE_C512_REQUIRED_WORKSPACE_BYTES
    ) {
      return ACE_VAE_C512_WINDOW_RUNTIME_PROFILE_CONTRACT;
    }
    if (available >= ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES) {
      return ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE_CONTRACT;
    }
  }
  return configured;
}

function requireLimitValue(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `ACE VAE window selection requires a positive integer ${name}`,
    );
  }
  return value;
}
