import { ACE_TURBO_V1_CORRECTNESS_PROFILE } from "../api.js";
import { aceSha256Hex } from "../model/sha256.js";

export const ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID =
  "ace-turbo-shift3-euler-8-exact-v1" as const;
export const ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID =
  "opt-0055-shift3-euler-6-diagnostic-v1" as const;
export const ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID =
  "opt-0065-shift3-euler-5-diagnostic-v1" as const;

export const ACE_DIT_SAMPLER_SCHEDULE_PROFILE_IDS = Object.freeze([
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
] as const);

export type AceDitSamplerScheduleProfileId =
  (typeof ACE_DIT_SAMPLER_SCHEDULE_PROFILE_IDS)[number];
export type AceDitSamplerEvaluationCount = 5 | 6 | 8;

export interface AceDitSamplerScheduleProfile {
  readonly schema: "ace-dit-sampler-schedule-contract-v1";
  readonly id: AceDitSamplerScheduleProfileId;
  readonly role: "production-default" | "diagnostic-only";
  readonly experimentId: null | "OPT-0055" | "OPT-0065";
  readonly evaluationCount: AceDitSamplerEvaluationCount;
  /** Shift-3 values before upstream materializes the schedule as BF16. */
  readonly declaredTimesteps: readonly number[];
  /** Exact JS numbers represented by upstream's BF16 schedule tensor. */
  readonly effectiveBfloat16Timesteps: readonly number[];
  /** Exact BF16 Euler `dt`, with final direct-x0 coefficient last. */
  readonly effectiveBfloat16UpdateCoefficients: readonly number[];
  /** SHA-256 of the ordered contract fields above, excluding this digest. */
  readonly contractSha256: string;
}

type SamplerContractPayload = Omit<
  AceDitSamplerScheduleProfile,
  "contractSha256"
>;

function freezeProfile(
  payload: SamplerContractPayload,
  contractSha256: string,
): Readonly<AceDitSamplerScheduleProfile> {
  return Object.freeze({
    ...payload,
    declaredTimesteps: Object.freeze([...payload.declaredTimesteps]),
    effectiveBfloat16Timesteps: Object.freeze([
      ...payload.effectiveBfloat16Timesteps,
    ]),
    effectiveBfloat16UpdateCoefficients: Object.freeze([
      ...payload.effectiveBfloat16UpdateCoefficients,
    ]),
    contractSha256,
  });
}

const EIGHT_PAYLOAD: SamplerContractPayload = Object.freeze({
  schema: "ace-dit-sampler-schedule-contract-v1",
  id: ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
  role: "production-default",
  experimentId: null,
  evaluationCount: 8,
  declaredTimesteps: ACE_TURBO_V1_CORRECTNESS_PROFILE.schedulerTimesteps,
  effectiveBfloat16Timesteps:
    ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16,
  effectiveBfloat16UpdateCoefficients: Object.freeze([
    0.046875,
    0.0546875,
    0.06640625,
    0.08203125,
    0.10546875,
    0.14453125,
    0.19921875,
    0.30078125,
  ]),
});

const SIX_PAYLOAD: SamplerContractPayload = Object.freeze({
  schema: "ace-dit-sampler-schedule-contract-v1",
  id: ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  role: "diagnostic-only",
  experimentId: "OPT-0055",
  evaluationCount: 6,
  declaredTimesteps: Object.freeze([
    1,
    0.9375,
    0.8571428571428571,
    0.75,
    0.6,
    0.375,
  ]),
  effectiveBfloat16Timesteps: Object.freeze([
    1,
    0.9375,
    0.85546875,
    0.75,
    0.6015625,
    0.375,
  ]),
  effectiveBfloat16UpdateCoefficients: Object.freeze([
    0.0625,
    0.08203125,
    0.10546875,
    0.1484375,
    0.2265625,
    0.375,
  ]),
});

const FIVE_PAYLOAD: SamplerContractPayload = Object.freeze({
  schema: "ace-dit-sampler-schedule-contract-v1",
  id: ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
  role: "diagnostic-only",
  experimentId: "OPT-0065",
  evaluationCount: 5,
  declaredTimesteps: Object.freeze([
    1,
    0.9230769230769231,
    0.8181818181818182,
    0.6666666666666666,
    0.42857142857142855,
  ]),
  effectiveBfloat16Timesteps: Object.freeze([
    1,
    0.921875,
    0.81640625,
    0.66796875,
    0.427734375,
  ]),
  effectiveBfloat16UpdateCoefficients: Object.freeze([
    0.078125,
    0.10546875,
    0.1484375,
    0.240234375,
    0.427734375,
  ]),
});

export const ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE = freezeProfile(
  EIGHT_PAYLOAD,
  "6e81531b45fff521fb1ea5f454cf1b9fd7559b7c8be0d78b7023ac4b6b9ddbd2",
);
export const ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE = freezeProfile(
  SIX_PAYLOAD,
  "c2d8a31df4d0023bb85b5a72f8eca3b781f006ddf763689c750622b231291bf0",
);
export const ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE = freezeProfile(
  FIVE_PAYLOAD,
  "a585b8bf567bd5cca30bd89bf60709afa6c72f2ef09d9c1015d2f83db7198324",
);

const PROFILE_BY_ID: Readonly<Record<
  AceDitSamplerScheduleProfileId,
  Readonly<AceDitSamplerScheduleProfile>
>> = Object.freeze({
  [ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID]:
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  [ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID]:
    ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE,
  [ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID]:
    ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE,
});

/** Resolve only one of the three registered, hash-authenticated schedules. */
export function resolveAceDitSamplerScheduleProfile(
  id: AceDitSamplerScheduleProfileId =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
): Readonly<AceDitSamplerScheduleProfile> {
  const profile = PROFILE_BY_ID[id];
  if (profile === undefined) {
    throw new TypeError(`ACE DiT sampler schedule ${String(id)} is not registered`);
  }
  authenticateProfile(profile);
  return profile;
}

/** Require the canonical object returned by the resolver, not a caller clone. */
export function requireResolvedAceDitSamplerScheduleProfile(
  value: Readonly<AceDitSamplerScheduleProfile>,
): Readonly<AceDitSamplerScheduleProfile> {
  const profile = PROFILE_BY_ID[value?.id];
  if (profile === undefined || profile !== value) {
    throw new Error(
      "ACE DiT sampler schedule must be the authenticated resolver object",
    );
  }
  authenticateProfile(profile);
  return profile;
}

function authenticateProfile(profile: AceDitSamplerScheduleProfile): void {
  const {
    contractSha256,
    schema,
    id,
    role,
    experimentId,
    evaluationCount,
    declaredTimesteps,
    effectiveBfloat16Timesteps,
    effectiveBfloat16UpdateCoefficients,
  } = profile;
  if (
    declaredTimesteps.length !== evaluationCount ||
    effectiveBfloat16Timesteps.length !== evaluationCount ||
    effectiveBfloat16UpdateCoefficients.length !== evaluationCount
  ) {
    throw new Error(`ACE DiT sampler schedule ${id} has inconsistent lengths`);
  }
  const payload: SamplerContractPayload = {
    schema,
    id,
    role,
    experimentId,
    evaluationCount,
    declaredTimesteps,
    effectiveBfloat16Timesteps,
    effectiveBfloat16UpdateCoefficients,
  };
  const actual = aceSha256Hex(
    new TextEncoder().encode(JSON.stringify(payload)),
  );
  if (actual !== contractSha256) {
    throw new Error(`ACE DiT sampler schedule ${id} failed authentication`);
  }
}
