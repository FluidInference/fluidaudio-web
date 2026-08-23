import { aceSeed, type AceGenerationRequest } from "../../src/api.js";

export const OPT_0080_PRODUCT_DURATION_SECONDS = 96 as const;
export const OPT_0080_PRODUCT_LATENT_FRAMES = 2_400 as const;
export const OPT_0080_PRODUCT_AUDIO_FRAMES = 4_608_000 as const;
export const OPT_0080_PRODUCT_RAW_BYTES = 36_864_000 as const;
export const OPT_0080_PRODUCT_WAV_BYTES = 36_864_044 as const;
export const OPT_0080_PRODUCT_WINDOW_COUNT = 2 as const;
export const OPT_0080_PRODUCT_STITCH_SEAM_LATENT_FRAME = 2_250 as const;
export const OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME = 4_320_000 as const;
export const OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES = 64 as const;
export const OPT_0080_PRODUCT_REQUEST_BYTES = 365 as const;
export const OPT_0080_PRODUCT_REQUEST_SHA256 =
  "ecc1d8d0fd7a87e14d0cf827563280fe35853526368becf883d98f4d42cb1ad4" as const;

export const OPT_0080_PRODUCT_MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const OPT_0080_PRODUCT_DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0080_PRODUCT_VAE_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7" as const;

export const OPT_0080_PRODUCT_REQUEST: AceGenerationRequest = Object.freeze({
  generationProfile: "ace-turbo-v1-correctness",
  prompt:
    "Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.",
  lyrics: "",
  instrumental: true,
  durationSeconds: OPT_0080_PRODUCT_DURATION_SECONDS,
  seed: aceSeed("c0ffee"),
  planner: Object.freeze({ mode: "disabled" as const }),
  metadata: Object.freeze({
    bpm: 104,
    keyScale: "D minor",
    timeSignature: "4",
  }),
});

export const OPT_0080_PRODUCT_ARM_ORDER = Object.freeze([
  Object.freeze({
    id: "control" as const,
    submissionPolicyOverride: "depth1-epoch1" as const,
    effectiveSubmissionPolicy: "depth1-epoch1" as const,
  }),
  Object.freeze({
    id: "candidate" as const,
    submissionPolicyOverride: "depth2-phase-epoch4" as const,
    effectiveSubmissionPolicy: "depth2-phase-epoch4" as const,
  }),
  Object.freeze({
    id: "production" as const,
    submissionPolicyOverride: undefined,
    effectiveSubmissionPolicy: "depth2-phase-epoch4" as const,
  }),
]);

export type Opt0080ProductArm =
  (typeof OPT_0080_PRODUCT_ARM_ORDER)[number];
export type Opt0080ProductArmId = Opt0080ProductArm["id"];
