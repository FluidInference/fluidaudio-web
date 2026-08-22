/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  DEFAULT_ACE_PLANNER_CONFIGURATION,
  type AceGenerationRequest,
  type AceGenerationResult,
} from "../../src/api.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import { compareAceRawAudioSnapshots } from
  "../../src/runtime/audio-output.js";
import { canonicalizeSeed } from "../../src/runtime/seed.js";
import type { AceWorkerConfiguration } from
  "../../src/runtime/protocol.js";
import {
  createAceWebGpuPipelineBackend,
  type AceSamplerScheduleDiagnosticEvidence,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
  type AceDitSamplerScheduleProfileId,
} from "../../src/webgpu/dit-sampler-profile.js";
import { ACE_VAE_C512_WINDOW_RUNTIME_PROFILE } from
  "../../src/webgpu/vae-window-profile.js";

const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
const DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f";
const VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949";
const SCHEDULES = Object.freeze([
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
] as const);
const BLIND_LABELS = Object.freeze(["A", "B", "C"] as const);

type ListeningMode = "direct12" | "vocal30";

interface StartMessage {
  readonly type: "start";
  readonly mode: ListeningMode;
}

interface RevealMessage {
  readonly type: "reveal";
}

interface ReleaseMessage {
  readonly type: "release";
}

type IncomingMessage = StartMessage | RevealMessage | ReleaseMessage;

interface CompletedArm {
  readonly result: AceGenerationResult;
  readonly evidence: AceSamplerScheduleDiagnosticEvidence;
}

let state: "idle" | "running" | "complete" | "revealed" | "released" =
  "idle";
let backend: ReturnType<typeof createAceWebGpuPipelineBackend> | undefined;
let completedArms: readonly CompletedArm[] = [];
let blindMapping: readonly Readonly<{
  readonly label: (typeof BLIND_LABELS)[number];
  readonly scheduleProfileId: AceDitSamplerScheduleProfileId;
}>[] = [];
let detailedReceipt: Readonly<Record<string, unknown>> | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "start" && state === "idle") {
    state = "running";
    void run(message.mode).catch(fail);
    return;
  }
  if (message.type === "reveal" && state === "complete") {
    state = "revealed";
    self.postMessage({
      type: "mapping",
      mapping: blindMapping,
      receipt: detailedReceipt,
    });
    return;
  }
  if (message.type === "release" &&
    (state === "complete" || state === "revealed")) {
    void release().catch(fail);
  }
});

async function run(mode: ListeningMode): Promise<void> {
  if (mode !== "direct12" && mode !== "vocal30") {
    throw new TypeError("OPT-0055/0065 listening mode is invalid");
  }
  const activeBackend = createAceWebGpuPipelineBackend();
  backend = activeBackend;
  const controller = new AbortController();
  postProgress("authenticating warm packages and initializing query8/rev7");
  await activeBackend.initialize(configuration(), {
    modelSource: "cache-only",
    signal: controller.signal,
    onProgress: () => undefined,
    onDiagnostic: failOnDiagnostic,
  });
  const request = requestFor(mode);
  const arms: CompletedArm[] = [];
  for (const [index, scheduleProfileId] of SCHEDULES.entries()) {
    postProgress(
      `running untimed arm ${index + 1}/3 (${scheduleProfileId})`,
    );
    let evidence: AceSamplerScheduleDiagnosticEvidence | undefined;
    const result = await activeBackend.generate(request, {
      signal: controller.signal,
      onProgress: () => undefined,
      onDiagnostic: failOnDiagnostic,
      opt0055Opt0065SamplerRun: {
        scheduleProfileId,
        onEvidence: (value) => {
          if (evidence !== undefined) {
            throw new Error("Sampler arm emitted duplicate evidence");
          }
          evidence = value;
        },
      },
    });
    if (evidence === undefined) {
      throw new Error("Sampler arm omitted its diagnostic evidence");
    }
    arms.push(Object.freeze({ result, evidence }));
  }
  completedArms = Object.freeze(arms);
  postProgress("checking exact eval0 identity and bounded raw-waveform drift");
  const baseline = arms[0]!.evidence;
  const six = arms[1]!.evidence;
  const five = arms[2]!.evidence;
  const eval0Six = compareExactTensor(
    baseline.evaluation0Velocity.values,
    six.evaluation0Velocity.values,
  );
  const eval0Five = compareExactTensor(
    baseline.evaluation0Velocity.values,
    five.evaluation0Velocity.values,
  );
  if (eval0Six.mismatchCount !== 0 || eval0Five.mismatchCount !== 0) {
    throw new Error("OPT-0055/0065 evaluation-0 denoiser identity failed");
  }
  const [rawSix, rawFive] = await Promise.all([
    compareAceRawAudioSnapshots(
      baseline.rawAudio.snapshot,
      six.rawAudio.snapshot,
      { signal: controller.signal },
    ),
    compareAceRawAudioSnapshots(
      baseline.rawAudio.snapshot,
      five.rawAudio.snapshot,
      { signal: controller.signal },
    ),
  ]);
  const wavHashes = await Promise.all(arms.map(async ({ evidence }) =>
    await hashBlob(evidence.wav)
  ));
  const permutation = shuffledIndices();
  blindMapping = Object.freeze(permutation.map((armIndex, labelIndex) =>
    Object.freeze({
      label: BLIND_LABELS[labelIndex]!,
      scheduleProfileId: SCHEDULES[armIndex]!,
    })
  ));
  const mappingCommitmentSha256 = hashText(JSON.stringify(blindMapping));
  detailedReceipt = Object.freeze({
    schema: "ace-opt-0055-0065-sampler-schedule-listening-v1",
    experimentIds: Object.freeze(["OPT-0055", "OPT-0065"]),
    status: "correctness-passed-listening-pending",
    mode,
    formalTimingCaptured: false,
    executionOrder: SCHEDULES,
    evaluation0Velocity: Object.freeze({
      eightVsSix: eval0Six,
      eightVsFive: eval0Five,
      exactIdentityRequired: true,
    }),
    arms: Object.freeze(arms.map(({ evidence }, index) => Object.freeze({
      scheduleProfileId: evidence.schedule.id,
      evaluationCount: evidence.schedule.evaluationCount,
      contractSha256: evidence.schedule.contractSha256,
      evaluation0Velocity: tensorReceipt(evidence.evaluation0Velocity),
      evaluations: Object.freeze(evidence.evaluations.map((value) =>
        Object.freeze({ evaluation: value.evaluation, ...tensorReceipt(value) })
      )),
      finalLatent: tensorReceipt(evidence.finalLatent),
      rawAudio: Object.freeze({
        byteLength: evidence.rawAudio.byteLength,
        peak: evidence.rawAudio.peak,
        finiteSamples: evidence.rawAudio.finiteSamples,
        interleavedSamples: evidence.rawAudio.interleavedSamples,
      }),
      wav: Object.freeze({
        byteLength: evidence.wav.size,
        sha256: wavHashes[index],
      }),
    }))),
    rawAudioComparisons: Object.freeze({ eightVsSix: rawSix, eightVsFive: rawFive }),
    blindMappingCommitmentSha256: mappingCommitmentSha256,
  });
  const artifacts = await Promise.all(permutation.map(async (armIndex, labelIndex) =>
    Object.freeze({
      label: BLIND_LABELS[labelIndex]!,
      fileName: `opt-0055-0065-${mode}-blind-${BLIND_LABELS[labelIndex]!.toLowerCase()}.wav`,
      wav: arms[armIndex]!.evidence.wav,
      byteLength: arms[armIndex]!.evidence.wav.size,
      sha256: wavHashes[armIndex]!,
    })
  ));
  await activeBackend.dispose();
  state = "complete";
  self.postMessage({
    type: "complete",
    publicReceipt: Object.freeze({
      schema: "ace-opt-0055-0065-blinded-listening-ready-v1",
      status: "blinded-listening-ready",
      mode,
      formalTimingCaptured: false,
      evaluation0ExactIdentityPassed: true,
      mappingCommitmentSha256,
      artifactCount: artifacts.length,
    }),
    artifacts,
  });
}

async function release(): Promise<void> {
  if (backend === undefined) throw new Error("Sampler backend is absent");
  await Promise.all(completedArms.map(async ({ result }) =>
    await backend!.releaseResult(result)
  ));
  completedArms = [];
  backend = undefined;
  state = "released";
  self.postMessage({ type: "released" });
}

function configuration(): AceWorkerConfiguration {
  return Object.freeze({
    manifestUrl: new URL("/model/files-reference/manifest.json", self.location.href).href,
    manifestSha256: MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-dit-rev7-oracle/manifest.json",
        self.location.href,
      ).href,
      manifestSha256: DENSE_MANIFEST_SHA256,
      runtimeProfile: "opt-0009-fp16-fp32-dense-v1",
    }),
    // Query8 is intentionally represented by omission. OPT62/70 physical quad
    // ownership is forbidden for this first quality gate.
    vaePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-vae-experimental/manifest.json",
        self.location.href,
      ).href,
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      windowRuntimeProfile: ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
      maxWindowFrames: 512,
    }),
  });
}

function requestFor(mode: ListeningMode): AceGenerationRequest {
  if (mode === "vocal30") {
    return Object.freeze({
      generationProfile: "ace-turbo-v1-correctness",
      prompt:
        "Intimate indie electronic pop, warm analog bass, crisp restrained drums, luminous synths, expressive close vocal, detailed stereo production.",
      lyrics:
        "[Verse]\nCity lights are folding into blue\nI keep a quiet signal back to you\n\n[Chorus]\nStay in the echo, stay for the dawn\nCarry the rhythm when the night is gone",
      instrumental: false,
      durationSeconds: 30,
      seed: canonicalizeSeed("0000000000c0ffee"),
      planner: DEFAULT_ACE_PLANNER_CONFIGURATION,
    });
  }
  return Object.freeze({
    generationProfile: "ace-turbo-v1-correctness",
    prompt:
      "Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.",
    lyrics: "",
    instrumental: true,
    durationSeconds: 12,
    seed: canonicalizeSeed("0000000000c0ffee"),
    planner: Object.freeze({ mode: "disabled" as const }),
    metadata: Object.freeze({
      bpm: 104,
      keyScale: "D minor",
      timeSignature: "4",
    }),
  });
}

function compareExactTensor(
  left: Float32Array<ArrayBuffer>,
  right: Float32Array<ArrayBuffer>,
): Readonly<{ readonly elementCount: number; readonly mismatchCount: number }> {
  if (left.length !== right.length) {
    throw new Error("Sampler eval0 tensor lengths diverged");
  }
  const leftWords = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const rightWords = new Uint32Array(right.buffer, right.byteOffset, right.length);
  let mismatchCount = 0;
  for (let index = 0; index < leftWords.length; index += 1) {
    if (leftWords[index] !== rightWords[index]) mismatchCount += 1;
  }
  return Object.freeze({ elementCount: left.length, mismatchCount });
}

function tensorReceipt(
  value: AceSamplerScheduleDiagnosticEvidence["finalLatent"],
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    byteLength: value.byteLength,
    elementCount: value.elementCount,
    sha256: value.sha256,
    nonFiniteCount: value.nonFiniteCount,
    nonzeroCount: value.nonzeroCount,
    maximumAbsolute: value.maximumAbsolute,
  });
}

function shuffledIndices(): readonly number[] {
  const values = [0, 1, 2];
  const random = new Uint32Array(2);
  crypto.getRandomValues(random);
  for (let index = values.length - 1, draw = 0; index > 0; index -= 1, draw += 1) {
    const selected = random[draw]! % (index + 1);
    [values[index], values[selected]] = [values[selected]!, values[index]!];
  }
  return Object.freeze(values);
}

async function hashBlob(blob: Blob): Promise<string> {
  const hash = new AceIncrementalSha256();
  const reader = blob.stream().getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return hash.digestHex();
      hash.update(value);
    }
  } finally {
    reader.releaseLock();
  }
}

function hashText(value: string): string {
  return new AceIncrementalSha256()
    .update(new TextEncoder().encode(value))
    .digestHex();
}

function failOnDiagnostic(diagnostic: { readonly severity: string; readonly code: string }): void {
  if (diagnostic.severity === "error") {
    throw new Error(`ACE diagnostic ${diagnostic.code}`);
  }
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

async function fail(error: unknown): Promise<void> {
  state = "released";
  try {
    await Promise.all(completedArms.map(async ({ result }) =>
      await backend?.releaseResult(result)
    ));
    await backend?.dispose();
  } catch {
    // Preserve the primary correctness/listening harness failure.
  }
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      name: error instanceof Error ? error.name : "Error",
      message: error instanceof Error ? error.message : String(error),
      ...(error instanceof Error && error.stack !== undefined
        ? { stack: error.stack }
        : {}),
    }),
  });
}
