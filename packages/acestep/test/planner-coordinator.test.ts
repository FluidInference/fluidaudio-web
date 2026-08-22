import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeAll, afterEach, describe, expect, it, vi } from "vitest";

import type { AceGenerationRequest } from "../src/api.js";
import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";
import {
  ACE_PLANNER_COORDINATOR_CONTRACT,
  ACE_PLANNER_COT_MAX_NEW_TOKENS,
  runAcePlannerCoordinator,
  type AcePlannerCoordinatorProgress,
} from "../src/runtime/planner-coordinator.js";
import type {
  AcePlannerDecodeBatch,
  AcePlannerGraphExecutor,
  AcePlannerLogitRange,
  AcePlannerPrefillBatch,
} from "../src/runtime/planner.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import { loadPinnedAceTokenizer } from "../src/tokenizer/loader.js";
import {
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_IM_END_TOKEN_ID,
  type AceQwenBpeTokenizer,
} from "../src/tokenizer/qwen-bpe.js";
import {
  AcePlannerGpuExecutor,
  type AcePlannerGpuExecutorProgress,
} from "../src/webgpu/planner-executor.js";

const PLANNER_ASSET_ROOT = resolve("model/files-reference/assets/planner");
const VOCABULARY_SIZE = 217_204;
const METADATA_FSM_VECTOR = JSON.parse(
  readFileSync(resolve("test/planner-metadata-fsm-vectors.json"), "utf8"),
) as Readonly<{
  readonly injectedTrace: Readonly<{
    readonly emittedTokenIds: readonly number[];
    readonly decoded: string;
  }>;
}>;

describe("ACE production planner coordinator", () => {
  let tokenizer: AceQwenBpeTokenizer;

  beforeAll(async () => {
    tokenizer = (await loadPinnedAceTokenizer("planner", {
      tokenizerJson: readFileSync(resolve(PLANNER_ASSET_ROOT, "tokenizer.json"), "utf8"),
      tokenizerConfigJson: readFileSync(
        resolve(PLANNER_ASSET_ROOT, "tokenizer_config.json"),
        "utf8",
      ),
      chatTemplate: readFileSync(
        resolve(PLANNER_ASSET_ROOT, "chat_template.jinja"),
        "utf8",
      ),
    })).tokenizer;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs the default real metadata FSM, then replaces it with a fresh semantic phase", async () => {
    const request = enabledRequest();
    const cotTokens = METADATA_FSM_VECTOR.injectedTrace.emittedTokenIds;
    const semanticTokens = [
      ...Array.from(
        { length: 60 },
        (_, code) => ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID + code,
      ),
      ACE_QWEN_IM_END_TOKEN_ID,
    ];
    const phaseDestroy = vi.fn();
    const ownedPhase = plannerPhase(phaseDestroy);
    const graph = new SequencedPlannerExecutor([cotTokens, semanticTokens], phaseDestroy);
    const create = vi.spyOn(AcePlannerGpuExecutor, "create").mockImplementation(
      (options) => {
        graph.onGpuProgress = options.onProgress;
        return graph as unknown as AcePlannerGpuExecutor;
      },
    );
    const progress: AcePlannerCoordinatorProgress[] = [];

    const result = await runAcePlannerCoordinator({
      request,
      resources: {
        kind: "owned-phase",
        device: {} as GPUDevice,
        modelProfile: "reference-bf16",
        tokenizer,
        ownedPlannerWeights: ownedPhase,
      },
      onProgress: (event) => progress.push(event),
    });

    expect(create).toHaveBeenCalledTimes(1);
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      modelProfile: "reference-bf16",
      ownedPlannerWeights: ownedPhase,
    });
    expect(graph.prefills).toHaveLength(2);
    expect(graph.prefills.map(({ rows }) => rows)).toEqual([1, 2]);
    expect(graph.operations.filter((operation) => operation === "prefill"))
      .toEqual(["prefill", "prefill"]);
    expect(graph.phaseDecodeCounts[0]).toBe(cotTokens.length - 1);
    expect(graph.phaseDecodeCounts[1]).toBe(60);

    expect(result.plannerMode).toBe("enabled");
    if (result.plannerMode !== "enabled") throw new Error("expected enabled result");
    expect(result.cot?.outputText).toBe(METADATA_FSM_VECTOR.injectedTrace.decoded);
    expect(result.conditioning).toMatchObject({
      caption: "Crisp drums.",
      vocalLanguage: "en",
      metadata: {
        bpm: 120,
        duration: 12,
        keyscale: "C major",
        language: "en",
        timesignature: "4",
      },
    });
    expect(result.downstream).toMatchObject({
      caption: "Crisp drums.",
      lyrics: "Lyrics",
      instrumental: false,
      durationSeconds: 12,
      vocalLanguage: "en",
    });
    expect(result.semantic.semanticCodeValues).toEqual(
      Array.from({ length: 60 }, (_, code) => code),
    );
    expect(result.semantic.emittedTokenIds).toHaveLength(61);
    expect(result.semantic.audioCodeText).toContain("<|audio_code_0|>");
    expect(result.semantic.audioCodeText).toContain("<|audio_code_59|>");
    expect(result.cot?.drawStart).toBe(0n);
    expect(result.semantic.drawStart).toBe(result.cot?.drawEnd);
    expect(result.sampling).toEqual({
      seed: request.seed,
      oracleId: "ace-browser-softmax-v1",
      distributionAuthority: "browser-defined",
      firstDraw: 0n,
      finalDraw: BigInt(cotTokens.length + semanticTokens.length),
    });

    const gpuPhases = progress
      .filter((event) => event.kind === "gpu")
      .map((event) => event.phase);
    expect(gpuPhases).toEqual(["cot", "semantic"]);
    const tokenEvents = progress.filter((event) => event.kind === "token");
    expect(tokenEvents.filter((event) => event.phase === "cot")).toHaveLength(
      cotTokens.length,
    );
    expect(tokenEvents.filter((event) => event.phase === "semantic")).toHaveLength(61);
    expect(progress).toContainEqual({
      kind: "boundary",
      phase: "release",
      state: "completed",
    });
    expect(graph.destroy).toHaveBeenCalledTimes(1);
    expect(phaseDestroy).toHaveBeenCalledTimes(1);
    expect(ACE_PLANNER_COT_MAX_NEW_TOKENS).toBe(1_700);
    expect(ACE_PLANNER_COORDINATOR_CONTRACT).toMatchObject({
      samplingDistributionAuthority: "browser-defined",
      torchLogitIdentityClaim: false,
      cotToSemanticTransition: "fresh-prefill-replaces-complete-kv-phase",
    });
  }, 30_000);

  it("bypasses planner acquisition explicitly for a disabled request", async () => {
    const create = vi.spyOn(AcePlannerGpuExecutor, "create");
    const progress: AcePlannerCoordinatorProgress[] = [];
    const result = await runAcePlannerCoordinator({
      request: {
        ...enabledRequest(),
        planner: { mode: "disabled" },
        metadata: {
          bpm: 92,
          keyScale: "D minor",
          timeSignature: "3",
          vocalLanguage: "fr",
        },
      },
      onProgress: (event) => progress.push(event),
    });

    expect(result).toEqual({
      plannerMode: "disabled",
      bypassReason: "planner-disabled-by-request",
      downstream: {
        caption: "Prompt",
        lyrics: "Lyrics",
        instrumental: false,
        durationSeconds: 12,
        vocalLanguage: "fr",
        metadata: {
          bpm: 92,
          duration: 12,
          keyscale: "D minor",
          timesignature: "3",
        },
      },
      conditioning: null,
      cot: null,
      semantic: null,
      sampling: null,
      runtime: {
        peakAccountedGpuBytes: 0,
        queueDrains: 0,
        cooperativeIdleMs: 0,
      },
    });
    expect(create).not.toHaveBeenCalled();
    expect(progress).toEqual([{
      kind: "boundary",
      phase: "planner-bypass",
      state: "skipped",
    }]);
  });

  it("canonicalizes explicit instrumental lyrics before both planner phases", async () => {
    const request = {
      ...enabledRequest(),
      instrumental: true,
      lyrics: "Conflicting vocal words",
      metadata: {
        bpm: 120,
        keyScale: "C major",
        timeSignature: "4",
      },
    } as const;
    const semanticTokens = [
      ...Array.from(
        { length: 60 },
        (_, code) => ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID + code,
      ),
      ACE_QWEN_IM_END_TOKEN_ID,
    ];
    const phaseDestroy = vi.fn();
    const graph = new SequencedPlannerExecutor([semanticTokens], phaseDestroy);
    vi.spyOn(AcePlannerGpuExecutor, "create").mockReturnValue(
      graph as unknown as AcePlannerGpuExecutor,
    );

    const result = await runAcePlannerCoordinator({
      request,
      resources: {
        kind: "owned-phase",
        device: {} as GPUDevice,
        modelProfile: "reference-bf16",
        tokenizer,
        ownedPlannerWeights: plannerPhase(phaseDestroy),
      },
    });

    expect(result.plannerMode).toBe("enabled");
    if (result.plannerMode !== "enabled") throw new Error("expected enabled");
    expect(result.downstream.lyrics).toBe("[Instrumental]");
    expect(result.conditioning.conditionalCodePrompt).toContain(
      "# Lyric\n[Instrumental]",
    );
    expect(result.conditioning.conditionalCodePrompt).not.toContain(
      "Conflicting vocal words",
    );
  });

  it("destroys transferred ownership when preflight fails before graph creation", async () => {
    const invalidDestroy = vi.fn();
    await expect(runAcePlannerCoordinator({
      request: { ...enabledRequest(), durationSeconds: 9 },
      resources: {
        kind: "owned-phase",
        device: {} as GPUDevice,
        modelProfile: "reference-bf16",
        tokenizer,
        ownedPlannerWeights: plannerPhase(invalidDestroy),
      },
    })).rejects.toThrow(/durationSeconds/);
    expect(invalidDestroy).toHaveBeenCalledTimes(1);

    const abortedDestroy = vi.fn();
    const controller = new AbortController();
    controller.abort(new DOMException("cancelled", "AbortError"));
    await expect(runAcePlannerCoordinator({
      request: enabledRequest(),
      resources: {
        kind: "owned-phase",
        device: {} as GPUDevice,
        modelProfile: "reference-bf16",
        tokenizer,
        ownedPlannerWeights: plannerPhase(abortedDestroy),
      },
      signal: controller.signal,
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(abortedDestroy).toHaveBeenCalledTimes(1);
  });

  it("destroys transferred planner ownership when cancellation interrupts CoT", async () => {
    const request = enabledRequest();
    const controller = new AbortController();
    const phaseDestroy = vi.fn();
    const graph = new SequencedPlannerExecutor(
      [[ACE_QWEN_IM_END_TOKEN_ID]],
      phaseDestroy,
      () => controller.abort(new DOMException("cancelled", "AbortError")),
    );
    vi.spyOn(AcePlannerGpuExecutor, "create").mockImplementation((options) => {
      graph.onGpuProgress = options.onProgress;
      return graph as unknown as AcePlannerGpuExecutor;
    });

    await expect(runAcePlannerCoordinator({
      request,
      resources: {
        kind: "owned-phase",
        device: {} as GPUDevice,
        modelProfile: "reference-bf16",
        tokenizer,
        ownedPlannerWeights: plannerPhase(phaseDestroy),
      },
      signal: controller.signal,
    })).rejects.toMatchObject({ name: "AbortError" });

    expect(graph.destroy).toHaveBeenCalledTimes(1);
    expect(phaseDestroy).toHaveBeenCalledTimes(1);
  });
});

class SequencedPlannerExecutor implements AcePlannerGraphExecutor {
  readonly prefills: AcePlannerPrefillBatch[] = [];
  readonly decodes: AcePlannerDecodeBatch[] = [];
  readonly operations: Array<"prefill" | "decode"> = [];
  readonly phaseDecodeCounts: number[] = [];
  readonly destroy = vi.fn(async () => {
    this.phaseDestroy();
  });
  onGpuProgress: ((progress: AcePlannerGpuExecutorProgress) => void) | undefined;

  private phase = -1;
  private step = 0;
  private readonly sharedLogits = new Float32Array(VOCABULARY_SIZE);

  constructor(
    private readonly sequences: readonly (readonly number[])[],
    private readonly phaseDestroy: () => void,
    private readonly onFirstPrefill?: () => void,
  ) {}

  async prefill(
    batch: AcePlannerPrefillBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.operations.push("prefill");
    this.prefills.push(batch);
    this.phase += 1;
    this.step = 0;
    this.phaseDecodeCounts[this.phase] = 0;
    if (this.phase === 0) this.onFirstPrefill?.();
    this.emitGpu(batch.kind);
    return this.nextLogits(batch.rows, logitRange);
  }

  async decode(
    batch: AcePlannerDecodeBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.operations.push("decode");
    this.decodes.push(batch);
    this.phaseDecodeCounts[this.phase] = (this.phaseDecodeCounts[this.phase] ?? 0) + 1;
    return this.nextLogits(batch.rows, logitRange);
  }

  private nextLogits(
    rows: number,
    logitRange?: AcePlannerLogitRange,
  ): readonly Float32Array[] {
    const preferred = this.sequences[this.phase]?.[this.step];
    if (preferred === undefined) {
      throw new Error(`planner test sequence exhausted at phase ${this.phase} step ${this.step}`);
    }
    this.step += 1;
    this.sharedLogits.fill(Number.NEGATIVE_INFINITY);
    this.sharedLogits[preferred] = 100;
    this.sharedLogits[ACE_QWEN_IM_END_TOKEN_ID] =
      preferred === ACE_QWEN_IM_END_TOKEN_ID ? 100 : -100;
    return Array.from({ length: rows }, () => logitRange === undefined
      ? this.sharedLogits
      : this.sharedLogits.slice(
          logitRange.firstTokenId,
          logitRange.firstTokenId + logitRange.tokenCount,
        ));
  }

  private emitGpu(phaseKind: "prefill" | "decode"): void {
    this.onGpuProgress?.({
      phaseKind,
      completedCommandBuffers: 1,
      totalCommandBuffers: 1,
      queueDrains: 1,
      cooperativeIdleMs: 0,
      stage: "readback",
      quantum: null,
      peakAccountedGpuBytes: 1,
      cumulativeQueueDrains: this.step + 1,
      cumulativeCooperativeIdleMs: 0,
    });
  }
}

function plannerPhase(destroy: () => void): AceGpuTensorPhase {
  return {
    phases: Object.freeze(["planner"]),
    residentBytes: 1,
    destroy,
  } as unknown as AceGpuTensorPhase;
}

function enabledRequest(): AceGenerationRequest {
  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt: "Prompt",
    lyrics: "Lyrics",
    instrumental: false,
    durationSeconds: 12,
    seed: canonicalizeSeed(0x1234),
    planner: {
      mode: "enabled",
      temperature: 0.85,
      guidanceScale: 2,
      topK: 0,
      topP: 0.9,
      constrainedDecoding: true,
      generateSemanticCodes: true,
      negativePrompt: "NO USER INPUT",
      thinking: {
        enabled: true,
        useCotCaption: true,
        useCotLanguage: true,
        useCotMissingMetadata: true,
      },
    },
    metadata: {
      bpm: 120,
      keyScale: "C major",
    },
  };
}
