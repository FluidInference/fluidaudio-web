import {
  aceCategoricalTokenFromWord,
  aceRandomWord,
  type AceSeed,
} from "./seed.js";

/**
 * Sampling order for the pinned eager-PyTorch planner oracle.
 *
 * Sources:
 * - ACE-Step `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`,
 *   `acestep/llm_inference.py:2618-2753 _generate_with_cfg_custom` and
 *   `2522-2595 _generate_with_constrained_decoding`.
 * - Repository capture contract `reference/native.py:PlannerWordInjector`.
 *
 * The browser intentionally owns categorical random words. It does not claim
 * that a numeric seed reproduces PyTorch's multinomial stream.
 */
export const ACE_PLANNER_SAMPLING_CONTRACT = Object.freeze({
  aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  referenceBackend: "pytorch-eager",
  order: Object.freeze([
    "code-allowed-subspace",
    "cfg",
    "constraint-mask",
    "repetition-penalty",
    "top-k",
    "top-p",
    "temperature",
    "fp32-softmax",
    "philox-categorical",
  ] as const),
  tieBreak: "descending-logit-then-ascending-token-id",
  randomStream: "planner-sampling",
  wordsPerEmittedToken: 1,
} as const);

/**
 * Numerical acceptance for the versioned browser-owned sampling distribution.
 *
 * `ace-browser-softmax-v1` uses a fixed binary64 polynomial software exp, FP32
 * storage/reductions, and stable token-ID tie order. It is accepted for the
 * browser-defined Philox categorical contract after differential checks
 * against Python Decimal exp, including a realistic sparse 217,204-token
 * vocabulary with all 64,000 semantic codes admitted. It deliberately does
 * not claim bit identity with a Torch device softmax. Teacher-forced Torch
 * differential capture remains a checkpoint validation, not a product gate.
 */
export const ACE_PLANNER_SOFTMAX_ACCEPTANCE = Object.freeze({
  status: "accepted-browser-v1" as const,
  productionOracleId: "ace-browser-softmax-v1",
  distributionAuthority: "browser-defined",
  independentReference: "scripts/planner_sampling_reference.py:Decimal.exp",
  acceptedCoverage: Object.freeze([
    "adversarial-dynamic-range",
    "top-p-boundaries",
    "realistic-64000-code-sparsity",
  ] as const),
  errorBounds: Object.freeze({
    normalExpRelative: 6.1e-8,
    subnormalExpAbsolute: 7.1e-46,
    realisticWeightAbsolute: 3e-7,
    realisticDistributionL1: 1e-5,
    realisticTopPKeepCountDelta: 4,
    realisticTopPRetainedMass: 1e-4,
  }),
  torchDifferential: "pending-teacher-forced-checkpoint" as const,
  torchDifferentialRisk:
    "near CDF boundaries browser-v1 may choose a different token than Torch FP32",
  torchDifferentialRequiredBeforeListeningCandidate: true,
  torchBitExactClaim: false,
  listeningCandidateMayUseBrowserV1: true,
  listeningCandidateMayUseDiagnosticOracle: false,
});

export interface AcePlannerSoftmaxOracle {
  readonly id: string;
  readonly status: "accepted" | "diagnostic";
  readonly reference: string;
  /** One means the original token survives the untempered top-p cutoff. */
  topPKeep(logits: Float32Array, topP: number): Uint8Array;
  weights(logits: Float32Array, temperature: number): Float32Array;
}

export type AcePlannerAllowedTokens =
  | Readonly<{ readonly kind: "all" }>
  | Readonly<{
      readonly kind: "ids";
      readonly tokenIds: readonly number[];
    }>
  | Readonly<{
      readonly kind: "range";
      readonly firstTokenId: number;
      readonly tokenCount: number;
      readonly additionalTokenIds?: readonly number[];
    }>;

export interface AcePlannerSamplingParameters {
  readonly temperature: number;
  readonly guidanceScale: number;
  readonly topK: number;
  readonly topP: number;
  readonly repetitionPenalty: number;
}

export interface AcePlannerTokenSampleInput {
  readonly conditionalLogits: ArrayLike<number>;
  readonly unconditionalLogits?: ArrayLike<number>;
  /** Conditional physical prompt row followed by prior emitted tokens. */
  readonly seenTokenIds: readonly number[];
  /**
   * Optional pre-CFG subspace. Code generation supplies all 64,000 semantic
   * tokens plus EOS here, then applies its count-dependent `allowedTokens`
   * constraint after CFG, exactly like the pinned eager loop.
   */
  readonly preCfgAllowedTokens?: AcePlannerAllowedTokens;
  readonly allowedTokens: AcePlannerAllowedTokens;
  readonly parameters: AcePlannerSamplingParameters;
  readonly word: number;
  /** Defaults to the accepted, versioned browser software implementation. */
  readonly softmax?: AcePlannerSoftmaxOracle;
}

/**
 * A contiguous, ascending slice of the global planner vocabulary.
 *
 * The caller has already applied the state-dependent model-head restriction:
 * every supplied row contains exactly the candidates admitted by the current
 * FSM state. `firstTokenId + localIndex` is therefore the global token ID.
 */
export interface AcePlannerCompactTokenSampleInput {
  readonly firstTokenId: number;
  readonly vocabularySize: number;
  readonly conditionalLogits: ArrayLike<number>;
  readonly unconditionalLogits?: ArrayLike<number>;
  /** Global token IDs; IDs outside this compact domain remain valid no-ops. */
  readonly seenTokenIds: readonly number[];
  readonly parameters: AcePlannerSamplingParameters;
  readonly word: number;
  /** Defaults to the accepted, versioned browser software implementation. */
  readonly softmax?: AcePlannerSoftmaxOracle;
}

export type AcePlannerFilterInput = Omit<
  AcePlannerTokenSampleInput,
  "word"
>;

export type AcePlannerCompactFilterInput = Omit<
  AcePlannerCompactTokenSampleInput,
  "word"
>;

export interface AcePlannerTokenSample {
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
}

export interface AcePlannerCursorSample extends AcePlannerTokenSample {
  readonly drawIndex: bigint;
}

/**
 * Allocation diagnostics for the benchmark-only OPT-0084 sampling workspace.
 *
 * `storageAllocationCount` changes only when a retained typed-array backing
 * store grows. The fixed 256-bin radix histogram is allocated once with the
 * workspace and is not included in that count.
 *
 * @internal
 */
export interface AceOpt0084PlannerSamplingWorkspaceStats {
  readonly candidateCapacity: number;
  readonly maskCapacity: number;
  readonly storageAllocationCount: number;
}

/**
 * Reusable candidate-domain storage for the exact OPT-0084 browser-v1 arm.
 *
 * This class is deliberately experiment-only. Ordinary production sampling
 * continues to use {@link sampleAcePlannerToken} and
 * {@link sampleAcePlannerCompactToken} until OPT-0084 clears its browser,
 * trajectory, and product gates.
 *
 * @internal
 */
export class AceOpt0084PlannerSamplingWorkspace {
  private candidateCapacityValue = 0;
  private maskCapacityValue = 0;
  private storageAllocationCountValue = 0;
  private logits = new Float32Array(0);
  private logitWords = new Uint32Array(0);
  private weights = new Float32Array(0);
  private weightWords = new Uint32Array(0);
  private radixA = new Uint32Array(0);
  private radixB = new Uint32Array(0);
  private active = new Uint8Array(0);
  private categoricalWeights = new Float32Array(0);
  private categoricalWeightCount = 0;
  private readonly radixHistogram = new Uint32Array(512);
  private lastSuccessfulCandidateCount = 0;
  private lastSuccessfulHadRadixOrder = false;
  private lastSuccessfulTokenIdOffset = 0;

  /** @internal */
  get stats(): AceOpt0084PlannerSamplingWorkspaceStats {
    return Object.freeze({
      candidateCapacity: this.candidateCapacityValue,
      maskCapacity: this.maskCapacityValue,
      storageAllocationCount: this.storageAllocationCountValue,
    });
  }

  /**
   * Benchmark Arm B for a logical full-vocabulary input.
   * @internal
   */
  sample(input: AcePlannerTokenSampleInput): AcePlannerTokenSample {
    this.beginSample();
    const candidateCount = this.prepareFullCandidates(input);
    return this.finishSample(
      candidateCount,
      input.conditionalLogits.length,
      input.seenTokenIds,
      input.parameters,
      input.word,
      input.softmax,
      false,
      0,
      input.conditionalLogits.length,
    );
  }

  /**
   * Benchmark Arm B for an ascending contiguous compact head domain.
   * @internal
   */
  sampleCompact(input: AcePlannerCompactTokenSampleInput): AcePlannerTokenSample {
    this.beginSample();
    const candidateCount = this.prepareCompactCandidates(input);
    return this.finishSample(
      candidateCount,
      input.vocabularySize,
      input.seenTokenIds,
      input.parameters,
      input.word,
      input.softmax,
      true,
      input.firstTokenId,
      input.conditionalLogits.length,
    );
  }

  /**
   * Copy global IDs from the most recent successful sample, in ID order.
   * @internal
   */
  copyLastCandidateTokenIds(): Float64Array {
    const count = this.requireLastSample();
    return Float64Array.from(
      this.radixB.subarray(0, count),
      (tokenId) => this.lastSuccessfulTokenIdOffset + tokenId,
    );
  }

  /**
   * Copy post-top-k/top-p candidate logits from the most recent successful
   * sample. Entries are in ascending global-ID order; removed entries are
   * represented by negative infinity exactly like Arm A.
   *
   * @internal
   */
  copyLastFilteredLogits(): Float32Array {
    const count = this.requireLastSample();
    const output = this.logits.slice(0, count);
    for (let ordinal = 0; ordinal < count; ordinal += 1) {
      if (this.active[ordinal] === 0) {
        output[ordinal] = Number.NEGATIVE_INFINITY;
      }
    }
    return output;
  }

  /**
   * Copy final browser-v1 FP32 weights in ascending global-ID order.
   * @internal
   */
  copyLastWeights(): Float32Array {
    const count = this.requireLastSample();
    return this.weights.slice(0, count);
  }

  /**
   * Copy the stable descending-logit radix order as global token IDs.
   * Throws when the most recent path did not need top-k or top-p ordering.
   *
   * @internal
   */
  copyLastRadixOrderTokenIds(): Float64Array {
    const count = this.requireLastSample();
    if (!this.lastSuccessfulHadRadixOrder) {
      throw new Error("OPT-0084 last sample did not require radix ordering");
    }
    const output = new Float64Array(count);
    for (let index = 0; index < count; index += 1) {
      output[index] = this.lastSuccessfulTokenIdOffset +
        this.radixB[this.radixA[index]!]!;
    }
    return output;
  }

  private beginSample(): void {
    this.lastSuccessfulCandidateCount = 0;
    this.lastSuccessfulHadRadixOrder = false;
    this.lastSuccessfulTokenIdOffset = 0;
  }

  private prepareFullCandidates(input: AcePlannerTokenSampleInput): number {
    const vocabularySize = requireNonEmptySafeLength(
      input.conditionalLogits,
      "ACE planner logits",
    );
    const conditionalOnly = input.unconditionalLogits === undefined;
    if (conditionalOnly && input.parameters.guidanceScale !== 1) {
      throw new TypeError("ACE planner CFG sampling requires unconditional logits");
    }

    validatePlannerLogitRow(input.conditionalLogits, "conditional logit");
    this.ensureMaskCapacity(vocabularySize);
    this.active.fill(0, 0, vocabularySize);

    if (conditionalOnly) {
      markAcePlannerAllowedTokens(
        input.allowedTokens,
        vocabularySize,
        this.active,
        OPT0084_FINAL_ALLOWED_BIT,
      );
      let candidateCount = 0;
      for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
        if ((this.active[tokenId]! & OPT0084_FINAL_ALLOWED_BIT) === 0) continue;
        const value = requirePlannerLogit(
          input.conditionalLogits[tokenId],
          `constraint input ${tokenId}`,
        );
        if (value === Number.NEGATIVE_INFINITY) continue;
        if (!Number.isFinite(value)) {
          throw new RangeError(
            `constraint input ${tokenId} must be finite or negative infinity`,
          );
        }
        candidateCount += 1;
      }
      if (candidateCount === 0) {
        throw new RangeError("ACE planner constraint removed every finite candidate");
      }
      this.ensureCandidateCapacity(candidateCount);
      let ordinal = 0;
      for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
        if ((this.active[tokenId]! & OPT0084_FINAL_ALLOWED_BIT) === 0) continue;
        const value = Math.fround(input.conditionalLogits[tokenId]!);
        if (!Number.isFinite(value)) continue;
        this.logits[ordinal] = value;
        this.weightWords[ordinal] = tokenId;
        ordinal += 1;
      }
      return candidateCount;
    }

    const unconditional = input.unconditionalLogits!;
    const preCfgAllowed = input.preCfgAllowedTokens ?? input.allowedTokens;
    markAcePlannerAllowedTokens(
      preCfgAllowed,
      vocabularySize,
      this.active,
      OPT0084_PRE_CFG_ALLOWED_BIT,
    );
    if (!hasFiniteAllowedRawLogit(
      input.conditionalLogits,
      this.active,
      OPT0084_PRE_CFG_ALLOWED_BIT,
    )) {
      throw new RangeError("ACE planner constraint removed every finite candidate");
    }

    validatePlannerLogitRow(unconditional, "unconditional logit");
    if (unconditional.length !== vocabularySize) {
      throw new RangeError(
        "ACE planner conditional and unconditional logits differ in length",
      );
    }
    if (!hasFiniteAllowedRawLogit(
      unconditional,
      this.active,
      OPT0084_PRE_CFG_ALLOWED_BIT,
    )) {
      throw new RangeError("ACE planner constraint removed every finite candidate");
    }
    const guidanceScale = requireAcePlannerGuidanceScale(
      input.parameters.guidanceScale,
    );
    let combinedFinite = 0;
    for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
      if ((this.active[tokenId]! & OPT0084_PRE_CFG_ALLOWED_BIT) === 0) continue;
      const combined = combineAcePlannerCfgCandidate(
        input.conditionalLogits[tokenId]!,
        unconditional[tokenId]!,
        guidanceScale,
      );
      if (combined !== Number.NEGATIVE_INFINITY) combinedFinite += 1;
    }
    if (combinedFinite === 0) {
      throw new RangeError("ACE planner CFG removed every finite candidate");
    }

    markAcePlannerAllowedTokens(
      input.allowedTokens,
      vocabularySize,
      this.active,
      OPT0084_FINAL_ALLOWED_BIT,
    );
    let candidateCount = 0;
    for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
      if (
        (this.active[tokenId]! & OPT0084_BOTH_ALLOWED_BITS) !==
          OPT0084_BOTH_ALLOWED_BITS
      ) {
        continue;
      }
      const combined = combineAcePlannerCfgCandidate(
        input.conditionalLogits[tokenId]!,
        unconditional[tokenId]!,
        guidanceScale,
      );
      if (combined !== Number.NEGATIVE_INFINITY) candidateCount += 1;
    }
    if (candidateCount === 0) {
      throw new RangeError("ACE planner constraint removed every finite candidate");
    }
    this.ensureCandidateCapacity(candidateCount);
    let ordinal = 0;
    for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
      if (
        (this.active[tokenId]! & OPT0084_BOTH_ALLOWED_BITS) !==
          OPT0084_BOTH_ALLOWED_BITS
      ) {
        continue;
      }
      const combined = combineAcePlannerCfgCandidate(
        input.conditionalLogits[tokenId]!,
        unconditional[tokenId]!,
        guidanceScale,
      );
      if (combined === Number.NEGATIVE_INFINITY) continue;
      this.logits[ordinal] = combined;
      this.weightWords[ordinal] = tokenId;
      ordinal += 1;
    }
    return candidateCount;
  }

  private prepareCompactCandidates(
    input: AcePlannerCompactTokenSampleInput,
  ): number {
    const sourceCount = requireNonEmptySafeLength(
      input.conditionalLogits,
      "ACE planner compact logits",
    );
    if (!Number.isSafeInteger(input.vocabularySize) || input.vocabularySize <= 0) {
      throw new RangeError("ACE planner compact vocabulary size must be positive");
    }
    requireTokenId(
      input.firstTokenId,
      input.vocabularySize,
      "ACE planner compact first token",
    );
    const domainEnd = input.firstTokenId + sourceCount;
    if (!Number.isSafeInteger(domainEnd) || domainEnd > input.vocabularySize) {
      throw new RangeError("ACE planner compact candidate domain exceeds the vocabulary");
    }
    if (
      input.unconditionalLogits !== undefined &&
      input.unconditionalLogits.length !== sourceCount
    ) {
      throw new RangeError(
        "ACE planner compact conditional and unconditional logits differ in length",
      );
    }
    requireAcePlannerTopK(input.parameters.topK, input.vocabularySize, true);
    for (const tokenId of input.seenTokenIds) {
      requireTokenId(tokenId, input.vocabularySize, "ACE planner seen token");
    }
    if (
      input.unconditionalLogits === undefined &&
      input.parameters.guidanceScale !== 1
    ) {
      throw new TypeError("ACE planner CFG sampling requires unconditional logits");
    }

    validatePlannerLogitRow(input.conditionalLogits, "constraint input");
    const conditional = input.conditionalLogits;
    const unconditional = input.unconditionalLogits;
    let guidanceScale = 1;
    if (unconditional !== undefined) {
      if (!hasFiniteRawLogit(conditional)) {
        throw new RangeError("ACE planner constraint removed every finite candidate");
      }
      validatePlannerLogitRow(unconditional, "constraint input");
      if (!hasFiniteRawLogit(unconditional)) {
        throw new RangeError("ACE planner constraint removed every finite candidate");
      }
      guidanceScale = requireAcePlannerGuidanceScale(
        input.parameters.guidanceScale,
      );
    }

    let candidateCount = 0;
    for (let local = 0; local < sourceCount; local += 1) {
      const value = unconditional === undefined
        ? requireFiniteOrNegativeInfinityF32(
          conditional[local]!,
          `constraint input ${local}`,
        )
        : combineAcePlannerCfgCandidate(
          conditional[local]!,
          unconditional[local]!,
          guidanceScale,
        );
      if (value !== Number.NEGATIVE_INFINITY) candidateCount += 1;
    }
    if (candidateCount === 0) {
      throw new RangeError(
        unconditional === undefined
          ? "ACE planner constraint removed every finite candidate"
          : "ACE planner CFG removed every finite candidate",
      );
    }
    this.ensureCandidateCapacity(candidateCount);
    // Compact sampling stores only source-row-local ordinals. Global IDs can
    // legitimately exceed uint32, and the workspace must stay proportional
    // to the compact row rather than the logical vocabulary.
    this.ensureMaskCapacity(sourceCount);
    this.active.fill(0, 0, sourceCount);
    let ordinal = 0;
    for (let local = 0; local < sourceCount; local += 1) {
      const value = unconditional === undefined
        ? Math.fround(conditional[local]!)
        : combineAcePlannerCfgCandidate(
          conditional[local]!,
          unconditional[local]!,
          guidanceScale,
        );
      if (!Number.isFinite(value)) continue;
      this.logits[ordinal] = value;
      this.weightWords[ordinal] = local;
      ordinal += 1;
    }
    return candidateCount;
  }

  private finishSample(
    candidateCount: number,
    vocabularySize: number,
    seenTokenIds: readonly number[],
    parameters: AcePlannerSamplingParameters,
    wordInput: number,
    softmax: AcePlannerSoftmaxOracle | undefined,
    seenAlreadyValidated: boolean,
    tokenIdOffset: number,
    seenDomainSize: number,
  ): AcePlannerTokenSample {
    candidateCount = this.applyRepetitionPenalty(
      candidateCount,
      vocabularySize,
      seenTokenIds,
      parameters.repetitionPenalty,
      seenAlreadyValidated,
      tokenIdOffset,
      seenDomainSize,
    );
    this.active.fill(1, 0, candidateCount);
    requireAcePlannerTopK(parameters.topK, vocabularySize, false);
    const topPRequiresRadix = parameters.topP < 1;
    if (topPRequiresRadix) requireAcePlannerTopP(parameters.topP);
    const topKRequiresRadix =
      parameters.topK > 0 && parameters.topK < candidateCount;
    const hasRadixOrder = topKRequiresRadix || topPRequiresRadix;
    if (hasRadixOrder) this.createStableRadixOrder(candidateCount);

    // The weight buffer holds ascending domain-relative token IDs during
    // preparation (global IDs for full inputs, local row ordinals for compact
    // inputs). Preserve that mapping before exponent generation reuses it.
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      this.radixB[ordinal] = this.weightWords[ordinal]!;
    }

    if (topKRequiresRadix) {
      const thresholdOrdinal = this.radixA[parameters.topK - 1]!;
      const threshold = this.logits[thresholdOrdinal]!;
      for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
        if (this.logits[ordinal]! < threshold) this.active[ordinal] = 0;
      }
    }
    if (topPRequiresRadix) {
      this.computeBrowserWeights(candidateCount, 1, false);
      const threshold = Math.fround(parameters.topP);
      let cumulative = 0;
      let previousCrossed = false;
      for (let index = 0; index < candidateCount; index += 1) {
        const ordinal = this.radixA[index]!;
        if (this.active[ordinal] === 0) continue;
        if (previousCrossed) this.active[ordinal] = 0;
        cumulative = Math.fround(cumulative + this.weights[ordinal]!);
        previousCrossed = cumulative > threshold;
      }
    }

    requireOpt0084BrowserSoftmax(softmax);
    this.computeBrowserWeights(candidateCount, parameters.temperature, true);
    const word = requireU32(wordInput, "ACE planner categorical word");
    const localToken = aceCategoricalTokenFromWord(
      this.getCategoricalWeights(candidateCount),
      word,
    );
    let positiveCandidateCount = 0;
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      if (this.weights[ordinal]! > 0) positiveCandidateCount += 1;
    }
    const tokenId = tokenIdOffset + this.radixB[localToken]!;
    this.lastSuccessfulCandidateCount = candidateCount;
    this.lastSuccessfulHadRadixOrder = hasRadixOrder;
    this.lastSuccessfulTokenIdOffset = tokenIdOffset;
    return Object.freeze({ tokenId, word, positiveCandidateCount });
  }

  private applyRepetitionPenalty(
    candidateCount: number,
    vocabularySize: number,
    seenTokenIds: readonly number[],
    penalty: number,
    seenAlreadyValidated: boolean,
    tokenIdOffset: number,
    seenDomainSize: number,
  ): number {
    if (!Number.isFinite(penalty) || penalty <= 0) {
      throw new RangeError(
        "ACE planner repetition penalty must be positive and finite",
      );
    }
    if (penalty === 1 || seenTokenIds.length === 0) return candidateCount;
    const roundedPenalty = Math.fround(penalty);
    for (const tokenId of seenTokenIds) {
      if (!seenAlreadyValidated) {
        requireTokenId(tokenId, vocabularySize, "ACE planner seen token");
      }
      const localTokenId = tokenId - tokenIdOffset;
      if (localTokenId < 0 || localTokenId >= seenDomainSize) continue;
      if ((this.active[localTokenId]! & OPT0084_SEEN_BIT) !== 0) continue;
      this.active[localTokenId] =
        this.active[localTokenId]! | OPT0084_SEEN_BIT;
      const ordinal = binarySearchUint32(
        this.weightWords,
        candidateCount,
        localTokenId,
      );
      if (ordinal < 0) continue;
      const value = this.logits[ordinal]!;
      const penalized = Math.fround(
        value < 0 ? value * roundedPenalty : value / roundedPenalty,
      );
      if (Number.isNaN(penalized) || penalized === Number.POSITIVE_INFINITY) {
        throw new RangeError(
          `repetition-penalty input ${ordinal} must be finite or negative infinity`,
        );
      }
      this.logits[ordinal] = penalized;
    }
    let retainedCount = 0;
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      const value = this.logits[ordinal]!;
      if (value === Number.NEGATIVE_INFINITY) continue;
      if (retainedCount !== ordinal) {
        this.logits[retainedCount] = value;
        this.weightWords[retainedCount] = this.weightWords[ordinal]!;
      }
      retainedCount += 1;
    }
    if (retainedCount === 0) {
      throw new RangeError(
        "ACE planner repetition penalty removed every finite candidate",
      );
    }
    return retainedCount;
  }

  private createStableRadixOrder(candidateCount: number): void {
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      this.radixA[ordinal] = ordinal;
    }
    let source = this.radixA;
    let destination = this.radixB;
    for (let pass = 0; pass < 4; pass += 1) {
      this.radixHistogram.fill(0, 0, 256);
      const shift = pass * 8;
      for (let index = 0; index < candidateCount; index += 1) {
        const ordinal = source[index]!;
        const bucket = (descendingFloatRadixKey(
          this.logitWords[ordinal]!,
        ) >>> shift) & 0xff;
        this.radixHistogram[bucket] = this.radixHistogram[bucket]! + 1;
      }
      let offset = 0;
      for (let bucket = 0; bucket < 256; bucket += 1) {
        const count = this.radixHistogram[bucket]!;
        this.radixHistogram[256 + bucket] = offset;
        offset += count;
      }
      for (let index = 0; index < candidateCount; index += 1) {
        const ordinal = source[index]!;
        const bucket = (descendingFloatRadixKey(
          this.logitWords[ordinal]!,
        ) >>> shift) & 0xff;
        const destinationIndex = this.radixHistogram[256 + bucket]!;
        destination[destinationIndex] = ordinal;
        this.radixHistogram[256 + bucket] = destinationIndex + 1;
      }
      const swap = source;
      source = destination;
      destination = swap;
    }
    if (source !== this.radixA) {
      throw new Error("OPT-0084 radix parity invariant failed");
    }
  }

  private computeBrowserWeights(
    candidateCount: number,
    temperature: number,
    applyTemperature: boolean,
  ): void {
    let roundedTemperature = 1;
    if (applyTemperature) {
      if (!Number.isFinite(temperature) || temperature <= 0) {
        throw new RangeError(
          "ACE planner temperature must be positive and finite",
        );
      }
      roundedTemperature = Math.fround(temperature);
    }
    let maximum = Number.NEGATIVE_INFINITY;
    this.weights.fill(0, 0, candidateCount);
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      if (this.active[ordinal] === 0) continue;
      const scaled = applyTemperature
        ? Math.fround(this.logits[ordinal]! / roundedTemperature)
        : this.logits[ordinal]!;
      if (scaled > maximum) maximum = scaled;
    }
    if (!Number.isFinite(maximum)) {
      throw new RangeError(
        "ACE planner browser-v1 softmax has no finite candidate",
      );
    }
    let total = 0;
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      if (this.active[ordinal] === 0) continue;
      const scaled = applyTemperature
        ? Math.fround(this.logits[ordinal]! / roundedTemperature)
        : this.logits[ordinal]!;
      const exponent = acePlannerBrowserExpF32(
        Math.fround(scaled - maximum),
      );
      this.weights[ordinal] = exponent;
      total = Math.fround(total + exponent);
    }
    if (!Number.isFinite(total) || total <= 0) {
      throw new RangeError("ACE planner browser-v1 FP32 softmax sum is invalid");
    }
    let positive = 0;
    for (let ordinal = 0; ordinal < candidateCount; ordinal += 1) {
      if (this.weights[ordinal] === 0) continue;
      this.weights[ordinal] = Math.fround(this.weights[ordinal]! / total);
      if (this.weights[ordinal]! > 0) positive += 1;
    }
    if (positive === 0) {
      throw new RangeError(
        "ACE planner browser-v1 softmax underflowed every candidate",
      );
    }
  }

  private ensureCandidateCapacity(required: number): void {
    if (required <= this.candidateCapacityValue) return;
    const capacity = nextOpt0084Capacity(required);
    this.logits = new Float32Array(capacity);
    this.logitWords = new Uint32Array(this.logits.buffer);
    this.weights = new Float32Array(capacity);
    this.weightWords = new Uint32Array(this.weights.buffer);
    this.categoricalWeights = new Float32Array(0);
    this.categoricalWeightCount = 0;
    this.radixA = new Uint32Array(capacity);
    this.radixB = new Uint32Array(capacity);
    this.candidateCapacityValue = capacity;
    this.storageAllocationCountValue += 4;
  }

  private ensureMaskCapacity(required: number): void {
    if (required <= this.maskCapacityValue) return;
    const capacity = nextOpt0084Capacity(required);
    this.active = new Uint8Array(capacity);
    this.maskCapacityValue = capacity;
    this.storageAllocationCountValue += 1;
  }

  private getCategoricalWeights(candidateCount: number): Float32Array {
    if (this.categoricalWeightCount !== candidateCount) {
      this.categoricalWeights = this.weights.subarray(0, candidateCount);
      this.categoricalWeightCount = candidateCount;
    }
    return this.categoricalWeights;
  }

  private requireLastSample(): number {
    if (this.lastSuccessfulCandidateCount === 0) {
      throw new Error("OPT-0084 workspace has no successful sample to inspect");
    }
    return this.lastSuccessfulCandidateCount;
  }
}

/**
 * Continuous Philox cursor shared by CoT and semantic-code phases.
 *
 * A draw advances only after the complete filtering pipeline has succeeded.
 * Forced one-candidate and terminal tokens still consume one word.
 */
export class AcePlannerSamplingCursor {
  readonly seed: AceSeed;
  private nextDrawIndex: bigint;
  /** @internal */
  private opt0084Workspace: AceOpt0084PlannerSamplingWorkspace | undefined;

  constructor(seed: AceSeed, firstDrawIndex: number | bigint = 0) {
    this.seed = seed;
    this.nextDrawIndex = requireNonNegativeSafeBigInt(
      firstDrawIndex,
      "ACE planner first draw index",
    );
  }

  get consumed(): bigint {
    return this.nextDrawIndex;
  }

  sample(input: Omit<AcePlannerTokenSampleInput, "word">): AcePlannerCursorSample {
    const drawIndex = this.nextDrawIndex;
    const word = aceRandomWord(this.seed, "planner-sampling", drawIndex);
    const sampled = sampleAcePlannerToken({ ...input, word });
    this.nextDrawIndex += 1n;
    return Object.freeze({ ...sampled, drawIndex });
  }

  /**
   * Sample an ascending contiguous candidate domain without reconstructing a
   * full-vocabulary vector. This consumes the same global Philox stream as
   * {@link sample} and commits the draw only after all filtering succeeds.
   */
  sampleCompact(
    input: Omit<AcePlannerCompactTokenSampleInput, "word">,
  ): AcePlannerCursorSample {
    const drawIndex = this.nextDrawIndex;
    const word = aceRandomWord(this.seed, "planner-sampling", drawIndex);
    const sampled = sampleAcePlannerCompactToken({ ...input, word });
    this.nextDrawIndex += 1n;
    return Object.freeze({ ...sampled, drawIndex });
  }

  /**
   * Benchmark-only OPT-0084 full/sparse candidate path.
   * @internal
   */
  sampleOpt0084(
    input: Omit<AcePlannerTokenSampleInput, "word">,
  ): AcePlannerCursorSample {
    const drawIndex = this.nextDrawIndex;
    const word = aceRandomWord(this.seed, "planner-sampling", drawIndex);
    const sampled = this.getOpt0084Workspace().sample({ ...input, word });
    this.nextDrawIndex += 1n;
    return Object.freeze({ ...sampled, drawIndex });
  }

  /**
   * Benchmark-only OPT-0084 compact contiguous candidate path.
   * @internal
   */
  sampleCompactOpt0084(
    input: Omit<AcePlannerCompactTokenSampleInput, "word">,
  ): AcePlannerCursorSample {
    const drawIndex = this.nextDrawIndex;
    const word = aceRandomWord(this.seed, "planner-sampling", drawIndex);
    const sampled = this.getOpt0084Workspace().sampleCompact({ ...input, word });
    this.nextDrawIndex += 1n;
    return Object.freeze({ ...sampled, drawIndex });
  }

  /** @internal */
  private getOpt0084Workspace(): AceOpt0084PlannerSamplingWorkspace {
    this.opt0084Workspace ??= new AceOpt0084PlannerSamplingWorkspace();
    return this.opt0084Workspace;
  }
}

/** FP32 CFG arithmetic: `uncond + scale * (cond - uncond)`. */
export function combineAcePlannerCfgLogits(
  conditional: ArrayLike<number>,
  unconditional: ArrayLike<number>,
  guidanceScale: number,
): Float32Array {
  const length = requireMatchingLogits(conditional, unconditional);
  if (!Number.isFinite(guidanceScale) || guidanceScale < 1) {
    throw new RangeError("ACE planner guidanceScale must be finite and at least one");
  }
  const scale = Math.fround(guidanceScale);
  const output = new Float32Array(length);
  for (let token = 0; token < length; token += 1) {
    const cond = requirePlannerLogit(conditional[token], `conditional logit ${token}`);
    const uncond = requirePlannerLogit(
      unconditional[token],
      `unconditional logit ${token}`,
    );
    if (!Number.isFinite(cond) || !Number.isFinite(uncond)) {
      throw new RangeError("ACE planner model logits must be finite before CFG");
    }
    const delta = Math.fround(cond - uncond);
    output[token] = Math.fround(uncond + Math.fround(scale * delta));
  }
  return output;
}

/** Pure whitelist mask; disallowed logits become negative infinity. */
export function maskAcePlannerLogits(
  logits: ArrayLike<number>,
  allowed: AcePlannerAllowedTokens,
): Float32Array {
  const source = copyPlannerLogits(logits, "constraint input");
  if (allowed.kind === "all") {
    requireFiniteCandidate(source, "ACE planner constraint");
    return source;
  }
  const output = new Float32Array(source.length);
  output.fill(Number.NEGATIVE_INFINITY);
  const admitted = new Uint8Array(source.length);
  const admit = (tokenId: number): void => {
    requireTokenId(tokenId, source.length, "ACE planner allowed token");
    if (admitted[tokenId] !== 0) {
      throw new RangeError(`ACE planner constraint repeats token ID ${tokenId}`);
    }
    admitted[tokenId] = 1;
    output[tokenId] = source[tokenId]!;
  };

  if (allowed.kind === "ids") {
    if (allowed.tokenIds.length === 0) {
      throw new RangeError("ACE planner constraint cannot have an empty token set");
    }
    for (const tokenId of allowed.tokenIds) admit(tokenId);
  } else {
    requireTokenId(allowed.firstTokenId, source.length, "ACE planner range start");
    if (!Number.isSafeInteger(allowed.tokenCount) || allowed.tokenCount <= 0) {
      throw new RangeError("ACE planner allowed tokenCount must be positive");
    }
    const rangeEnd = allowed.firstTokenId + allowed.tokenCount;
    if (!Number.isSafeInteger(rangeEnd) || rangeEnd > source.length) {
      throw new RangeError("ACE planner allowed token range exceeds the vocabulary");
    }
    for (let tokenId = allowed.firstTokenId; tokenId < rangeEnd; tokenId += 1) {
      admit(tokenId);
    }
    for (const tokenId of allowed.additionalTokenIds ?? []) admit(tokenId);
  }
  requireFiniteCandidate(output, "ACE planner constraint");
  return output;
}

/** Transformers repetition penalty, applied once to each seen token ID. */
export function applyAcePlannerRepetitionPenalty(
  logits: ArrayLike<number>,
  seenTokenIds: readonly number[],
  penalty: number,
): Float32Array {
  const output = copyPlannerLogits(logits, "repetition-penalty input");
  if (!Number.isFinite(penalty) || penalty <= 0) {
    throw new RangeError("ACE planner repetition penalty must be positive and finite");
  }
  if (penalty === 1 || seenTokenIds.length === 0) return output;
  const roundedPenalty = Math.fround(penalty);
  const visited = new Uint8Array(output.length);
  for (const tokenId of seenTokenIds) {
    requireTokenId(tokenId, output.length, "ACE planner seen token");
    if (visited[tokenId] !== 0) continue;
    visited[tokenId] = 1;
    const value = output[tokenId]!;
    if (value === Number.NEGATIVE_INFINITY) continue;
    output[tokenId] = Math.fround(
      value < 0 ? value * roundedPenalty : value / roundedPenalty,
    );
  }
  requireFiniteCandidate(output, "ACE planner repetition penalty");
  return output;
}

/** Upstream threshold semantics retain every tie at the kth-largest logit. */
export function applyAcePlannerTopK(
  logits: ArrayLike<number>,
  topK: number,
): Float32Array {
  const output = copyPlannerLogits(logits, "top-k input");
  if (!Number.isSafeInteger(topK) || topK < 0 || topK > output.length) {
    throw new RangeError("ACE planner topK must be between zero and vocabulary size");
  }
  if (topK === 0) {
    requireFiniteCandidate(output, "ACE planner top-k");
    return output;
  }
  const order = stableFiniteTokenOrder(output);
  if (order.length === 0) throw new RangeError("ACE planner top-k has no finite candidate");
  if (topK >= order.length) return output;
  const threshold = output[order[topK - 1]!]!;
  for (let tokenId = 0; tokenId < output.length; tokenId += 1) {
    if (output[tokenId]! < threshold) output[tokenId] = Number.NEGATIVE_INFINITY;
  }
  return output;
}

/**
 * Diagnostic host-exp reproduction of the stable upstream nucleus cutoff.
 *
 * The first token whose cumulative probability crosses top-p is retained by
 * shifting the removal mask one position to the right. `topP` is rounded to
 * FP32 before comparison, matching a Torch scalar/tensor comparison. Product
 * generation uses {@link createAcePlannerBrowserTopPKeep} instead.
 */
export function applyAcePlannerTopP(
  logits: ArrayLike<number>,
  topP: number,
): Float32Array {
  const output = copyPlannerLogits(logits, "top-p input");
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new RangeError("ACE planner topP must be in the interval (0, 1]");
  }
  const order = stableFiniteTokenOrder(output);
  if (order.length === 0) throw new RangeError("ACE planner top-p has no finite candidate");
  if (topP === 1) return output;

  const probabilities = softmaxF32(output);
  const threshold = Math.fround(topP);
  let cumulative = 0;
  let previousCrossed = false;
  for (const tokenId of order) {
    const remove = previousCrossed;
    cumulative = Math.fround(cumulative + probabilities[tokenId]!);
    previousCrossed = cumulative > threshold;
    if (remove) output[tokenId] = Number.NEGATIVE_INFINITY;
  }
  requireFiniteCandidate(output, "ACE planner top-p");
  return output;
}

/** Diagnostic host-exp temperature division followed by FP32 softmax. */
export function createAcePlannerSamplingWeights(
  logits: ArrayLike<number>,
  temperature: number,
): Float32Array {
  const scaled = copyPlannerLogits(logits, "temperature input");
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("ACE planner temperature must be positive and finite");
  }
  const roundedTemperature = Math.fround(temperature);
  for (let tokenId = 0; tokenId < scaled.length; tokenId += 1) {
    if (scaled[tokenId] !== Number.NEGATIVE_INFINITY) {
      scaled[tokenId] = Math.fround(scaled[tokenId]! / roundedTemperature);
    }
  }
  return softmaxF32(scaled);
}

/**
 * Accepted browser-v1 temperature/softmax.
 *
 * Every transcendental result comes from {@link acePlannerBrowserExpF32}; no
 * host `Math.exp`, WGSL `exp`, or device-dependent reduction is involved.
 * Inputs, exponent weights, the sequential sum, and outputs are all rounded to
 * binary32 at their declared boundaries.
 */
export function createAcePlannerBrowserSamplingWeights(
  logits: ArrayLike<number>,
  temperature: number,
): Float32Array {
  const scaled = copyPlannerLogits(logits, "browser-v1 temperature input");
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("ACE planner temperature must be positive and finite");
  }
  const roundedTemperature = Math.fround(temperature);
  for (let tokenId = 0; tokenId < scaled.length; tokenId += 1) {
    if (scaled[tokenId] !== Number.NEGATIVE_INFINITY) {
      scaled[tokenId] = Math.fround(scaled[tokenId]! / roundedTemperature);
    }
  }
  return browserSoftmaxF32(scaled);
}

/**
 * Deterministic software exp for a non-positive binary32 input.
 *
 * The input is reduced to `2^n * exp(r)`, `0 <= r < ln(2)`. `exp(r)` is the
 * fixed degree-12 Taylor polynomial evaluated in the listed binary64 order;
 * the exact power-of-two table is built without a host transcendental. The
 * result is rounded once to binary32. Degree, coefficients, range reduction,
 * and underflow behavior are part of `ace-browser-softmax-v1`.
 */
export function acePlannerBrowserExpF32(value: number): number {
  const input = Math.fround(value);
  if (!Number.isFinite(input) || input > 0) {
    throw new RangeError(
      "ACE planner browser exp requires a finite non-positive binary32 input",
    );
  }
  if (input === 0) return 1;
  const exponent = Math.floor(input / BROWSER_SOFTMAX_LN2);
  // exponent=-151 is needed to round values immediately around half of the
  // smallest positive binary32 subnormal. Smaller bins always round to zero.
  if (exponent < -151) return 0;
  const remainder = input - exponent * BROWSER_SOFTMAX_LN2;
  let polynomial = BROWSER_EXP_TAYLOR[12]!;
  for (let degree = 11; degree >= 0; degree -= 1) {
    polynomial = polynomial * remainder + BROWSER_EXP_TAYLOR[degree]!;
  }
  return Math.fround(polynomial * BROWSER_NEGATIVE_POWERS_OF_TWO[-exponent]!);
}

/** Stable browser-v1 nucleus keep mask over untempered logits. */
export function createAcePlannerBrowserTopPKeep(
  logits: ArrayLike<number>,
  topP: number,
): Uint8Array {
  const source = copyPlannerLogits(logits, "browser-v1 top-p input");
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new RangeError("ACE planner topP must be in the interval (0, 1]");
  }
  const order = stableFiniteTokenOrder(source);
  if (order.length === 0) {
    throw new RangeError("ACE planner browser-v1 top-p has no finite candidate");
  }
  const keep = new Uint8Array(source.length);
  if (topP === 1) {
    for (const tokenId of order) keep[tokenId] = 1;
    return keep;
  }

  const probabilities = browserSoftmaxF32(source);
  const threshold = Math.fround(topP);
  let cumulative = 0;
  let previousCrossed = false;
  for (const tokenId of order) {
    if (!previousCrossed) keep[tokenId] = 1;
    cumulative = Math.fround(cumulative + probabilities[tokenId]!);
    previousCrossed = cumulative > threshold;
  }
  return keep;
}

/** Accepted production implementation for browser-owned planner sampling. */
export const ACE_BROWSER_SOFTMAX_V1: AcePlannerSoftmaxOracle = Object.freeze({
  id: ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId,
  status: "accepted" as const,
  reference: ACE_PLANNER_SOFTMAX_ACCEPTANCE.independentReference,
  topPKeep(logits: Float32Array, topP: number): Uint8Array {
    return createAcePlannerBrowserTopPKeep(logits, topP);
  },
  weights(logits: Float32Array, temperature: number): Float32Array {
    return createAcePlannerBrowserSamplingWeights(logits, temperature);
  },
});

/** Execute the complete, output-affecting planner sampling pipeline. */
export function sampleAcePlannerToken(
  input: AcePlannerTokenSampleInput,
): AcePlannerTokenSample {
  const softmax = input.softmax ?? ACE_BROWSER_SOFTMAX_V1;
  const logits = createAcePlannerFilteredLogits(input);
  requireAcceptedSoftmax(softmax);
  const weights = validateAcePlannerSamplingWeights(
    softmax.weights(logits.slice(), input.parameters.temperature),
    logits.length,
    softmax.id,
  );
  const word = requireU32(input.word, "ACE planner categorical word");
  const tokenId = aceCategoricalTokenFromWord(weights, word);
  let positiveCandidateCount = 0;
  for (const weight of weights) {
    if (weight > 0) positiveCandidateCount += 1;
  }
  return Object.freeze({ tokenId, word, positiveCandidateCount });
}

/**
 * Benchmark-only OPT-0084 full/sparse candidate-domain sampler.
 *
 * Pass a retained workspace for allocation-free steady-state storage. The
 * default is convenient for differential tests, not representative timing.
 *
 * @internal
 */
export function sampleAcePlannerTokenOpt0084(
  input: AcePlannerTokenSampleInput,
  workspace = new AceOpt0084PlannerSamplingWorkspace(),
): AcePlannerTokenSample {
  return workspace.sample(input);
}

/**
 * Exact full-vector-equivalent sampling over one contiguous global-ID domain.
 *
 * Omitted vocabulary entries are the `-inf` entries that the ordinary path
 * would remove before top-k/top-p/softmax. Because the retained domain is in
 * ascending global-token order, compact FP32 reductions and categorical
 * traversal visit every nonzero value in the same order as the full vector.
 */
export function sampleAcePlannerCompactToken(
  input: AcePlannerCompactTokenSampleInput,
): AcePlannerTokenSample {
  const prepared = prepareAcePlannerCompactInput(input);
  const sampled = sampleAcePlannerToken(prepared.localInput);
  return Object.freeze({
    ...sampled,
    tokenId: prepared.firstTokenId + sampled.tokenId,
  });
}

/**
 * Benchmark-only OPT-0084 compact contiguous candidate-domain sampler.
 * @internal
 */
export function sampleAcePlannerCompactTokenOpt0084(
  input: AcePlannerCompactTokenSampleInput,
  workspace = new AceOpt0084PlannerSamplingWorkspace(),
): AcePlannerTokenSample {
  return workspace.sampleCompact(input);
}

/** Compact-domain form of {@link createAcePlannerFilteredLogits}. */
export function createAcePlannerCompactFilteredLogits(
  input: AcePlannerCompactFilterInput,
): Float32Array {
  return createAcePlannerFilteredLogits(
    prepareAcePlannerCompactInput({ ...input, word: 0 }).localInput,
  );
}

/** Execute every untempered filtering step and return logical FP32 logits. */
export function createAcePlannerFilteredLogits(
  input: AcePlannerFilterInput,
): Float32Array {
  const parameters = input.parameters;
  const softmax = input.softmax ?? ACE_BROWSER_SOFTMAX_V1;
  let logits: Float32Array;
  if (input.unconditionalLogits === undefined) {
    if (parameters.guidanceScale !== 1) {
      throw new TypeError("ACE planner CFG sampling requires unconditional logits");
    }
    logits = maskAcePlannerLogits(
      input.conditionalLogits,
      input.allowedTokens,
    );
  } else {
    // Pinned code generation selects the valid audio-code subspace before
    // CFG, avoiding undefined `-inf - -inf` work over the other 153k rows.
    // Applying the same final whitelist here also fuses the duration/EOS mask;
    // every retained token has the identical FP32 CFG result as the upstream
    // select-combine-mask sequence.
    const conditionalAllowed = maskAcePlannerLogits(
      input.conditionalLogits,
      input.preCfgAllowedTokens ?? input.allowedTokens,
    );
    const unconditionalAllowed = maskAcePlannerLogits(
      input.unconditionalLogits,
      input.preCfgAllowedTokens ?? input.allowedTokens,
    );
    logits = combineAcePlannerCfgLogitsOnAllowedSubspace(
      conditionalAllowed,
      unconditionalAllowed,
      parameters.guidanceScale,
    );
    // The FSM is invoked after CFG. In semantic mode this is where EOS stays
    // blocked for the first N draws, then becomes the only admitted token.
    logits = maskAcePlannerLogits(logits, input.allowedTokens);
  }
  logits = applyAcePlannerRepetitionPenalty(
    logits,
    input.seenTokenIds,
    parameters.repetitionPenalty,
  );
  logits = applyAcePlannerTopK(logits, parameters.topK);
  if (parameters.topP < 1) {
    requireAcceptedSoftmax(softmax);
    logits = applyAcePlannerAcceptedTopP(logits, parameters.topP, softmax);
  }
  return logits;
}

/**
 * Diagnostic host softmax used only by cross-language contract vectors.
 * Production sampling rejects this receipt because `Math.exp` is not part of
 * the versioned browser distribution.
 */
export const ACE_DIAGNOSTIC_HOST_SOFTMAX: AcePlannerSoftmaxOracle =
  Object.freeze({
    id: "diagnostic-host-math-exp-f32-v1",
    status: "diagnostic" as const,
    reference: "scripts/planner_sampling_reference.py",
    topPKeep(logits: Float32Array, topP: number): Uint8Array {
      const filtered = applyAcePlannerTopP(logits, topP);
      return Uint8Array.from(filtered, (value) => Number.isFinite(value) ? 1 : 0);
    },
    weights(logits: Float32Array, temperature: number): Float32Array {
      return createAcePlannerSamplingWeights(logits, temperature);
    },
  });

function applyAcePlannerAcceptedTopP(
  logits: Float32Array,
  topP: number,
  oracle: AcePlannerSoftmaxOracle,
): Float32Array {
  const output = logits.slice();
  const keep = oracle.topPKeep(output.slice(), topP);
  if (!(keep instanceof Uint8Array) || keep.length !== output.length) {
    throw new RangeError(
      `ACE planner softmax oracle ${oracle.id} returned the wrong top-p mask shape`,
    );
  }
  for (let tokenId = 0; tokenId < output.length; tokenId += 1) {
    const value = keep[tokenId]!;
    if (value !== 0 && value !== 1) {
      throw new RangeError(
        `ACE planner softmax oracle ${oracle.id} returned invalid top-p mask value`,
      );
    }
    if (value === 0) output[tokenId] = Number.NEGATIVE_INFINITY;
  }
  requireFiniteCandidate(output, "ACE planner accepted top-p");
  return output;
}

function requireAcceptedSoftmax(oracle: AcePlannerSoftmaxOracle): void {
  if (oracle.status !== "accepted") {
    throw new Error(
      `ACE planner softmax oracle ${oracle.id} is diagnostic-only; ` +
        `use ${ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId} for generation`,
    );
  }
}

function prepareAcePlannerCompactInput(
  input: AcePlannerCompactTokenSampleInput,
): Readonly<{
  readonly firstTokenId: number;
  readonly localInput: AcePlannerTokenSampleInput;
}> {
  const candidateCount = input.conditionalLogits.length;
  if (!Number.isSafeInteger(candidateCount) || candidateCount <= 0) {
    throw new RangeError("ACE planner compact logits must have a non-empty safe length");
  }
  if (
    !Number.isSafeInteger(input.vocabularySize) ||
    input.vocabularySize <= 0
  ) {
    throw new RangeError("ACE planner compact vocabulary size must be positive");
  }
  requireTokenId(
    input.firstTokenId,
    input.vocabularySize,
    "ACE planner compact first token",
  );
  const domainEnd = input.firstTokenId + candidateCount;
  if (!Number.isSafeInteger(domainEnd) || domainEnd > input.vocabularySize) {
    throw new RangeError("ACE planner compact candidate domain exceeds the vocabulary");
  }
  if (
    input.unconditionalLogits !== undefined &&
    input.unconditionalLogits.length !== candidateCount
  ) {
    throw new RangeError(
      "ACE planner compact conditional and unconditional logits differ in length",
    );
  }
  if (
    !Number.isSafeInteger(input.parameters.topK) ||
    input.parameters.topK < 0 ||
    input.parameters.topK > input.vocabularySize
  ) {
    throw new RangeError(
      "ACE planner compact topK must be between zero and vocabulary size",
    );
  }

  const localSeenTokenIds: number[] = [];
  for (const tokenId of input.seenTokenIds) {
    requireTokenId(tokenId, input.vocabularySize, "ACE planner seen token");
    const local = tokenId - input.firstTokenId;
    if (local >= 0 && local < candidateCount) localSeenTokenIds.push(local);
  }
  // In the full vector, topK greater than or equal to the number of finite
  // candidates is a no-op. The compact helper's local vocabulary is smaller,
  // so express that same state with the existing topK=0 convention.
  const compactTopK = input.parameters.topK >= candidateCount
    ? 0
    : input.parameters.topK;
  const parameters: AcePlannerSamplingParameters = compactTopK ===
      input.parameters.topK
    ? input.parameters
    : Object.freeze({ ...input.parameters, topK: compactTopK });
  const localInput: AcePlannerTokenSampleInput = {
    conditionalLogits: input.conditionalLogits,
    ...(input.unconditionalLogits === undefined
      ? {}
      : { unconditionalLogits: input.unconditionalLogits }),
    seenTokenIds: localSeenTokenIds,
    allowedTokens: { kind: "all" },
    parameters,
    word: input.word,
    ...(input.softmax === undefined ? {} : { softmax: input.softmax }),
  };
  return Object.freeze({ firstTokenId: input.firstTokenId, localInput });
}

function combineAcePlannerCfgLogitsOnAllowedSubspace(
  conditional: Float32Array,
  unconditional: Float32Array,
  guidanceScale: number,
): Float32Array {
  const length = requireMatchingLogits(conditional, unconditional);
  if (!Number.isFinite(guidanceScale) || guidanceScale < 1) {
    throw new RangeError("ACE planner guidanceScale must be finite and at least one");
  }
  const scale = Math.fround(guidanceScale);
  const output = new Float32Array(length);
  output.fill(Number.NEGATIVE_INFINITY);
  for (let token = 0; token < length; token += 1) {
    const cond = conditional[token]!;
    const uncond = unconditional[token]!;
    if (cond === Number.NEGATIVE_INFINITY && uncond === Number.NEGATIVE_INFINITY) {
      continue;
    }
    if (!Number.isFinite(cond) || !Number.isFinite(uncond)) {
      throw new RangeError("ACE planner CFG allowed subspace differs between rows");
    }
    output[token] = Math.fround(
      uncond + Math.fround(scale * Math.fround(cond - uncond)),
    );
  }
  requireFiniteCandidate(output, "ACE planner CFG");
  return output;
}

function validateAcePlannerSamplingWeights(
  weights: Float32Array,
  expectedLength: number,
  oracleId: string,
): Float32Array {
  if (!(weights instanceof Float32Array) || weights.length !== expectedLength) {
    throw new RangeError(
      `ACE planner softmax oracle ${oracleId} returned the wrong weight shape`,
    );
  }
  let total = 0;
  let positive = 0;
  for (let tokenId = 0; tokenId < weights.length; tokenId += 1) {
    const value = weights[tokenId]!;
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(
        `ACE planner softmax oracle ${oracleId} returned invalid weight ${tokenId}`,
      );
    }
    if (value > 0) positive += 1;
    total += value;
  }
  if (positive === 0 || !Number.isFinite(total) || total <= 0) {
    throw new RangeError(`ACE planner softmax oracle ${oracleId} has no probability mass`);
  }
  return weights;
}

const OPT0084_PRE_CFG_ALLOWED_BIT = 1;
const OPT0084_FINAL_ALLOWED_BIT = 2;
const OPT0084_BOTH_ALLOWED_BITS =
  OPT0084_PRE_CFG_ALLOWED_BIT | OPT0084_FINAL_ALLOWED_BIT;
const OPT0084_SEEN_BIT = 4;

function requireNonEmptySafeLength(
  values: ArrayLike<number>,
  label: string,
): number {
  if (!Number.isSafeInteger(values.length) || values.length <= 0) {
    throw new RangeError(`${label} must have a non-empty safe length`);
  }
  return values.length;
}

function validatePlannerLogitRow(
  logits: ArrayLike<number>,
  label: string,
): void {
  requireNonEmptySafeLength(logits, label);
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    requirePlannerLogit(logits[tokenId], `${label} ${tokenId}`);
  }
}

function requireFiniteOrNegativeInfinityF32(
  value: number,
  label: string,
): number {
  const rounded = Math.fround(value);
  if (Number.isNaN(rounded) || rounded === Number.POSITIVE_INFINITY) {
    throw new RangeError(`${label} must be finite or negative infinity`);
  }
  return rounded;
}

function markAcePlannerAllowedTokens(
  allowed: AcePlannerAllowedTokens,
  vocabularySize: number,
  flags: Uint8Array,
  bit: number,
): void {
  const admit = (tokenId: number): void => {
    requireTokenId(tokenId, vocabularySize, "ACE planner allowed token");
    if ((flags[tokenId]! & bit) !== 0) {
      throw new RangeError(`ACE planner constraint repeats token ID ${tokenId}`);
    }
    flags[tokenId] = flags[tokenId]! | bit;
  };
  if (allowed.kind === "all") {
    for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
      flags[tokenId] = flags[tokenId]! | bit;
    }
    return;
  }
  if (allowed.kind === "ids") {
    if (allowed.tokenIds.length === 0) {
      throw new RangeError("ACE planner constraint cannot have an empty token set");
    }
    for (const tokenId of allowed.tokenIds) admit(tokenId);
    return;
  }
  requireTokenId(
    allowed.firstTokenId,
    vocabularySize,
    "ACE planner range start",
  );
  if (!Number.isSafeInteger(allowed.tokenCount) || allowed.tokenCount <= 0) {
    throw new RangeError("ACE planner allowed tokenCount must be positive");
  }
  const rangeEnd = allowed.firstTokenId + allowed.tokenCount;
  if (!Number.isSafeInteger(rangeEnd) || rangeEnd > vocabularySize) {
    throw new RangeError("ACE planner allowed token range exceeds the vocabulary");
  }
  for (let tokenId = allowed.firstTokenId; tokenId < rangeEnd; tokenId += 1) {
    admit(tokenId);
  }
  for (const tokenId of allowed.additionalTokenIds ?? []) admit(tokenId);
}

function hasFiniteAllowedRawLogit(
  logits: ArrayLike<number>,
  flags: Uint8Array,
  bit: number,
): boolean {
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    if ((flags[tokenId]! & bit) === 0) continue;
    if (Number.isFinite(Math.fround(logits[tokenId]!))) return true;
  }
  return false;
}

function hasFiniteRawLogit(logits: ArrayLike<number>): boolean {
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    if (Number.isFinite(Math.fround(logits[tokenId]!))) return true;
  }
  return false;
}

function requireAcePlannerGuidanceScale(guidanceScale: number): number {
  if (!Number.isFinite(guidanceScale) || guidanceScale < 1) {
    throw new RangeError(
      "ACE planner guidanceScale must be finite and at least one",
    );
  }
  return Math.fround(guidanceScale);
}

function combineAcePlannerCfgCandidate(
  conditional: number,
  unconditional: number,
  roundedGuidanceScale: number,
): number {
  const cond = requireFiniteOrNegativeInfinityF32(
    conditional,
    "ACE planner conditional CFG candidate",
  );
  const uncond = requireFiniteOrNegativeInfinityF32(
    unconditional,
    "ACE planner unconditional CFG candidate",
  );
  if (
    cond === Number.NEGATIVE_INFINITY &&
    uncond === Number.NEGATIVE_INFINITY
  ) {
    return Number.NEGATIVE_INFINITY;
  }
  if (!Number.isFinite(cond) || !Number.isFinite(uncond)) {
    throw new RangeError("ACE planner CFG allowed subspace differs between rows");
  }
  const combined = Math.fround(
    uncond + Math.fround(
      roundedGuidanceScale * Math.fround(cond - uncond),
    ),
  );
  if (Number.isNaN(combined) || combined === Number.POSITIVE_INFINITY) {
    throw new RangeError(
      "ACE planner CFG output must be finite or negative infinity",
    );
  }
  return combined;
}

function requireAcePlannerTopK(
  topK: number,
  vocabularySize: number,
  compact: boolean,
): void {
  if (
    !Number.isSafeInteger(topK) ||
    topK < 0 ||
    topK > vocabularySize
  ) {
    throw new RangeError(
      compact
        ? "ACE planner compact topK must be between zero and vocabulary size"
        : "ACE planner topK must be between zero and vocabulary size",
    );
  }
}

function requireAcePlannerTopP(topP: number): void {
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new RangeError("ACE planner topP must be in the interval (0, 1]");
  }
}

function requireOpt0084BrowserSoftmax(
  oracle: AcePlannerSoftmaxOracle | undefined,
): void {
  if (oracle === undefined || oracle === ACE_BROWSER_SOFTMAX_V1) return;
  requireAcceptedSoftmax(oracle);
  throw new Error(
    "OPT-0084 benchmark arm requires the exact ace-browser-softmax-v1 oracle",
  );
}

function binarySearchUint32(
  values: Uint32Array,
  length: number,
  target: number,
): number {
  let low = 0;
  let high = length;
  while (low < high) {
    const middle = low + ((high - low) >>> 1);
    const value = values[middle]!;
    if (value < target) low = middle + 1;
    else high = middle;
  }
  return low < length && values[low] === target ? low : -1;
}

/**
 * Ascending integer key for descending binary32 order. Both signed zero
 * encodings intentionally collapse to the +0 key so LSD stability supplies
 * the ascending-token-ID tie break.
 */
function descendingFloatRadixKey(rawWord: number): number {
  let word = rawWord >>> 0;
  if ((word & 0x7fff_ffff) === 0) word = 0;
  return (word & 0x8000_0000) !== 0
    ? word
    : (word ^ 0x7fff_ffff) >>> 0;
}

function nextOpt0084Capacity(required: number): number {
  let capacity = 1;
  while (capacity < required) capacity *= 2;
  if (!Number.isSafeInteger(capacity)) {
    throw new RangeError("OPT-0084 workspace capacity exceeds safe storage");
  }
  return capacity;
}

const BROWSER_SOFTMAX_LN2 = 0.6931471805599453;

// Binary64 literals for 1 / degree!. Their order and degree are versioned.
const BROWSER_EXP_TAYLOR = Object.freeze([
  1,
  1,
  0.5,
  0.16666666666666666,
  0.041666666666666664,
  0.008333333333333333,
  0.001388888888888889,
  0.0001984126984126984,
  0.0000248015873015873,
  0.0000027557319223985893,
  2.755731922398589e-7,
  2.505210838544172e-8,
  2.08767569878681e-9,
] as const);

/** Exact binary64 values 2^-0 through 2^-151, without `Math.pow`. */
const BROWSER_NEGATIVE_POWERS_OF_TWO = (() => {
  const powers = new Float64Array(152);
  powers[0] = 1;
  for (let index = 1; index < powers.length; index += 1) {
    powers[index] = powers[index - 1]! * 0.5;
  }
  return powers;
})();

function browserSoftmaxF32(logits: Float32Array): Float32Array {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    if (Number.isNaN(value) || value === Number.POSITIVE_INFINITY) {
      throw new RangeError(
        "ACE planner browser-v1 logits must be finite or negative infinity",
      );
    }
    if (value > maximum) maximum = value;
  }
  if (!Number.isFinite(maximum)) {
    throw new RangeError("ACE planner browser-v1 softmax has no finite candidate");
  }

  const weights = new Float32Array(logits.length);
  let total = 0;
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const value = logits[tokenId]!;
    if (value === Number.NEGATIVE_INFINITY) continue;
    const exponent = acePlannerBrowserExpF32(Math.fround(value - maximum));
    weights[tokenId] = exponent;
    total = Math.fround(total + exponent);
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new RangeError("ACE planner browser-v1 FP32 softmax sum is invalid");
  }
  let positive = 0;
  for (let tokenId = 0; tokenId < weights.length; tokenId += 1) {
    if (weights[tokenId] === 0) continue;
    weights[tokenId] = Math.fround(weights[tokenId]! / total);
    if (weights[tokenId]! > 0) positive += 1;
  }
  if (positive === 0) {
    throw new RangeError(
      "ACE planner browser-v1 softmax underflowed every candidate",
    );
  }
  return weights;
}

function softmaxF32(logits: Float32Array): Float32Array {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    if (Number.isNaN(value) || value === Number.POSITIVE_INFINITY) {
      throw new RangeError("ACE planner softmax logits must be finite or negative infinity");
    }
    if (value > maximum) maximum = value;
  }
  if (!Number.isFinite(maximum)) {
    throw new RangeError("ACE planner softmax has no finite candidate");
  }

  const weights = new Float32Array(logits.length);
  let total = 0;
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const value = logits[tokenId]!;
    if (value === Number.NEGATIVE_INFINITY) continue;
    const exponent = Math.fround(Math.exp(Math.fround(value - maximum)));
    weights[tokenId] = exponent;
    total = Math.fround(total + exponent);
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new RangeError("ACE planner FP32 softmax sum is invalid");
  }
  let positive = 0;
  for (let tokenId = 0; tokenId < weights.length; tokenId += 1) {
    if (weights[tokenId] === 0) continue;
    weights[tokenId] = Math.fround(weights[tokenId]! / total);
    if (weights[tokenId]! > 0) positive += 1;
  }
  if (positive === 0) {
    throw new RangeError("ACE planner FP32 softmax underflowed every candidate");
  }
  return weights;
}

function stableFiniteTokenOrder(logits: Float32Array): number[] {
  const order: number[] = [];
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    if (Number.isFinite(logits[tokenId])) order.push(tokenId);
  }
  order.sort((left, right) => {
    const difference = logits[right]! - logits[left]!;
    return difference === 0 ? left - right : difference;
  });
  return order;
}

function copyPlannerLogits(logits: ArrayLike<number>, label: string): Float32Array {
  if (!Number.isSafeInteger(logits.length) || logits.length <= 0) {
    throw new RangeError(`${label} must have a non-empty safe length`);
  }
  const output = new Float32Array(logits.length);
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    output[tokenId] = requirePlannerLogit(logits[tokenId], `${label} ${tokenId}`);
  }
  return output;
}

function requireMatchingLogits(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): number {
  if (!Number.isSafeInteger(left.length) || left.length <= 0) {
    throw new RangeError("ACE planner logits must have a non-empty safe length");
  }
  if (right.length !== left.length) {
    throw new RangeError("ACE planner conditional and unconditional logits differ in length");
  }
  return left.length;
}

function requirePlannerLogit(value: number | undefined, label: string): number {
  if (value === undefined || Number.isNaN(value) || value === Number.POSITIVE_INFINITY) {
    throw new RangeError(`${label} must be finite or negative infinity`);
  }
  return Math.fround(value);
}

function requireFiniteCandidate(logits: Float32Array, label: string): void {
  for (const value of logits) {
    if (Number.isFinite(value)) return;
  }
  throw new RangeError(`${label} removed every finite candidate`);
}

function requireTokenId(tokenId: number, vocabularySize: number, label: string): void {
  if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
    throw new RangeError(`${label} ${String(tokenId)} is outside the vocabulary`);
  }
}

function requireU32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer`);
  }
  return value >>> 0;
}

function requireNonNegativeSafeBigInt(
  value: number | bigint,
  label: string,
): bigint {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${label} must be a non-negative safe integer`);
    }
    return BigInt(value);
  }
  if (value < 0n) throw new RangeError(`${label} must be non-negative`);
  return value;
}
