export {
  ACE_CHANNEL_COUNT,
  ACE_DIRECT_DCW_CONFIGURATION,
  ACE_GENERATION_PROFILE_IDS,
  ACE_MAX_DURATION_SECONDS,
  ACE_MIN_DURATION_SECONDS,
  ACE_SAMPLE_RATE_HZ,
  ACE_TURBO_DENOISING_EVALUATIONS,
  ACE_TURBO_V1_CORRECTNESS_PROFILE,
  ACE_THINKING_DCW_CONFIGURATION,
  DEFAULT_ACE_PLANNER_CONFIGURATION,
  aceSeed,
  assertAceGenerationRequest,
  checkSupport,
  isAceGenerationRequest,
  resolveAceDynamicConditionalWeighting,
} from "./api.js";
export type {
  AceDynamicConditionalWeightingConfiguration,
  AceGenerateOptions,
  AceGenerationMetrics,
  AceGenerationProfileId,
  AceGenerationRequest,
  AceGenerationResult,
  AceMusicMetadata,
  AcePlannerConfiguration,
  AcePlannerDisabled,
  AcePlannerEnabled,
  AceSupportOptions,
} from "./api.js";

export {
  ACE_AUDIO_ACOUSTIC_CHANNELS,
  ACE_CONTEXT_CHANNELS,
  ACE_DIT_INPUT_CHANNELS,
  ACE_DIT_LAYER_TYPES,
  ACE_DIT_PATCH_SIZE,
  ACE_GRAPH_CONTRACT,
  ACE_LATENT_RATE_HZ,
  ACE_SEMANTIC_RATE_HZ,
  ACE_VAE_TEMPORAL_STRIDE,
  assertAceGraphContract,
  deriveAceDurationGraphShape,
} from "./model/graph-contract.js";
export type { AceDurationGraphShape } from "./model/graph-contract.js";

export {
  ACE_SEED_CONTRACT,
  aceCategoricalTokenFromWord,
  aceGaussianF32FromWord,
  aceOpenUnitFloat64FromWord,
  acePlannerCategoricalToken,
  aceRandomWord,
  aceRandomWords,
  canonicalizeSeed,
  fillAceDiffusionNoise,
  isAceSeed,
  philox4x32_10,
} from "./runtime/seed.js";
export type {
  AceRandomStream,
  AceSeed,
  PhiloxWords,
} from "./runtime/seed.js";

export {
  ACE_GENERATION_STAGES,
  ACE_INITIALIZATION_STAGES,
  AceInitializationProgressSequence,
  AceProgressSequence,
  generationStagePlan,
  isAceGenerationProgress,
  isAceGenerationStage,
  isAceInitializationProgress,
  isAceInitializationStage,
  isAceProgressUnit,
} from "./runtime/stages.js";
export type {
  AceGenerationProgress,
  AceGenerationStage,
  AceInitializationProgress,
  AceInitializationStage,
  AceProgressUnit,
  AceStageTiming,
} from "./runtime/stages.js";

export type {
  AceDiagnostic,
  AceDiagnosticSeverity,
  AceDiagnosticValue,
  AceFailureContext,
  AceRuntimeDiagnostics,
} from "./runtime/diagnostics.js";
export {
  ACE_PLANNER_SOURCE_REVISION,
  ACE_SOURCE_REVISION,
  PARAKEET_REFERENCE_REVISION,
} from "./runtime/diagnostics.js";

export {
  ACE_FATAL_GPU_ERROR_CODES,
  isAceClientMessage,
  isAceFatalGpuErrorCode,
  isAceWorkerMessage,
  serializeAceWorkerError,
} from "./runtime/protocol.js";
export type {
  AceClientMessage,
  AceFatalGpuErrorCode,
  AceModelLoadSource,
  AceSerializedWorkerError,
  AceWorkerConfiguration,
  AceWorkerVaePackageConfiguration,
  AceWorkerMessage,
} from "./runtime/protocol.js";

export type {
  AceGenerationContext,
  AceInitializationContext,
  AcePipelineBackend,
} from "./runtime/pipeline.js";

export {
  AceWebGpuPipelineBackend,
  createAceWebGpuPipelineBackend,
} from "./runtime/webgpu-pipeline.js";
export type {
  AceWebGpuPipelineOptions,
} from "./runtime/webgpu-pipeline.js";

export {
  releaseAceAudioOutput,
} from "./runtime/audio-output.js";
export type {
  AceAudioOutputStorage,
  AceCommittedAudioOutput,
} from "./runtime/audio-output.js";

export {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  AceFifoGraphOwner,
  submitAceCommandBuffersCooperatively,
} from "./runtime/scheduler.js";
export type {
  AceCooperativeSubmissionOptions,
  AceGpuSchedulingProgress,
  AceGpuSchedulingResult,
  AceGraphLease,
  AceRunGpuGraphOptions,
} from "./runtime/scheduler.js";

export {
  ACE_MAX_WEIGHT_SHARD_BYTES,
  ACE_MODEL_SNAPSHOT_REVISION,
  ACE_PACKAGE_ALIGNMENT_BYTES,
  ACE_PACKAGE_CONVERTER_REVISION,
  ACE_PACKAGE_FORMAT,
  ACE_PLANNER_SNAPSHOT_REVISION,
  ACE_PORTABLE_STORAGE_BINDING_BYTES,
  ACE_REFERENCE_SOURCE_REVISION,
  AcePackageManifestError,
  parseAcePackageManifest,
} from "./model/manifest.js";

export {
  ACE_MODEL_STORAGE_HEADROOM_BYTES,
  aceRuntimePackageFiles,
  acquireAceModelFiles,
  acquireAceModelFilesFromOpfs,
  planAceModelAcquisition,
  requestAceModelStoragePersistence,
} from "./model/acquire.js";
export type {
  AceAcquireModelOptions,
  AceAcquiredModelFiles,
  AceModelAcquisitionPlan,
  AceModelAcquisitionProgress,
  AceModelCacheBackend,
} from "./model/acquire.js";
export type {
  AcePackageAccounting,
  AcePackageFileKind,
  AcePackageFileRecord,
  AcePackageLicenseRecord,
  AcePackageManifest,
  AcePackageProfile,
  AcePackageProvenance,
  AcePackageSourceRecord,
  AcePackageTensorRecord,
  AceTensorDtype,
  AceTensorLifetime,
  AceTensorPhase,
} from "./model/manifest.js";

export {
  ACE_MAX_MANIFEST_BYTES,
  ACE_MODEL_TRANSPORT_CHUNK_BYTES,
  AceModelTransportError,
  fetchAceModelAsset,
  loadAcePackageManifest,
} from "./model/package.js";
export type {
  AceFetchModelAssetOptions,
  AceLoadPackageManifestOptions,
  AceLoadedPackageManifest,
  AceManifestLoadProgress,
  AceModelAssetProgress,
  AceModelAssetTransaction,
} from "./model/package.js";

export {
  AceOpfsModelCache,
  deleteAceModelCache,
  inspectAceModelCache,
  inspectAceModelStorage,
} from "./model/cache.js";

export {
  ACE_GPU_UPLOAD_CHUNK_BYTES,
  uploadAcePackageFileToGpu,
} from "./model/gpu-upload.js";
export type {
  AceGpuUploadOptions,
  AceGpuUploadProgress,
} from "./model/gpu-upload.js";

export {
  AceGpuTensorPhase,
} from "./model/gpu-tensors.js";
export type {
  AceGpuLogicalTensor,
  AceGpuTensorPart,
  AceGpuTensorPhaseProgress,
  AceLoadGpuTensorPhaseOptions,
} from "./model/gpu-tensors.js";

export {
  ACE_PINNED_TOKENIZER_ASSETS,
  ACE_PLANNER_AUDIO_CODE_COUNT,
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_BASE_ADDED_TOKENS,
  ACE_QWEN_BASE_VOCABULARY_SIZE,
  ACE_QWEN_IM_END_TOKEN_ID,
  ACE_QWEN_IM_START_TOKEN_ID,
  ACE_QWEN_PAD_TOKEN_ID,
  ACE_QWEN_TEXT_POST_TOKEN_ID,
  AceQwenBpeTokenizer,
  aceQwenMergeKey,
  formatAceTextEncoderCaptionInput,
  formatAceTextEncoderLyricsInput,
  loadPinnedAceTokenizer,
  renderAceQwenChat,
} from "./tokenizer/index.js";
export type {
  AceChatMessage,
  AceChatRole,
  AceChatTemplateOptions,
  AceQwenBpeDefinition,
  AceTokenizedBatch,
  AceTokenizerAssetBundle,
  AceTokenizerAssetSource,
  AceTokenizerBatchOptions,
  AceTokenizerDecodeOptions,
  AceTokenizerEncodeOptions,
  AceTokenizerKind,
  LoadedAceTokenizer,
} from "./tokenizer/index.js";
export type {
  AceModelCacheInfo,
  AceOpfsPartialAsset,
  AceStoredModelInfo,
} from "./model/cache.js";

export * from "./runtime/planner-sampling.js";
export * from "./runtime/planner.js";
export * from "./runtime/planner-metadata-fsm.js";

export {
  AceWorkerRuntime,
  AceWorkerRuntimeError,
  installAceWorkerRuntime,
} from "./runtime/worker.js";
export type {
  AceWorkerRuntimeOptions,
  AceWorkerScope,
  AceWorkerState,
} from "./runtime/worker.js";

export {
  ACE_FP16_PORTABLE_PROFILE,
  ACE_MODEL_PROFILE_IDS,
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  ACE_REQUIRED_SUBGROUP_SIZE,
  ACE_REQUIRED_WEBGPU_LIMITS,
  ACE_STOCK_WEBGPU_FEATURES,
  ACE_WEBGPU_LIMIT_NAMES,
  AceWebGpuUnavailableError,
  findAceWebGpuLimitDeficits,
  inspectWebGpuSupport,
  isAceModelProfileId,
  selectAceExecutionProfile,
  snapshotAceWebGpuLimits,
} from "./webgpu/capabilities.js";

export {
  AceGpuArena,
  planAceLifetimeArena,
} from "./webgpu/arena.js";
export type {
  AceArenaBufferPlan,
  AceArenaSlice,
  AceLifetimeAllocation,
  AceLifetimeArenaPlan,
} from "./webgpu/arena.js";

export {
  AceUniformPool,
} from "./webgpu/uniform-pool.js";
export type { AceUniformAllocation } from "./webgpu/uniform-pool.js";

export {
  AceWebGpuDeviceContext,
  requestAceWebGpuDevice,
} from "./webgpu/device.js";
export type {
  AceGpuRuntimeEvent,
  AceRequestWebGpuDeviceOptions,
} from "./webgpu/device.js";

export {
  ACE_PLANNER_QWEN3_CONFIG,
  ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS,
  ACE_QWEN3_ROPE_REFERENCE_PROVENANCE,
  ACE_TEXT_QWEN3_CONFIG,
  AceCorrectnessQwen3Runtime,
  createAceQwen3CausalControlData,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  validateAceQwen3Config,
  validateAceQwen3TiedOutputBindings,
} from "./webgpu/qwen3.js";

export * from "./webgpu/text-encoder.js";
export * from "./webgpu/ace-dit.js";
export * from "./webgpu/ace-dit-package.js";
export * from "./webgpu/vae-decoder.js";
export * from "./webgpu/vae-chunks.js";
export * from "./webgpu/vae-wav.js";
export type {
  AceQwen3AttentionBindings,
  AceQwen3BlockBindings,
  AceQwen3BlockDispatch,
  AceQwen3BlockPlan,
  AceQwen3BlockShape,
  AceQwen3BlockScratch,
  AceQwen3BlockWeights,
  AceQwen3CachedBindings,
  AceQwen3CausalControlData,
  AceQwen3CausalControlInput,
  AceQwen3Config,
  AceQwen3RopeTableShape,
  AceQwen3RopeTables,
  AceQwen3TiedOutputDispatch,
  AceQwen3TiedWeightShard,
  AceQwen3UncachedBindings,
} from "./webgpu/qwen3.js";

export {
  AceCorrectnessEncoderRuntime,
  aceEncoderLayerAttentionMode,
  createAceEncoderControlData,
  createAceEncoderFullControlData,
  createAceEncoderRopeTables,
  planAceEncoderBlock,
  validateAceEncoderBlockBindingAliases,
  validateAceEncoderConfig,
} from "./webgpu/ace-encoder.js";
export type {
  AceEncoderAttentionMode,
  AceEncoderBlockBindings,
  AceEncoderBlockDispatch,
  AceEncoderBlockPlan,
  AceEncoderBlockScratch,
  AceEncoderBlockShape,
  AceEncoderBlockWeights,
  AceEncoderConfig,
  AceEncoderControlData,
} from "./webgpu/ace-encoder.js";

export {
  ACE_CONDITION_ENCODER_CONFIG,
  ACE_DIRECT_CONDITIONER_TENSOR_NAMES,
  ACE_SEMANTIC_TENSOR_NAMES,
  AceCorrectnessSemanticConditionerRuntime,
  aceEncoderLayerTensorNames,
  createAceDirectV1ChunkMask,
  createAceNoReferenceTimbreControls,
  expandAceDetokenizerPatchesCpu,
  packAceSequencesCpu,
  planAceDirectConditioner,
  planAceSemanticDecode,
  validateAceDirectConditionerBindingAliases,
  validateAceSemanticDecodeBindingAliases,
} from "./webgpu/semantic-conditioner.js";
export type {
  AceConditionerQuantum,
  AceDirectConditionerBindings,
  AceDirectConditionerDispatch,
  AceDirectConditionerPlan,
  AceDirectConditionerShape,
  AceEncoderLayerTensorNames,
  AceGraphStage,
  AcePackedSequencesCpu,
  AceSemanticDecodeBindings,
  AceSemanticDecodeDispatch,
  AceSemanticDecodePlan,
  AceSemanticDecodeShape,
} from "./webgpu/semantic-conditioner.js";
export * from "./webgpu/semantic-conditioner-package.js";
export type {
  AceAdapterInfoSnapshot,
  AceExecutionProfile,
  AceExecutionProfileId,
  AceKernelBackend,
  AceModelProfileId,
  AceOptionalWebGpuFeature,
  AceRequiredWebGpuFeature,
  AceSchedulingProfile,
  AceStockWebGpuFeature,
  AceSupportReport,
  AceWebGpuCapabilityReport,
  AceWebGpuFeatureReport,
  AceWebGpuFeatureStatus,
  AceWebGpuLimitDeficit,
  AceWebGpuLimitName,
  AceWebGpuLimits,
  AceWebGpuUnavailableCode,
} from "./webgpu/capabilities.js";
