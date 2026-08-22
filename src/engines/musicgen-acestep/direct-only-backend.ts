import {
  AceQwenBpeTokenizer,
  createAceWebGpuPipelineBackend,
  type AceAcquiredModelFiles,
  type AceGenerationContext,
  type AceGenerationRequest,
  type AceGenerationResult,
  type AceInitializationContext,
  type AceLoadedPackageManifest,
  type AcePackageTensorRecord,
  type AcePipelineBackend,
  type AceQwenBpeDefinition,
  type AceRuntimeDiagnostics,
  type AceTokenizerAssetBundle,
  type AceTokenizerKind,
  type AceWebGpuPipelineOptions,
  type AceWorkerConfiguration,
  type LoadedAceTokenizer,
} from "ace-step-1.5.wgsl";

import directSemanticValidationTensors from
  "./generated/direct-semantic-validation-tensors.json";

const DIRECT_ONLY_DEPENDENCY_PROPERTY = "dependencies";
const PLANNER_TOKENIZER_ALIASES = Object.freeze([
  ["assets/planner/tokenizer.json", "assets/qwen/tokenizer.json"],
  [
    "assets/planner/tokenizer_config.json",
    "assets/qwen/tokenizer_config.json",
  ],
  ["assets/planner/chat_template.jinja", "assets/qwen/chat_template.jinja"],
] as const);

const FORBIDDEN_DIRECT_FILE =
  /^(?:assets\/planner\/|weights\/(?:planner|semantic)\/|licenses\/ACE-Step-acestep-5Hz-lm-0\.6B-README\.md$)/u;
const DIRECT_SEMANTIC_VALIDATION_TENSOR_COUNT = 30;

interface DirectOnlyAcquireOptions {
  readonly kind: "main" | "dit-dense" | "vae";
  readonly [key: string]: unknown;
}

type ManifestLoader = (
  configuration: AceWorkerConfiguration,
  signal: AbortSignal,
) => Promise<AceLoadedPackageManifest>;

type ModelAcquirer = (
  options: DirectOnlyAcquireOptions,
) => Promise<AceAcquiredModelFiles>;

type TokenizerLoader = (
  kind: AceTokenizerKind,
  assets: AceTokenizerAssetBundle,
) => Promise<LoadedAceTokenizer>;

interface DirectOnlyDependencySeam {
  loadManifest: ManifestLoader;
  acquireModel: ModelAcquirer;
  loadTokenizer: TokenizerLoader;
}

type StockBackendFactory = (
  options: AceWebGpuPipelineOptions,
) => AcePipelineBackend;

/**
 * Build the production backend with the demo's direct-only boundary installed.
 *
 * The ACE package intentionally keeps its dependency object private. This demo
 * is pinned to that implementation, so the structural seam is checked at
 * runtime and fails closed if the package layout changes.
 */
export function createAceDirectOnlyWebGpuPipelineBackend(
  options: AceWebGpuPipelineOptions = {},
  stockBackendFactory: StockBackendFactory = createAceWebGpuPipelineBackend,
): AcePipelineBackend {
  const stock = stockBackendFactory(options);
  installDirectOnlyDependencies(stock);
  return new DirectOnlyBackend(stock);
}

class DirectOnlyBackend implements AcePipelineBackend {
  constructor(private readonly stock: AcePipelineBackend) {}

  async initialize(
    configuration: AceWorkerConfiguration,
    context: AceInitializationContext,
  ): Promise<AceRuntimeDiagnostics> {
    return await this.stock.initialize(configuration, context);
  }

  async generate(
    request: AceGenerationRequest,
    context: AceGenerationContext,
  ): Promise<AceGenerationResult> {
    if (request.planner.mode !== "disabled") {
      throw new DOMException(
        "This ACE-Step demo supports direct generation only",
        "NotSupportedError",
      );
    }
    return await this.stock.generate(request, context);
  }

  async releaseResult(result: AceGenerationResult): Promise<void> {
    await this.stock.releaseResult(result);
  }

  async dispose(): Promise<void> {
    await this.stock.dispose();
  }
}

function installDirectOnlyDependencies(stock: AcePipelineBackend): void {
  const dependencies = requireDependencySeam(stock);
  const loadManifest = dependencies.loadManifest.bind(dependencies);
  const acquireModel = dependencies.acquireModel.bind(dependencies);
  const loadTokenizer = dependencies.loadTokenizer.bind(dependencies);

  const replacements: DirectOnlyDependencySeam = {
    loadManifest: async (configuration, signal) => {
      const loaded = await loadManifest(configuration, signal);
      assertDirectOnlyManifest(loaded);
      return addInertSemanticValidationMetadata(loaded);
    },
    acquireModel: async (options) => {
      const acquired = await acquireModel(options);
      if (options.kind !== "main") return acquired;
      return addUnusedPlannerTokenizerAliases(acquired);
    },
    loadTokenizer: async (kind, assets) => {
      if (kind === "text") return await loadTokenizer(kind, assets);
      return unusedPlannerTokenizer();
    },
  };

  try {
    dependencies.loadManifest = replacements.loadManifest;
    dependencies.acquireModel = replacements.acquireModel;
    dependencies.loadTokenizer = replacements.loadTokenizer;
  } catch (error) {
    throw new Error(
      "ACE direct-only dependency seam is not writable",
      { cause: error },
    );
  }

  if (
    dependencies.loadManifest !== replacements.loadManifest ||
    dependencies.acquireModel !== replacements.acquireModel ||
    dependencies.loadTokenizer !== replacements.loadTokenizer
  ) {
    throw new Error("ACE direct-only dependency seam installation failed");
  }
}

function requireDependencySeam(
  stock: AcePipelineBackend,
): DirectOnlyDependencySeam {
  const record = stock as unknown as Record<string, unknown>;
  const value = record[DIRECT_ONLY_DEPENDENCY_PROPERTY];
  if (
    typeof value !== "object" ||
    value === null ||
    typeof (value as Record<string, unknown>).loadManifest !== "function" ||
    typeof (value as Record<string, unknown>).acquireModel !== "function" ||
    typeof (value as Record<string, unknown>).loadTokenizer !== "function"
  ) {
    throw new Error(
      "ACE backend no longer exposes the pinned direct-only dependency seam",
    );
  }
  return value as DirectOnlyDependencySeam;
}

function assertDirectOnlyManifest(loaded: AceLoadedPackageManifest): void {
  const forbiddenFile = loaded.manifest.files.find((file) =>
    FORBIDDEN_DIRECT_FILE.test(file.name)
  );
  const forbiddenTensor = Object.entries(loaded.manifest.tensors).find(
    ([, tensor]) => tensor.phase === "planner" || tensor.phase === "semantic",
  );
  const forbiddenSource = loaded.manifest.source.find(
    (source) => source.repository === "ACE-Step/acestep-5Hz-lm-0.6B",
  );
  if (
    forbiddenFile !== undefined ||
    forbiddenTensor !== undefined ||
    forbiddenSource !== undefined
  ) {
    const identity = forbiddenFile?.name ?? forbiddenTensor?.[0] ??
      forbiddenSource?.key ?? "unknown planner material";
    throw new Error(
      `ACE direct-only manifest contains forbidden planner material: ${identity}`,
    );
  }
}

function addInertSemanticValidationMetadata(
  loaded: AceLoadedPackageManifest,
): AceLoadedPackageManifest {
  const validationTensors = directSemanticValidationTensors as unknown as
    Readonly<Record<string, AcePackageTensorRecord>>;
  const entries = Object.entries(validationTensors);
  if (entries.length !== DIRECT_SEMANTIC_VALIDATION_TENSOR_COUNT) {
    throw new Error(
      "ACE direct-only semantic validation metadata has the wrong tensor count",
    );
  }

  const fileNames = new Set(loaded.manifest.files.map((file) => file.name));
  const sourceKeys = new Set(loaded.manifest.source.map((source) => source.key));
  const tensors: Record<string, AcePackageTensorRecord> = {
    ...loaded.manifest.tensors,
  };
  for (const [name, tensor] of entries) {
    const sourceKey = tensor.source.split(":", 1)[0]!;
    if (
      tensor.phase !== "semantic" ||
      Object.hasOwn(tensors, name) ||
      fileNames.has(tensor.shard) ||
      !sourceKeys.has(sourceKey)
    ) {
      throw new Error(
        `ACE direct-only semantic validation metadata is invalid: ${name}`,
      );
    }
    tensors[name] = Object.freeze({ ...tensor });
  }

  const manifest = Object.freeze({
    ...loaded.manifest,
    tensors: Object.freeze(tensors),
  });
  return Object.freeze({ ...loaded, manifest });
}

function addUnusedPlannerTokenizerAliases(
  acquired: AceAcquiredModelFiles,
): AceAcquiredModelFiles {
  const forbidden = [...acquired.files.keys()].find((name) =>
    FORBIDDEN_DIRECT_FILE.test(name)
  );
  if (forbidden !== undefined) {
    throw new Error(
      `ACE direct-only acquisition unexpectedly retained ${forbidden}`,
    );
  }

  const files = new Map(acquired.files);
  for (const [plannerName, textName] of PLANNER_TOKENIZER_ALIASES) {
    const textAsset = files.get(textName);
    if (textAsset === undefined) {
      throw new Error(`ACE direct-only package is missing ${textName}`);
    }
    files.set(plannerName, textAsset);
  }
  return Object.freeze({ ...acquired, files });
}

function unusedPlannerTokenizer(): LoadedAceTokenizer {
  const definition: AceQwenBpeDefinition = Object.freeze({
    vocabulary: new Map<string, number>(),
    tokensById: Object.freeze([]),
    mergeRanks: new Map<string, number>(),
  });
  return Object.freeze({
    tokenizer: new AceQwenBpeTokenizer("planner", definition),
    chatTemplate: "",
    assetIdentity: Object.freeze({
      tokenizerSha256: "direct-only-unused",
      tokenizerConfigSha256: "direct-only-unused",
      chatTemplateSha256: "direct-only-unused",
    }),
  });
}
