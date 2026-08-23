import type {
  AceGenerationRequest,
  AceGenerationResult,
} from "../api.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from "./diagnostics.js";
import type {
  AceGenerationProgress,
  AceInitializationProgress,
} from "./stages.js";
import type {
  AceModelLoadSource,
  AceWorkerConfiguration,
} from "./protocol.js";

export interface AceInitializationContext {
  readonly modelSource: AceModelLoadSource;
  readonly signal: AbortSignal;
  readonly onProgress: (progress: AceInitializationProgress) => void;
  readonly onDiagnostic: (diagnostic: AceDiagnostic) => void;
}

export interface AceGenerationContext {
  readonly signal: AbortSignal;
  readonly onProgress: (progress: AceGenerationProgress) => void;
  readonly onDiagnostic: (diagnostic: AceDiagnostic) => void;
  /**
   * Opt-in Stage-1 provenance capture. This may hash bounded CPU tensors and
   * emit the complete planner semantic sequence; normal generation leaves it
   * disabled to avoid diagnostic work and retaining user text.
   */
  readonly captureTrace?: boolean;
}

/**
 * Dependency boundary for the dedicated worker shell. Implementations own all
 * model-cache, WebGPU, scheduling, WASM, and audio resources. No placeholder
 * backend is supplied: a worker cannot claim readiness until a real backend is
 * injected.
 */
export interface AcePipelineBackend {
  initialize(
    configuration: AceWorkerConfiguration,
    context: AceInitializationContext,
  ): Promise<AceRuntimeDiagnostics>;
  generate(
    request: AceGenerationRequest,
    context: AceGenerationContext,
  ): Promise<AceGenerationResult>;
  /**
   * Reclaim a generated result that the worker did not successfully publish.
   * Published results transfer this responsibility to the UI, which releases
   * the OPFS audio identity after playback/download is finished.
   */
  releaseResult(result: AceGenerationResult): void | Promise<void>;
  /**
   * Release every partially or fully initialized resource. This operation must
   * be idempotent. After it resolves, the backend must be clean enough for a
   * fresh `initialize` call; the worker relies on that transaction boundary to
   * permit retry after failed or cancelled initialization.
   */
  dispose(): void | Promise<void>;
}
