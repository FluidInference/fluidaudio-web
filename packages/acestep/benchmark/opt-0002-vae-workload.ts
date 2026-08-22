import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderOperation,
  type AceVaeDecoderQuantumWorkPolicy,
} from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0001_VAE_LEGACY_UNIFORM_QUANTUM_WORK_POLICY,
  ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
} from "./opt-0001-vae-workload.js";
import { ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS } from
  "../src/runtime/scheduler.js";

export type AceOpt0002VaeOperationFamily = AceVaeDecoderOperation["kind"];

export interface AceOpt0002VaePolicyWorkload {
  readonly quantumWorkPolicy: AceVaeDecoderQuantumWorkPolicy;
  readonly decoderQuantumCount: number;
  readonly primitiveDispatchCount: number;
  readonly commandBufferCountIncludingReadback: number;
  readonly configuredCooperativeIdleMilliseconds: number;
  readonly familyQuantumCounts: Readonly<
    Record<AceOpt0002VaeOperationFamily, number>
  >;
  readonly maximumLogicalOutputElements: number;
  readonly maximumEstimatedMultiplyAccumulates: number;
  readonly outputBudgetViolationCount: number;
  readonly convolutionMacBudgetViolationCount: number;
}

export interface AceOpt0002VaeWorkloadComparison {
  readonly schemaVersion: 1;
  readonly reportKind: "ace-opt-0002-vae-work-aware-quanta";
  readonly latentWindowFrames: 256;
  readonly baseline: AceOpt0002VaePolicyWorkload;
  readonly candidate: AceOpt0002VaePolicyWorkload;
  readonly decoderQuantumReduction: number;
  readonly decoderQuantumReductionRatio: number;
  readonly configuredCooperativeIdleReductionMilliseconds: number;
}

/** Static, production-planner comparison for the authenticated 256-frame window. */
export function createAceOpt0002VaeWorkloadComparison():
  AceOpt0002VaeWorkloadComparison {
  const graph = planAceVaeDecoder(256);
  const baseline = summarizePolicy(
    graph,
    ACE_OPT_0001_VAE_LEGACY_UNIFORM_QUANTUM_WORK_POLICY,
  );
  const candidate = summarizePolicy(graph, ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY);
  return Object.freeze({
    schemaVersion: 1,
    reportKind: "ace-opt-0002-vae-work-aware-quanta",
    latentWindowFrames: 256,
    baseline,
    candidate,
    decoderQuantumReduction:
      baseline.decoderQuantumCount - candidate.decoderQuantumCount,
    decoderQuantumReductionRatio:
      baseline.decoderQuantumCount / candidate.decoderQuantumCount,
    configuredCooperativeIdleReductionMilliseconds:
      baseline.configuredCooperativeIdleMilliseconds -
      candidate.configuredCooperativeIdleMilliseconds,
  });
}

function summarizePolicy(
  graph: ReturnType<typeof planAceVaeDecoder>,
  policy: AceVaeDecoderQuantumWorkPolicy,
): AceOpt0002VaePolicyWorkload {
  const cooperative = planAceVaeDecoderQuanta(
    graph,
    ACE_OPT_0001_VAE_TRANSPOSE_PARTS,
    policy,
  );
  const familyQuantumCounts: Record<AceOpt0002VaeOperationFamily, number> = {
    conv1d: 0,
    "conv-transpose1d": 0,
    snake: 0,
    add: 0,
  };
  let maximumLogicalOutputElements = 0;
  let maximumEstimatedMultiplyAccumulates = 0;
  let outputBudgetViolationCount = 0;
  let convolutionMacBudgetViolationCount = 0;
  for (const quantum of cooperative.quanta) {
    familyQuantumCounts[quantum.operationKind] += 1;
    maximumLogicalOutputElements = Math.max(
      maximumLogicalOutputElements,
      quantum.logicalOutputCount,
    );
    maximumEstimatedMultiplyAccumulates = Math.max(
      maximumEstimatedMultiplyAccumulates,
      quantum.estimatedMaximumMultiplyAccumulates,
    );
    if (quantum.logicalOutputCount > policy.maximumOutputElements) {
      outputBudgetViolationCount += 1;
    }
    if (
      quantum.estimatedMaximumMultiplyAccumulates >
        policy.maximumConvolutionMultiplyAccumulates
    ) {
      convolutionMacBudgetViolationCount += 1;
    }
  }
  return Object.freeze({
    quantumWorkPolicy: cooperative.quantumWorkPolicy,
    decoderQuantumCount: cooperative.quantumCount,
    primitiveDispatchCount: cooperative.primitiveDispatchCount,
    commandBufferCountIncludingReadback: cooperative.quantumCount + 1,
    configuredCooperativeIdleMilliseconds:
      cooperative.quantumCount * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    familyQuantumCounts: Object.freeze(familyQuantumCounts),
    maximumLogicalOutputElements,
    maximumEstimatedMultiplyAccumulates,
    outputBudgetViolationCount,
    convolutionMacBudgetViolationCount,
  });
}
