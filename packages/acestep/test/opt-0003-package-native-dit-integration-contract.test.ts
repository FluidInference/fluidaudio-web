import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  OPT_0003_DIT_FILE_COUNT,
  OPT_0003_DIT_LOGICAL_TENSOR_BYTES,
  OPT_0003_DIT_RESIDENT_FILE_BYTES,
  OPT_0003_DIT_TILE_MAJOR_GEMM_COUNT,
  OPT_0003_DIT_TENSOR_COUNT,
  OPT_0003_FINAL_LATENT_ELEMENTS,
  OPT_0003_PACKAGE_NATIVE_SHAPE,
  OPT_0003_PORTABLE_COMMAND_BUFFER_COUNT,
  OPT_0003_REFERENCE_MANIFEST_SHA256,
  OPT_0003_SUBGROUP_COMMAND_BUFFER_COUNT,
} from "./browser/opt-0003-package-native-dit-integration.js";
import {
  planAceDitGpuBackendMemory,
  planAceDitPhysicalCommandBufferCount,
} from "../src/webgpu/dit-backend.js";

describe("OPT-0003 package-native DiT browser integration contract", () => {
  it("pins the regenerated rev4 package inventory and bounded real graph", () => {
    expect(OPT_0003_REFERENCE_MANIFEST_SHA256).toBe(
      "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
    );
    expect(OPT_0003_PACKAGE_NATIVE_SHAPE).toEqual({
      batch: 1,
      latentFrames: 129,
      conditionTokens: 1,
    });
    expect(OPT_0003_DIT_TENSOR_COUNT).toBe(476);
    expect(OPT_0003_DIT_TILE_MAJOR_GEMM_COUNT).toBe(271);
    expect(OPT_0003_DIT_FILE_COUNT).toBe(50);
    expect(OPT_0003_DIT_LOGICAL_TENSOR_BYTES).toBe(3_150_917_760);
    expect(OPT_0003_DIT_RESIDENT_FILE_BYTES).toBe(3_150_917_888);
    expect(OPT_0003_FINAL_LATENT_ELEMENTS).toBe(8_256);
  });

  it("pins backend-specific physical command and memory accounting", () => {
    expect(planAceDitPhysicalCommandBufferCount(
      OPT_0003_PACKAGE_NATIVE_SHAPE,
      "portable",
    ) + 1).toBe(OPT_0003_PORTABLE_COMMAND_BUFFER_COUNT);
    expect(planAceDitPhysicalCommandBufferCount(
      OPT_0003_PACKAGE_NATIVE_SHAPE,
      "fixed32-subgroups",
    ) + 1).toBe(OPT_0003_SUBGROUP_COMMAND_BUFFER_COUNT);
    for (const backend of ["portable", "fixed32-subgroups"] as const) {
      const memory = planAceDitGpuBackendMemory(
        "reference-bf16",
        OPT_0003_PACKAGE_NATIVE_SHAPE,
        OPT_0003_DIT_RESIDENT_FILE_BYTES,
        backend,
      );
      expect(memory).toMatchObject({
        modelProfile: "reference-bf16",
        gemmBackend: backend,
        residentWeightBytes: 3_150_917_888,
        readbackBufferBytes: 33_024,
        accountedGpuBytes: 3_170_141_952,
        boundedCpuBytes: 173_876,
      });
      expect(memory.arena.allocatedArenaBytes).toBe(19_191_040);
    }
  });

  it("runs sequential production backends and compares every final bit", () => {
    const source = readFileSync(new URL(
      "./browser/opt-0003-package-native-dit-integration.ts",
      import.meta.url,
    ), "utf8");
    const portableCall = source.indexOf("ACE_REFERENCE_PORTABLE_PROFILE");
    const subgroupCall = source.indexOf("ACE_REFERENCE_SUBGROUP_PROFILE", portableCall + 1);
    expect(portableCall).toBeGreaterThan(-1);
    expect(subgroupCall).toBeGreaterThan(portableCall);
    expect(source).toContain("AceGpuTensorPhase.load(");
    expect(source).toContain("AceDitGpuBackend.create({");
    expect(source).toContain("phase = undefined");
    expect(source).toContain("await backend?.destroy()");
    expect(source).toContain("await device.queue.onSubmittedWorkDone()");
    expect(source).toContain("bitMismatchCount !== 0 || !identicalSha256");
    expect(source).toContain("unchangedInitialBitCount !== 0");
    expect(source).toContain("completeDitGraph: true");
    expect(source).toContain("denoisingEvaluations: 8");
    expect(source).toContain("simultaneousHeavyweightPhaseCount: 1");
  });

  it("authenticates bounded acquisition, exact progress, and heartbeat scope", () => {
    const source = readFileSync(new URL(
      "./browser/opt-0003-package-native-dit-integration.ts",
      import.meta.url,
    ), "utf8");
    expect(source).toContain("files: inventory.ditFiles");
    expect(source).toContain("acquireAceModelFiles({");
    expect(source).toContain("expectedManifestSha256: OPT_0003_REFERENCE_MANIFEST_SHA256");
    expect(source).toContain("event.compiledQuanta !== index + 1");
    expect(source).toContain("event.graph.completedQuanta !== completed");
    expect(source).toContain("event.graph.commandId");
    expect(source).toContain("evaluationLayerCounts.size !== 24 * 8");
    expect(source).toContain("compileProgressIntervalsMilliseconds");
    expect(source).toContain("runProgressIntervalsMilliseconds");
    expect(source).toContain("result.cooperativeIdleMs !== memory.commandBufferCount - 1");
    expect(source).toContain("const compileHeartbeat = startHeartbeat()");
    expect(source).toContain("const runHeartbeat = startHeartbeat()");
    expect(source).toContain("validateHeartbeat(runHeartbeatResult");
    expect(source).toContain("runtimeEvents.length !== 0");
    expect(source).toContain("context.destroy()");
  });

  it("keeps the browser page explicit about evidence scope", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0003-package-native-dit-integration.html",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("24-layer");
    expect(html).toContain("eight-evaluation");
    expect(html).toContain("never keeps two");
    expect(html).toContain("not a reportable performance benchmark");
  });
});
