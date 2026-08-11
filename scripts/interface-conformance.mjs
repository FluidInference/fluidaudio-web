// Backend ↔ type-layer conformance gate: parses ComputeContext out of
// src/gpu/compute.d.ts (the authoritative interface) and asserts BOTH backend
// classes implement every required method. Catches the drift class the d.ts
// rewrite fixed (a member declared but missing on one backend, or implemented
// but never declared) — tsc can't, because allowJs/checkJs are off.
//   node scripts/interface-conformance.mjs
import { GpuContext } from "../src/gpu/compute.js";
import { WasmContext } from "../src/gpu/wasm-context.js";
import { computeInterfaceMethods } from "./lib/compute-interface.mjs";

const { required, optional } = computeInterfaceMethods();

const backends = [
  ["GpuContext", GpuContext],
  ["WasmContext", WasmContext],
];
let failed = false;
for (const [name, cls] of backends) {
  const missing = required.filter((m) => typeof cls.prototype[m] !== "function");
  if (missing.length) {
    failed = true;
    console.error(`✗ ${name} is missing required ComputeContext methods: ${missing.join(", ")}`);
  } else {
    console.log(`✓ ${name} implements all ${required.length} required ComputeContext methods`);
  }
}

// Reverse direction: public methods on a backend that the interface never
// declares (drift where the impl grew a member nobody typed). GPU-only members
// must at least appear in the optional block.
const declared = new Set([...required, ...optional]);
for (const [name, cls] of backends) {
  const undeclared = Object.getOwnPropertyNames(cls.prototype).filter(
    (m) => m !== "constructor" && !m.startsWith("_") && typeof cls.prototype[m] === "function" && !declared.has(m),
  );
  if (undeclared.length) {
    failed = true;
    console.error(`✗ ${name} has public methods missing from ComputeContext (add as required or optional): ${undeclared.join(", ")}`);
  }
}

if (failed) process.exit(1);
console.log("INTERFACE CONFORMANCE OK");
