// Shared parser for the authoritative compute interface: extracts the method
// names of ComputeContext from src/gpu/compute.d.ts. Used by the structural
// gate (interface-conformance.mjs) and the numeric gate (backend-conformance.mjs).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import ts from "typescript";

export function computeInterfaceMethods() {
  const dtsPath = fileURLToPath(new URL("../../src/gpu/compute.d.ts", import.meta.url));
  const src = ts.createSourceFile(dtsPath, readFileSync(dtsPath, "utf8"), ts.ScriptTarget.Latest, true);
  let required = null;
  let optional = null;
  src.forEachChild((node) => {
    if (ts.isInterfaceDeclaration(node) && node.name.text === "ComputeContext") {
      required = [];
      optional = [];
      for (const m of node.members) {
        if (!ts.isMethodSignature(m) || !ts.isIdentifier(m.name)) continue;
        (m.questionToken ? optional : required).push(m.name.text);
      }
    }
  });
  if (!required) throw new Error("ComputeContext interface not found in compute.d.ts");
  return { required, optional };
}
