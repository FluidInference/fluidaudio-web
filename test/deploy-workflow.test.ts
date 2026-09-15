import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { parse } from "yaml";

it("gates Pages on every check for the same main revision", () => {
  const ci = parse(readFileSync(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8"));
  const deploy = parse(readFileSync(new URL("../.github/workflows/deploy.yml", import.meta.url), "utf8"));
  expect(ci.jobs.deploy.needs.toSorted()).toEqual(
    Object.keys(ci.jobs)
      .filter((name) => name !== "deploy")
      .toSorted(),
  );
  expect(ci.jobs.deploy.uses).toBe("./.github/workflows/deploy.yml");
  expect(ci.jobs.deploy.if).toBe("github.ref == 'refs/heads/main' && (github.event_name == 'push' || github.event_name == 'workflow_dispatch')");
  expect(Object.keys(deploy.on)).toEqual(["workflow_call"]);
  expect(ci.jobs.build.steps.map((step: { run?: string }) => step.run)).toEqual(expect.arrayContaining(["npm run test:integration", "npm run sdk:test"]));
  // Checkout must use the caller's revision, not fetch a newer main after tests.
  expect(deploy.jobs.build.steps.find((step: { uses?: string }) => step.uses?.startsWith("actions/checkout@")).with?.ref).toBeUndefined();
});
