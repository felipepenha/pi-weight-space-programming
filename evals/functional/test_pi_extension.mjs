import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const piVersion = fs.readFileSync(path.join(os.homedir(), ".pi/agent/install/current-version"), "utf-8").trim();
const piBase = path.join(
  os.homedir(),
  `.pi/agent/install/releases/${piVersion}/node_modules/@earendil-works/pi-coding-agent/dist`
);

const { loadExtensions } = await import(path.join(piBase, "core/extensions/loader.js"));
const { ExtensionRunner } = await import(path.join(piBase, "core/extensions/runner.js"));
const { loadSkillsFromDir } = await import(path.join(piBase, "core/skills.js"));
const { DefaultResourceLoader } = await import(path.join(piBase, "core/resource-loader.js"));
const { SettingsManager } = await import(path.join(piBase, "core/settings-manager.js"));

const extResult = await loadExtensions([path.join(repoRoot, "extensions/index.ts")], repoRoot);
if (extResult.errors.length > 0) {
  throw new Error(`Extension load errors: ${JSON.stringify(extResult.errors)}`);
}

const ext = extResult.extensions[0];
const tools = Array.from(ext.tools.keys());
const commands = Array.from(ext.commands.keys());
const flags = Array.from(ext.flags.keys());
const handlers = Array.from(ext.handlers.keys());

const skillsResult = loadSkillsFromDir({
  dir: path.join(repoRoot, "skills"),
  source: "pi-weight-space-programming",
});

// Verify Option 1 (<workspace>/.pi/settings.json) and Option 2 (-e ./extensions/index.ts) resource discovery
const tmpWorkspace = fs.mkdtempSync(path.join(os.tmpdir(), "wsp-ext-test-"));
const agentDir = path.join(os.homedir(), ".pi/agent");
const settingsManager = SettingsManager.create(tmpWorkspace, agentDir, { projectTrusted: true });
const resourceLoader = new DefaultResourceLoader({
  cwd: tmpWorkspace,
  agentDir,
  settingsManager,
  additionalExtensionPaths: [path.join(repoRoot, "extensions/index.ts")],
  projectTrusted: true,
});
await resourceLoader.reload();

const runner = new ExtensionRunner(extResult.extensions, extResult.runtime, tmpWorkspace, null, null);
const discovered = await runner.emitResourcesDiscover(tmpWorkspace, "startup");
const wrapMeta = (items) =>
  (items || []).map((item) => ({
    path: item.path,
    metadata: {
      source: item.extensionPath,
      scope: "temporary",
      origin: "top-level",
      baseDir: path.dirname(item.path),
    },
  }));
resourceLoader.extendResources({
  skillPaths: wrapMeta(discovered.skillPaths),
  promptPaths: wrapMeta(discovered.promptPaths),
  themePaths: wrapMeta(discovered.themePaths),
});

const summary = {
  passed: true,
  extension_path: "extensions/index.ts",
  load_errors: extResult.errors,
  tools_registered: tools,
  commands_registered: commands,
  cli_flags_registered: flags,
  event_hooks_registered: handlers,
  skills_loaded: skillsResult.skills.map((s) => s.name),
  skill_diagnostics: skillsResult.diagnostics,
  prompts_loaded: resourceLoader.getPrompts().prompts.map((p) => p.name),
};

fs.rmSync(tmpWorkspace, { recursive: true, force: true });
fs.writeFileSync(path.join(__dirname, "extension_loader_results.json"), JSON.stringify(summary, null, 2) + "\n");
console.log(JSON.stringify(summary, null, 2));
