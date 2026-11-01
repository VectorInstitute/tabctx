// tabctx tools for pi (https://github.com/badlogic/pi-mono), which has no MCP.
//
// Install:  tabctx-mcp pi-extension > ~/.pi/agent/extensions/tabctx.ts
//           (or .pi/extensions/tabctx.ts in a project)
//
// At startup it asks `tabctx-mcp tools` for the MCP server's tools (names,
// descriptions, JSON Schemas) and registers each as a pi tool; a call runs
// `tabctx-mcp call <tool> <json>`, which goes through the same server code
// as an MCP client would. No shell is involved: pi can run without bash.
// TABCTX_MCP_BIN overrides the command (default: tabctx-mcp on PATH).
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";

const BIN = process.env.TABCTX_MCP_BIN || "tabctx-mcp";
const TIMEOUT_MS = 10 * 60 * 1000;

interface ToolSpec {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

interface ExecResult {
  stdout: string;
  stderr: string;
  code: number;
  killed?: boolean;
}

// The command's JSON; a timeout, a crash or stray output as an error that
// says which (with stderr), not just "no JSON".
function parse(result: ExecResult, what: string): any {
  if (result.killed) {
    throw new Error(`${what}: ${BIN} was stopped (timeout or cancelled) ${result.stderr.slice(-500)}`);
  }
  try {
    return JSON.parse(result.stdout);
  } catch {
    throw new Error(
      `${what}: ${BIN} exited ${result.code} without JSON: ${(result.stderr || result.stdout).slice(-500)}`,
    );
  }
}

export default async function (pi: ExtensionAPI) {
  const listed = await pi.exec(BIN, ["tools"], { timeout: 60_000 });
  if (listed.code !== 0) {
    throw new Error(`tabctx: '${BIN} tools' failed: ${listed.stderr || listed.stdout}`);
  }
  const specs: ToolSpec[] = parse(listed, "tools").tools;

  for (const spec of specs) {
    pi.registerTool({
      name: `tabctx_${spec.name}`,
      label: `tabctx ${spec.name}`,
      description: spec.description,
      parameters: Type.Unsafe(spec.input_schema),
      async execute(_id, params, signal) {
        const result = await pi.exec(BIN, ["call", spec.name, JSON.stringify(params ?? {})], {
          signal,
          timeout: TIMEOUT_MS,
        });
        const out = parse(result, spec.name);
        if (result.code !== 0 || out.error) {
          throw new Error(out.error ?? result.stderr);
        }
        return {
          content: [{ type: "text", text: JSON.stringify(out, null, 2) }],
          details: out,
        };
      },
    });
  }
}
