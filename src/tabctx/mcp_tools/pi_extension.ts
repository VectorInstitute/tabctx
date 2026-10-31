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

function parse(stdout: string, what: string): any {
  try {
    return JSON.parse(stdout);
  } catch {
    throw new Error(`${what}: ${BIN} printed no JSON: ${stdout.slice(0, 500)}`);
  }
}

export default async function (pi: ExtensionAPI) {
  const listed = await pi.exec(BIN, ["tools"], { timeout: 60_000 });
  if (listed.code !== 0) {
    throw new Error(`tabctx: '${BIN} tools' failed: ${listed.stderr || listed.stdout}`);
  }
  const specs: ToolSpec[] = parse(listed.stdout, "tools").tools;

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
        const out = parse(result.stdout, spec.name);
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
