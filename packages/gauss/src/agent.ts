import type { Message, AgentOptions, AgentResult, ToolDef } from './types.js';
import type { Provider } from './provider.js';
import { getBackend } from './backend.js';

/** Agent wraps a Rust-powered AI agent with tool loop. */
export class Agent {
  private name: string;
  private provider: Provider;
  private options: AgentOptions;

  constructor(name: string, provider: Provider, options: AgentOptions = {}) {
    this.name = name;
    this.provider = provider;
    this.options = options;
  }

  /** Run the agent with the given messages. */
  async run(messages: Message[]): Promise<AgentResult> {
    const backend = getBackend();
    const tools = this.options.tools ?? [];
    const toolSchemas = tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }));

    const hasExecuteFns = tools.some((t) => typeof t.execute === 'function');

    if (hasExecuteFns) {
      // Build executor map for tools with execute functions
      const executorMap = new Map<string, NonNullable<ToolDef['execute']>>();
      for (const t of tools) {
        if (t.execute) executorMap.set(t.name, t.execute);
      }

      const toolExecutor = async (callJson: string): Promise<string> => {
        const { tool, args } = JSON.parse(callJson);
        const fn = executorMap.get(tool);
        if (!fn) throw new Error(`No execute function for tool: ${tool}`);
        const result = await fn(args);
        return JSON.stringify(result);
      };

      return backend.agentRunWithToolExecutor(
        this.name,
        this.provider.getHandle(),
        toolSchemas,
        messages,
        this.options,
        toolExecutor,
      );
    }

    return backend.agentRun(
      this.name,
      this.provider.getHandle(),
      toolSchemas,
      messages,
      this.options,
    );
  }
}

/** Create an agent instance. */
export function createAgent(
  name: string,
  provider: Provider,
  options?: AgentOptions,
): Agent {
  return new Agent(name, provider, options);
}
