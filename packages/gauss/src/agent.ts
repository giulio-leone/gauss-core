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
    const tools = (this.options.tools ?? []).map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }));

    return backend.agentRun(
      this.name,
      this.provider.getHandle(),
      tools,
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
