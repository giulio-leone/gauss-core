/** Supported provider types. */
export type ProviderType = 'openai' | 'anthropic' | 'google' | 'groq' | 'ollama' | 'deepseek';

/** Provider configuration. */
export interface ProviderOptions {
  apiKey: string;
  baseUrl?: string;
  timeoutMs?: number;
  maxRetries?: number;
  organization?: string;
}

/** Chat message. */
export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** Tool definition. */
export interface ToolDef {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
  execute?: (args: Record<string, unknown>) => Promise<unknown>;
}

/** Agent configuration. */
export interface AgentOptions {
  instructions?: string;
  maxSteps?: number;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  seed?: number;
  stopOnTool?: string;
  outputSchema?: Record<string, unknown>;
  tools?: ToolDef[];
}

/** Agent execution result. */
export interface AgentResult {
  text: string;
  steps: number;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  structuredOutput?: unknown;
}

/** Direct generation result. */
export interface GenerateResult {
  text: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  finishReason: string;
}
