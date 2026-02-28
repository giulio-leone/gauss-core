/**
 * Backend abstraction — auto-detects NAPI (Node.js) or WASM (Browser/Edge).
 * Falls back to WASM if NAPI is unavailable.
 */

import type { ProviderType, ProviderOptions, Message, AgentOptions, AgentResult, GenerateResult, GenerateWithToolsResult, ToolDef } from './types.js';

/** Backend interface — both NAPI and WASM implement this. */
export interface Backend {
  createProvider(type: ProviderType, model: string, options: ProviderOptions): number;
  destroyProvider(handle: number): void;
  generate(
    handle: number,
    messages: Message[],
    temperature?: number,
    maxTokens?: number,
  ): Promise<GenerateResult>;
  generateWithTools(
    handle: number,
    messages: Message[],
    tools: Array<Omit<ToolDef, 'execute'>>,
    temperature?: number,
    maxTokens?: number,
  ): Promise<GenerateWithToolsResult>;
  agentRun(
    name: string,
    providerHandle: number,
    tools: Array<Omit<ToolDef, 'execute'>>,
    messages: Message[],
    options: AgentOptions,
  ): Promise<AgentResult>;
  agentRunWithToolExecutor(
    name: string,
    providerHandle: number,
    tools: Array<Omit<ToolDef, 'execute'>>,
    messages: Message[],
    options: AgentOptions,
    toolExecutor: (callJson: string) => Promise<string>,
  ): Promise<AgentResult>;
}

let _backend: Backend | null = null;

/** Get the active backend (auto-detected on first call). */
export function getBackend(): Backend {
  if (!_backend) {
    _backend = detectBackend();
  }
  return _backend;
}

/** Override the backend (for testing or custom implementations). */
export function setBackend(backend: Backend): void {
  _backend = backend;
}

function detectBackend(): Backend {
  // Try NAPI first (Node.js)
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const napi = require('@giulio-leone/gauss-core-napi');
    return createNapiBackend(napi);
  } catch {
    // NAPI not available
  }

  // Try WASM
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const wasm = require('@giulio-leone/gauss-core-wasm');
    return createWasmBackend(wasm);
  } catch {
    // WASM not available
  }

  throw new Error(
    'No Gauss backend found. Install @giulio-leone/gauss-core-napi (Node.js) or @giulio-leone/gauss-core-wasm (Browser/Edge).',
  );
}

/* eslint-disable @typescript-eslint/no-explicit-any */

function createNapiBackend(napi: any): Backend {
  return {
    createProvider(type, model, options) {
      return napi.createProvider(type, model, {
        apiKey: options.apiKey,
        baseUrl: options.baseUrl ?? null,
        timeoutMs: options.timeoutMs ?? null,
        maxRetries: options.maxRetries ?? null,
        organization: options.organization ?? null,
      });
    },

    destroyProvider(handle) {
      napi.destroyProvider(handle);
    },

    async generate(handle, messages, temperature, maxTokens) {
      const result = await napi.generate(
        handle,
        messages.map((m: Message) => ({ role: m.role, content: m.content })),
        temperature ?? null,
        maxTokens ?? null,
      );
      return result as GenerateResult;
    },

    async agentRun(name, providerHandle, tools, messages, options) {
      const result = await napi.agentRun(
        name,
        providerHandle,
        tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters ?? null,
        })),
        messages.map((m) => ({ role: m.role, content: m.content })),
        {
          instructions: options.instructions ?? null,
          maxSteps: options.maxSteps ?? null,
          temperature: options.temperature ?? null,
          topP: options.topP ?? null,
          maxTokens: options.maxTokens ?? null,
          seed: options.seed ?? null,
          stopOnTool: options.stopOnTool ?? null,
          outputSchema: options.outputSchema ?? null,
        },
      );
      return {
        text: result.text,
        steps: result.steps,
        usage: {
          inputTokens: result.inputTokens,
          outputTokens: result.outputTokens,
        },
        structuredOutput: result.structuredOutput ?? undefined,
      };
    },

    async generateWithTools(handle, messages, tools, temperature, maxTokens) {
      const result = await napi.generateWithTools(
        handle,
        messages.map((m: Message) => ({ role: m.role, content: m.content })),
        tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters ?? null,
        })),
        temperature ?? null,
        maxTokens ?? null,
      );
      return result as GenerateWithToolsResult;
    },

    async agentRunWithToolExecutor(name, providerHandle, tools, messages, options, toolExecutor) {
      const result = await napi.agentRunWithToolExecutor(
        name,
        providerHandle,
        tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters ?? null,
        })),
        messages.map((m) => ({ role: m.role, content: m.content })),
        {
          instructions: options.instructions ?? null,
          maxSteps: options.maxSteps ?? null,
          temperature: options.temperature ?? null,
          topP: options.topP ?? null,
          maxTokens: options.maxTokens ?? null,
          seed: options.seed ?? null,
          stopOnTool: options.stopOnTool ?? null,
          outputSchema: options.outputSchema ?? null,
        },
        toolExecutor,
      );
      return {
        text: result.text,
        steps: result.steps,
        usage: {
          inputTokens: result.inputTokens,
          outputTokens: result.outputTokens,
        },
        structuredOutput: result.structuredOutput ?? undefined,
      };
    },
  };
}

function createWasmBackend(wasm: any): Backend {
  return {
    createProvider(type, model, options) {
      return wasm.createProvider(type, model, options.apiKey, options.baseUrl ?? undefined);
    },

    destroyProvider(handle) {
      wasm.destroyProvider(handle);
    },

    async generate(handle, messages, temperature, maxTokens) {
      const json = await wasm.generate(
        handle,
        JSON.stringify(messages),
        temperature ?? undefined,
        maxTokens ?? undefined,
      );
      return JSON.parse(json) as GenerateResult;
    },

    async agentRun(name, providerHandle, _tools, messages, options) {
      const json = await wasm.agentRun(
        name,
        providerHandle,
        JSON.stringify(messages),
        JSON.stringify({
          instructions: options.instructions,
          maxSteps: options.maxSteps,
          temperature: options.temperature,
          topP: options.topP,
          maxTokens: options.maxTokens,
          stopOnTool: options.stopOnTool,
          outputSchema: options.outputSchema,
        }),
      );
      return JSON.parse(json) as AgentResult;
    },

    async generateWithTools(handle, messages, _tools, temperature, maxTokens) {
      // WASM fallback: generate without tools
      const json = await wasm.generate(
        handle,
        JSON.stringify(messages),
        temperature ?? undefined,
        maxTokens ?? undefined,
      );
      return { ...JSON.parse(json), toolCalls: [] } as GenerateWithToolsResult;
    },

    async agentRunWithToolExecutor(name, providerHandle, _tools, messages, options, _toolExecutor) {
      // WASM fallback: run without tool executor (tools not supported in WASM)
      return this.agentRun(name, providerHandle, _tools, messages, options);
    },
  };
}
