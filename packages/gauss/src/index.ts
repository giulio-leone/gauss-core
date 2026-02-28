/**
 * Gauss — Cross-platform AI Agent Engine
 *
 * TypeScript SDK that auto-detects the best backend:
 * 1. NAPI (Node.js native) — fastest, requires @giulio-leone/gauss-core-napi
 * 2. WASM (Browser/Edge/Deno/Bun) — universal, requires @giulio-leone/gauss-core-wasm
 */

// Re-export types
export type { ProviderType, ProviderOptions, Message, AgentOptions, AgentResult, GenerateResult, GenerateWithToolsResult, ToolDef } from './types.js';
export type { GaussPlugin } from './plugin.js';

// Re-export classes
export { Agent } from './agent.js';
export { Provider } from './provider.js';

// Backend (for advanced usage)
export { getBackend, setBackend } from './backend.js';
export type { Backend } from './backend.js';

// Factory functions
export { createProvider } from './provider.js';
export { createAgent } from './agent.js';
