/* tslint:disable */
/* eslint-disable */
/* auto-generated – DO NOT EDIT */

// ============ Shared Types ============

export interface ProviderOptions {
  apiKey: string
  baseUrl?: string
  timeoutMs?: number
  maxRetries?: number
  organization?: string
}

export interface ToolDef {
  name: string
  description: string
  parameters?: any
}

export interface JsMessage {
  role: string
  content: string
}

export interface AgentOptions {
  instructions?: string
  maxSteps?: number
  temperature?: number
  topP?: number
  maxTokens?: number
  seed?: number
  stopOnTool?: string
  outputSchema?: any
  thinkingBudget?: number
  cacheControl?: boolean
  codeExecution?: {
    python?: boolean
    javascript?: boolean
    bash?: boolean
    timeoutSecs?: number
    workingDir?: string
    sandbox?: string
    unified?: boolean
  }
  /** Enable Google Search grounding (Gemini only). */
  grounding?: boolean
  /** Enable native code execution / Gemini code interpreter. */
  nativeCodeExecution?: boolean
  /** Response modalities (e.g. ["TEXT", "IMAGE"] for Gemini image generation). */
  responseModalities?: string[]
}

export interface AgentResult {
  text: string
  steps: number
  inputTokens: number
  outputTokens: number
  structuredOutput?: any
  thinking?: string
  citations?: Array<{
    citationType: string
    citedText?: string
    documentTitle?: string
    start?: number
    end?: number
  }>
  groundingMetadata?: any
}

// ============ Version ============

export function version(): string

// ============ Provider ============

export function create_provider(providerType: string, model: string, options: ProviderOptions): number
export function destroy_provider(handle: number): void

// ============ Agent ============

export function agent_run(
  name: string,
  providerHandle: number,
  tools: ToolDef[],
  messages: JsMessage[],
  options?: AgentOptions | undefined | null
): Promise<AgentResult>

export function agent_run_with_tool_executor(
  name: string,
  providerHandle: number,
  tools: ToolDef[],
  messages: JsMessage[],
  options?: AgentOptions | undefined | null,
  toolExecutor?: (callJson: string) => Promise<string>
): Promise<AgentResult>

export function agent_stream_with_tool_executor(
  name: string,
  providerHandle: number,
  tools: ToolDef[],
  messages: JsMessage[],
  options?: AgentOptions | undefined | null,
  streamCallback?: (eventJson: string) => void,
  toolExecutor?: (callJson: string) => Promise<string>
): Promise<AgentResult>

// ============ Generate (raw provider call) ============

export function generate(
  providerHandle: number,
  messages: JsMessage[],
  temperature?: number | undefined | null,
  maxTokens?: number | undefined | null,
  thinkingBudget?: number | undefined | null,
  cacheControl?: boolean | undefined | null
): Promise<any>

export function generate_with_tools(
  providerHandle: number,
  messages: JsMessage[],
  tools: ToolDef[],
  temperature?: number | undefined | null,
  maxTokens?: number | undefined | null
): Promise<any>

// ============ Provider Capabilities ============

export function get_provider_capabilities(providerHandle: number): any

// ============ Code Execution (PTC) ============

export function execute_code(
  language: string,
  code: string,
  timeoutSecs?: number | undefined | null,
  workingDir?: string | undefined | null,
  sandbox?: string | undefined | null
): Promise<any>

export function available_runtimes(): Promise<string[]>

// ============ Image Generation ============

export function generate_image(
  providerHandle: number,
  prompt: string,
  model?: string | undefined | null,
  size?: string | undefined | null,
  quality?: string | undefined | null,
  style?: string | undefined | null,
  aspectRatio?: string | undefined | null,
  n?: number | undefined | null,
  responseFormat?: string | undefined | null
): Promise<any>

// ============ Memory ============

export function create_memory(): number
export function memory_store(handle: number, entryJson: string): Promise<void>
export function memory_recall(handle: number, optionsJson?: string | undefined | null): Promise<any>
export function memory_clear(handle: number, sessionId?: string | undefined | null): Promise<void>
export function memory_stats(handle: number): Promise<any>
export function destroy_memory(handle: number): void

// ============ Context / Tokens ============

export function count_tokens(text: string): number
export function count_tokens_for_model(text: string, model: string): number
export function count_message_tokens(messages: JsMessage[]): number
export function get_context_window_size(model: string): number

// ============ RAG / Vector Store ============

export function create_vector_store(): number
export function vector_store_upsert(handle: number, chunksJson: string): Promise<void>
export function vector_store_search(handle: number, embeddingJson: string, topK: number): Promise<any>
export function destroy_vector_store(handle: number): void
export function cosine_similarity(a: number[], b: number[]): number

// ============ MCP ============

export function create_mcp_server(name: string, versionStr: string): number
export function mcp_server_add_tool(handle: number, toolJson: string): void
export function mcpServerAddResource(handle: number, resourceJson: string): void
export function mcpServerAddPrompt(handle: number, promptJson: string): void
export function mcp_server_handle(handle: number, messageJson: string): Promise<any>
export function destroy_mcp_server(handle: number): void

// ============ Network (Multi-Agent) ============

export function create_network(): number
export function network_add_agent(
  handle: number,
  name: string,
  providerHandle: number,
  instructions?: string | undefined | null
): void
export function network_set_supervisor(handle: number, agentName: string): void
export function network_delegate(
  handle: number,
  fromAgent: string,
  toAgent: string,
  prompt: string
): Promise<any>
export function network_agent_cards(handle: number): any
export function destroy_network(handle: number): void

// ============ Middleware ============

export function create_middleware_chain(): number
export function middleware_use_logging(handle: number): void
export function middleware_use_caching(handle: number, ttlMs: number): void
export function destroy_middleware_chain(handle: number): void

// ============ HITL — Approval ============

export function create_approval_manager(): number
export function approval_request(
  handle: number,
  toolName: string,
  argsJson: string,
  sessionId: string
): string
export function approval_approve(
  handle: number,
  requestId: string,
  modifiedArgs?: string | undefined | null
): void
export function approval_deny(
  handle: number,
  requestId: string,
  reason?: string | undefined | null
): void
export function approval_list_pending(handle: number): any
export function destroy_approval_manager(handle: number): void

// ============ HITL — Checkpoints ============

export function create_checkpoint_store(): number
export function checkpoint_save(handle: number, checkpointJson: string): Promise<void>
export function checkpoint_load(handle: number, checkpointId: string): Promise<any>
export function checkpoint_load_latest(handle: number, sessionId: string): Promise<any>
export function destroy_checkpoint_store(handle: number): void

// ============ Eval ============

export function create_eval_runner(threshold?: number | undefined | null): number
export function eval_add_scorer(handle: number, scorerType: string): void
export function load_dataset_jsonl(jsonl: string): any
export function load_dataset_json(jsonStr: string): any
export function destroy_eval_runner(handle: number): void

// ============ Telemetry ============

export function create_telemetry(): number
export function telemetry_record_span(handle: number, spanJson: string): void
export function telemetry_export_spans(handle: number): any
export function telemetry_export_metrics(handle: number): any
export function telemetry_clear(handle: number): void
export function destroy_telemetry(handle: number): void

// ============ Guardrails ============

export function create_guardrail_chain(): number
export function guardrail_chain_add_content_moderation(
  handle: number,
  blockPatterns: string[],
  warnPatterns: string[]
): void
export function guardrail_chain_add_pii_detection(handle: number, action: string): void
export function guardrail_chain_add_token_limit(
  handle: number,
  maxInput?: number | undefined | null,
  maxOutput?: number | undefined | null
): void
export function guardrail_chain_add_regex_filter(
  handle: number,
  blockRules: string[],
  warnRules: string[]
): void
export function guardrail_chain_add_schema(handle: number, schemaJson: string): void
export function guardrail_chain_list(handle: number): string[]
export function destroy_guardrail_chain(handle: number): void

// ============ Resilience ============

export function create_fallback_provider(providerHandles: number[]): number
export function create_circuit_breaker(
  providerHandle: number,
  failureThreshold?: number | undefined | null,
  recoveryTimeoutMs?: number | undefined | null
): number
export function create_resilient_provider(
  primaryHandle: number,
  fallbackHandles: number[],
  enableCircuitBreaker?: boolean | undefined | null
): number

// ============ Plugin System ============

export function create_plugin_registry(): number
export function plugin_registry_add_telemetry(handle: number): void
export function plugin_registry_add_memory(handle: number): void
export function plugin_registry_list(handle: number): string[]
export function plugin_registry_emit(handle: number, eventJson: string): void
export function destroy_plugin_registry(handle: number): void

// ============ Tool Validator ============

export function create_tool_validator(strategies?: string[] | undefined | null): number
export function tool_validator_validate(handle: number, input: string, schema: string): string
export function destroy_tool_validator(handle: number): void

// ============ Config ============

export function agent_config_from_json(jsonStr: string): string
export function agent_config_resolve_env(value: string): string

// ============ Graph ============

export function create_graph(): number
export function graph_add_node(
  handle: number,
  nodeId: string,
  agentName: string,
  providerHandle: number,
  instructions?: string | undefined | null,
  tools?: ToolDef[]
): void
export function graph_add_edge(handle: number, from: string, to: string): void
export interface ForkAgentDef {
  agentName: string
  providerHandle: number
  instructions?: string | undefined | null
}
export function graph_add_fork_node(
  handle: number,
  nodeId: string,
  agents: ForkAgentDef[],
  consensus: string
): void
export function graph_run(handle: number, prompt: string): Promise<any>
export function destroy_graph(handle: number): void

// ============ Workflow ============

export function create_workflow(): number
export function workflow_add_step(
  handle: number,
  stepId: string,
  agentName: string,
  providerHandle: number,
  instructions?: string | undefined | null,
  tools?: ToolDef[]
): void
export function workflow_add_dependency(handle: number, stepId: string, dependsOn: string): void
export function workflow_run(handle: number, prompt: string): Promise<any>
export function destroy_workflow(handle: number): void

// ============ Stream Utils ============

export function parse_partial_json(text: string): string | null

// ============ Team ============

export function create_team(name: string): number
export function team_add_agent(
  handle: number,
  agentName: string,
  providerHandle: number,
  instructions?: string | undefined | null
): void
export function team_set_strategy(handle: number, strategy: string): void
export function team_run(handle: number, messagesJson: string): Promise<any>
export function destroy_team(handle: number): void

// ============ AGENTS.MD & SKILL.MD Parsers ============

export function parseAgentsMd(content: string): any
export function discoverAgents(dir: string): any
export function parseSkillMd(content: string): any

// ============ A2A Protocol ============

export function createA2aClient(baseUrl: string, authToken?: string): any
export function a2aDiscover(baseUrl: string, authToken?: string): Promise<any>
export function a2aSendMessage(baseUrl: string, authToken?: string | null, messageJson: string, configJson?: string | null): Promise<any>
export function a2aAsk(baseUrl: string, authToken?: string | null, text: string): Promise<string>
export function a2aGetTask(baseUrl: string, authToken?: string | null, taskId: string, historyLength?: number | null): Promise<any>
export function a2aCancelTask(baseUrl: string, authToken?: string | null, taskId: string): Promise<any>
export function a2aHandleRequest(agentCardJson: string, requestBody: string): Promise<string>

// ============ Tool Registry ============

export function createToolRegistry(): number
export function toolRegistryAdd(handle: number, toolJson: string): void
export function toolRegistrySearch(handle: number, query: string): any
export function toolRegistryByTag(handle: number, tag: string): any
export function toolRegistryList(handle: number): any
export function destroyToolRegistry(handle: number): void
