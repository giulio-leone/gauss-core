/** Gauss plugin interface for extensibility. */
export interface GaussPlugin {
  /** Plugin name. */
  name: string;

  /** Called when the plugin is registered. */
  init?(): void | Promise<void>;

  /** Middleware for generate calls. */
  onGenerate?(ctx: {
    messages: Array<{ role: string; content: string }>;
    provider: string;
    model: string;
  }): void | Promise<void>;

  /** Middleware for agent step completions. */
  onStepFinish?(ctx: {
    step: number;
    text: string;
    toolCalls: Array<{ name: string; arguments: string }>;
  }): void | Promise<void>;
}
