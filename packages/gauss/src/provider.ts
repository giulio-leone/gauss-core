import type { ProviderType, ProviderOptions, Message, GenerateResult } from './types.js';
import { getBackend } from './backend.js';

/** Provider wraps a Rust-powered LLM provider. */
export class Provider {
  private handle: number;
  private destroyed = false;

  /** Use `createProvider()` factory instead. */
  constructor(handle: number) {
    this.handle = handle;
  }

  /** Get the native handle (for internal use). */
  getHandle(): number {
    if (this.destroyed) throw new Error('Provider has been destroyed');
    return this.handle;
  }

  /** Generate a response from messages. */
  async generate(
    messages: Message[],
    options?: { temperature?: number; maxTokens?: number },
  ): Promise<GenerateResult> {
    if (this.destroyed) throw new Error('Provider has been destroyed');
    const backend = getBackend();
    return backend.generate(this.handle, messages, options?.temperature, options?.maxTokens);
  }

  /** Release native resources. */
  destroy(): void {
    if (!this.destroyed) {
      const backend = getBackend();
      backend.destroyProvider(this.handle);
      this.destroyed = true;
    }
  }
}

/** Create a provider instance. */
export function createProvider(
  type: ProviderType,
  model: string,
  options: ProviderOptions,
): Provider {
  const backend = getBackend();
  const handle = backend.createProvider(type, model, options);
  return new Provider(handle);
}
