/**
 * codex-cli/tests/disableResponseStorage.agentLoop.test.ts
 *
 * Verifies AgentLoop's request-building logic for both values of
 * disableResponseStorage.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { AgentLoop } from "../src/utils/agent/agent-loop";
import type { AppConfig } from "../src/utils/config";
import { ReviewDecision } from "../src/utils/agent/review";

/* ─────────── 1.  Spy + module mock ─────────────────────────────── */

// Spy for the responses.create method
const mockResponsesCreate = vi.fn();

// Mock OpenAI client instance that will be returned by createOpenAIClient
const mockOpenAIClientInstance = {
  responses: { create: mockResponsesCreate },
};

// Mock for the createOpenAIClient function
const mockCreateOpenAIClient = vi.fn().mockReturnValue(mockOpenAIClientInstance);

vi.mock("../src/utils/openai-client.js", () => ({ 
  createOpenAIClient: mockCreateOpenAIClient,
}));

vi.mock("openai", async (importOriginal) => {
    const original = await importOriginal() as any; 
    return {
        ...original,
        APIConnectionTimeoutError: class extends Error { constructor(message?: string) { super(message); this.name = "APIConnectionTimeoutError";}},
    };
});


/* ─────────── 2.  Parametrised tests for disableResponseStorage ─────────────────────────────── */
describe.each([
  { flag: true, title: "omits previous_response_id & sets store:false" },
  { flag: false, title: "sends previous_response_id & allows store:true" },
])("AgentLoop with disableResponseStorage=%s (provider: openai)", ({ flag, title }) => {
  const cfg: AppConfig = {
    model: "gpt-4", // Using a common model for these general tests
    provider: "openai",
    instructions: "",
    disableResponseStorage: flag,
    notify: false,
  };

  beforeEach(() => {
    mockResponsesCreate.mockClear();
    mockCreateOpenAIClient.mockClear();
    mockResponsesCreate.mockResolvedValue({ 
        [Symbol.asyncIterator]() { return { async next() { return { done: true, value: undefined }; } }; },
        controller: { abort: vi.fn() } 
    });
  });

  it(title, async () => {
    const loop = new AgentLoop({
      model: cfg.model,
      provider: cfg.provider,
      config: cfg, 
      instructions: "",
      approvalPolicy: "suggest",
      disableResponseStorage: flag, 
      additionalWritableRoots: [],
      onItem() {},
      onLoading() {},
      getCommandConfirmation: async () => ({ review: ReviewDecision.YES }),
      onLastResponseId() {},
    });

    expect(mockCreateOpenAIClient).toHaveBeenCalledTimes(1);
    expect(mockCreateOpenAIClient).toHaveBeenCalledWith(expect.objectContaining({
      provider: cfg.provider,
      model: cfg.model,
    }));

    await loop.run([{ type: "message", role: "user", content: [{ type: "input_text", text: "hello" }] }]);
    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);

    const payload: any = mockResponsesCreate.mock.calls[0][0];

    if (flag) {
      expect(payload).not.toHaveProperty("previous_response_id");
      expect(payload.store).toBe(false);
      expect(payload.input).toEqual(expect.arrayContaining([expect.objectContaining({ role: "user" })]));
    } else {
      expect(payload).toHaveProperty("previous_response_id"); 
      expect(payload.store === undefined ? true : payload.store).toBe(true);
      expect(payload.input).toEqual(expect.arrayContaining([expect.objectContaining({ role: "user" })]));
    }
  });
});


/* ─────────── 3. Ollama provider specific parameter handling ───────────────────────── */
describe("AgentLoop with Ollama provider specific parameter handling", () => {
  beforeEach(() => {
    mockResponsesCreate.mockClear();
    mockCreateOpenAIClient.mockClear();
    // Default mock for responses.create for Ollama tests
    mockResponsesCreate.mockResolvedValue({
        [Symbol.asyncIterator]() { return { async next() { return { done: true, value: undefined }; } }; },
        controller: { abort: vi.fn() }
    });
  });

  it("should include Ollama sampling parameters when provider is ollama and parameters are set", async () => {
    const ollamaCfg: AppConfig = {
      model: "ollama-model",
      provider: "ollama",
      instructions: "",
      disableResponseStorage: false,
      notify: false,
      ollamaTemperature: 0.7,
      ollamaTopK: 40,
      ollamaTopP: 0.9,
      ollamaRepeatPenalty: 1.1,
      ollamaMinP: 0.1,
    };

    const loop = new AgentLoop({
      model: ollamaCfg.model,
      provider: ollamaCfg.provider,
      config: ollamaCfg,
      instructions: "",
      approvalPolicy: "suggest",
      additionalWritableRoots: [],
      onItem() {}, onLoading() {},
      getCommandConfirmation: async () => ({ review: ReviewDecision.YES }),
      onLastResponseId() {},
    });

    await loop.run([{ type: "message", role: "user", content: [{ type: "input_text", text: "hello ollama" }] }]);
    
    expect(mockCreateOpenAIClient).toHaveBeenCalledWith(expect.objectContaining({ provider: "ollama", model: "ollama-model" }));
    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
    
    const payload: any = mockResponsesCreate.mock.calls[0][0];
    // @ts-expect-error - Accessing custom/Ollama params
    expect(payload.temperature).toBe(0.7);
    // @ts-expect-error
    expect(payload.top_k).toBe(40);
    // @ts-expect-error
    expect(payload.top_p).toBe(0.9);
    // @ts-expect-error
    expect(payload.repeat_penalty).toBe(1.1);
    // @ts-expect-error
    expect(payload.min_p).toBe(0.1);
  });

  it("should not include Ollama parameters if they are not set in config, even if provider is ollama", async () => {
    const ollamaCfgNoParams: AppConfig = {
      model: "ollama-model-no-params",
      provider: "ollama",
      instructions: "",
      disableResponseStorage: false,
      notify: false,
      // Ollama params are deliberately not set here
    };

    const loop = new AgentLoop({
      model: ollamaCfgNoParams.model,
      provider: ollamaCfgNoParams.provider,
      config: ollamaCfgNoParams,
      instructions: "",
      approvalPolicy: "suggest",
      additionalWritableRoots: [],
      onItem() {}, onLoading() {},
      getCommandConfirmation: async () => ({ review: ReviewDecision.YES }),
      onLastResponseId() {},
    });

    await loop.run([{ type: "message", role: "user", content: [{ type: "input_text", text: "hello ollama no params" }] }]);
    
    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
    const payload: any = mockResponsesCreate.mock.calls[0][0];

    // Standard OpenAI params might be present with defaults or undefined, but Ollama-specific ones should not.
    // Temperature and top_p are standard, so they might appear if OpenAI SDK defaults them.
    // We are primarily interested in ollama-specific top_k, repeat_penalty, min_p
    // @ts-expect-error
    expect(payload.top_k).toBeUndefined();
    // @ts-expect-error
    expect(payload.repeat_penalty).toBeUndefined();
    // @ts-expect-error
    expect(payload.min_p).toBeUndefined();
  });

  it("should NOT include Ollama specific parameters when provider is not ollama, even if set in config", async () => {
    const openaiCfgWithOllamaParams: AppConfig = {
      model: "gpt-4",
      provider: "openai", // Not ollama
      instructions: "",
      disableResponseStorage: false,
      notify: false,
      ollamaTemperature: 0.77, // These should be ignored
      ollamaTopK: 44,
      ollamaTopP: 0.99,
      ollamaRepeatPenalty: 1.11,
      ollamaMinP: 0.11,
    };

    const loop = new AgentLoop({
      model: openaiCfgWithOllamaParams.model,
      provider: openaiCfgWithOllamaParams.provider,
      config: openaiCfgWithOllamaParams,
      instructions: "",
      approvalPolicy: "suggest",
      additionalWritableRoots: [],
      onItem() {}, onLoading() {},
      getCommandConfirmation: async () => ({ review: ReviewDecision.YES }),
      onLastResponseId() {},
    });

    await loop.run([{ type: "message", role: "user", content: [{ type: "input_text", text: "hello openai" }] }]);
    
    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
    const payload: any = mockResponsesCreate.mock.calls[0][0];

    // Standard temperature and top_p might exist if they are part of the default OpenAI request or if
    // the config had general 'temperature' / 'top_p' fields, but not the ollama* prefixed ones.
    // For this test, we ensure the ollama-specific ones are not passed.
    // @ts-expect-error
    expect(payload.top_k).toBeUndefined();
    // @ts-expect-error
    expect(payload.repeat_penalty).toBeUndefined();
    // @ts-expect-error
    expect(payload.min_p).toBeUndefined();
    
    // If a generic 'temperature' or 'top_p' were set in AppConfig and used by OpenAI,
    // they should not be from ollamaTemperature / ollamaTopP.
    // For this test, if they are undefined, it's fine. If they have values, they should not match ollama* values.
    // @ts-expect-error
    if (payload.temperature !== undefined) expect(payload.temperature).not.toBe(0.77);
    // @ts-expect-error
    if (payload.top_p !== undefined) expect(payload.top_p).not.toBe(0.99);
  });
});
