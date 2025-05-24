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
  // Add other methods or properties if AgentLoop uses them directly, though it mainly uses responses.create
};

// Mock for the createOpenAIClient function
const mockCreateOpenAIClient = vi.fn().mockReturnValue(mockOpenAIClientInstance);

vi.mock("../src/utils/openai-client.js", () => ({ // Adjusted path
  createOpenAIClient: mockCreateOpenAIClient,
  // If customFetchForOllamaWithCreds is also in openai-client.js and needed by other parts, mock it too or ensure it's not called.
  // For this test, we primarily care about createOpenAIClient.
}));

// Mock error classes from "openai" if AgentLoop directly imports/uses them.
// AgentLoop does import APIConnectionTimeoutError from "openai".
vi.mock("openai", async (importOriginal) => {
    const original = await importOriginal() as any; // Cast to any to allow modification
    return {
        ...original,
        APIConnectionTimeoutError: class extends Error { constructor(message?: string) { super(message); this.name = "APIConnectionTimeoutError";}},
        // Remove the default class mock for new OpenAI()
        // default: class { public responses = { create: createSpy }; }, // This line is removed
    };
});


/* ─────────── 2.  Parametrised tests ─────────────────────────────── */
describe.each([
  { flag: true, title: "omits previous_response_id & sets store:false" },
  { flag: false, title: "sends previous_response_id & allows store:true" },
])("AgentLoop with disableResponseStorage=%s", ({ flag, title }) => {
  /* build a fresh config for each case */
  const cfg: AppConfig = {
    model: "codex-mini-latest",
    provider: "openai",
    instructions: "",
    disableResponseStorage: flag,
    notify: false,
    // Add other necessary AppConfig fields if AgentLoop's constructor or createOpenAIClient expects them
  };

  beforeEach(() => {
    // Reset spies per iteration
    mockResponsesCreate.mockClear();
    mockCreateOpenAIClient.mockClear();
    mockResponsesCreate.mockResolvedValue({ // Ensure it returns a promise-like structure for the stream
        // Simulate a minimal stream-like object or the actual expected structure if more complex
        [Symbol.asyncIterator]() {
            return {
                async next() {
                    // Simulate the end of the stream or specific events if needed by the test
                    return { done: true, value: undefined };
                }
            };
        },
        // If AgentLoop interacts with the controller of the stream directly:
        controller: { abort: vi.fn() } 
    });
  });

  it(title, async () => {
    const loop = new AgentLoop({
      model: cfg.model,
      provider: cfg.provider,
      config: cfg, // Pass the full AppConfig
      instructions: "",
      approvalPolicy: "suggest",
      disableResponseStorage: flag, // This is now part of cfg, but AgentLoop also takes it directly
      additionalWritableRoots: [],
      onItem() {},
      onLoading() {},
      getCommandConfirmation: async () => ({ review: ReviewDecision.YES }),
      onLastResponseId() {},
    });

    // Assert that createOpenAIClient was called correctly
    expect(mockCreateOpenAIClient).toHaveBeenCalledTimes(1);
    expect(mockCreateOpenAIClient).toHaveBeenCalledWith(expect.objectContaining({
      provider: cfg.provider,
      model: cfg.model,
      // instructions: cfg.instructions, // createOpenAIClient doesn't use instructions
      // apiKey: cfg.apiKey, // createOpenAIClient gets this from config or env
    }));


    await loop.run([
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hello" }],
      },
    ]);

    expect(mockResponsesCreate).toHaveBeenCalledTimes(1);

    const call = mockResponsesCreate.mock.calls[0];
    if (!call) {
      throw new Error("Expected mockResponsesCreate to have been called at least once");
    }
    const payload: any = call[0];

    if (flag) {
      /* behaviour when ZDR is *on* (disableResponseStorage = true) */
      expect(payload).not.toHaveProperty("previous_response_id");
      expect(payload.store).toBe(false); // Explicitly check store:false
      // When disableResponseStorage is true, the input should contain the full transcript.
      // For the first message, this means the input itself.
      expect(payload.input).toEqual(expect.arrayContaining([
        expect.objectContaining({
            role: "user", 
            content: expect.arrayContaining([expect.objectContaining({text: "hello"})])
        })
      ]));

    } else {
      /* behaviour when ZDR is *off* (disableResponseStorage = false) */
      // previous_response_id might be empty for the very first call, which is fine
      expect(payload).toHaveProperty("previous_response_id"); 
      expect(payload.store === undefined ? true : payload.store).toBe(true); // store should be true or undefined
      // When disableResponseStorage is false, input contains only the current message for the first turn.
       expect(payload.input).toEqual(expect.arrayContaining([
        expect.objectContaining({
            role: "user", 
            content: expect.arrayContaining([expect.objectContaining({text: "hello"})])
        })
      ]));
    }
  });
});
