import { describe, it, expect, vi, beforeEach } from "vitest";
import { createOpenAIClient } from "../src/utils/openai-client";
import * as config from "../src/utils/config";
import OpenAI from "openai";

// Mock the config functions
vi.mock("../src/utils/config", () => ({
  getBaseUrl: vi.fn(),
  getApiKey: vi.fn(),
  AZURE_OPENAI_API_VERSION: "2023-07-01-preview",
  OPENAI_TIMEOUT_MS: 10000,
  OPENAI_ORGANIZATION: undefined,
  OPENAI_PROJECT: undefined,
}));

// Mock the OpenAI constructor
vi.mock("openai", () => {
  const actualOpenAI = vi.importActual("openai");
  return {
    ...actualOpenAI,
    __esModule: true,
    default: vi.fn(),
    AzureOpenAI: vi.fn(),
  };
});

const mockOpenAI = OpenAI as vi.MockedFunction<typeof OpenAI>;

describe("createOpenAIClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock implementations for general cases
    vi.mocked(config.getApiKey).mockReturnValue("test-api-key");
    vi.mocked(config.getBaseUrl).mockImplementation((provider?: string) => {
      if (provider === "azure") return "https://azure.example.com";
      return "https://api.openai.com/v1";
    });
    mockOpenAI.mockClear();
  });

  it("should create a standard OpenAI client for 'openai' provider", () => {
    vi.mocked(config.getBaseUrl).mockReturnValue("https://api.openai.com/v1");
    createOpenAIClient({ provider: "openai" });
    expect(mockOpenAI).toHaveBeenCalledWith({
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
      timeout: 10000,
      defaultHeaders: {},
    });
  });

  describe("Ollama provider", () => {
    it("should handle Ollama URL with username and password", () => {
      const ollamaUrlWithCreds = "http://user:pass@localhost:11434/v1";
      const expectedBaseUrl = "http://localhost:11434/v1";
      const expectedAuthHeader = "Basic " + Buffer.from("user:pass").toString("base64");

      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama"); // Ollama typically doesn't use API keys in the same way

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {
          Authorization: expectedAuthHeader,
        },
      });
    });

    it("should handle Ollama URL with username and password containing special characters", () => {
      const ollamaUrlWithSpecialChars = "http://us%20er:p%40ss@localhost:11434/api";
      const expectedBaseUrl = "http://localhost:11434/api";
      // "us er":"p@ss"
      const expectedAuthHeader = "Basic " + Buffer.from("us er:p@ss").toString("base64");
      
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithSpecialChars);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {
          Authorization: expectedAuthHeader,
        },
      });
    });
    
    it("should handle Ollama URL with username only", () => {
      const ollamaUrlWithUserOnly = "http://admin@localhost:11434";
      const expectedBaseUrl = "http://localhost:11434";
      const expectedAuthHeader = "Basic " + Buffer.from("admin:").toString("base64");

      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithUserOnly);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {
          Authorization: expectedAuthHeader,
        },
      });
    });

    it("should handle Ollama URL without credentials", () => {
      const ollamaUrlWithoutCreds = "http://localhost:11434/v1";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithoutCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithoutCreds,
        timeout: 10000,
        defaultHeaders: {}, // No Authorization header expected
      });
    });

    it("should handle Ollama URL without credentials and non-standard path", () => {
      const ollamaUrlWithPath = "http://localhost:11434/custom/path";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithPath);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithPath,
        timeout: 10000,
        defaultHeaders: {},
      });
    });
  });

  it("should pass through OPENAI_ORGANIZATION and OPENAI_PROJECT if set", () => {
    vi.mocked(config.getBaseUrl).mockReturnValue("https://api.openai.com/v1");
    // @ts-expect-error - We are testing the effect of these values
    config.OPENAI_ORGANIZATION = "org-123";
    // @ts-expect-error - We are testing the effect of these values
    config.OPENAI_PROJECT = "proj-abc";

    createOpenAIClient({ provider: "openai" });
    expect(mockOpenAI).toHaveBeenCalledWith({
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
      timeout: 10000,
      defaultHeaders: {
        "OpenAI-Organization": "org-123",
        "OpenAI-Project": "proj-abc",
      },
    });
    // @ts-expect-error - Reset for other tests
    config.OPENAI_ORGANIZATION = undefined;
    // @ts-expect-error - Reset for other tests
    config.OPENAI_PROJECT = undefined;
  });

  // Test for Azure still needed if desired, but not directly related to Ollama change
  it("should create an AzureOpenAI client for 'azure' provider", () => {
    const mockAzureOpenAI = vi.mocked(OpenAI.AzureOpenAI);
    vi.mocked(config.getBaseUrl).mockReturnValue("https://azure.example.com");
    vi.mocked(config.getApiKey).mockReturnValue("azure-api-key");
    
    createOpenAIClient({ provider: "azure" });
    
    expect(mockAzureOpenAI).toHaveBeenCalledWith({
      apiKey: "azure-api-key",
      baseURL: "https://azure.example.com",
      apiVersion: "2023-07-01-preview",
      timeout: 10000,
      defaultHeaders: {},
    });
  });
});
