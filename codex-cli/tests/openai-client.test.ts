import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createOpenAIClient, customFetchForOllamaWithCreds } from "../src/utils/openai-client";
import * as config from "../src/utils/config";
import OpenAI from "openai";
import { Readable } from "stream";
import axios from 'axios'; // Import to access actual isAxiosError for the mock

// Mock axios
const mockAxios = vi.fn();
vi.mock('axios', async (importOriginal) => {
  const actualAxios = await importOriginal() as typeof import('axios'); // Cast to actual type
  return {
    ...actualAxios,
    default: mockAxios,
    isAxiosError: (payload: any): payload is import('axios').AxiosError => actualAxios.isAxiosError(payload),
    // Keep HttpStatusCode or other enums/types if they are used by the code under test
    // For this specific test, we primarily care about default and isAxiosError.
  };
});


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
    vi.clearAllMocks(); // Clears all mocks, including mockAxios
    // Reset mock implementations for general cases for createOpenAIClient tests
    vi.mocked(config.getApiKey).mockReturnValue("test-api-key");
    vi.mocked(config.getBaseUrl).mockImplementation((provider?: string) => {
      if (provider === "azure") return "https://azure.example.com";
      return "https://api.openai.com/v1";
    });
    // mockOpenAI.mockClear(); // mockOpenAI is cleared by vi.clearAllMocks()
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
    it("should use custom fetch for Ollama URL with username and password", () => {
      const ollamaUrlWithCreds = "http://user:pass@localhost:11434/v1";
      const expectedBaseUrl = ollamaUrlWithCreds; 

      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {}, 
        fetch: customFetchForOllamaWithCreds, // Check if it's the actual function
      });
    });

    it("should use custom fetch for Ollama URL with username and password containing special characters", () => {
      const ollamaUrlWithSpecialChars = "http://us%20er:p%40ss@localhost:11434/api";
      const expectedBaseUrl = ollamaUrlWithSpecialChars;
      
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithSpecialChars);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {}, 
        fetch: customFetchForOllamaWithCreds, 
      });
    });
    
    it("should use custom fetch for Ollama URL with username only", () => {
      const ollamaUrlWithUserOnly = "http://admin@localhost:11434";
      const expectedBaseUrl = ollamaUrlWithUserOnly;

      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithUserOnly);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: expectedBaseUrl,
        timeout: 10000,
        defaultHeaders: {}, 
        fetch: customFetchForOllamaWithCreds, 
      });
    });

    it("should handle Ollama URL without credentials (no custom fetch)", () => {
      const ollamaUrlWithoutCreds = "http://localhost:11434/v1";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithoutCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      const expectedCall = {
        apiKey: "ollama",
        baseURL: ollamaUrlWithoutCreds,
        timeout: 10000,
        defaultHeaders: {},
      };
      expect(mockOpenAI).toHaveBeenCalledWith(expect.objectContaining(expectedCall));
      const actualCall = mockOpenAI.mock.calls[0][0];
      expect(actualCall.fetch).toBeUndefined();
    });

    it("should handle Ollama URL without credentials and non-standard path (no custom fetch)", () => {
      const ollamaUrlWithPath = "http://localhost:11434/custom/path";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithPath);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");

      createOpenAIClient({ provider: "ollama" });

      const expectedCall = {
        apiKey: "ollama",
        baseURL: ollamaUrlWithPath,
        timeout: 10000,
        defaultHeaders: {},
      };
      expect(mockOpenAI).toHaveBeenCalledWith(expect.objectContaining(expectedCall));
      const actualCall = mockOpenAI.mock.calls[0][0];
      expect(actualCall.fetch).toBeUndefined();
    });
  });

  it("should pass through OPENAI_ORGANIZATION and OPENAI_PROJECT if set (standard OpenAI)", () => {
    vi.mocked(config.getBaseUrl).mockReturnValue("https://api.openai.com/v1");
    // @ts-expect-error - We are testing the effect of these values
    config.OPENAI_ORGANIZATION = "org-123";
    // @ts-expect-error - We are testing the effect of these values
    config.OPENAI_PROJECT = "proj-abc";

    createOpenAIClient({ provider: "openai" });
    const expectedCall = {
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
      timeout: 10000,
      defaultHeaders: {
        "OpenAI-Organization": "org-123",
        "OpenAI-Project": "proj-abc",
      },
    };
    expect(mockOpenAI).toHaveBeenCalledWith(expect.objectContaining(expectedCall));
    const actualCall = mockOpenAI.mock.calls[0][0];
    expect(actualCall.fetch).toBeUndefined();

    // @ts-expect-error - Reset for other tests
    config.OPENAI_ORGANIZATION = undefined;
    // @ts-expect-error - Reset for other tests
    config.OPENAI_PROJECT = undefined;
  });

  it("should create an AzureOpenAI client for 'azure' provider (no custom fetch)", () => {
    const mockAzureOpenAI = vi.mocked(OpenAI.AzureOpenAI);
    vi.mocked(config.getBaseUrl).mockReturnValue("https://azure.example.com");
    vi.mocked(config.getApiKey).mockReturnValue("azure-api-key");
    
    createOpenAIClient({ provider: "azure" });
    
    const expectedCall = {
      apiKey: "azure-api-key",
      baseURL: "https://azure.example.com",
      apiVersion: "2023-07-01-preview",
      timeout: 10000,
      defaultHeaders: {},
    };
    expect(mockAzureOpenAI).toHaveBeenCalledWith(expect.objectContaining(expectedCall));
    const actualCall = mockAzureOpenAI.mock.calls[0][0];
    expect(actualCall.fetch).toBeUndefined();
  });
});


describe('customFetchForOllamaWithCreds', () => {
  beforeEach(() => {
    mockAxios.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks(); // Restore any other mocks if necessary
  });

  it('should make a successful GET request and return a Response object', async () => {
    const mockResponseData = 'success';
    const mockStatus = 200;
    const mockStatusText = 'OK';
    const mockHeaders = { 'content-type': 'text/plain', 'x-custom-header': 'value' };

    mockAxios.mockResolvedValueOnce({
      data: mockResponseData,
      status: mockStatus,
      statusText: mockStatusText,
      headers: mockHeaders,
    });

    const url = 'http://localhost/test';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(mockAxios).toHaveBeenCalledWith({
      url: url,
      method: 'GET',
      headers: {}, // Assuming no headers passed in init
      data: undefined,
      signal: undefined,
      responseType: undefined, // Default for non-streaming
    });
    expect(response.status).toBe(mockStatus);
    expect(response.statusText).toBe(mockStatusText);
    expect(response.headers.get('content-type')).toBe('text/plain');
    expect(response.headers.get('x-custom-header')).toBe('value');
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should make a successful POST request with JSON body and return a Response object', async () => {
    const requestBody = { key: 'value' };
    const mockResponseData = { message: 'created' };
    const mockStatus = 201;
    const mockStatusText = 'Created';
    const mockHeaders = { 'content-type': 'application/json' };

    mockAxios.mockResolvedValueOnce({
      data: mockResponseData,
      status: mockStatus,
      statusText: mockStatusText,
      headers: mockHeaders,
    });

    const url = 'http://localhost/create';
    const response = await customFetchForOllamaWithCreds(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    expect(mockAxios).toHaveBeenCalledWith({
      url: url,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      data: JSON.stringify(requestBody),
      signal: undefined,
      responseType: undefined,
    });
    expect(response.status).toBe(mockStatus);
    expect(await response.json()).toEqual(mockResponseData);
  });

  it('should handle streaming response for text/event-stream', async () => {
    const mockNodeReadableStream = Readable.from(['data: event1\n\n', 'data: event2\n\n']);
    mockAxios.mockResolvedValueOnce({
      data: mockNodeReadableStream,
      status: 200,
      statusText: 'OK',
      headers: { 'content-type': 'text/event-stream' },
    });

    const url = 'http://localhost/stream';
    const response = await customFetchForOllamaWithCreds(url, {
      method: 'GET',
      headers: { 'Accept': 'text/event-stream' },
    });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      responseType: 'stream',
      headers: { 'Accept': 'text/event-stream' },
    }));
    expect(response.body).toBe(mockNodeReadableStream);
    expect(response.status).toBe(200);
  });

  it('should handle AxiosError with a response', async () => {
    const errorResponseData = 'Not Found';
    const errorStatus = 404;
    const errorStatusText = 'Not Found';
    const errorHeaders = { 'x-error-header': 'error-val' };

    const axiosError = {
      isAxiosError: true,
      response: {
        data: errorResponseData,
        status: errorStatus,
        statusText: errorStatusText,
        headers: errorHeaders,
      },
      config: {},
      name: 'AxiosError',
      message: 'Request failed with status code 404',
    };
    mockAxios.mockRejectedValueOnce(axiosError);

    const url = 'http://localhost/notfound';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(response.status).toBe(errorStatus);
    expect(response.statusText).toBe(errorStatusText);
    expect(response.headers.get('x-error-header')).toBe('error-val');
    expect(await response.text()).toBe(errorResponseData);
  });

  it('should re-throw AxiosError if it has no response (network error)', async () => {
    const networkError = {
      isAxiosError: true,
      request: {}, // Indicates no response was received
      message: 'Network Error',
      name: 'AxiosError',
      config: {},
    };
    mockAxios.mockRejectedValueOnce(networkError);

    const url = 'http://localhost/networkissue';
    await expect(customFetchForOllamaWithCreds(url, { method: 'GET' }))
      .rejects.toThrow('Network Error');
  });

  it('should re-throw non-Axios error', async () => {
    const genericError = new Error('Something else failed');
    mockAxios.mockRejectedValueOnce(genericError);

    const url = 'http://localhost/genericfailure';
    await expect(customFetchForOllamaWithCreds(url, { method: 'GET' }))
      .rejects.toThrow('Something else failed');
  });
  
  it('should pass AbortSignal to axios config', async () => {
    const controller = new AbortController();
    const signal = controller.signal;

    mockAxios.mockResolvedValueOnce({
      data: 'success',
      status: 200,
      statusText: 'OK',
      headers: {},
    });

    const url = 'http://localhost/abort-test';
    await customFetchForOllamaWithCreds(url, { method: 'GET', signal });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      signal: signal,
    }));
    
    // Optional: Test abortion (if axios mock supports it or if you want to simulate it)
    // This part is tricky as axios's internal handling of AbortSignal might be complex.
    // A simple way is to make mockAxios check the signal.
    mockAxios.mockImplementation(async (config) => {
      if (config.signal?.aborted) {
        const error = new Error("Request aborted");
        // @ts-ignore
        error.name = 'AbortError'; // Or whatever axios throws
        throw error;
      }
      return { data: 'success', status: 200, statusText: 'OK', headers: {} };
    });

    const controller2 = new AbortController();
    const promise = customFetchForOllamaWithCreds(url, { method: 'GET', signal: controller2.signal });
    controller2.abort();
    
    // This expectation depends on how AbortError is structured by axios or the mock
    // For now, checking if it throws any error due to abort.
    await expect(promise).rejects.toThrow(); 
  });
});
