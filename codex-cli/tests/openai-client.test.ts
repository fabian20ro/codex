import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createOpenAIClient, customFetchForOllamaWithCreds } from "../src/utils/openai-client";
import * as config from "../src/utils/config";
import OpenAI from "openai";
import { Readable } from "stream";
import axios from 'axios'; // Import to access actual isAxiosError for the mock

// Mock axios
const mockAxios = vi.fn();
vi.mock('axios', async (importOriginal) => {
  const actualAxios = await importOriginal() as typeof import('axios');
  return {
    ...actualAxios,
    default: mockAxios,
    isAxiosError: (payload: any): payload is import('axios').AxiosError => actualAxios.isAxiosError(payload),
  };
});

// Mock fs and https
const mockReadFileSync = vi.fn();
const mockHttpsAgent = vi.fn();

vi.mock('fs', () => ({
  readFileSync: mockReadFileSync,
}));

vi.mock('https', async (importOriginal) => {
  const actualHttps = await importOriginal() as typeof import('https');
  return {
    ...actualHttps, 
    Agent: mockHttpsAgent, 
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
    vi.clearAllMocks();
    vi.mocked(config.getApiKey).mockReturnValue("test-api-key");
    vi.mocked(config.getBaseUrl).mockImplementation((provider?: string) => {
      if (provider === "azure") return "https://azure.example.com";
      return "https://api.openai.com/v1";
    });
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
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");
      createOpenAIClient({ provider: "ollama" });
      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithCreds,
        timeout: 10000,
        defaultHeaders: {}, 
        fetch: customFetchForOllamaWithCreds,
      });
    });

    it("should use custom fetch for Ollama URL with username and password containing special characters", () => {
      const ollamaUrlWithSpecialChars = "http://us%20er:p%40ss@localhost:11434/api";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithSpecialChars);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");
      createOpenAIClient({ provider: "ollama" });
      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithSpecialChars,
        timeout: 10000,
        defaultHeaders: {}, 
        fetch: customFetchForOllamaWithCreds, 
      });
    });
    
    it("should use custom fetch for Ollama URL with username only", () => {
      const ollamaUrlWithUserOnly = "http://admin@localhost:11434";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithUserOnly);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");
      createOpenAIClient({ provider: "ollama" });
      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithUserOnly,
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
  let originalOllamaCaPath: string | undefined;
  let originalNodeTlsRejectUnauthorized: string | undefined;
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>;


  beforeEach(() => {
    mockAxios.mockReset();
    mockReadFileSync.mockReset();
    mockHttpsAgent.mockReset();
    
    originalOllamaCaPath = process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
    originalNodeTlsRejectUnauthorized = process.env['NODE_TLS_REJECT_UNAUTHORIZED'];
    // Clear them for tests that don't set them explicitly
    delete process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
    delete process.env['NODE_TLS_REJECT_UNAUTHORIZED'];

    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks(); // This will also restore console.warn
    if (originalOllamaCaPath !== undefined) {
      process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = originalOllamaCaPath;
    } else {
      delete process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
    }
    if (originalNodeTlsRejectUnauthorized !== undefined) {
      process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = originalNodeTlsRejectUnauthorized;
    } else {
      delete process.env['NODE_TLS_REJECT_UNAUTHORIZED'];
    }
  });

  it('should make a successful GET request and return a Response object', async () => {
    const mockResponseData = 'success';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: { 'content-type': 'text/plain' } });
    const url = 'http://localhost/test';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET' });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ url, method: 'GET' }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should make a successful POST request with JSON body and return a Response object', async () => {
    const requestBody = { key: 'value' };
    const mockResponseData = { message: 'created' };
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 201, statusText: 'Created', headers: { 'content-type': 'application/json' } });
    const url = 'http://localhost/create';
    const response = await customFetchForOllamaWithCreds(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody) });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ url, method: 'POST', data: JSON.stringify(requestBody) }));
    expect(response.status).toBe(201);
    expect(await response.json()).toEqual(mockResponseData);
  });

  it('should handle streaming response for text/event-stream', async () => {
    const mockNodeReadableStream = Readable.from(['data: event1\n\n']);
    mockAxios.mockResolvedValueOnce({ data: mockNodeReadableStream, status: 200, headers: { 'content-type': 'text/event-stream' } });
    const url = 'http://localhost/stream';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET', headers: { 'Accept': 'text/event-stream' } });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ responseType: 'stream' }));
    expect(response.body).toBe(mockNodeReadableStream);
  });

  it('should handle AxiosError with a response', async () => {
    const axiosError = { isAxiosError: true, response: { data: 'Not Found', status: 404, statusText: 'Not Found', headers: {} }, config: {}, name: 'AxiosError', message: 'Request failed' };
    mockAxios.mockRejectedValueOnce(axiosError);
    const url = 'http://localhost/notfound';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET' });
    expect(response.status).toBe(404);
    expect(await response.text()).toBe('Not Found');
  });

  it('should re-throw AxiosError if it has no response (network error)', async () => {
    const networkError = { isAxiosError: true, request: {}, message: 'Network Error', name: 'AxiosError', config: {} };
    mockAxios.mockRejectedValueOnce(networkError);
    const url = 'http://localhost/networkissue';
    await expect(customFetchForOllamaWithCreds(url, { method: 'GET' })).rejects.toThrow('Network Error');
  });

  it('should re-throw non-Axios error', async () => {
    const genericError = new Error('Something else failed');
    mockAxios.mockRejectedValueOnce(genericError);
    const url = 'http://localhost/genericfailure';
    await expect(customFetchForOllamaWithCreds(url, { method: 'GET' })).rejects.toThrow('Something else failed');
  });
  
  it('should pass AbortSignal to axios config', async () => {
    const controller = new AbortController();
    mockAxios.mockResolvedValueOnce({ data: 'success', status: 200, headers: {} });
    const url = 'http://localhost/abort-test';
    await customFetchForOllamaWithCreds(url, { method: 'GET', signal: controller.signal });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ signal: controller.signal }));
  });

  // New tests for CA certificate handling
  it('HTTPS URL with valid CA cert path: should configure https.Agent', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    const dummyCertContent = Buffer.from('dummy_ca_cert_content');
    mockReadFileSync.mockReturnValueOnce(dummyCertContent);
    const mockAgentInstance = { type: 'mockAgent' }; // Dummy object for agent instance
    mockHttpsAgent.mockReturnValueOnce(mockAgentInstance);
    mockAxios.mockResolvedValueOnce({ data: 'success', status: 200, headers: {} });

    const url = 'https://localhost/secure';
    await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(mockReadFileSync).toHaveBeenCalledWith('/path/to/ca.pem');
    expect(mockHttpsAgent).toHaveBeenCalledWith({ ca: dummyCertContent });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      httpsAgent: mockAgentInstance,
    }));
  });

  it('HTTPS URL with invalid CA cert path (fs.readFileSync throws): should throw error', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/bad_ca.pem';
    mockReadFileSync.mockImplementationOnce(() => { throw new Error('File not found'); });
    
    const url = 'https://localhost/secure-fail';
    await expect(customFetchForOllamaWithCreds(url, { method: 'GET' }))
      .rejects.toThrow('Failed to load CA certificate from /path/to/bad_ca.pem: File not found');
    
    expect(mockAxios).not.toHaveBeenCalled();
  });

  it('HTTP URL with CA cert path set: should NOT configure https.Agent', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    mockAxios.mockResolvedValueOnce({ data: 'success', status: 200, headers: {} });

    const url = 'http://localhost/insecure'; // HTTP URL
    await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(mockReadFileSync).not.toHaveBeenCalled();
    expect(mockHttpsAgent).not.toHaveBeenCalled();
    const axiosCallOptions = mockAxios.mock.calls[0][0];
    expect(axiosCallOptions.httpsAgent).toBeUndefined();
  });

  it('HTTPS URL without CA cert path set: should NOT configure https.Agent', async () => {
    // OLLAMA_CUSTOM_CA_CERT_PATH is ensured to be undefined by beforeEach
    mockAxios.mockResolvedValueOnce({ data: 'success', status: 200, headers: {} });

    const url = 'https://localhost/secure-no-ca';
    await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(mockReadFileSync).not.toHaveBeenCalled();
    expect(mockHttpsAgent).not.toHaveBeenCalled();
    const axiosCallOptions = mockAxios.mock.calls[0][0];
    expect(axiosCallOptions.httpsAgent).toBeUndefined();
  });

  it('HTTPS URL with CA cert and NODE_TLS_REJECT_UNAUTHORIZED=0: should warn and configure https.Agent', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0';
    const dummyCertContent = Buffer.from('dummy_ca_cert_content_tls_warning');
    mockReadFileSync.mockReturnValueOnce(dummyCertContent);
    const mockAgentInstance = { type: 'mockAgentWarning' };
    mockHttpsAgent.mockReturnValueOnce(mockAgentInstance);
    mockAxios.mockResolvedValueOnce({ data: 'success', status: 200, headers: {} });
    
    const url = 'https://localhost/secure-warning';
    await customFetchForOllamaWithCreds(url, { method: 'GET' });

    expect(consoleWarnSpy).toHaveBeenCalledWith(expect.stringContaining('[Codex CLI] Warning: OLLAMA_CUSTOM_CA_CERT_PATH is set, but NODE_TLS_REJECT_UNAUTHORIZED is also set to 0.'));
    expect(mockReadFileSync).toHaveBeenCalledWith('/path/to/ca.pem');
    expect(mockHttpsAgent).toHaveBeenCalledWith({ ca: dummyCertContent });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      httpsAgent: mockAgentInstance,
    }));
  });
});
