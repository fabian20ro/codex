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
    // These tests for createOpenAIClient are simplified as the credential handling
    // is now delegated to customFetchForOllamaWithCreds.
    // The main thing to check here is that the correct fetch function and original baseURL are passed.
    it("should use custom fetch for Ollama URL with username and password", () => {
      const ollamaUrlWithCreds = "http://user:pass@localhost:11434/v1";
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");
      createOpenAIClient({ provider: "ollama" });
      expect(mockOpenAI).toHaveBeenCalledWith({
        apiKey: "ollama",
        baseURL: ollamaUrlWithCreds, // Original URL is passed
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
        baseURL: ollamaUrlWithSpecialChars, // Original URL is passed
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
        baseURL: ollamaUrlWithUserOnly, // Original URL is passed
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
    delete process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
    delete process.env['NODE_TLS_REJECT_UNAUTHORIZED'];

    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks(); 
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

  it('should make a successful GET request without credentials and return a Response object', async () => {
    const mockResponseData = 'success';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: { 'content-type': 'text/plain' } });
    const url = 'http://localhost/test';
    const response = await customFetchForOllamaWithCreds(url, { method: 'GET' });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ url, method: 'GET', auth: undefined }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should handle URL with credentials, passing them to axios auth and using clean URL', async () => {
    const mockResponseData = 'success_with_auth';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: {} });
    
    const originalUrl = 'http://user:pass@localhost/authtest/v1';
    const cleanUrl = 'http://localhost/authtest/v1';
    const response = await customFetchForOllamaWithCreds(originalUrl, { method: 'POST' });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      method: 'POST',
      auth: { username: 'user', password: 'pass' },
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });
  
  it('should handle URL with URI-encoded credentials, decoding them for axios auth', async () => {
    const mockResponseData = 'success_encoded_auth';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: {} });
    
    const originalUrl = 'http://user%20name:p%40ss%23word@localhost/encoded/api';
    const cleanUrl = 'http://localhost/encoded/api'; // Path part should remain as is
    const response = await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      method: 'GET',
      auth: { username: 'user name', password: 'p@ss#word' },
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should handle URL with username only, passing empty password to axios auth', async () => {
    mockAxios.mockResolvedValueOnce({ data: 'success_user_only', status: 200, headers: {} });
    const originalUrl = 'http://admin@localhost/useronly';
    const cleanUrl = 'http://localhost/useronly';
    await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      auth: { username: 'admin', password: '' }
    }));
  });

  it('should handle streaming response for text/event-stream with credentials', async () => {
    const mockNodeReadableStream = Readable.from(['data: event1\n\n']);
    mockAxios.mockResolvedValueOnce({ data: mockNodeReadableStream, status: 200, headers: { 'content-type': 'text/event-stream' } });
    
    const originalUrl = 'http://streamuser:streampass@localhost/stream';
    const cleanUrl = 'http://localhost/stream';
    const response = await customFetchForOllamaWithCreds(originalUrl, {
      method: 'GET',
      headers: { 'Accept': 'text/event-stream' },
    });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      responseType: 'stream',
      auth: { username: 'streamuser', password: 'streampass' },
      headers: { 'Accept': 'text/event-stream' },
    }));
    expect(response.body).toBe(mockNodeReadableStream);
  });

  it('should handle AxiosError with a response when using credentials', async () => {
    const axiosError = { isAxiosError: true, response: { data: 'Auth Error', status: 401, statusText: 'Unauthorized', headers: {} }, config: {}, name: 'AxiosError', message: 'Request failed' };
    mockAxios.mockRejectedValueOnce(axiosError);
    
    const originalUrl = 'http://wrong:user@localhost/authfail';
    const cleanUrl = 'http://localhost/authfail';
    const response = await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      auth: { username: 'wrong', password: 'user' },
    }));
    expect(response.status).toBe(401);
    expect(await response.text()).toBe('Auth Error');
  });

  // CA Cert tests adjusted for new auth logic
  it('HTTPS URL with CA cert and credentials: should configure https.Agent and axios auth', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    const dummyCertContent = Buffer.from('dummy_ca_cert_content');
    mockReadFileSync.mockReturnValueOnce(dummyCertContent);
    const mockAgentInstance = { type: 'mockAgent' };
    mockHttpsAgent.mockReturnValueOnce(mockAgentInstance);
    mockAxios.mockResolvedValueOnce({ data: 'success_ca_auth', status: 200, headers: {} });

    const originalUrl = 'https://secureuser:securepass@localhost/secure-auth';
    const cleanUrl = 'https://localhost/secure-auth';
    await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(mockReadFileSync).toHaveBeenCalledWith('/path/to/ca.pem');
    expect(mockHttpsAgent).toHaveBeenCalledWith({ ca: dummyCertContent });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      httpsAgent: mockAgentInstance,
      auth: { username: 'secureuser', password: 'securepass' },
    }));
  });

  it('HTTPS URL with invalid CA cert path and credentials: should throw error before axios call', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/bad_ca.pem';
    mockReadFileSync.mockImplementationOnce(() => { throw new Error('File not found'); });
    
    const originalUrl = 'https://user:pass@localhost/secure-fail-auth';
    await expect(customFetchForOllamaWithCreds(originalUrl, { method: 'GET' }))
      .rejects.toThrow('Failed to load CA certificate from /path/to/bad_ca.pem: File not found');
    
    expect(mockAxios).not.toHaveBeenCalled(); // Axios should not be called if CA loading fails
  });

  it('HTTP URL with CA cert path and credentials: should NOT configure https.Agent but set auth', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    mockAxios.mockResolvedValueOnce({ data: 'success_http_auth_ca', status: 200, headers: {} });

    const originalUrl = 'http://user:pass@localhost/insecure-auth';
    const cleanUrl = 'http://localhost/insecure-auth';
    await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(mockReadFileSync).not.toHaveBeenCalled();
    expect(mockHttpsAgent).not.toHaveBeenCalled();
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      httpsAgent: undefined,
      auth: { username: 'user', password: 'pass' },
    }));
  });

  it('HTTPS URL without CA cert path but with credentials: should NOT configure https.Agent but set auth', async () => {
    mockAxios.mockResolvedValueOnce({ data: 'success_https_auth_no_ca', status: 200, headers: {} });

    const originalUrl = 'https://user:pass@localhost/secure-auth-no-ca';
    const cleanUrl = 'https://localhost/secure-auth-no-ca';
    await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(mockReadFileSync).not.toHaveBeenCalled();
    expect(mockHttpsAgent).not.toHaveBeenCalled();
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      httpsAgent: undefined,
      auth: { username: 'user', password: 'pass' },
    }));
  });

  it('HTTPS URL with CA cert, NODE_TLS_REJECT_UNAUTHORIZED=0, and credentials: should warn, configure https.Agent, and set auth', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0';
    const dummyCertContent = Buffer.from('dummy_ca_cert_content_tls_warning');
    mockReadFileSync.mockReturnValueOnce(dummyCertContent);
    const mockAgentInstance = { type: 'mockAgentWarning' };
    mockHttpsAgent.mockReturnValueOnce(mockAgentInstance);
    mockAxios.mockResolvedValueOnce({ data: 'success_warn_ca_auth', status: 200, headers: {} });
    
    const originalUrl = 'https://user:pass@localhost/secure-warning-auth';
    const cleanUrl = 'https://localhost/secure-warning-auth';
    await customFetchForOllamaWithCreds(originalUrl, { method: 'GET' });

    expect(consoleWarnSpy).toHaveBeenCalledWith(expect.stringContaining('[Codex CLI] Warning: OLLAMA_CUSTOM_CA_CERT_PATH is set, but NODE_TLS_REJECT_UNAUTHORIZED is also set to 0.'));
    expect(mockReadFileSync).toHaveBeenCalledWith('/path/to/ca.pem');
    expect(mockHttpsAgent).toHaveBeenCalledWith({ ca: dummyCertContent });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      httpsAgent: mockAgentInstance,
      auth: { username: 'user', password: 'pass' },
    }));
  });
});
