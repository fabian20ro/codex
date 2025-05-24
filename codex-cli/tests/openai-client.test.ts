import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as openaiClientModule from "../src/utils/openai-client"; // Import the module to spy on its exports
import * as config from "../src/utils/config";
import OpenAI from "openai";
import { Readable } from "stream";
import axios from 'axios'; 

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
const mockOpenAIConstructor = vi.fn();
vi.mock("openai", () => {
  const actualOpenAI = vi.importActual("openai");
  return {
    ...actualOpenAI,
    __esModule: true,
    default: mockOpenAIConstructor, // Use the new spy here
    AzureOpenAI: vi.fn(),
  };
});

// We are spying on the actual customFetchForOllamaWithCreds for some tests
let customFetchSpy: ReturnType<typeof vi.spyOn>;

describe("createOpenAIClient", () => {
  beforeEach(() => {
    vi.clearAllMocks(); // This clears mockOpenAIConstructor, mockAxios, etc.
    vi.mocked(config.getApiKey).mockReturnValue("test-api-key");
    vi.mocked(config.getBaseUrl).mockImplementation((provider?: string) => {
      if (provider === "azure") return "https://azure.example.com";
      return "https://api.openai.com/v1";
    });

    // Ensure spy is set up before each test that might need it
    customFetchSpy = vi.spyOn(openaiClientModule, 'customFetchForOllamaWithCreds');
  });

  afterEach(() => {
    customFetchSpy.mockRestore(); // Restore the spy
  });


  it("should create a standard OpenAI client for 'openai' provider", () => {
    vi.mocked(config.getBaseUrl).mockReturnValue("https://api.openai.com/v1");
    openaiClientModule.createOpenAIClient({ provider: "openai" });
    expect(mockOpenAIConstructor).toHaveBeenCalledWith({
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
      timeout: 10000,
      defaultHeaders: {},
    });
  });

  describe("Ollama provider with credentials", () => {
    it("should pass a clean baseURL to OpenAI and use fetcherForThisInstance that calls customFetchForOllamaWithCreds with correct credentials", async () => {
      const ollamaUrlWithCreds = "http://user123:pass456@ollamahost.com/api";
      const cleanOllamaBaseURL = "http://ollamahost.com/api";
      
      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama_api_key"); // Ollama uses this as a dummy

      // Call createOpenAIClient which sets up the OpenAI client with the fetcher
      openaiClientModule.createOpenAIClient({ provider: "ollama" });

      // Verify OpenAI constructor call
      expect(mockOpenAIConstructor).toHaveBeenCalledTimes(1);
      const constructorOptions = mockOpenAIConstructor.mock.calls[0][0];
      expect(constructorOptions.baseURL).toBe(cleanOllamaBaseURL);
      expect(constructorOptions.apiKey).toBe("ollama_api_key");
      expect(typeof constructorOptions.fetch).toBe('function');

      // Now, test the fetcherForThisInstance closure
      const fetcherForThisInstance = constructorOptions.fetch;
      const sampleFetchUrl = `${cleanOllamaBaseURL}/chat/completions`; // Example endpoint
      const sampleFetchInit = { method: 'POST', body: JSON.stringify({ model: "llama2" }) };
      
      // Mock the underlying customFetchForOllamaWithCreds to avoid actual network call
      customFetchSpy.mockResolvedValueOnce(new Response("mocked response"));

      await fetcherForThisInstance(sampleFetchUrl, sampleFetchInit);

      expect(customFetchSpy).toHaveBeenCalledTimes(1);
      expect(customFetchSpy).toHaveBeenCalledWith(
        sampleFetchUrl,
        sampleFetchInit,
        { username: 'user123', password: 'pass456' }
      );
    });
    
    it("should correctly decode URI components in credentials for fetcherForThisInstance", async () => {
      const ollamaUrlWithEncodedCreds = "http://user%20name:p%40ss%23w%C3%B6rd@ollamahost.com/api/v2";
      const cleanOllamaBaseURL = "http://ollamahost.com/api/v2";

      vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithEncodedCreds);
      vi.mocked(config.getApiKey).mockReturnValue("ollama");
      
      openaiClientModule.createOpenAIClient({ provider: "ollama" });
      
      const constructorOptions = mockOpenAIConstructor.mock.calls[0][0];
      const fetcherForThisInstance = constructorOptions.fetch;
      
      customFetchSpy.mockResolvedValueOnce(new Response("mocked response"));
      await fetcherForThisInstance(`${cleanOllamaBaseURL}/generate`, {});

      expect(customFetchSpy).toHaveBeenCalledWith(
        `${cleanOllamaBaseURL}/generate`,
        {},
        { username: 'user name', password: 'p@ss#wörd' }
      );
    });
  });
  
  it("Ollama provider without credentials: should call OpenAI constructor with original baseURL and no fetch override", () => {
    const ollamaUrlWithoutCreds = "http://localhost:11434/v1";
    vi.mocked(config.getBaseUrl).mockReturnValue(ollamaUrlWithoutCreds);
    vi.mocked(config.getApiKey).mockReturnValue("ollama");
    openaiClientModule.createOpenAIClient({ provider: "ollama" });
    
    const expectedCall = {
      apiKey: "ollama",
      baseURL: ollamaUrlWithoutCreds,
      timeout: 10000,
      defaultHeaders: {},
    };
    expect(mockOpenAIConstructor).toHaveBeenCalledWith(expect.objectContaining(expectedCall));
    const actualCall = mockOpenAIConstructor.mock.calls[0][0];
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
    if (originalOllamaCaPath !== undefined) process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = originalOllamaCaPath;
    else delete process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
    if (originalNodeTlsRejectUnauthorized !== undefined) process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = originalNodeTlsRejectUnauthorized;
    else delete process.env['NODE_TLS_REJECT_UNAUTHORIZED'];
  });

  it('should make a successful GET request without credentials and return a Response object', async () => {
    const mockResponseData = 'success';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: { 'content-type': 'text/plain' } });
    const cleanUrl = 'http://localhost/test';
    const response = await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'GET' }); 
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({ url: cleanUrl, method: 'GET', auth: undefined }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should handle credentials passed via creds object, using them for axios auth', async () => {
    const mockResponseData = 'success_with_auth';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: {} });
    
    const cleanUrl = 'http://localhost/authtest/v1';
    const creds = { username: 'user', password: 'pass' };
    const response = await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'POST' }, creds);

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      method: 'POST',
      auth: creds,
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });
  
  it('should remove pre-existing Authorization header and use creds for auth', async () => {
    mockAxios.mockResolvedValueOnce({ data: 'auth_override_success', status: 200, headers: {} });
    
    const cleanUrl = 'http://localhost/auth-override-test';
    const initHeaders = {
      'Authorization': 'Bearer some-token-that-should-be-removed',
      'Content-Type': 'application/json',
      'X-Custom-Header': 'keep-this',
    };
    const creds = { username: 'ollamauser', password: 'ollamapass' };
    
    await openaiClientModule.customFetchForOllamaWithCreds(
      cleanUrl, 
      { method: 'POST', headers: initHeaders, body: JSON.stringify({ data: 'payload' }) }, 
      creds
    );

    expect(mockAxios).toHaveBeenCalledTimes(1);
    const axiosCallArgs = mockAxios.mock.calls[0][0];
    
    // Check that Authorization header is removed (both casings)
    expect(axiosCallArgs.headers.Authorization).toBeUndefined();
    expect(axiosCallArgs.headers.authorization).toBeUndefined();
    
    // Check that other headers are preserved
    expect(axiosCallArgs.headers['Content-Type']).toBe('application/json');
    expect(axiosCallArgs.headers['X-Custom-Header']).toBe('keep-this');
    
    // Check that new auth from creds is used
    expect(axiosCallArgs.auth).toEqual(creds);
    
    // Check other parameters
    expect(axiosCallArgs.url).toBe(cleanUrl);
    expect(axiosCallArgs.method).toBe('POST');
    expect(axiosCallArgs.data).toBe(JSON.stringify({ data: 'payload' }));
  });

  it('should handle URI-encoded credentials passed via creds object (assuming they are pre-decoded by caller)', async () => {
    const mockResponseData = 'success_encoded_auth';
    mockAxios.mockResolvedValueOnce({ data: mockResponseData, status: 200, statusText: 'OK', headers: {} });
    
    const cleanUrl = 'http://localhost/encoded/api';
    const decodedCreds = { username: 'user name', password: 'p@ss#word' }; 
    const response = await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'GET' }, decodedCreds);

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      method: 'GET',
      auth: decodedCreds,
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(mockResponseData);
  });

  it('should handle username only in creds object, passing empty password to axios auth', async () => {
    mockAxios.mockResolvedValueOnce({ data: 'success_user_only', status: 200, headers: {} });
    const cleanUrl = 'http://localhost/useronly';
    const creds = { username: 'admin', password: '' }; 
    await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'GET' }, creds);
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      auth: creds
    }));
  });

  it('should handle streaming response for text/event-stream with credentials in creds object', async () => {
    const mockNodeReadableStream = Readable.from(['data: event1\n\n']);
    mockAxios.mockResolvedValueOnce({ data: mockNodeReadableStream, status: 200, headers: { 'content-type': 'text/event-stream' } });
    
    const cleanUrl = 'http://localhost/stream';
    const creds = { username: 'streamuser', password: 'streampass' };
    const response = await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, 
      { method: 'GET', headers: { 'Accept': 'text/event-stream' } }, 
      creds
    );

    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      responseType: 'stream',
      auth: creds,
      headers: { 'Accept': 'text/event-stream' }, // Authorization should not be here
    }));
    expect(response.body).toBe(mockNodeReadableStream);
  });

  it('HTTPS URL with CA cert and credentials in creds: should configure https.Agent and axios auth', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/ca.pem';
    const dummyCertContent = Buffer.from('dummy_ca_cert_content');
    mockReadFileSync.mockReturnValueOnce(dummyCertContent);
    const mockAgentInstance = { type: 'mockAgent' };
    mockHttpsAgent.mockReturnValueOnce(mockAgentInstance);
    mockAxios.mockResolvedValueOnce({ data: 'success_ca_auth', status: 200, headers: {} });

    const cleanUrl = 'https://localhost/secure-auth';
    const creds = { username: 'secureuser', password: 'securepass' };
    await openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'GET' }, creds);

    expect(mockReadFileSync).toHaveBeenCalledWith('/path/to/ca.pem');
    expect(mockHttpsAgent).toHaveBeenCalledWith({ ca: dummyCertContent });
    expect(mockAxios).toHaveBeenCalledWith(expect.objectContaining({
      url: cleanUrl,
      httpsAgent: mockAgentInstance,
      auth: creds,
    }));
  });

  it('HTTPS URL with invalid CA cert path and credentials in creds: should throw error before axios call', async () => {
    process.env['OLLAMA_CUSTOM_CA_CERT_PATH'] = '/path/to/bad_ca.pem';
    mockReadFileSync.mockImplementationOnce(() => { throw new Error('File not found'); });
    
    const cleanUrl = 'https://localhost/secure-fail-auth';
    const creds = { username: 'user', password: 'pass' };
    await expect(openaiClientModule.customFetchForOllamaWithCreds(cleanUrl, { method: 'GET' }, creds))
      .rejects.toThrow('Failed to load CA certificate from /path/to/bad_ca.pem: File not found');
    
    expect(mockAxios).not.toHaveBeenCalled();
  });
});
