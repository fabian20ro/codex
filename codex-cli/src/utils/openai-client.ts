import type { AppConfig } from "./config.js";
import { URL } from "url";
import axios, { AxiosError, AxiosRequestConfig, Method } from 'axios';
import * as https from 'https';
import * as fs from 'fs';

import {
  getBaseUrl,
  getApiKey,
  AZURE_OPENAI_API_VERSION,
  OPENAI_TIMEOUT_MS,
  OPENAI_ORGANIZATION,
  OPENAI_PROJECT,
} from "./config.js";
import OpenAI, { AzureOpenAI } from "openai";

type OpenAIClientConfig = {
  provider: string;
};

export async function customFetchForOllamaWithCreds(
  url: RequestInfo, // This will be a string from the OpenAI library
  init?: RequestInit,
): Promise<Response> {
  const { method, headers, body, signal, ...restOfInit } = init || {};
  const urlString = url.toString();
  const parsedUrl = new URL(urlString);

  let cleanUrlString = urlString;
  let authOption: AxiosRequestConfig['auth'] = undefined;

  if (parsedUrl.username || parsedUrl.password) {
    const username = decodeURIComponent(parsedUrl.username);
    const password = decodeURIComponent(parsedUrl.password); // Will be empty string if not present
    authOption = { username, password };

    // Reconstruct the URL without userinfo
    parsedUrl.username = '';
    parsedUrl.password = '';
    cleanUrlString = parsedUrl.toString();
  }

  // Type assertion for method, as Axios expects specific strings
  const axiosMethod = method as Method | undefined;

  const axiosConfig: AxiosRequestConfig = {
    url: cleanUrlString, // Use the potentially cleaned URL
    method: axiosMethod || 'GET', // Default to GET if method is not specified
    headers: headers as Record<string, string>, // Assuming headers is Record<string, string>
    data: body, // Axios uses 'data' for the request body
    signal: signal as AbortSignal, // Pass AbortSignal if present
    // timeout: /* map from init.timeout if necessary, though OpenAI client has its own timeout */
  };

  if (authOption) {
    axiosConfig.auth = authOption;
  }

  // Handle streaming responses for Server-Sent Events (SSE)
  const acceptHeader = (headers as Record<string, string>)?.['Accept']?.toLowerCase();
  if (acceptHeader === 'text/event-stream') {
    axiosConfig.responseType = 'stream';
  }

  const customCaCertPath = process.env['OLLAMA_CUSTOM_CA_CERT_PATH'];
  // For CA decision, use the original URL's protocol
  const caDecisionUrl = new URL(urlString); 

  if (caDecisionUrl.protocol === 'https:' && customCaCertPath) {
    try {
      const caCert = fs.readFileSync(customCaCertPath);
      axiosConfig.httpsAgent = new https.Agent({ ca: caCert });
      // Advise against NODE_TLS_REJECT_UNAUTHORIZED=0 if possible,
      // but the code itself doesn't set/unset it.
      // Log a warning if NODE_TLS_REJECT_UNAUTHORIZED is set to 0, as it might interfere.
      if (process.env['NODE_TLS_REJECT_UNAUTHORIZED'] === '0') {
        console.warn('[Codex CLI] Warning: OLLAMA_CUSTOM_CA_CERT_PATH is set, but NODE_TLS_REJECT_UNAUTHORIZED is also set to 0. This may lead to unexpected behavior or bypass custom CA validation. It is recommended to remove NODE_TLS_REJECT_UNAUTHORIZED=0 when using a custom CA certificate.');
      }
    } catch (e: any) {
      console.error(`[Codex CLI] Error: Could not read CA certificate file from OLLAMA_CUSTOM_CA_CERT_PATH at ${customCaCertPath}: ${e.message}`);
      // Decide if we should throw the error or proceed without custom CA.
      // Proceeding without it will likely cause a connection failure by default if the cert is self-signed.
      // Throwing seems more appropriate to alert the user to a config issue.
      throw new Error(`Failed to load CA certificate from ${customCaCertPath}: ${e.message}`);
    }
  }

  try {
    const response = await axios(axiosConfig);

    const responseHeaders = new Headers();
    for (const [key, value] of Object.entries(response.headers)) {
      if (typeof value === 'string') {
        responseHeaders.append(key, value);
      } else if (Array.isArray(value)) {
        value.forEach((v) => responseHeaders.append(key, v));
      }
    }
    
    // If response.data is a stream (due to responseType: 'stream'),
    // it needs to be wrapped in a ReadableStream for the Fetch API Response.
    // Node.js streams are not directly compatible with Web API ReadableStream.
    // However, for Node.js environments, often the stream from axios IS a Node.js stream
    // which might be directly usable by the version of `Response` in that env,
    // or might need a simple conversion. Let's assume for now it's compatible enough
    // or the OpenAI lib handles this. If not, a conversion utility like `Readable.toWeb` might be needed.
    const responseBody = response.data;

    return new Response(responseBody, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      const errorResponse = error.response;
      const errorResponseHeaders = new Headers();
      for (const [key, value] of Object.entries(errorResponse.headers)) {
         if (typeof value === 'string') {
          errorResponseHeaders.append(key, value);
        } else if (Array.isArray(value)) {
          value.forEach((v) => errorResponseHeaders.append(key, v));
        }
      }
      // Similar stream consideration for errorResponse.data
      return new Response(errorResponse.data, {
        status: errorResponse.status,
        statusText: errorResponse.statusText,
        headers: errorResponseHeaders,
      });
    }
    // For non-Axios errors or errors without a response, rethrow
    // so it's not swallowed and can be handled by the OpenAI client or calling code.
    // console.error('Error in customFetchForOllamaWithCreds:', error);
    throw error;
  }
}

/**
 * Creates an OpenAI client instance based on the provided configuration.
 * Handles both standard OpenAI and Azure OpenAI configurations.
 *
 * @param config The configuration containing provider information
 * @returns An instance of either OpenAI or AzureOpenAI client
 */
export function createOpenAIClient(
  config: OpenAIClientConfig | AppConfig,
): OpenAI | AzureOpenAI {
  const baseURLString = getBaseUrl(config.provider);
  const headers: Record<string, string> = {};

  if (OPENAI_ORGANIZATION) {
    headers["OpenAI-Organization"] = OPENAI_ORGANIZATION;
  }
  if (OPENAI_PROJECT) {
    headers["OpenAI-Project"] = OPENAI_PROJECT;
  }

  if (
    config.provider?.toLowerCase() === "ollama" &&
    baseURLString &&
    baseURLString.includes("@")
  ) {
    // If Ollama and URL has credentials, use the custom fetch with the original URL
    return new OpenAI({
      apiKey: getApiKey(config.provider), // Still pass API key (dummy for Ollama)
      baseURL: baseURLString,             // Original URL with credentials
      timeout: OPENAI_TIMEOUT_MS,
      defaultHeaders: headers,
      fetch: customFetchForOllamaWithCreds, // The custom fetch function
    });
  } else {
    // For all other cases (non-Ollama, or Ollama without credentials in URL):
    // Use the standard OpenAI client setup.
    if (config.provider?.toLowerCase() === "azure") {
      return new AzureOpenAI({
        apiKey: getApiKey(config.provider),
        baseURL: baseURLString, // Use original baseURLString
        apiVersion: AZURE_OPENAI_API_VERSION,
        timeout: OPENAI_TIMEOUT_MS,
        defaultHeaders: headers,
      });
    }

    return new OpenAI({
      apiKey: getApiKey(config.provider),
      baseURL: baseURLString, // Use original baseURLString
      timeout: OPENAI_TIMEOUT_MS,
      defaultHeaders: headers,
    });
  }
}
