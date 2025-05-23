import type { AppConfig } from "./config.js";
import { URL } from "url";
import axios, { AxiosError, AxiosRequestConfig, Method } from 'axios';

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

  // Type assertion for method, as Axios expects specific strings
  const axiosMethod = method as Method | undefined;

  const axiosConfig: AxiosRequestConfig = {
    url: url.toString(),
    method: axiosMethod || 'GET', // Default to GET if method is not specified
    headers: headers as Record<string, string>, // Assuming headers is Record<string, string>
    data: body, // Axios uses 'data' for the request body
    signal: signal as AbortSignal, // Pass AbortSignal if present
    // timeout: /* map from init.timeout if necessary, though OpenAI client has its own timeout */
  };

  // Handle streaming responses for Server-Sent Events (SSE)
  const acceptHeader = (headers as Record<string, string>)?.['Accept']?.toLowerCase();
  if (acceptHeader === 'text/event-stream') {
    axiosConfig.responseType = 'stream';
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
