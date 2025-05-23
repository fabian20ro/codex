import type { AppConfig } from "./config.js";
import { URL } from "url";

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
  const headers: Record<string, string> = {};
  let processedBaseURL: string | undefined;
  const baseURLString = getBaseUrl(config.provider);

  if (OPENAI_ORGANIZATION) {
    headers["OpenAI-Organization"] = OPENAI_ORGANIZATION;
  }
  if (OPENAI_PROJECT) {
    headers["OpenAI-Project"] = OPENAI_PROJECT;
  }

  if (config.provider?.toLowerCase() === "azure") {
    return new AzureOpenAI({
      apiKey: getApiKey(config.provider),
      baseURL: baseURLString,
      apiVersion: AZURE_OPENAI_API_VERSION,
      timeout: OPENAI_TIMEOUT_MS,
      defaultHeaders: headers,
    });
  }

  processedBaseURL = baseURLString;

  if (
    config.provider?.toLowerCase() === "ollama" &&
    baseURLString &&
    baseURLString.includes("@")
  ) {
    const parsedUrl = new URL(baseURLString);
    const username = parsedUrl.username;
    const password = parsedUrl.password;

    processedBaseURL =
      parsedUrl.protocol +
      "//" +
      parsedUrl.host +
      (parsedUrl.pathname === "/" ? "" : parsedUrl.pathname);

    if (username) {
      const basicAuthHeader =
        "Basic " +
        Buffer.from(
          decodeURIComponent(username) +
            ":" +
            (password ? decodeURIComponent(password) : ""),
        ).toString("base64");
      headers["Authorization"] = basicAuthHeader;
    }
  }

  return new OpenAI({
    apiKey: getApiKey(config.provider),
    baseURL: processedBaseURL,
    timeout: OPENAI_TIMEOUT_MS,
    defaultHeaders: headers,
  });
}
