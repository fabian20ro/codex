import type * as fsType from "fs";

import {
  loadConfig,
  saveConfig,
  loadConfig as originalLoadConfig, // Keep original for other tests
  saveConfig,
  DEFAULT_SHELL_MAX_BYTES,
  DEFAULT_SHELL_MAX_LINES,
  getBaseUrl, // Import getBaseUrl
} from "../src/utils/config.js";
import * as LogModule from "../src/utils/logger/log.js"; // For spying on log
import { AutoApprovalMode } from "../src/utils/auto-approval-mode.js";
import { tmpdir } from "os";
import { join } from "path";
import { test, expect, beforeEach, afterEach, vi } from "vitest";
import { providers as defaultProviders } from "../src/utils/providers";

// In‑memory FS store
let memfs: Record<string, string> = {};

// Mock out the parts of "fs" that our config module uses:
// Also, mock loadConfig from "../src/utils/config.js" for getBaseUrl tests
vi.mock("fs", async (importOriginalFs) => {
  const realFs = (await importOriginalFs()) as typeof fsType;
  return {
    ...realFs,
    existsSync: (path: string) => memfs[path] !== undefined,
    readFileSync: (path: string) => {
      if (memfs[path] === undefined) {
        throw new Error("ENOENT");
      }
      return memfs[path];
    },
    writeFileSync: (path: string, data: string) => {
      memfs[path] = data;
    },
    mkdirSync: () => {
      // no-op in in‑memory store
    },
    rmSync: (path: string) => {
      // recursively delete any key under this prefix
      const prefix = path.endsWith("/") ? path : path + "/";
      for (const key of Object.keys(memfs)) {
        if (key === path || key.startsWith(prefix)) {
          delete memfs[key];
        }
      }
    },
  };
});

let testDir: string;
let testConfigPath: string;
let testInstructionsPath: string;

beforeEach(() => {
  memfs = {}; // reset in‑memory store
  testDir = tmpdir(); // use the OS temp dir as our "cwd"
  testConfigPath = join(testDir, "config.json");
  testInstructionsPath = join(testDir, "instructions.md");
});

afterEach(() => {
  memfs = {};
});

test("loads default config if files don't exist", () => {
  const config = loadConfig(testConfigPath, testInstructionsPath, {
    disableProjectDoc: true,
  });
  // Keep the test focused on just checking that default model and instructions are loaded
  // so we need to make sure we check just these properties
  expect(config.model).toBe("codex-mini-latest");
  expect(config.instructions).toBe("");
});

test("saves and loads config correctly", () => {
  const testConfig = {
    model: "test-model",
    instructions: "test instructions",
    notify: false,
  };
  saveConfig(testConfig, testConfigPath, testInstructionsPath);

  // Our in‑memory fs should now contain those keys:
  expect(memfs[testConfigPath]).toContain(`"model": "test-model"`);
  expect(memfs[testInstructionsPath]).toBe("test instructions");

  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });
  // Check just the specified properties that were saved
  expect(loadedConfig.model).toBe(testConfig.model);
  expect(loadedConfig.instructions).toBe(testConfig.instructions);
});

test("loads user instructions + project doc when codex.md is present", () => {
  // 1) seed memfs: a config JSON, an instructions.md, and a codex.md in the cwd
  const userInstr = "here are user instructions";
  const projectDoc = "# Project Title\n\nSome project‑specific doc";
  // first, make config so loadConfig will see storedConfig
  memfs[testConfigPath] = JSON.stringify({ model: "mymodel" }, null, 2);
  // then user instructions:
  memfs[testInstructionsPath] = userInstr;
  // and now our fake codex.md in the cwd:
  const codexPath = join(testDir, "codex.md");
  memfs[codexPath] = projectDoc;

  // 2) loadConfig without disabling project‑doc, but with cwd=testDir
  const cfg = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    cwd: testDir,
  });

  // 3) assert we got both pieces concatenated
  expect(cfg.model).toBe("mymodel");
  expect(cfg.instructions).toBe(
    userInstr + "\n\n--- project-doc ---\n\n" + projectDoc,
  );
});

test("loads and saves approvalMode correctly", () => {
  // Setup config with approvalMode
  memfs[testConfigPath] = JSON.stringify(
    {
      model: "mymodel",
      approvalMode: AutoApprovalMode.AUTO_EDIT,
    },
    null,
    2,
  );
  memfs[testInstructionsPath] = "test instructions";

  // Load config and verify approvalMode
  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  // Check approvalMode was loaded correctly
  expect(loadedConfig.approvalMode).toBe(AutoApprovalMode.AUTO_EDIT);

  // Modify approvalMode and save
  const updatedConfig = {
    ...loadedConfig,
    approvalMode: AutoApprovalMode.FULL_AUTO,
  };

  saveConfig(updatedConfig, testConfigPath, testInstructionsPath);

  // Verify saved config contains updated approvalMode
  expect(memfs[testConfigPath]).toContain(
    `"approvalMode": "${AutoApprovalMode.FULL_AUTO}"`,
  );

  // Load again and verify updated value
  const reloadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });
  expect(reloadedConfig.approvalMode).toBe(AutoApprovalMode.FULL_AUTO);
});

test("loads and saves providers correctly", () => {
  // Setup custom providers configuration
  const customProviders = {
    openai: {
      name: "Custom OpenAI",
      baseURL: "https://custom-api.openai.com/v1",
      envKey: "CUSTOM_OPENAI_API_KEY",
    },
    anthropic: {
      name: "Anthropic",
      baseURL: "https://api.anthropic.com",
      envKey: "ANTHROPIC_API_KEY",
    },
  };

  // Create config with providers
  const testConfig = {
    model: "test-model",
    provider: "anthropic",
    providers: customProviders,
    instructions: "test instructions",
    notify: false,
  };

  // Save the config
  saveConfig(testConfig, testConfigPath, testInstructionsPath);

  // Verify saved config contains providers
  expect(memfs[testConfigPath]).toContain(`"providers"`);
  expect(memfs[testConfigPath]).toContain(`"Custom OpenAI"`);
  expect(memfs[testConfigPath]).toContain(`"Anthropic"`);
  expect(memfs[testConfigPath]).toContain(`"provider": "anthropic"`);

  // Load config and verify providers were loaded correctly
  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  // Check providers were loaded correctly
  expect(loadedConfig.provider).toBe("anthropic");
  expect(loadedConfig.providers).toEqual({
    ...defaultProviders,
    ...customProviders,
  });

  // Test merging with built-in providers
  // Create a config with only one custom provider
  const partialProviders = {
    customProvider: {
      name: "Custom Provider",
      baseURL: "https://custom-api.example.com",
      envKey: "CUSTOM_API_KEY",
    },
  };

  const partialConfig = {
    model: "test-model",
    provider: "customProvider",
    providers: partialProviders,
    instructions: "test instructions",
    notify: false,
  };

  // Save the partial config
  saveConfig(partialConfig, testConfigPath, testInstructionsPath);

  // Load config and verify providers were merged with built-in providers
  const mergedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  // Check providers is defined
  expect(mergedConfig.providers).toBeDefined();

  // Use bracket notation to access properties
  if (mergedConfig.providers) {
    expect(mergedConfig.providers["customProvider"]).toBeDefined();
    expect(mergedConfig.providers["customProvider"]).toEqual(
      partialProviders.customProvider,
    );
    // Built-in providers should still be there (like openai)
    expect(mergedConfig.providers["openai"]).toBeDefined();
  }
});

test("saves and loads instructions with project doc separator correctly", () => {
  const userInstructions = "user specific instructions";
  const projectDoc = "project specific documentation";
  const combinedInstructions = `${userInstructions}\n\n--- project-doc ---\n\n${projectDoc}`;

  const testConfig = {
    model: "test-model",
    instructions: combinedInstructions,
    notify: false,
  };

  saveConfig(testConfig, testConfigPath, testInstructionsPath);

  expect(memfs[testInstructionsPath]).toBe(userInstructions);

  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });
  expect(loadedConfig.instructions).toBe(userInstructions);
});

test("handles empty user instructions when saving with project doc separator", () => {
  const projectDoc = "project specific documentation";
  const combinedInstructions = `\n\n--- project-doc ---\n\n${projectDoc}`;

  const testConfig = {
    model: "test-model",
    instructions: combinedInstructions,
    notify: false,
  };

  saveConfig(testConfig, testConfigPath, testInstructionsPath);

  expect(memfs[testInstructionsPath]).toBe("");

  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });
  expect(loadedConfig.instructions).toBe("");
});

test("loads default shell config when not specified", () => {
  // Setup config without shell settings
  memfs[testConfigPath] = JSON.stringify(
    {
      model: "mymodel",
    },
    null,
    2,
  );
  memfs[testInstructionsPath] = "test instructions";

  // Load config and verify default shell settings
  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  // Check shell settings were loaded with defaults
  expect(loadedConfig.tools).toBeDefined();
  expect(loadedConfig.tools?.shell).toBeDefined();
  expect(loadedConfig.tools?.shell?.maxBytes).toBe(DEFAULT_SHELL_MAX_BYTES);
  expect(loadedConfig.tools?.shell?.maxLines).toBe(DEFAULT_SHELL_MAX_LINES);
});

test("loads and saves custom shell config", () => {
  // Setup config with custom shell settings
  const customMaxBytes = 12_410;
  const customMaxLines = 500;

  memfs[testConfigPath] = JSON.stringify(
    {
      model: "mymodel",
      tools: {
        shell: {
          maxBytes: customMaxBytes,
          maxLines: customMaxLines,
        },
      },
    },
    null,
    2,
  );
  memfs[testInstructionsPath] = "test instructions";

  // Load config and verify custom shell settings
  const loadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  // Check shell settings were loaded correctly
  expect(loadedConfig.tools?.shell?.maxBytes).toBe(customMaxBytes);
  expect(loadedConfig.tools?.shell?.maxLines).toBe(customMaxLines);

  // Modify shell settings and save
  const updatedMaxBytes = 20_000;
  const updatedMaxLines = 1_000;

  const updatedConfig = {
    ...loadedConfig,
    tools: {
      shell: {
        maxBytes: updatedMaxBytes,
        maxLines: updatedMaxLines,
      },
    },
  };

  saveConfig(updatedConfig, testConfigPath, testInstructionsPath);

  // Verify saved config contains updated shell settings
  expect(memfs[testConfigPath]).toContain(`"maxBytes": ${updatedMaxBytes}`);
  expect(memfs[testConfigPath]).toContain(`"maxLines": ${updatedMaxLines}`);

  // Load again and verify updated values
  const reloadedConfig = originalLoadConfig(testConfigPath, testInstructionsPath, { // Use originalLoadConfig here
    disableProjectDoc: true,
  });

  expect(reloadedConfig.tools?.shell?.maxBytes).toBe(updatedMaxBytes);
  expect(reloadedConfig.tools?.shell?.maxLines).toBe(updatedMaxLines);
});

// Mock loadConfig for getBaseUrl tests specifically
// We need to be careful here. If other tests in this file also call getBaseUrl,
// they might be affected if we don't scope this mock properly or reset it.
// However, getBaseUrl is primarily tested via env vars first.
const mockedLoadConfig = vi.fn();
vi.mock("../src/utils/config.js", async (importOriginal) => {
  const original = await importOriginal() as any;
  return {
    ...original,
    loadConfig: mockedLoadConfig, // Replace loadConfig with our mock
  };
});


describe("getBaseUrl", () => {
  let originalOpenaiBaseUrl: string | undefined;
  let originalProviderBaseUrl: string | undefined;
  let logSpy: any;

  beforeEach(() => {
    // Store original env vars
    originalOpenaiBaseUrl = process.env.OPENAI_BASE_URL;
    originalProviderBaseUrl = process.env.FOO_PROVIDER_BASE_URL; // Using FOO as a sample provider

    // Reset env vars
    delete process.env.OPENAI_BASE_URL;
    delete process.env.FOO_PROVIDER_BASE_URL;
    
    mockedLoadConfig.mockReset();
    logSpy = vi.spyOn(LogModule, 'log'); // Spy on the imported log function
  });

  afterEach(() => {
    // Restore original env vars
    if (originalOpenaiBaseUrl !== undefined) {
      process.env.OPENAI_BASE_URL = originalOpenaiBaseUrl;
    } else {
      delete process.env.OPENAI_BASE_URL;
    }
    if (originalProviderBaseUrl !== undefined) {
      process.env.FOO_PROVIDER_BASE_URL = originalProviderBaseUrl;
    } else {
      delete process.env.FOO_PROVIDER_BASE_URL;
    }
    logSpy.mockRestore();
  });

  test("should return undefined if no URL is configured", () => {
    mockedLoadConfig.mockReturnValue({}); // No providers in config
    expect(getBaseUrl("openai")).toBeUndefined();
    expect(getBaseUrl("foo_provider")).toBeUndefined();
  });

  describe("with environment variables", () => {
    test("strips userinfo from FOO_PROVIDER_BASE_URL", () => {
      process.env.FOO_PROVIDER_BASE_URL = "http://user:pass@example.com/api";
      expect(getBaseUrl("foo_provider")).toBe("http://example.com/api");
    });

    test("strips userinfo from OPENAI_BASE_URL", () => {
      process.env.OPENAI_BASE_URL = "https://user@another.com/path";
      expect(getBaseUrl("openai")).toBe("https://another.com/path");
    });
    
    test("strips userinfo and includes port from OPENAI_BASE_URL", () => {
      process.env.OPENAI_BASE_URL = "https://user:secret@localhost:8080/path";
      expect(getBaseUrl("openai")).toBe("https://localhost:8080/path");
    });

    test("returns URL as is if no userinfo in OPENAI_BASE_URL", () => {
      process.env.OPENAI_BASE_URL = "https://clean.com/api";
      expect(getBaseUrl("openai")).toBe("https://clean.com/api");
    });

    test("removes trailing slash from OPENAI_BASE_URL", () => {
      process.env.OPENAI_BASE_URL = "https://example.com/api/";
      expect(getBaseUrl("openai")).toBe("https://example.com/api");
    });
  });

  describe("with config file (mocked loadConfig)", () => {
    test("strips userinfo from provider's baseURL in config", () => {
      mockedLoadConfig.mockReturnValue({
        providers: {
          foo_provider: { baseURL: "http://user:pass@config.com/api" },
        },
      });
      expect(getBaseUrl("foo_provider")).toBe("http://config.com/api");
    });

    test("returns URL as is from config if no userinfo", () => {
      mockedLoadConfig.mockReturnValue({
        providers: {
          foo_provider: { baseURL: "https://config.com/path" },
        },
      });
      expect(getBaseUrl("foo_provider")).toBe("https://config.com/path");
    });
    
    test("removes trailing slash from config URL", () => {
      mockedLoadConfig.mockReturnValue({
        providers: {
          foo_provider: { baseURL: "https://config.com/another/" },
        },
      });
      expect(getBaseUrl("foo_provider")).toBe("https://config.com/another");
    });

    test("falls back to OPENAI_BASE_URL env var if provider not in config", () => {
      mockedLoadConfig.mockReturnValue({ providers: {} }); // Empty providers
      process.env.OPENAI_BASE_URL = "https://user:pass@env-fallback.com/api";
      expect(getBaseUrl("some_provider")).toBe("https://env-fallback.com/api");
    });
  });

  describe("invalid URLs", () => {
    test("handles malformed URL from env var and logs warning", () => {
      process.env.OPENAI_BASE_URL = "htp://invalid-url";
      expect(getBaseUrl("openai")).toBeUndefined();
      expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("[codex] Warning: Malformed URL"));
    });
    
    test("handles empty string URL from env var and logs warning", () => {
      process.env.OPENAI_BASE_URL = "";
      // The URL constructor might throw on empty string, or it might parse it as relative to current location.
      // The current implementation of getBaseUrl would pass "" to new URL(), which errors.
      expect(getBaseUrl("openai")).toBeUndefined();
      expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("[codex] Warning: Malformed URL"));
    });

    test("handles malformed URL from config and logs warning", () => {
      mockedLoadConfig.mockReturnValue({
        providers: {
          foo_provider: { baseURL: "badscheme://config.com" },
        },
      });
      expect(getBaseUrl("foo_provider")).toBeUndefined();
      expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("[codex] Warning: Malformed URL"));
    });

    test("handles undefined URL from config (e.g. provider exists but no baseURL)", () => {
      mockedLoadConfig.mockReturnValue({
        providers: {
          foo_provider: { name: "Foo" /* no baseURL */ },
        },
      });
      // This should then fall back to OPENAI_BASE_URL or return undefined
      process.env.OPENAI_BASE_URL = "https://fallback.com/api";
      expect(getBaseUrl("foo_provider")).toBe("https://fallback.com/api");

      delete process.env.OPENAI_BASE_URL;
      expect(getBaseUrl("foo_provider")).toBeUndefined();
    });
  });
});
