"use strict";

// ============================================================
// NVIDIA NIM -> OPENAI-COMPATIBLE CHAT PROXY
// JanitorAI + ChubAI/Venus
// ONE ACTIVE MODEL. NO AUTOMATIC MODEL FALLBACK.
// ============================================================
//
// Dependencies: express, cors, axios
//
// Required Render environment variable:
//   NIM_API_KEY       NVIDIA API key
//
// Optional Render environment variables:
//   ACTIVE_MODEL      Exact model ID from MODEL_PROFILES
//   NIM_API_BASE      Defaults to NVIDIA Integrate API
//   PROXY_API_KEY     Protects this proxy from unauthorized callers
//   NIM_TIMEOUT_MS    Upstream timeout; default 150000
//   MAX_CONCURRENT_REQUESTS  Simultaneous chat requests; default 4
//
// Client authentication:
//   If PROXY_API_KEY is configured, clients must send:
//   Authorization: Bearer YOUR_PROXY_API_KEY
//
// The proxy key and NVIDIA key are completely separate.
//
// Streaming:
//   stream:false or omitted = normal JSON response
//   stream:true              = SSE passthrough
//
// Model selection:
//   The server ALWAYS uses ACTIVE_MODEL.
//   The client's model field cannot change it.
//   Reserve models are never automatically selected.
// ============================================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const crypto = require("crypto");

const app = express();
const PORT = Number(process.env.PORT) || 3000;

const NIM_API_BASE = (
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY || "";

const PROXY_API_KEY = process.env.PROXY_API_KEY || "";

const REQUEST_TIMEOUT_MS = Math.max(
  10000,
  Number(process.env.NIM_TIMEOUT_MS) || 150000
);

const MAX_OUTPUT_TOKENS = 8192;

const MAX_CONCURRENT_REQUESTS = Math.max(
  1,
  Math.min(100, Number(process.env.MAX_CONCURRENT_REQUESTS) || 4)
);

let activeRequests = 0;

// ============================================================
// MODEL LIBRARY
// ALL 11 ORIGINAL MODEL PROFILES PRESERVED.
// ONLY ACTIVE_MODEL RECEIVES CHAT REQUESTS.
// ============================================================

const MODEL_PROFILES = Object.freeze({

  "mistralai/mistral-large-4-1t-a52b": {
    maxTokens: 4096,
    temperature: 0.7,
    advanced: [
      "frequency_penalty",
      "presence_penalty",
      "tools",
      "tool_choice"
    ]
  },

  "z-ai/glm-5.3-flash": {
    maxTokens: 4096,
    temperature: 0.7,
    extra: {
      reasoning_effort: "low",
      chat_template_kwargs: {
        clear_thinking: true
      }
    },
    advanced: [
      "frequency_penalty",
      "presence_penalty",
      "tools",
      "tool_choice"
    ]
  },

  "minimaxai/minimax-m3.1-flash": {
    maxTokens: 6144,
    temperature: 1.0,
    topP: 0.95,
    advanced: [
      "tools",
      "tool_choice"
    ]
  },

  "moonshotai/kimi-k2.6": {
    maxTokens: 4096,
    temperature: 0.7,
    extra: {
      chat_template_kwargs: {
        thinking: false
      }
    },
    advanced: [
      "tools",
      "tool_choice"
    ]
  },

  "nvidia/nemotron-3.5-lightning-30b-a3b": {
    maxTokens: 3072,
    temperature: 1.0,
    topP: 0.95,
    extra: {
      chat_template_kwargs: {
        enable_thinking: false
      }
    }
  },

  "nvidia/nemotron-3.5-bolt-e4b": {
    maxTokens: 3072,
    temperature: 1.0,
    topP: 0.95,
    extra: {
      chat_template_kwargs: {
        enable_thinking: false
      }
    },
    advanced: [
      "tools",
      "tool_choice",
      "response_format"
    ]
  },

  "google/gemma-4-31b-it": {
    maxTokens: 3072,
    temperature: 0.8,
    extra: {
      chat_template_kwargs: {
        enable_thinking: false
      }
    }
  },

  "writer/palmyra-creative-122b": {
    maxTokens: 3072,
    temperature: 0.8
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    maxTokens: 4096,
    temperature: 0.7
  },

  "stepfun-ai/step-5-preview": {
    maxTokens: 4096,
    temperature: 0.5,
    topP: 1,
    advanced: [
      "frequency_penalty",
      "presence_penalty",
      "tools",
      "tool_choice"
    ]
  },

  "deepseek-ai/deepseek-v4.1-flash": {
    maxTokens: 4096,
    temperature: 0.7
  }

});

const DEFAULT_ACTIVE_MODEL = "stepfun-ai/step-5-preview";

const ACTIVE_MODEL = (
  process.env.ACTIVE_MODEL || DEFAULT_ACTIVE_MODEL
).trim();

const ACTIVE_PROFILE = Object.prototype.hasOwnProperty.call(
  MODEL_PROFILES,
  ACTIVE_MODEL
)
  ? MODEL_PROFILES[ACTIVE_MODEL]
  : null;

const RESERVE_MODELS = Object.keys(MODEL_PROFILES).filter(
  (name) => name !== ACTIVE_MODEL
);

// ============================================================
// EXPRESS MIDDLEWARE
// ============================================================

app.use((req, res, next) => {
  req.requestId = crypto.randomUUID();
  req.startedAt = Date.now();

  res.setHeader("X-Request-ID", req.requestId);

  next();
});

app.use(cors());

app.use(express.json({
  limit: "25mb"
}));

app.use(express.urlencoded({
  extended: true,
  limit: "25mb"
}));

// ============================================================
// SHARED HELPERS
// ============================================================

function apiError(res, status, message, type, requestId) {
  const error = {
    message,
    type,
    code: status
  };

  if (requestId) {
    error.request_id = requestId;
  }

  return res.status(status).json({
    error
  });
}

function isObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function nonEmptyText(content) {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === "string") {
        return part;
      }

      if (isObject(part) && typeof part.text === "string") {
        return part.text;
      }

      return "";
    }).join("");
  }

  return "";
}

function configurationError() {
  if (!NIM_API_KEY) {
    return "Missing NIM_API_KEY (or NVIDIA_API_KEY) in Render.";
  }

  if (!ACTIVE_PROFILE) {
    return (
      "Unknown ACTIVE_MODEL: " +
      ACTIVE_MODEL +
      ". Choose an exact model ID from MODEL_PROFILES."
    );
  }

  return null;
}

function getBearerToken(req) {
  const header = req.headers.authorization || "";

  const match = header.match(/^Bearer\s+(.+)$/i);

  return match ? match[1].trim() : "";
}

// Optional authentication.
// Disabled completely when PROXY_API_KEY is not configured.
//
// This protects the proxy endpoint, not NVIDIA itself.
// The NVIDIA API key is never sent to the client.

function proxyAuthentication(req, res, next) {
  if (!PROXY_API_KEY) {
    return next();
  }

  const suppliedKey = getBearerToken(req);

  const suppliedBuffer = Buffer.from(suppliedKey);
  const expectedBuffer = Buffer.from(PROXY_API_KEY);

  const valid =
    suppliedBuffer.length === expectedBuffer.length &&
    crypto.timingSafeEqual(suppliedBuffer, expectedBuffer);

  if (!valid) {
    return apiError(
      res,
      401,
      "Invalid or missing proxy API key.",
      "authentication_error",
      req.requestId
    );
  }

  next();
}

app.use(["/v1", "/nvidia"], proxyAuthentication);

// ============================================================
// PAYLOAD BUILDER
// ============================================================

function buildPayload(body, streaming) {
  const profile = ACTIVE_PROFILE;

  const payload = {
    model: ACTIVE_MODEL,
    messages: body.messages,
    stream: Boolean(streaming),
    max_tokens: profile.maxTokens
  };

  if (Number.isFinite(profile.temperature)) {
    payload.temperature = profile.temperature;
  }

  if (Number.isFinite(profile.topP)) {
    payload.top_p = profile.topP;
  }

  // Compatible frontend parameters.
  // Avoid blindly forwarding unsupported sampling/reasoning options.

  for (const name of [
    "temperature",
    "top_p",
    "stop",
    "seed"
  ]) {
    if (body[name] !== undefined && body[name] !== null) {
      payload[name] = body[name];
    }
  }

  // Only parameters explicitly permitted by the active model profile.

  for (const name of profile.advanced || []) {
    if (body[name] !== undefined && body[name] !== null) {
      payload[name] = body[name];
    }
  }

  // Some frontends send token counts as strings or floats.
  const clientTokens = Math.floor(
    Number(body.max_tokens ?? body.max_completion_tokens)
  );

  if (
    Number.isInteger(clientTokens) &&
    clientTokens > 0
  ) {
    payload.max_tokens = Math.min(
      clientTokens,
      MAX_OUTPUT_TOKENS
    );
  } else {
    payload.max_tokens = Math.min(
      profile.maxTokens,
      MAX_OUTPUT_TOKENS
    );
  }

  // Model-specific settings always win.
  // We do not inject generic reasoning flags into other models.

  if (profile.extra) {
    Object.assign(payload, profile.extra);
  }

  return payload;
}

// ============================================================
// REQUEST VALIDATION
// Keep this intentionally permissive for JanitorAI and Venus.
// ============================================================

function validateChatRequest(body) {
  if (!isObject(body)) {
    return "Request body must be a JSON object.";
  }

  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0
  ) {
    return "Provide a nonempty messages array.";
  }

  for (let i = 0; i < body.messages.length; i++) {
    const message = body.messages[i];

    if (!isObject(message)) {
      return "Each message must be an object.";
    }

    if (
      typeof message.role !== "string" ||
      message.role.trim().length === 0
    ) {
      return "Each message must have a valid role.";
    }
  }

  // Be permissive: some frontends send "true"/"false" strings or null.
  if (body.stream === "true") body.stream = true;
  if (body.stream === "false" || body.stream === null) body.stream = false;

  if (
    body.stream !== undefined &&
    typeof body.stream !== "boolean"
  ) {
    return "The stream parameter must be true or false.";
  }

  return null;
}

// ============================================================
// CONCURRENCY CONTROL
// No queue and no silent request dropping.
// Busy server returns a clear retryable error.
// ============================================================

function acquireRequestSlot() {
  if (activeRequests >= MAX_CONCURRENT_REQUESTS) {
    return false;
  }

  activeRequests++;
  return true;
}

function releaseRequestSlot() {
  activeRequests = Math.max(0, activeRequests - 1);
}

// ============================================================
// REQUEST DIAGNOSTICS
// Never log prompts, messages, API keys, or character content.
// ============================================================

function logRequestStart(req, payload) {
  console.log(
    "[REQUEST START]",
    "id:", req.requestId,
    "model:", ACTIVE_MODEL,
    "stream:", payload.stream,
    "tokens:", payload.max_tokens,
    "active:", activeRequests + "/" + MAX_CONCURRENT_REQUESTS
  );
}

function logRequestEnd(req, status, outcome) {
  const elapsed = Date.now() - req.startedAt;

  console.log(
    "[REQUEST END]",
    "id:", req.requestId,
    "model:", ACTIVE_MODEL,
    "status:", status,
    "result:", outcome,
    "elapsed_ms:", elapsed,
    "active:", activeRequests + "/" + MAX_CONCURRENT_REQUESTS
  );
}

// ============================================================
// CLIENT DISCONNECT TRACKING
// ============================================================

function createDisconnectTracker(req, res, controller) {
  let disconnected = false;

  const onAborted = () => {
    disconnected = true;
    controller.abort();
  };

  const onClose = () => {
    if (!res.writableEnded) {
      disconnected = true;
      controller.abort();
    }
  };

  req.on("aborted", onAborted);
  res.on("close", onClose);

  return {
    get disconnected() {
      return disconnected;
    },

    cleanup() {
      req.removeListener("aborted", onAborted);
      res.removeListener("close", onClose);
    }
  };
}

// ============================================================
// UPSTREAM ERROR BODY READER
// Used for non-200 streaming responses.
// Limits diagnostic body size.
// ============================================================

function readStreamText(stream, maxBytes = 12000) {
  return new Promise((resolve) => {
    let output = "";
    let size = 0;

    stream.on("data", (chunk) => {
      if (size >= maxBytes) {
        return;
      }

      const buffer = Buffer.isBuffer(chunk)
        ? chunk
        : Buffer.from(chunk);

      const remaining = maxBytes - size;
      const slice = buffer.subarray(0, remaining);

      output += slice.toString("utf8");
      size += slice.length;
    });

    stream.on("end", () => resolve(output));

    stream.on("error", () => resolve(output));

    stream.on("close", () => resolve(output));
  });
}

// ============================================================
// UPSTREAM ERROR MESSAGE EXTRACTION
// ============================================================

function extractRemoteMessage(data) {
  if (typeof data === "string") {
    return data;
  }

  if (!isObject(data)) {
    return "";
  }

  return (
    data.error?.message ||
    data.detail ||
    data.message ||
    ""
  );
}

// ============================================================
// NVIDIA REQUEST HEADERS
// ============================================================

function upstreamHeaders(streaming) {
  return {
    Authorization: "Bearer " + NIM_API_KEY,
    "Content-Type": "application/json",
    Accept: streaming
      ? "text/event-stream"
      : "application/json"
  };
}

// ============================================================
// ROOT / HEALTH
// ============================================================

app.get("/", (_req, res) => {
  res.json({
    service: "NVIDIA NIM chat proxy",
    active_model: ACTIVE_MODEL,
    streaming: true,
    authentication_required: Boolean(PROXY_API_KEY)
  });
});

// Configuration health is deliberately separate from upstream
// connectivity. This endpoint does not make an NVIDIA request.

app.get("/health", (_req, res) => {
  const problem = configurationError();

  res.status(problem ? 503 : 200).json({
    status: problem ? "misconfigured" : "ok",
    active_model: ACTIVE_MODEL,
    reserve_models: RESERVE_MODELS.length,
    api_key_configured: Boolean(NIM_API_KEY),
    proxy_authentication_enabled: Boolean(PROXY_API_KEY),
    active_requests: activeRequests,
    max_concurrent_requests: MAX_CONCURRENT_REQUESTS,
    timeout_ms: REQUEST_TIMEOUT_MS,
    ...(problem ? { error: problem } : {})
  });
});

// ============================================================
// NVIDIA MODEL CATALOG CACHE
// ============================================================

let fullCatalog = null;
let fullCatalogFetchedAt = 0;

const CATALOG_TTL_MS = 10 * 60 * 1000;
const HEALTH_CACHE_TTL_MS = 60 * 1000;

let upstreamHealthCache = null;
let upstreamHealthCheckedAt = 0;

async function fetchNvidiaCatalog(timeout = 30000) {
  return axios.get(NIM_API_BASE + "/models", {
    headers: {
      Authorization: "Bearer " + NIM_API_KEY
    },
    timeout,
    validateStatus: () => true
  });
}

// ============================================================
// LIVE UPSTREAM HEALTH CHECK
//
// GET /health/upstream
//
// This checks NVIDIA's model catalog endpoint.
// It does NOT generate text and does NOT consume inference credits.
//
// A successful catalog response confirms connectivity and API-key
// acceptance, but does not guarantee that every model is available.
// ============================================================

app.get("/health/upstream", async (req, res) => {
  const problem = configurationError();

  if (problem) {
    return res.status(503).json({
      status: "misconfigured",
      error: problem,
      request_id: req.requestId
    });
  }

  if (
    upstreamHealthCache &&
    Date.now() - upstreamHealthCheckedAt < HEALTH_CACHE_TTL_MS
  ) {
    return res.status(
      upstreamHealthCache.status === "reachable" ? 200 : 502
    ).json({
      ...upstreamHealthCache,
      cached: true
    });
  }

  const started = Date.now();

  try {
    const response = await fetchNvidiaCatalog(15000);

    const reachable =
      response.status === 200 &&
      Array.isArray(response.data?.data);

    const result = {
      status: reachable ? "reachable" : "upstream_error",
      upstream_http_status: response.status,
      checked_at: new Date().toISOString(),
      elapsed_ms: Date.now() - started,
      inference_tested: false
    };

    if (!reachable) {
      result.message =
        "NVIDIA's catalog endpoint did not return a valid model list.";
    }

    upstreamHealthCache = result;
    upstreamHealthCheckedAt = Date.now();

    return res.status(reachable ? 200 : 502).json(result);

  } catch (err) {
    const result = {
      status: "unreachable",
      error_code: err.code || "UNKNOWN",
      elapsed_ms: Date.now() - started,
      inference_tested: false
    };

    console.error(
      "[UPSTREAM HEALTH FAILURE]",
      "id:", req.requestId,
      "code:", err.code || err.message
    );

    return res.status(502).json(result);
  }
});

// ============================================================
// FULL NVIDIA MODEL CATALOG
//
// Informational only.
// Catalog models are never automatically activated.
// ============================================================

app.get("/nvidia/models", async (req, res) => {
  if (!NIM_API_KEY) {
    return apiError(
      res,
      503,
      "Missing NVIDIA API key.",
      "configuration_error",
      req.requestId
    );
  }

  if (
    fullCatalog &&
    Date.now() - fullCatalogFetchedAt < CATALOG_TTL_MS
  ) {
    return res.json(fullCatalog);
  }

  try {
    const response = await fetchNvidiaCatalog(30000);

    if (
      response.status !== 200 ||
      !Array.isArray(response.data?.data)
    ) {
      console.error(
        "[CATALOG ERROR]",
        "id:", req.requestId,
        "HTTP:", response.status
      );

      return apiError(
        res,
        502,
        "Could not retrieve NVIDIA's model catalog.",
        "upstream_error",
        req.requestId
      );
    }

    fullCatalog = response.data;
    fullCatalogFetchedAt = Date.now();

    return res.json(fullCatalog);

  } catch (err) {
    console.error(
      "[CATALOG NETWORK FAILURE]",
      "id:", req.requestId,
      "code:", err.code || err.message
    );

    return apiError(
      res,
      502,
      "Could not retrieve NVIDIA's model catalog.",
      "upstream_error",
      req.requestId
    );
  }
});

// ============================================================
// OPENAI-COMPATIBLE MODEL LIST
// ONLY THE ACTIVE MODEL IS ADVERTISED.
// ============================================================

app.get("/v1/models", (req, res) => {
  if (!ACTIVE_PROFILE) {
    return apiError(
      res,
      503,
      configurationError(),
      "configuration_error",
      req.requestId
    );
  }

  return res.json({
    object: "list",
    data: [{
      id: ACTIVE_MODEL,
      object: "model",
      created: 0,
      owned_by: ACTIVE_MODEL.split("/")[0]
    }]
  });
});

// ============================================================
// NVIDIA UPSTREAM ERROR HANDLING
// ============================================================

function handleUpstreamHttpError(res, req, status, remoteMessage) {
  const detail = String(remoteMessage || "")
    .replace(/[\r\n]+/g, " ")
    .slice(0, 300);

  console.error(
    "[NVIDIA HTTP ERROR]",
    "id:", req.requestId,
    "model:", ACTIVE_MODEL,
    "HTTP:", status,
    "details:", detail || "(no details)"
  );

  if (status === 401 || status === 403) {
    return apiError(
      res,
      status,
      "NVIDIA rejected the API key or access to this model. Check Render credentials and NVIDIA permissions.",
      "authentication_error",
      req.requestId
    );
  }

  if (status === 402) {
    return apiError(
      res,
      402,
      "NVIDIA reports insufficient API credits.",
      "billing_error",
      req.requestId
    );
  }

  if (status === 429) {
    const retryAfter = req.upstreamRetryAfter;

    if (retryAfter) {
      res.set("Retry-After", retryAfter);
    }

    return apiError(
      res,
      429,
      "NVIDIA rate-limited the active model. No fallback attempted.",
      "rate_limit_error",
      req.requestId
    );
  }

  if (status === 400 || status === 422) {
    return apiError(
      res,
      502,
      "NVIDIA rejected the active model request: " +
        (detail || "Check model-specific request settings."),
      "upstream_request_error",
      req.requestId
    );
  }

  return apiError(
    res,
    502,
    "NVIDIA returned HTTP " + status +
      " for the active model. No fallback attempted.",
    "upstream_error",
    req.requestId
  );
}

// ============================================================
// NETWORK FAILURE HANDLING
// ============================================================

function handleNetworkError(res, req, err) {
  const code = err.code || "UNKNOWN";

  const timedOut =
    code === "ECONNABORTED" ||
    code === "ETIMEDOUT";

  console.error(
    "[NVIDIA NETWORK FAILURE]",
    "id:", req.requestId,
    "model:", ACTIVE_MODEL,
    "code:", code,
    "message:", String(err.message || "").slice(0, 250)
  );

  if (timedOut) {
    return apiError(
      res,
      504,
      "The active model timed out after " +
        REQUEST_TIMEOUT_MS +
        " ms. No reserve model was attempted.",
      "upstream_timeout",
      req.requestId
    );
  }

  return apiError(
    res,
    502,
    "Could not reach NVIDIA. No reserve model was attempted.",
    "upstream_error",
    req.requestId
  );
}

// ============================================================
// NON-STREAMING RESPONSE NORMALIZATION
//
// Keep reasoning_content separate from content.
// Preserve tool calls and other message fields.
// Never merge reasoning into character dialogue.
// ============================================================

function normalizeCompletion(result, req) {
  const choice = Array.isArray(result?.choices)
    ? result.choices[0]
    : null;

  const message = isObject(choice?.message)
    ? choice.message
    : null;

  if (!choice || !message) {
    return {
      error: "The active model returned an invalid completion structure."
    };
  }

  const content = nonEmptyText(message.content);

  const toolCalls = Array.isArray(message.tool_calls)
    ? message.tool_calls
    : [];

  const reasoning = message.reasoning_content;

  // A reasoning-only response is not a character reply.
  // Keep the reasoning field intact, but do not put it in content.

  if (!content.trim() && toolCalls.length === 0) {
    const finish = typeof choice.finish_reason === "string"
      ? choice.finish_reason
      : "unknown";

    console.error(
      "[EMPTY MODEL RESPONSE]",
      "id:", req.requestId,
      "model:", ACTIVE_MODEL,
      "finish_reason:", finish,
      "completion_tokens:",
      result?.usage?.completion_tokens ?? "unknown",
      "reasoning_present:",
      reasoning !== undefined && reasoning !== null
    );

    return {
      error:
        "The active model returned no visible reply (finish_reason: " +
        finish +
        "). Try increasing max_tokens in your chat app."
    };
  }

  // Preserve provider-returned message fields, while normalizing
  // content and keeping reasoning_content as its own field.

  const outgoingMessage = {
    ...message,
    role: "assistant",
    content: content || null
  };

  if (reasoning !== undefined) {
    outgoingMessage.reasoning_content = reasoning;
  }

  return {
    completion: {
      id: typeof result.id === "string"
        ? result.id
        : "chatcmpl-" + Date.now(),

      object: "chat.completion",

      created: Number.isInteger(result.created)
        ? result.created
        : Math.floor(Date.now() / 1000),

      model: ACTIVE_MODEL,

      choices: [{
        index: Number.isInteger(choice.index)
          ? choice.index
          : 0,

        message: outgoingMessage,

        finish_reason:
          choice.finish_reason ||
          (toolCalls.length ? "tool_calls" : "stop")
      }],

      usage: isObject(result.usage)
        ? result.usage
        : {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
          }
    }
  };
}

// ============================================================
// STREAMING RESPONSE HANDLER
//
// NVIDIA's SSE response is passed through without reconstructing
// chunks. This preserves:
//   - content deltas
//   - reasoning_content deltas
//   - tool_calls deltas
//   - finish_reason
//   - usage chunks, when provided
//   - [DONE]
//
// We deliberately do not merge or reinterpret any streamed fields.
// ============================================================

async function forwardStreamingResponse(upstream, req, res) {
  res.status(200);

  res.set({
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Request-ID": req.requestId
  });

  // Tell Express/Node to begin the SSE response immediately.

  if (typeof res.flushHeaders === "function") {
    res.flushHeaders();
  }

  return new Promise((resolve) => {
    let settled = false;

    const finish = () => {
      if (settled) return;
      settled = true;

      if (!res.writableEnded && !res.destroyed) {
        res.end();
      }

      resolve();
    };

    upstream.data.on("error", (err) => {
      console.error(
        "[STREAM UPSTREAM ERROR]",
        "id:", req.requestId,
        "code:", err.code || "STREAM_ERROR",
        "message:", String(err.message || "").slice(0, 200)
      );

      finish();
    });

    upstream.data.on("end", finish);

    res.on("finish", () => {
      if (settled) return;
      settled = true;
      resolve();
    });

    res.on("close", () => {
      if (settled) return;
      settled = true;
      resolve();
    });

    upstream.data.pipe(res);
  });
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post("/v1/chat/completions", async (req, res) => {
  const problem = configurationError();

  if (problem) {
    return apiError(
      res,
      503,
      problem,
      "configuration_error",
      req.requestId
    );
  }

  const validationError = validateChatRequest(req.body);

  if (validationError) {
    return apiError(
      res,
      400,
      validationError,
      "invalid_request_error",
      req.requestId
    );
  }

  if (!acquireRequestSlot()) {
    res.set("Retry-After", "5");

    return apiError(
      res,
      503,
      "The proxy is currently handling its maximum number of simultaneous requests. Please retry shortly.",
      "server_overloaded",
      req.requestId
    );
  }

  const body = req.body;
  const streaming = body.stream === true;

  const payload = buildPayload(body, streaming);

  const controller = new AbortController();

  const disconnect = createDisconnectTracker(
    req,
    res,
    controller
  );

  let outcome = "unknown";
  let responseStatus = 500;

  if (
    typeof body.model === "string" &&
    body.model !== ACTIVE_MODEL
  ) {
    console.log(
      "[MODEL OVERRIDE IGNORED]",
      "id:", req.requestId,
      "requested:", body.model,
      "using:", ACTIVE_MODEL
    );
  }

  logRequestStart(req, payload);

  try {
    const upstream = await axios.post(
      NIM_API_BASE + "/chat/completions",
      payload,
      {
        headers: upstreamHeaders(streaming),

        timeout: REQUEST_TIMEOUT_MS,

        validateStatus: () => true,

        responseType: streaming
          ? "stream"
          : "json",

        signal: controller.signal

        // Axios must decompress: if NVIDIA gzips the SSE body and we
        // pipe raw bytes without a Content-Encoding header, clients
        // receive garbage.
      }
    );

    if (disconnect.disconnected) {
      outcome = "client_disconnected";
      responseStatus = 499;
      return;
    }

    if (upstream.status !== 200) {
      let errorData = upstream.data;

      if (streaming && errorData && typeof errorData.on === "function") {
        errorData = await readStreamText(errorData);
      }

      const remoteMessage = extractRemoteMessage(errorData);

      req.upstreamRetryAfter =
        upstream.headers?.["retry-after"] || null;

      responseStatus = upstream.status >= 500
        ? 502
        : upstream.status;

      outcome = "upstream_http_" + upstream.status;

      return handleUpstreamHttpError(
        res,
        req,
        upstream.status,
        remoteMessage
      );
    }

    // ========================================================
    // STREAMING PATH
    // ========================================================

    if (streaming) {
      await forwardStreamingResponse(
        upstream,
        req,
        res
      );

      responseStatus = 200;
      outcome = disconnect.disconnected
        ? "client_disconnected_during_stream"
        : "stream_completed";

      return;
    }

    // ========================================================
    // NON-STREAMING PATH
    // ========================================================

    const normalized = normalizeCompletion(
      upstream.data,
      req
    );

    if (normalized.error) {
      responseStatus = 502;
      outcome = "empty_or_invalid_model_response";

      return apiError(
        res,
        502,
        normalized.error,
        "empty_model_response",
        req.requestId
      );
    }

    const completion = normalized.completion;

    const returnedMessage = completion.choices[0].message;

    console.log(
      "[MODEL SUCCESS]",
      "id:", req.requestId,
      "model:", ACTIVE_MODEL,
      "characters:", nonEmptyText(returnedMessage.content).length,
      "reasoning_present:",
      returnedMessage.reasoning_content !== undefined,
      "tool_calls:",
      Array.isArray(returnedMessage.tool_calls)
        ? returnedMessage.tool_calls.length
        : 0,
      "finish_reason:",
      completion.choices[0].finish_reason
    );

    responseStatus = 200;
    outcome = "completed";

    return res.json(completion);

  } catch (err) {
    if (
      disconnect.disconnected ||
      err.code === "ERR_CANCELED" ||
      err.name === "CanceledError"
    ) {
      outcome = "client_disconnected";
      responseStatus = 499;
      return;
    }

    responseStatus =
      err.code === "ECONNABORTED" ||
      err.code === "ETIMEDOUT"
        ? 504
        : 502;

    outcome = "network_failure";

    if (!res.headersSent) {
      return handleNetworkError(
        res,
        req,
        err
      );
    }

    // If streaming headers have already been sent, a JSON error
    // would corrupt the SSE response. Close the stream instead.

    console.error(
      "[STREAM TERMINATED]",
      "id:", req.requestId,
      "code:", err.code || err.message
    );

    if (!res.writableEnded && !res.destroyed) {
      res.end();
    }

  } finally {
    disconnect.cleanup();

    releaseRequestSlot();

    logRequestEnd(
      req,
      responseStatus,
      outcome
    );
  }
});

// ============================================================
// EXPRESS ERROR HANDLER
// ============================================================

app.use((err, req, res, _next) => {
  console.error(
    "[PROXY ERROR]",
    "id:", req.requestId || "unknown",
    "message:", String(err.message || err).slice(0, 300)
  );

  if (res.headersSent) {
    if (!res.writableEnded && !res.destroyed) {
      res.end();
    }
    return;
  }

  if (err.type === "entity.too.large") {
    return apiError(
      res,
      413,
      "Request exceeds the 25 MB limit.",
      "invalid_request_error",
      req.requestId
    );
  }

  if (err instanceof SyntaxError && "body" in err) {
    return apiError(
      res,
      400,
      "Malformed JSON request body.",
      "invalid_request_error",
      req.requestId
    );
  }

  return apiError(
    res,
    500,
    "Internal proxy error.",
    "server_error",
    req.requestId
  );
});

// ============================================================
// SERVER STARTUP
// ============================================================

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log("================================================");
  console.log("NVIDIA NIM CHAT PROXY ONLINE");
  console.log("================================================");

  console.log("PORT:", PORT);
  console.log("ACTIVE MODEL:", ACTIVE_MODEL);

  console.log(
    "RESERVE MODELS:",
    RESERVE_MODELS.length,
    "(disabled; no fallback)"
  );

  console.log("STREAMING: ENABLED");
  console.log(
    "PROXY AUTH:",
    PROXY_API_KEY ? "ENABLED" : "DISABLED"
  );

  console.log(
    "MAX CONCURRENT REQUESTS:",
    MAX_CONCURRENT_REQUESTS
  );

  console.log(
    "UPSTREAM TIMEOUT:",
    REQUEST_TIMEOUT_MS,
    "ms"
  );

  console.log("MAX OUTPUT TOKENS:", MAX_OUTPUT_TOKENS);
  console.log("REQUEST BODY LIMIT: 25 MB");

  const problem = configurationError();

  if (problem) {
    console.error("CONFIGURATION ERROR:", problem);
  } else {
    console.log("CONFIGURATION: VALID");
  }

  console.log("================================================");
});

// ============================================================
// GRACEFUL SHUTDOWN
// ============================================================

function shutdown(signal) {
  console.log("[SHUTDOWN]", signal, "received.");

  server.close(() => {
    console.log("[SHUTDOWN] HTTP server closed.");
    process.exit(0);
  });

  // Do not allow a stuck connection to prevent Render shutdown
  // indefinitely. Render can restart the instance normally.

  setTimeout(() => {
    console.error("[SHUTDOWN] Forced exit.");
    process.exit(1);
  }, 10000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
