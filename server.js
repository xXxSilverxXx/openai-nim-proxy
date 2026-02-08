// server.js — OpenAI-compatible NVIDIA NIM proxy (Janitor AI optimized)

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 8080;

// ─────────────────────────────────────────────
// Middleware
// ─────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY || "dummy";

// Janitor AI prefers no hidden reasoning tokens
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping (OpenAI → NIM)
const MODEL_MAPPING = {
  "gpt-3.5-turbo": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
  "gpt-4": "qwen/qwen3-coder-480b-a35b-instruct",
  "gpt-4-turbo": "moonshotai/kimi-k2-instruct-0905",
  "gpt-4o": "deepseek-ai/deepseek-v3.1",
  "claude-3-opus": "openai/gpt-oss-120b",
  "claude-3-sonnet": "openai/gpt-oss-20b",
  "gemini-pro": "qwen/qwen3-next-80b-a3b-thinking"
};

// ─────────────────────────────────────────────
// Root + OpenAI compatibility probes
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.send("NVIDIA NIM OpenAI-compatible proxy is running");
});

// Janitor AI requires this to exist
app.get("/v1", (req, res) => {
  res.json({ object: "api", status: "ok" });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI → NVIDIA NIM Proxy",
    janitor_compatible: true
  });
});

// ─────────────────────────────────────────────
// OpenAI-compatible model list
// ─────────────────────────────────────────────
app.get("/v1/models", (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((id) => ({
    id,
    object: "model",
    created: Date.now(),
    owned_by: "nvidia-nim-proxy"
  }));

  res.json({ object: "list", data: models });
});

// ─────────────────────────────────────────────
// Chat completions proxy
// ─────────────────────────────────────────────
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      const m = (model || "").toLowerCase();
      if (m.includes("4") || m.includes("opus") || m.includes("405")) {
        nimModel = "meta/llama-3.1-405b-instruct";
      } else if (m.includes("claude") || m.includes("gemini") || m.includes("70")) {
        nimModel = "meta/llama-3.1-70b-instruct";
      } else {
        nimModel = "meta/llama-3.1-8b-instruct";
      }
    }

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 4096,
      stream: !!stream,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json"
        },
        responseType: stream ? "stream" : "json"
      }
    );

    // ───────── Streaming (Janitor-safe SSE) ─────────
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      response.data.on("data", (chunk) => {
        res.write(chunk);
      });

      response.data.on("end", () => res.end());
      response.data.on("error", () => res.end());
      return;
    }

    // ───────── Non-streaming response ─────────
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map((c, i) => ({
        index: i,
        message: {
          role: "assistant",
          content: c.message?.content || ""
        },
        finish_reason: c.finish_reason || "stop"
      })),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });
  } catch (error) {
    console.error("Proxy error:", error?.message || error);

    res.status(error?.response?.status || 500).json({
      error: {
        message: error?.message || "Internal server error",
        type: "invalid_request_error",
        code: error?.response?.status || 500
      }
    });
  }
});

// ─────────────────────────────────────────────
// Fallback (prevents Janitor false negatives)
// ─────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Endpoint not found",
      type: "invalid_request_error",
      code: 404
    }
  });
});

// ─────────────────────────────────────────────
// 🚀 Start server (Railway REQUIRED bind)
// ─────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Janitor AI compatible`);
});
