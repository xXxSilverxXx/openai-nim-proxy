// server.js — Stable OpenAI-compatible NVIDIA NIM proxy

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// Middleware
// ─────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ limit: "25mb", extended: true }));

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─────────────────────────────────────────────
// Model Mapping (OpenAI → NIM safe equivalents)
// ─────────────────────────────────────────────
const MODEL_MAPPING = {
  "gpt-3.5-turbo": "meta/llama-3.1-8b-instruct",
  "gpt-4": "qwen/qwen3-coder-32b-instruct",
  "gpt-4-turbo": "moonshotai/kimi-k2-instruct",
  "gpt-4o": "deepseek-ai/deepseek-v3.1",
  "claude-3-opus": "anthropic/claude-3.5-sonnet",
  "claude-3-sonnet": "anthropic/claude-3.5-haiku",
  "gemini-pro": "qwen/qwen3-next-80b-a3b-thinking"
};

// ─────────────────────────────────────────────
// Health / basic routes
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.send("NIM Proxy Running");
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/v1", (req, res) => {
  res.json({ object: "api", status: "ok" });
});

// ─────────────────────────────────────────────
// REAL Models Route (FIXED)
// ─────────────────────────────────────────────
app.get("/v1/models", async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`
      },
      timeout: 15000
    });

    const models = (r.data?.data || []).map((m) => ({
      id: m.id,
      object: "model",
      created: m.created || Date.now(),
      owned_by: "nvidia-nim"
    }));

    res.json({
      object: "list",
      data: models
    });
  } catch (err) {
    console.error("Failed to fetch models:", err.message);

    // safe fallback so frontend never breaks
    res.json({
      object: "list",
      data: []
    });
  }
});

// ─────────────────────────────────────────────
// Chat Completions
// ─────────────────────────────────────────────
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    const mappedModel =
      MODEL_MAPPING[model] || "deepseek-ai/deepseek-v3.1";

    const fallbackModels = [
      mappedModel,
      "deepseek-ai/deepseek-v3.1",
      "qwen/qwen3-coder-32b-instruct",
      "meta/llama-3.1-8b-instruct"
    ];

    let lastError = null;

    for (const modelToTry of fallbackModels) {
      try {
        console.log("Trying model:", modelToTry);

        const result = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          {
            model: modelToTry,
            messages,
            temperature: temperature ?? 0.7,
            max_tokens: max_tokens ?? 300,
            top_p: 0.9,
            stream: false
          },
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              "Content-Type": "application/json"
            },
            timeout: 60000
          }
        );

        if (result.data?.choices) {
          return res.json({
            id: `chatcmpl-${Date.now()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: result.data.choices.map((c, i) => ({
              index: i,
              message: {
                role: "assistant",
                content: c.message?.content || ""
              },
              finish_reason: c.finish_reason
            })),
            usage: result.data.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
          });
        }
      } catch (err) {
        lastError = err;
        console.warn(
          "Model failed:",
          modelToTry,
          err?.response?.data || err.message
        );

        await sleep(1000);
      }
    }

    // final fallback response (never fails API contract)
    return res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "The model service is temporarily unavailable. Please try again."
          },
          finish_reason: "stop"
        }
      ],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });
  } catch (error) {
    console.error("Fatal proxy error:", error.message);

    return res.status(500).json({
      error: {
        message: error.message || "Internal server error",
        type: "server_error",
        code: 500
      }
    });
  }
});

// ─────────────────────────────────────────────
// 404 handler
// ─────────────────────────────────────────────
app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Route ${req.path} not found`,
      type: "invalid_request_error",
      code: 404
    }
  });
});

// ─────────────────────────────────────────────
// Start server
// ─────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`NIM Proxy running on port ${PORT}`);
});
