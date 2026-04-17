// server.js — Production-stable OpenAI-compatible proxy for NVIDIA NIM

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

const ENABLE_THINKING_MODE = true;

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─────────────────────────────────────────────
// Model Mapping
// ─────────────────────────────────────────────
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
// Routes
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.send("Proxy is running");
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/v1", (req, res) => {
  res.json({ object: "api", status: "ok" });
});

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
// Chat Completions
// ─────────────────────────────────────────────
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    let nimModel = MODEL_MAPPING[model] || "deepseek-ai/deepseek-v3.1";

    const supportsThinking =
      nimModel.includes("deepseek") || nimModel.includes("qwen");

    const baseRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 300,
      top_p: 0.9,
      stream: false,
      ...(ENABLE_THINKING_MODE && supportsThinking
        ? { extra_body: { chat_template_kwargs: { thinking: true } } }
        : {})
    };

    // 🔥 STABLE FALLBACK ORDER (most reliable first)
    const fallbackModels = [
      "deepseek-ai/deepseek-v3.1",
      "moonshotai/kimi-k2-instruct-0905",
      "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    ];

    let response = null;
    let lastError = null;

    for (const modelToTry of fallbackModels) {
      try {
        console.log("Trying model:", modelToTry);

        const attempt = {
          ...baseRequest,
          model: modelToTry
        };

        const result = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          attempt,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              "Content-Type": "application/json"
            },
            timeout: 60000
          }
        );

        if (result.data && result.data.choices) {
          response = result;
          break;
        }
      } catch (err) {
        lastError = err;

        console.warn(
          "Model failed:",
          modelToTry,
          err?.response?.data || err.message
        );

        // 🔥 critical delay (lets degraded models recover)
        await sleep(1200);
      }
    }

    // ❌ If ALL models fail → return valid OpenAI response (NOT error)
    if (!response || !response.data || !response.data.choices) {
      console.error(
        "All models failed:",
        lastError?.response?.data || lastError
      );

      return res.json({
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content:
                "[The AI is temporarily overloaded. Please retry your message in a moment.]"
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
    }

    // ✅ Normal success response
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map((choice, index) => ({
        index,
        message: {
          role: "assistant",
          content: choice.message?.content || ""
        },
        finish_reason: choice.finish_reason
      })),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error("Proxy error:", error?.response?.data || error.message);

    res.status(error?.response?.status || 500).json({
      error: {
        message:
          error?.response?.data?.detail ||
          error?.response?.data?.error?.message ||
          error.message ||
          "Internal server error",
        type: "invalid_request_error",
        code: error?.response?.status || 500
      }
    });
  }
});

// ─────────────────────────────────────────────
// 404 fallback
// ─────────────────────────────────────────────
app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: "invalid_request_error",
      code: 404
    }
  });
});

// ─────────────────────────────────────────────
// Start Server
// ─────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Proxy running on port ${PORT}`);
});
