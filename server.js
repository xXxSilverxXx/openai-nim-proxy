// server.js — OpenAI-compatible proxy for NVIDIA NIM API (Railway-ready)

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

// ─────────────────────────────────────────────
// Middleware (🚨 BODY LIMIT FIX FOR JANITOR AI)
// ─────────────────────────────────────────────
app.use(cors());

// Janitor AI sends LARGE payloads (chat history, character data, etc)
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ limit: "25mb", extended: true }));

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggle reasoning display (<think> tags)
const SHOW_REASONING = false;

// Enable thinking mode for supported models
const ENABLE_THINKING_MODE = false;

// Model mapping
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
// Root + Health
// ─────────────────────────────────────────────
app.get("/", (req, res) => {
  res.send("NVIDIA NIM OpenAI-compatible proxy is running");
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI → NVIDIA NIM Proxy",
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// OpenAI base compatibility endpoint (Janitor AI expects this)
app.get("/v1", (req, res) => {
  res.json({
    object: "api",
    status: "ok"
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

    // Smart fallback if model not mapped
    if (!nimModel) {
      const modelLower = model?.toLowerCase?.() || "";

      if (
        modelLower.includes("gpt-4") ||
        modelLower.includes("claude-opus") ||
        modelLower.includes("405b")
      ) {
        nimModel = "meta/llama-3.1-405b-instruct";
      } else if (
        modelLower.includes("claude") ||
        modelLower.includes("gemini") ||
        modelLower.includes("70b")
      ) {
        nimModel = "meta/llama-3.1-70b-instruct";
      } else {
        nimModel = "meta/llama-3.1-8b-instruct";
      }
    }

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
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

        // 🚨 REQUIRED FOR LARGE REQUESTS / RESPONSES
        maxBodyLength: Infinity,
        maxContentLength: Infinity,

        responseType: stream ? "stream" : "json"
      }
    );

    // ───────── Streaming (SSE) ─────────
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      let buffer = "";
      let reasoningOpen = false;

      response.data.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          if (line.includes("[DONE]")) {
            res.write(line + "\n\n");
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING && reasoning) {
                delta.content = reasoningOpen
                  ? reasoning
                  : `<think>\n${reasoning}`;
                reasoningOpen = true;
              }

              if (content) {
                delta.content = reasoningOpen
                  ? `</think>\n\n${content}`
                  : content;
                reasoningOpen = false;
              }

              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + "\n\n");
          }
        }
      });

      response.data.on("end", () => res.end());
      response.data.on("error", () => res.end());
      return;
    }

    // ───────── Non-streaming ─────────
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map((choice, index) => {
        let content = choice.message?.content || "";

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content =
            `<think>\n${choice.message.reasoning_content}\n</think>\n\n` +
            content;
        }

        return {
          index,
          message: { role: "assistant", content },
          finish_reason: choice.finish_reason
        };
      }),
      usage: response.data.usage ?? {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
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
// Fallback
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
// 🚀 Start server (Railway REQUIRED bind)
// ─────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: /health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? "ON" : "OFF"}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? "ON" : "OFF"}`);
});
