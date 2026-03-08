#!/usr/bin/env node
"use strict";

const { init } = require("@heyputer/puter.js/src/init.cjs");

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

function extractText(response) {
  if (typeof response === "string") {
    return response.trim();
  }
  if (!response || typeof response !== "object") {
    return "";
  }

  if (typeof response.text === "string") {
    return response.text.trim();
  }

  if (response.message && typeof response.message === "object") {
    const content = response.message.content;
    if (typeof content === "string") {
      return content.trim();
    }
    if (Array.isArray(content)) {
      const chunks = [];
      for (const item of content) {
        if (typeof item === "string") {
          chunks.push(item);
        } else if (item && typeof item === "object" && typeof item.text === "string") {
          chunks.push(item.text);
        }
      }
      return chunks.join(" ").trim();
    }
  }

  return "";
}

function parsePayload(raw) {
  try {
    const data = JSON.parse(raw || "{}");
    return {
      systemPrompt: String(data.system_prompt || ""),
      userPrompt: String(data.user_prompt || ""),
      model: String(data.model || process.env.PUTER_MODEL || "google/gemini-2.5-flash-lite"),
      maxNewTokens: Number(data.max_new_tokens || 256),
    };
  } catch (err) {
    throw new Error(`Invalid JSON payload: ${err.message}`);
  }
}

async function main() {
  const authToken = (process.env.PUTER_AUTH_TOKEN || "").trim();
  if (!authToken) {
    throw new Error("Missing PUTER_AUTH_TOKEN.");
  }

  const payload = parsePayload(await readStdin());
  const puter = init(authToken);

  const prompt =
    `System Instructions:\n${payload.systemPrompt}\n\n` +
    `User Request:\n${payload.userPrompt}\n\n` +
    `Keep the output within about ${Math.max(32, Math.min(1024, payload.maxNewTokens))} tokens.`;

  const response = await puter.ai.chat(prompt, {
    model: payload.model,
    stream: false,
  });

  const text = extractText(response);
  if (!text) {
    throw new Error("Empty response from Puter model.");
  }

  process.stdout.write(`${JSON.stringify({ text })}\n`);
}

main().catch((err) => {
  process.stderr.write(`${err.message || String(err)}\n`);
  process.exit(1);
});
