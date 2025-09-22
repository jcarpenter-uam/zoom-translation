import "dotenv/config.js";
import express from "express";
import path from "path";
import helmet from "helmet";
import { fileURLToPath } from "url";
import { createProxyMiddleware } from "http-proxy-middleware/dist/index.js";
import fs from "fs";

import { startRtmsHandler } from "./rtms/rtms-handler.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logDir = path.join(__dirname, "logs");
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir);
  console.log(`Log directory created at: ${logDir}`);
}

const transcriptionUrl = process.env.TRANSCRIPTION_URL;
const translationUrl = process.env.TRANSLATION_URL;
const transcriptionServer = process.env.TRANSCRIPTION_SERVER_URL;

if (!transcriptionUrl || !translationUrl || !transcriptionServer) {
  throw new Error(
    "Missing required environment variables: Ensure TRANSCRIPTION_URL, TRANSLATION_URL, and TRANSCRIPTION_SERVER_URL are set.",
  );
}

const app = express();
const port = 3000;
const rtmsPort = 8080;

app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "appssdk.zoom.us"],
        connectSrc: ["'self'", "wss:", "ws:"],
        styleSrc: ["'self'"],
      },
    },
  }),
);

app.use(
  "/zoomrtms",
  createProxyMiddleware({
    target: `http://localhost:${rtmsPort}`,
    changeOrigin: true,
    pathRewrite: {
      [`^/zoomrtms`]: "",
    },
  }),
);

app.get("/api/config", (req, res) => {
  res.json({
    transcription_url: transcriptionUrl,
    translation_url: translationUrl,
  });
});

app.use(express.static(path.join(__dirname, "frontend")));

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);

  startRtmsHandler({
    transcriptionServer: transcriptionServer,
  });

  console.log(`RTMS webhook handler is running ${rtmsPort}`);
});
