import react from "@vitejs/plugin-react";
import os from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const webRoot = fileURLToPath(new URL(".", import.meta.url));
const monorepoRoot = resolve(webRoot, "../..");
const apiTarget = process.env.SPATIALDINO_DEV_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "spatialdino-dev-template-vars",
      apply: "serve",
      transformIndexHtml(html) {
        const hostname = process.env.SERVER_HOSTNAME ?? os.hostname();
        return html.replaceAll("__SPATIALDINO_SERVER_HOSTNAME__", hostname);
      }
    }
  ],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    fs: {
      allow: [monorepoRoot]
    },
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true
      }
    }
  }
});
