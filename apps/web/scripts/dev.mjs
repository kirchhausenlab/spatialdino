import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const monorepoRoot = resolve(scriptDir, "../../..");
const serverDir = resolve(monorepoRoot, "apps/server");
const webDir = resolve(monorepoRoot, "apps/web");
const backendHost = process.env.SPATIALDINO_DEV_API_HOST ?? "127.0.0.1";
const backendPortRaw = process.env.SPATIALDINO_DEV_API_PORT ?? "8000";
const hasPinnedBackendPort = process.env.SPATIALDINO_DEV_API_PORT != null;
const explicitBackendTarget = process.env.SPATIALDINO_DEV_API_TARGET ?? null;
const uvCacheDir = process.env.SPATIALDINO_UV_CACHE_DIR ?? resolve(monorepoRoot, ".uv-cache");
let isShuttingDown = false;

function startProcess(command, args, cwd, envExtras = {}) {
  const child = spawn(command, args, {
    stdio: "inherit",
    cwd,
    env: {
      ...process.env,
      UV_CACHE_DIR: uvCacheDir,
      ...envExtras
    }
  });
  return child;
}

let backend;
let frontend;

async function isPortAvailable(host, port) {
  return await new Promise((resolvePromise) => {
    const server = createServer();
    server.once("error", () => resolvePromise(false));
    server.once("listening", () => {
      server.close(() => resolvePromise(true));
    });
    server.listen(port, host);
  });
}

function parsePort(text) {
  const parsed = Number.parseInt(String(text), 10);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) {
    throw new Error(`Invalid SPATIALDINO_DEV_API_PORT: ${text}`);
  }
  return parsed;
}

async function chooseBackendPort(host, preferredPort, hasPinnedPort) {
  if (hasPinnedPort) return preferredPort;
  const maxOffset = 100;
  for (let offset = 0; offset <= maxOffset; offset += 1) {
    const candidate = preferredPort + offset;
    if (candidate > 65535) break;
    // Probe for an available local port to avoid startup failures on common defaults.
    // A tiny race remains between probing and bind, but this removes most conflicts.
    // If a race still occurs, process monitoring below will fail fast.
    // eslint-disable-next-line no-await-in-loop
    if (await isPortAvailable(host, candidate)) {
      return candidate;
    }
  }
  throw new Error(`No available backend port found in range ${preferredPort}-${Math.min(preferredPort + maxOffset, 65535)}.`);
}

function shutdown(exitCode = 0) {
  if (isShuttingDown) return;
  isShuttingDown = true;
  process.exitCode = exitCode;
  if (backend && !backend.killed) backend.kill("SIGINT");
  if (frontend && !frontend.killed) frontend.kill("SIGINT");
}

function watchProcess(name, child) {
  child.on("error", (error) => {
    console.error(`[dev] ${name} failed to start:`, error);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (isShuttingDown) return;
    const exitCode = code ?? 1;
    const reason = signal ? `signal ${signal}` : `code ${exitCode}`;
    console.error(`[dev] ${name} exited (${reason}). Stopping dev environment.`);
    shutdown(exitCode === 0 ? 1 : exitCode);
  });
}

async function main() {
  const preferredBackendPort = parsePort(backendPortRaw);
  const selectedBackendPort = await chooseBackendPort(backendHost, preferredBackendPort, hasPinnedBackendPort);
  if (!hasPinnedBackendPort && selectedBackendPort !== preferredBackendPort) {
    console.warn(
        `[dev] Port ${preferredBackendPort} is in use; using ${selectedBackendPort}. ` +
        "Set SPATIALDINO_DEV_API_PORT to pin a specific backend port."
    );
  }

  const backendTarget = explicitBackendTarget ?? `http://${backendHost}:${selectedBackendPort}`;
  console.log(`[dev] API target: ${backendTarget}`);

  backend = startProcess(
    "uv",
    [
      "run",
      "--all-packages",
      "spatialdino-server",
      "--host",
      backendHost,
      "--port",
      String(selectedBackendPort),
      "--reload"
    ],
    monorepoRoot
  );
  frontend = startProcess("npm", ["run", "dev:ui"], webDir, { SPATIALDINO_DEV_API_TARGET: backendTarget });

  watchProcess("backend", backend);
  watchProcess("frontend", frontend);
}

main().catch((error) => {
  console.error("[dev] Failed to start development environment:", error);
  process.exit(1);
});

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
