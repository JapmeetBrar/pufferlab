import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(webRoot, "..");
const temporaryRoot = await mkdtemp(path.join(tmpdir(), "pufferlab-browser-gate-"));
const dataDirectory = path.join(temporaryRoot, "data");
const guardMarker = path.join(temporaryRoot, "guard-tripped.txt");
const inheritedEnvironment = Object.fromEntries(
  Object.entries(process.env).filter(([name]) => !name.startsWith("VITE_")),
);

function allocatedPort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        server.close();
        reject(new Error("failed to allocate a loopback port"));
        return;
      }
      const port = address.port;
      server.close((error) => error === undefined ? resolve(port) : reject(error));
    });
  });
}

function start(command, args, { cwd, env }) {
  const child = spawn(command, args, {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  const append = (chunk) => {
    output = `${output}${chunk.toString()}`.slice(-200_000);
  };
  child.stdout.on("data", append);
  child.stderr.on("data", append);
  const exited = new Promise((resolve) => {
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
  return { child, exited, output: () => output };
}

async function run(command, args, options) {
  const running = start(command, args, options);
  const result = await running.exited;
  if (result.code !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed\n${running.output()}`);
  }
  return running.output();
}

async function waitForUrl(url, processHandle, label) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const exited = await Promise.race([
      processHandle.exited.then((result) => ({ exited: result })),
      new Promise((resolve) => setTimeout(() => resolve(null), 100)),
    ]);
    if (exited !== null) {
      throw new Error(`${label} exited before readiness\n${processHandle.output()}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // The owned child has not bound its allocated loopback port yet.
    }
  }
  throw new Error(`${label} did not become ready\n${processHandle.output()}`);
}

async function stop(processHandle) {
  if (processHandle.child.exitCode !== null || processHandle.child.signalCode !== null) return;
  processHandle.child.kill("SIGINT");
  const stopped = await Promise.race([
    processHandle.exited.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 5_000)),
  ]);
  if (!stopped) {
    processHandle.child.kill("SIGKILL");
    await processHandle.exited;
  }
}

let apiProcess;
let webProcess;
let failure;

try {
  const apiPort = await allocatedPort();
  let webPort = await allocatedPort();
  while (webPort === apiPort) webPort = await allocatedPort();
  if (apiPort === 8000 || apiPort === 5173 || webPort === 8000 || webPort === 5173) {
    throw new Error("allocated ports must not use the documented user-server defaults");
  }
  const apiOrigin = `http://127.0.0.1:${apiPort}`;
  const webOrigin = `http://127.0.0.1:${webPort}`;
  const cleanEnvironment = {
    ...inheritedEnvironment,
    PUFFERLAB_DATA_DIR: dataDirectory,
    PUFFERLAB_CORS_ORIGINS: webOrigin,
    PUFFERLAB_ENVIRONMENT: "browser-gate",
    PUFFERLAB_FIXTURE_DIR: path.join(repositoryRoot, "fixtures", "tiny-corpus"),
    PUFFERLAB_SEARCH_NAMESPACE: "",
    PUFFERLAB_E2E_GUARD_MARKER: guardMarker,
    TURBOPUFFER_API_KEY: "",
    TURBOPUFFER_REGION: "gcp-us-central1",
  };

  const seedOutput = await run("uv", ["run", "pufferlab", "demo", "seed"], {
    cwd: repositoryRoot,
    env: cleanEnvironment,
  });
  const runId = /run_id=([0-9a-f-]{36})/.exec(seedOutput)?.[1];
  if (runId === undefined) throw new Error("synthetic seed did not report its stable run ID");

  await run("pnpm", ["build"], {
    cwd: webRoot,
    env: { ...cleanEnvironment, VITE_API_BASE_URL: apiOrigin },
  });

  apiProcess = start(
    "uv",
    [
      "run",
      "uvicorn",
      "scripts.browser_gate_api:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(apiPort),
      "--workers",
      "1",
      "--no-access-log",
      "--log-level",
      "warning",
    ],
    { cwd: repositoryRoot, env: cleanEnvironment },
  );
  webProcess = start(
    "pnpm",
    ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(webPort), "--strictPort"],
    { cwd: webRoot, env: cleanEnvironment },
  );

  await Promise.all([
    waitForUrl(`${apiOrigin}/api/v1/health`, apiProcess, "API"),
    waitForUrl(webOrigin, webProcess, "built frontend"),
  ]);

  await run("pnpm", ["exec", "playwright", "test", "--config", "playwright.config.ts"], {
    cwd: webRoot,
    env: {
      ...cleanEnvironment,
      PUFFERLAB_E2E_BASE_URL: webOrigin,
      PUFFERLAB_E2E_RUN_ID: runId,
    },
  });

  if (apiProcess.child.exitCode !== null || apiProcess.child.signalCode !== null) {
    throw new Error(`API exited during the browser journey\n${apiProcess.output()}`);
  }
  if (webProcess.child.exitCode !== null || webProcess.child.signalCode !== null) {
    throw new Error(`built frontend exited during the browser journey\n${webProcess.output()}`);
  }

  try {
    await stat(guardMarker);
    throw new Error(`provider-free API guard tripped: ${await readFile(guardMarker, "utf-8")}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
} catch (error) {
  failure = error;
} finally {
  if (webProcess !== undefined) await stop(webProcess);
  if (apiProcess !== undefined) await stop(apiProcess);
  await rm(temporaryRoot, { recursive: true, force: true });
}

if (failure !== undefined) {
  process.stderr.write(`${failure instanceof Error ? failure.stack ?? failure.message : String(failure)}\n`);
  process.exitCode = 1;
}
