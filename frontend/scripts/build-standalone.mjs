import { cp, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const frontendDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nextCli = path.join(frontendDirectory, "node_modules", "next", "dist", "bin", "next");
const standaloneDirectory = path.join(frontendDirectory, ".next", "standalone");

const exitCode = await new Promise((resolve, reject) => {
  const buildProcess = spawn(process.execPath, [nextCli, "build"], {
    cwd: frontendDirectory,
    stdio: "inherit",
  });

  buildProcess.once("error", reject);
  buildProcess.once("close", resolve);
});

if (exitCode !== 0) {
  process.exit(exitCode ?? 1);
}

await mkdir(path.join(standaloneDirectory, ".next"), { recursive: true });
await cp(
  path.join(frontendDirectory, ".next", "static"),
  path.join(standaloneDirectory, ".next", "static"),
  { force: true, recursive: true },
);

try {
  await stat(path.join(frontendDirectory, "public"));
  await cp(path.join(frontendDirectory, "public"), path.join(standaloneDirectory, "public"), {
    force: true,
    recursive: true,
  });
} catch (error) {
  if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
    process.exit(0);
  }
  throw error;
}
