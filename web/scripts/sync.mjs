// Copy the static export into the Python package, where `shilp serve` finds it.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "../out");
const target = resolve(here, "../../src/shilp/server/webui");

if (!existsSync(resolve(out, "index.html"))) {
  console.error(`no export at ${out}; did next build run with output: "export"?`);
  process.exit(1);
}
rmSync(target, { recursive: true, force: true });
cpSync(out, target, { recursive: true });
console.log(`→ copied ${out} to ${target}`);
