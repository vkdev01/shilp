// Record a scripted tour of the UI against a running server, then encode a GIF
// and an MP4 with ffmpeg. Drives the locally installed Chrome; nothing is downloaded.
//
//   node scripts/record-demo.mjs --url=http://127.0.0.1:8000 --out=assets --width=1200
//
// Frames come from Chrome's screencast with their real timestamps, so the output
// plays at the speed things actually happened, streaming included.

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, ...v] = a.replace(/^--/, "").split("=");
    return [k, v.join("=")];
  }),
);
const BASE = args.url || "http://127.0.0.1:8000";
const OUT = resolve(here, "../..", args.out || "assets");
const NAME = args.name || "shilp-ui";
const GIF_WIDTH = Number(args.width || 1200);
const W = 1440;
const H = 900;
// Long idle stretches are shortened so the GIF stays watchable.
const MAX_FRAME_S = 1.6;

const CURSOR = `(() => {
  const install = () => {
    if (document.getElementById("__demo_cursor")) return;
    const c = document.createElement("div");
    c.id = "__demo_cursor";
    c.style.cssText = "position:fixed;left:-40px;top:-40px;width:24px;height:24px;margin:-2px 0 0 -4px;z-index:2147483647;pointer-events:none;transition:transform .1s";
    c.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M5 2.5l14 10.2-6.4 1.2-3.6 6.6z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    const s = document.createElement("style");
    s.textContent = "@keyframes __ripple{from{transform:scale(.3);opacity:.9}to{transform:scale(1.7);opacity:0}}";
    document.documentElement.append(s, c);
    addEventListener("mousemove", (e) => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; }, true);
    addEventListener("mousedown", (e) => {
      c.style.transform = "scale(.85)";
      const r = document.createElement("div");
      r.style.cssText = "position:fixed;left:" + (e.clientX - 16) + "px;top:" + (e.clientY - 16) + "px;width:32px;height:32px;border-radius:50%;border:2px solid rgba(234,88,12,.9);z-index:2147483646;pointer-events:none;animation:__ripple .5s ease-out forwards";
      document.documentElement.append(r);
      setTimeout(() => r.remove(), 550);
    }, true);
    addEventListener("mouseup", () => { c.style.transform = ""; }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", install);
  else install();
})();`;

async function pickModels() {
  const res = await fetch(`${BASE}/api/tags`);
  if (!res.ok) throw new Error(`${BASE}/api/tags → HTTP ${res.status}; is the server running?`);
  const names = (await res.json()).models.map((m) => m.name);
  if (!names.length) throw new Error("the server has no models; pull one first");
  const primary = args.model || names.find((n) => n.includes("gemma")) || names[0];
  const secondary = args.model2 || primary;
  return { primary, secondary };
}

async function main() {
  const { primary, secondary } = await pickModels();
  console.log(`→ recording ${BASE} with ${primary}${secondary !== primary ? ` and ${secondary}` : ""}`);

  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: W, height: H },
    deviceScaleFactor: 1,
    colorScheme: "light",
  });
  await context.addInitScript(CURSOR);
  const page = await context.newPage();

  const wait = (ms) => page.waitForTimeout(ms);
  const moveTo = (x, y, steps = 22) => page.mouse.move(x, y, { steps });
  async function point(locator, steps) {
    await locator.scrollIntoViewIfNeeded();
    const b = await locator.boundingBox();
    if (!b) throw new Error(`not visible: ${locator}`);
    await moveTo(b.x + b.width / 2, b.y + b.height / 2, steps);
    return b;
  }
  async function click(locator, pause = 220) {
    await point(locator);
    await wait(pause);
    await page.mouse.down();
    await page.mouse.up();
  }
  async function scroll(dy, steps = 14) {
    for (let i = 0; i < steps; i += 1) {
      await page.mouse.wheel(0, dy / steps);
      await wait(28);
    }
  }
  async function sweep(locator, from = 0.08, to = 0.95, steps = 36) {
    const b = await point(locator, 18);
    for (let i = 0; i <= steps; i += 1) {
      const f = from + ((to - from) * i) / steps;
      await page.mouse.move(b.x + b.width * f, b.y + b.height * 0.45);
      await wait(45);
    }
  }
  const streamed = () =>
    page.waitForSelector('[aria-label="Stop generating"]', { state: "detached", timeout: 120_000 });

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Decode speed (p50)");

  const cdp = await context.newCDPSession(page);
  const frames = [];
  cdp.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
    frames.push({ data, ts: metadata.timestamp });
    cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 90, maxWidth: W, maxHeight: H });
  await moveTo(W * 0.55, H * 0.35, 1);

  // --- Overview -------------------------------------------------------
  await wait(1800);
  await sweep(page.locator('section:has(h2:text-is("Latency")) svg').first());
  await sweep(page.locator('section:has(h2:text-is("Traffic")) svg').first(), 0.3, 0.95, 24);
  await click(page.getByRole("radio", { name: "15m", exact: true }));
  await wait(1600);
  await scroll(820);
  await wait(1400);
  await scroll(760);
  await wait(1400);
  await scroll(-1580, 18);
  await wait(500);

  // --- Playground: Ollama surface, streaming ----------------------------
  await click(page.locator("aside").getByRole("link", { name: "Playground" }));
  await page.waitForSelector('select[aria-label="Model"]:not([disabled])');
  await point(page.getByRole("combobox", { name: "Model" }));
  await page.getByRole("combobox", { name: "Model" }).selectOption(primary);
  await wait(500);
  await click(page.getByRole("textbox", { name: "Message" }));
  await page.keyboard.type("Write a haiku about running a language model on a laptop.", { delay: 28 });
  await wait(300);
  await page.keyboard.press("Enter");
  await streamed();
  await wait(1600);
  await click(page.getByRole("radio", { name: "Request" }));
  await wait(1800);
  await click(page.getByRole("radio", { name: "Response" }));
  await wait(1800);

  // --- Playground: OpenAI surface, markdown + code ----------------------
  await click(page.getByRole("radio", { name: "OpenAI" }));
  if (secondary !== primary) {
    await point(page.getByRole("combobox", { name: "Model" }));
    await page.getByRole("combobox", { name: "Model" }).selectOption(secondary);
  }
  await wait(400);
  await click(page.getByRole("textbox", { name: "Message" }));
  await page.keyboard.type("Give me a Python one-liner to reverse a string, in a code block.", { delay: 26 });
  await page.keyboard.press("Enter");
  await streamed();
  await wait(800);
  await click(page.getByRole("radio", { name: "Metrics" }));
  await wait(2200);

  // --- Models -----------------------------------------------------------
  await click(page.locator("aside").getByRole("link", { name: "Models" }));
  await page.waitForSelector("text=Requests 24h");
  await wait(1200);
  await click(page.getByRole("button", { name: /Details/ }).first());
  await page.waitForSelector("text=Capabilities");
  await scroll(420);
  await wait(1400);
  await click(page.getByRole("radio", { name: "Shilpfile" }));
  await wait(1600);

  // --- Requests ---------------------------------------------------------
  await click(page.locator("aside").getByRole("link", { name: "Requests" }));
  await page.waitForSelector("text=Filtered summary");
  await wait(1400);
  await click(page.getByRole("radio", { name: "tok/s" }));
  await wait(1200);
  await click(page.getByRole("radio", { name: "Total" }));
  await wait(1200);
  await click(page.getByRole("radio", { name: "Failed" }));
  await wait(1800);
  await click(page.getByRole("radio", { name: "All", exact: true }));
  await click(page.getByPlaceholder(/Filter by model/));
  await page.keyboard.type(primary.split(/[-:_]/)[0], { delay: 45 });
  await wait(1500);

  // --- Dark mode, back to the overview -----------------------------------
  await click(page.locator('button[aria-label="Toggle colour theme"]:visible'));
  await wait(900);
  await click(page.locator("aside").getByRole("link", { name: "Overview" }));
  await page.waitForSelector("text=Decode speed (p50)");
  await wait(1200);
  await sweep(page.locator('section:has(h2:text-is("Decode throughput")) svg').first(), 0.2, 0.95, 26);
  await wait(1500);

  await cdp.send("Page.stopScreencast");
  await browser.close();
  encode(frames);
}

function encode(frames) {
  if (frames.length < 2) throw new Error("no frames captured");
  const dir = mkdtempSync(join(tmpdir(), "shilp-demo-"));
  const name = (i) => `f${String(i).padStart(5, "0")}.jpg`;
  let list = "";
  frames.forEach((f, i) => {
    writeFileSync(join(dir, name(i)), Buffer.from(f.data, "base64"));
    const next = frames[i + 1]?.ts ?? f.ts + 1.5;
    list += `file '${name(i)}'\nduration ${Math.min(MAX_FRAME_S, Math.max(0.01, next - f.ts)).toFixed(3)}\n`;
  });
  list += `file '${name(frames.length - 1)}'\n`; // the concat demuxer ignores the last duration otherwise
  writeFileSync(join(dir, "frames.txt"), list);

  mkdirSync(OUT, { recursive: true });
  const gif = join(OUT, `${NAME}.gif`);
  const mp4 = join(OUT, `${NAME}.mp4`);
  const input = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", join(dir, "frames.txt")];
  execFileSync("ffmpeg", [
    ...input,
    "-vf",
    `fps=12,scale=${GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle`,
    "-loop",
    "0",
    gif,
  ]);
  execFileSync("ffmpeg", [
    ...input,
    "-vf",
    "fps=30,format=yuv420p",
    "-c:v",
    "libx264",
    "-crf",
    "20",
    "-preset",
    "slow",
    "-movflags",
    "+faststart",
    mp4,
  ]);
  rmSync(dir, { recursive: true, force: true });

  const seconds = frames.reduce((a, f, i) => a + Math.min(MAX_FRAME_S, (frames[i + 1]?.ts ?? f.ts) - f.ts), 0);
  const mb = (p) => (statSync(p).size / 1e6).toFixed(1);
  console.log(`→ ${frames.length} frames, ${seconds.toFixed(0)} s`);
  console.log(`→ ${gif} (${mb(gif)} MB)`);
  console.log(`→ ${mp4} (${mb(mp4)} MB)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
