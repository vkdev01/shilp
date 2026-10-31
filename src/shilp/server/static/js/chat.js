// Chat client for both API surfaces.
//
// It deliberately uses the public endpoints and nothing else, so what you see
// here is what any Ollama or OpenAI client would get. The "Last request" panel
// shows the exact call as curl, which makes the page a manual API test bench.
(function () {
  "use strict";

  const SETTINGS_KEY = "shilp-chat-settings";
  const RAW_LIMIT = 20000;

  const $ = (selector) => document.querySelector(selector);
  const els = {
    settings: $("#settings"),
    model: $("#model"),
    modelHint: $("#model-hint"),
    refresh: $("#refresh-models"),
    system: $("#system"),
    temperature: $("#temperature"),
    temperatureValue: $("#temperature-value"),
    maxTokens: $("#max-tokens"),
    stream: $("#stream"),
    newChat: $("#new-chat"),
    messages: $("#messages"),
    emptyState: $("#empty-state"),
    composer: $("#composer"),
    prompt: $("#prompt"),
    send: $("#send"),
    stop: $("#stop"),
    template: $("#message-template"),
    inspector: $("#inspector"),
    reqMeta: $("#req-meta"),
    reqCurl: $("#req-curl"),
    resRaw: $("#res-raw"),
  };

  const state = {
    history: [],
    controller: null,
    busy: false,
    // Bumped by "New chat" so a reply that finishes late cannot write into it.
    conversation: 0,
  };

  // --- settings ---------------------------------------------------------
  function surface() {
    return els.settings.querySelector('input[name="surface"]:checked').value;
  }

  function readSettings() {
    const max = parseInt(els.maxTokens.value, 10);
    return {
      surface: surface(),
      model: els.model.value,
      system: els.system.value.trim(),
      temperature: parseFloat(els.temperature.value),
      maxTokens: Number.isFinite(max) && max > 0 ? max : null,
      stream: els.stream.checked,
    };
  }

  function saveSettings() {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(readSettings()));
    } catch (_) {
      /* storage unavailable: settings just won't persist */
    }
  }

  function restoreSettings() {
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    } catch (_) {
      saved = {};
    }
    if (saved.surface) {
      const radio = els.settings.querySelector(`input[name="surface"][value="${saved.surface}"]`);
      if (radio) radio.checked = true;
    }
    if (typeof saved.system === "string") els.system.value = saved.system;
    if (typeof saved.temperature === "number") els.temperature.value = String(saved.temperature);
    if (saved.maxTokens) els.maxTokens.value = String(saved.maxTokens);
    if (typeof saved.stream === "boolean") els.stream.checked = saved.stream;
    els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2);
    return saved;
  }

  // --- models -----------------------------------------------------------
  function humanSize(bytes) {
    if (!Number.isFinite(bytes)) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let n = bytes;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
  }

  async function loadModels(preferred) {
    let models;
    try {
      const res = await fetch("/api/tags");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      models = (await res.json()).models || [];
    } catch (err) {
      els.modelHint.hidden = false;
      els.modelHint.textContent = `Could not load models: ${err.message}`;
      return;
    }

    const keep = preferred || els.model.value;
    els.model.replaceChildren();
    if (!models.length) {
      els.model.append(new Option("No models stored", ""));
      els.model.disabled = true;
      els.modelHint.hidden = false;
      els.modelHint.innerHTML = "Pull one first: <code>make pull</code>";
    } else {
      for (const m of models) {
        els.model.append(new Option(`${m.name} · ${humanSize(m.size)}`, m.name));
      }
      els.model.disabled = false;
      els.modelHint.hidden = true;
      if (keep && models.some((m) => m.name === keep)) els.model.value = keep;
    }
    updateComposer();
  }

  // --- rendering --------------------------------------------------------
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }

  // Input is escaped first, so only these tags can ever reach the DOM.
  function renderInline(escaped) {
    return escaped
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  }

  function renderMarkdown(text) {
    return text
      .split("```")
      .map((part, i) => {
        if (i % 2 === 1) {
          const newline = part.indexOf("\n");
          const code = newline >= 0 ? part.slice(newline + 1) : part;
          return `<pre class="code"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
        }
        return part
          .split(/\n{2,}/)
          .filter((p) => p.trim())
          .map((p) => `<p>${renderInline(escapeHtml(p)).replace(/\n/g, "<br>")}</p>`)
          .join("");
      })
      .join("");
  }

  function nearBottom() {
    const m = els.messages;
    return m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  }

  function scrollToBottom(force) {
    if (force || nearBottom()) els.messages.scrollTop = els.messages.scrollHeight;
  }

  function addMessage(role, text, label) {
    els.emptyState.hidden = true;
    const node = els.template.content.firstElementChild.cloneNode(true);
    node.classList.add(`message-${role}`);
    node.querySelector(".avatar").textContent = role === "user" ? "You" : "AI";
    node.querySelector(".role").textContent = label || (role === "user" ? "You" : "Assistant");
    const content = node.querySelector(".content");
    if (role === "user") {
      content.classList.add("plain-text");
      content.textContent = text;
    } else {
      content.innerHTML = renderMarkdown(text);
    }
    els.messages.append(node);
    scrollToBottom(true);
    return node;
  }

  function enableCopy(node, getText) {
    const button = node.querySelector(".copy-message");
    button.hidden = false;
    button.addEventListener("click", () => copyText(getText(), button));
  }

  async function copyText(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      flash(button, "Copied");
    } catch (_) {
      flash(button, "Copy failed");
    }
  }

  function flash(button, label) {
    const original = button.dataset.label || button.textContent;
    button.dataset.label = original;
    button.textContent = label;
    setTimeout(() => {
      button.textContent = original;
    }, 1200);
  }

  // --- request building -------------------------------------------------
  function buildRequest(settings) {
    const messages = [];
    if (settings.system) messages.push({ role: "system", content: settings.system });
    messages.push(...state.history);

    if (settings.surface === "ollama") {
      const options = { temperature: settings.temperature };
      if (settings.maxTokens) options.num_predict = settings.maxTokens;
      return {
        url: "/api/chat",
        body: { model: settings.model, messages, stream: settings.stream, options },
      };
    }
    const body = {
      model: settings.model,
      messages,
      stream: settings.stream,
      temperature: settings.temperature,
    };
    if (settings.maxTokens) body.max_tokens = settings.maxTokens;
    return { url: "/v1/chat/completions", body };
  }

  function toCurl(url, body) {
    const json = JSON.stringify(body, null, 2).replace(/'/g, "'\\''");
    const flag = body.stream ? "-N " : "";
    return `curl ${flag}${location.origin}${url} \\\n  -H 'content-type: application/json' \\\n  -d '${json}'`;
  }

  // --- response parsing -------------------------------------------------
  async function* readLines(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, index).replace(/\r$/, "");
        buffer = buffer.slice(index + 1);
        yield line;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  }

  // Normalise one stream event from either surface into {delta, done, stats}.
  function parseEvent(kind, line) {
    if (kind === "ollama") {
      if (!line.trim()) return null;
      const obj = JSON.parse(line);
      if (obj.error) throw new Error(obj.error);
      return {
        delta: (obj.message && obj.message.content) || "",
        done: Boolean(obj.done),
        stats: obj.done
          ? {
              completion: obj.eval_count,
              prompt: obj.prompt_eval_count,
              evalNs: obj.eval_duration,
              reason: obj.done_reason,
            }
          : null,
      };
    }
    if (!line.startsWith("data:")) return null;
    const payload = line.slice(5).trim();
    if (payload === "[DONE]") return { delta: "", done: true, stats: null };
    const obj = JSON.parse(payload);
    if (obj.error) throw new Error(typeof obj.error === "string" ? obj.error : obj.error.message);
    const choice = (obj.choices && obj.choices[0]) || {};
    return {
      delta: (choice.delta && choice.delta.content) || "",
      done: false,
      stats: obj.usage
        ? {
            completion: obj.usage.completion_tokens,
            prompt: obj.usage.prompt_tokens,
            reason: choice.finish_reason,
          }
        : null,
    };
  }

  function parseFull(kind, obj) {
    if (kind === "ollama") {
      return {
        text: (obj.message && obj.message.content) || "",
        stats: {
          completion: obj.eval_count,
          prompt: obj.prompt_eval_count,
          evalNs: obj.eval_duration,
          reason: obj.done_reason,
        },
      };
    }
    const choice = (obj.choices && obj.choices[0]) || {};
    return {
      text: (choice.message && choice.message.content) || "",
      stats: {
        completion: obj.usage && obj.usage.completion_tokens,
        prompt: obj.usage && obj.usage.prompt_tokens,
        reason: choice.finish_reason,
      },
    };
  }

  async function errorMessage(response) {
    const text = await response.text();
    try {
      const obj = JSON.parse(text);
      if (obj.error) return typeof obj.error === "string" ? obj.error : obj.error.message;
      if (obj.detail) return typeof obj.detail === "string" ? obj.detail : JSON.stringify(obj.detail);
    } catch (_) {
      /* not JSON */
    }
    return text || `HTTP ${response.status}`;
  }

  function formatMeta(settings, timing, stats, stopped) {
    const parts = [settings.surface === "ollama" ? "Ollama" : "OpenAI"];
    const tokens = stats && stats.completion;
    if (tokens) parts.push(`${tokens} tokens`);

    let tps = null;
    if (tokens && timing.first && timing.end > timing.first) {
      tps = tokens / ((timing.end - timing.first) / 1000);
    } else if (tokens && stats.evalNs) {
      tps = tokens / (stats.evalNs / 1e9);
    }
    if (tps && Number.isFinite(tps)) parts.push(`${tps.toFixed(1)} tok/s`);
    if (timing.first) parts.push(`TTFT ${Math.round(timing.first - timing.start)} ms`);
    parts.push(`${((timing.end - timing.start) / 1000).toFixed(2)} s`);
    if (stopped) parts.push("stopped");
    else if (stats && stats.reason && stats.reason !== "stop") parts.push(stats.reason);
    return parts.join(" · ");
  }

  // --- send -------------------------------------------------------------
  function setBusy(busy) {
    state.busy = busy;
    els.send.hidden = busy;
    els.stop.hidden = !busy;
    els.prompt.disabled = busy;
    updateComposer();
  }

  function updateComposer() {
    els.send.disabled = state.busy || !els.model.value || !els.prompt.value.trim();
  }

  async function send(text) {
    const settings = readSettings();
    if (!settings.model || state.busy) return;
    const conversation = state.conversation;

    state.history.push({ role: "user", content: text });
    addMessage("user", text);
    const reply = addMessage("assistant", "", settings.model);
    reply.classList.add("is-pending");
    const content = reply.querySelector(".content");
    const meta = reply.querySelector(".meta");

    const { url, body } = buildRequest(settings);
    els.reqCurl.textContent = toCurl(url, body);
    els.reqMeta.textContent = `POST ${url} · sending…`;
    els.resRaw.textContent = "";

    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);

    const timing = { start: performance.now(), first: 0, end: 0 };
    let answer = "";
    let stats = null;
    let raw = "";
    let stopped = false;
    let frame = 0;

    const paint = () => {
      frame = 0;
      content.innerHTML = renderMarkdown(answer);
      scrollToBottom(false);
    };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      els.reqMeta.textContent = `POST ${url} · ${response.status} ${response.statusText}`;
      if (!response.ok) throw new Error(await errorMessage(response));

      if (settings.stream) {
        for await (const line of readLines(response)) {
          if (raw.length < RAW_LIMIT) raw += line + "\n";
          const event = parseEvent(settings.surface, line);
          if (!event) continue;
          if (event.delta) {
            if (!timing.first) {
              timing.first = performance.now();
              reply.classList.remove("is-pending");
            }
            answer += event.delta;
            if (!frame) frame = requestAnimationFrame(paint);
          }
          if (event.stats) stats = event.stats;
          if (event.done) break;
        }
      } else {
        const obj = await response.json();
        raw = JSON.stringify(obj, null, 2);
        const parsed = parseFull(settings.surface, obj);
        answer = parsed.text;
        stats = parsed.stats;
      }
    } catch (err) {
      if (frame) cancelAnimationFrame(frame);
      if (conversation !== state.conversation) {
        finish();
        return;
      }
      if (err.name === "AbortError") {
        stopped = true;
      } else {
        timing.end = performance.now();
        reply.classList.remove("is-pending");
        reply.classList.add("is-error");
        content.textContent = err.message;
        meta.hidden = false;
        meta.textContent = `Request failed after ${((timing.end - timing.start) / 1000).toFixed(2)} s`;
        els.resRaw.textContent = raw || err.message;
        // Drop the failed turn so a retry does not resend it twice.
        state.history.pop();
        if (!els.prompt.value) els.prompt.value = text;
        finish();
        return;
      }
    }

    timing.end = performance.now();
    if (frame) cancelAnimationFrame(frame);
    if (conversation !== state.conversation) {
      finish();
      return;
    }
    reply.classList.remove("is-pending");
    content.innerHTML = answer ? renderMarkdown(answer) : '<p class="muted">(empty response)</p>';
    if (answer) state.history.push({ role: "assistant", content: answer });
    enableCopy(reply, () => answer);
    meta.hidden = false;
    meta.textContent = formatMeta(settings, timing, stats, stopped);
    els.resRaw.textContent = raw.length >= RAW_LIMIT ? raw + "\n… (truncated)" : raw;
    finish();
  }

  function finish() {
    state.controller = null;
    setBusy(false);
    els.prompt.focus();
    scrollToBottom(true);
  }

  function newChat() {
    state.conversation += 1;
    if (state.controller) state.controller.abort();
    state.history = [];
    els.messages.querySelectorAll(".message").forEach((node) => node.remove());
    els.emptyState.hidden = false;
    els.prompt.value = "";
    autoGrow();
    updateComposer();
    els.prompt.focus();
  }

  function autoGrow() {
    els.prompt.style.height = "auto";
    els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 220)}px`;
  }

  function submit() {
    const text = els.prompt.value.trim();
    if (!text || els.send.disabled) return;
    els.prompt.value = "";
    autoGrow();
    send(text);
  }

  // --- wiring -----------------------------------------------------------
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });

  els.prompt.addEventListener("input", () => {
    autoGrow();
    updateComposer();
  });

  els.stop.addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });

  els.newChat.addEventListener("click", newChat);
  els.refresh.addEventListener("click", () => loadModels());

  els.temperature.addEventListener("input", () => {
    els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2);
  });

  els.settings.addEventListener("change", () => {
    saveSettings();
    updateComposer();
  });
  els.settings.addEventListener("submit", (event) => event.preventDefault());

  document.querySelectorAll("[data-suggest]").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.prompt.value = chip.dataset.suggest;
      autoGrow();
      updateComposer();
      submit();
    });
  });

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", () => copyText($(button.dataset.copy).textContent, button));
  });

  const saved = restoreSettings();
  const fromUrl = new URLSearchParams(location.search).get("model");
  loadModels(fromUrl || saved.model);
  updateComposer();
  els.prompt.focus();
})();
