// tabctx agent demo: one page, no framework. The server streams each
// agent's transcript as normalized events (../transcripts.py); each tabctx
// tool gets a card built around one visual, with details on hover.

const $ = (sel) => document.querySelector(sel);
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
// A tooltip: data-tip holds HTML (the tooltip sets innerHTML from it), and
// the browser unescapes an attribute once, so the HTML is escaped again
// here. Untrusted parts of it must already be esc()-ed: tipAttr(esc(v)).
const tipAttr = (html) => `data-tip="${esc(html)}"`;
const fmt = (n, d = 0) =>
  n == null || Number.isNaN(Number(n)) ? "–" : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (p, d = 0) => (p == null ? "–" : `${fmt(p * 100, d)}%`);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const base = (p) => String(p ?? "").split("/").pop();
const METRICS = { roc_auc: "ROC AUC", r2: "R²", log_loss: "log loss", accuracy: "accuracy", rmse: "RMSE", mae: "MAE" };
const metricName = (m) => METRICS[m] || m;

const STEPS = [
  ["inspect_table", "Inspect"],
  ["evaluate", "Evaluate"],
  ["fit_model", "Fit"],
  ["predict", "Predict"],
  ["explain", "Explain"],
  ["what_if", "What-if"],
];
const TITLES = {
  inspect_table: "Inspect",
  evaluate: "Evaluate",
  fit_model: "Fit",
  predict: "Predict",
  what_if: "What-if",
  explain: "Explain",
  list_models: "Models",
};
const ICONS = {
  inspect_table: '<path d="M4 5h16v14H4zM4 10h16M10 10v9" />',
  evaluate: '<path d="M5 19V9M12 19V5M19 19v-7" />',
  fit_model: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" /><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />',
  predict: '<path d="M4 18l5-6 4 3 7-9" /><path d="M15 6h5v5" />',
  explain: '<path d="M5 6h14M5 12h9M5 18h5" />',
  what_if: '<path d="M7 7h10l-3-3M17 17H7l3 3" />',
  list_models: '<path d="M5 6h14M5 12h14M5 18h14" />',
};
const svg = (paths, size = 15) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;

const state = {
  setup: null,
  agent: null,
  source: null,
  calls: new Map(),
  tabctxCalls: 0,
  used: new Set(),
  started: 0,
  timer: null,
  queue: [],
  replaying: false,
  stats: null,
};

// ---- class colors: the entity keeps its color everywhere -------------------
// The positive class ("Yes", the one ranked) is slot 2; the others slot 1.
const positiveOf = (classes) => (classes.includes("Yes") ? "Yes" : classes.at(-1));
const classVar = (cls, classes) => (cls === positiveOf(classes) ? "var(--s2)" : "var(--s1)");

// ---- theme + tooltip --------------------------------------------------------

function initTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem("theme");
  } catch {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme").onclick = () => {
    const dark =
      document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {}
  };
}

function initTooltips() {
  const tip = $("#tip");
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) return tip.classList.remove("on");
    tip.innerHTML = el.dataset.tip;
    tip.classList.add("on");
  });
  document.addEventListener("mousemove", (e) => {
    if (!tip.classList.contains("on")) return;
    const w = tip.offsetWidth;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - w - 8)}px`;
    tip.style.top = `${e.clientY + 16}px`;
  });
}

// ---- setup ---------------------------------------------------------------

async function init() {
  initTheme();
  initTooltips();
  const setup = await loadSetup();
  renderDataset(setup.dataset);
  renderTask(setup.dataset);
  $("#prompt").value = setup.prompt;
  $("#prompt").readOnly = !setup.prompt_editable;
  $("#reset-prompt").hidden = !setup.prompt_editable;
  state.agent = setup.running || setup.runs.at(-1)?.agent || setup.agents.find((a) => a.available)?.id;
  renderAgents();
  $("#reset-prompt").onclick = () => ($("#prompt").value = setup.prompt);
  $("#run").onclick = start;
  $("#replay").onclick = () => watch(state.agent, true);
  $("#login").onsubmit = signIn;
  $("#logout").onclick = signOut;
  if (setup.running) watch(setup.running, false);
}

async function loadSetup() {
  const setup = await (await fetch("/api/setup")).json();
  state.setup = setup;
  renderGateway(setup);
  renderAccount(setup);
  return setup;
}

function renderAccount(setup) {
  $("#account").hidden = !setup.hosted;
  const signed = Boolean(setup.signed_in);
  $("#login").hidden = signed;
  $("#account .signed").hidden = !signed;
  $("#h-account").textContent = signed ? "Signed in" : "Gateway key";
  if (signed) $("#key-id").textContent = setup.signed_in.key_id;
}

async function signIn(e) {
  e.preventDefault();
  notice("");
  const key = $("#key").value.trim();
  if (!key) return;
  const button = $("#login button");
  button.disabled = true;
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  button.disabled = false;
  $("#key").value = "";
  if (!res.ok) {
    notice((await res.json()).detail || "Could not sign in");
    return;
  }
  await loadSetup();
  renderAgents();
}

async function signOut() {
  await fetch("/api/logout", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  await loadSetup();
  renderAgents();
}

function renderGateway(setup) {
  const host = setup.gateway.replace(/^https?:\/\//, "");
  $("#gateway").innerHTML = `<span class="dot ${setup.key_set ? "" : "off"}"></span>${esc(host)}`;
  $("#gateway").dataset.tip = setup.hosted
    ? setup.key_set
      ? "Signed in: runs use your key"
      : "Sign in with your gateway key to run an agent"
    : setup.key_set
      ? "Gateway key set"
      : "No TABCTX_API_KEY: runs will fail";
}

function renderDataset(d) {
  if (!d) {
    $("#dataset").innerHTML = `<p class="muted">Run <code>prepare_data.py</code> first.</p>`;
    return;
  }
  const yes = d.positive_rate;
  const classes = [d.negative, d.positive];
  $("#dataset").innerHTML = `
    <div class="ds">
      <div class="v">${fmt(d.history_rows)}</div><div class="k">${esc(d.history)}</div>
      <div class="split">${segBar([[d.negative, 1 - yes], [d.positive, yes]], classes, 8)}</div>
      <div class="f" ${tipAttr(`${esc(d.history_file)}: ${pct(yes, 1)} ${esc(d.outcome)} ${esc(d.window)}`)}>${pct(yes, 1)} ${esc(d.outcome)}</div>
    </div>
    <div class="ds">
      <div class="v">${fmt(d.current_rows)}</div><div class="k">${esc(d.current)}</div>
      <div class="locked" data-tip="Kept from the agent; used only for scoring">${svg('<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>', 12)} outcome hidden</div>
      <div class="f" ${tipAttr(esc(d.columns.join(", ")))}>${d.features} columns</div>
    </div>`;
}

// The task (PROMPT.md's steps, from task.py) as icons; the prompt is folded away.
const REPORT_ICON = '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>';
function renderTask(d) {
  const steps = d?.steps || [];
  $("#task-flow").innerHTML = steps
    .map(
      ([tool, label, tip]) =>
        `<div class="task-step" ${tipAttr(esc(tip))}><span class="tool-icon">${svg(ICONS[tool] || REPORT_ICON)}</span>${esc(label)}</div>`,
    )
    .join("");
  $("#prompt-box").open = Boolean(state.setup.prompt_editable);
}

function renderAgents() {
  const scores = Object.fromEntries(state.setup.runs.map((r) => [r.agent, r.score]));
  const hosted = state.setup.hosted;
  $("#agents").innerHTML = state.setup.agents
    .map((a) => {
      const s = scores[a.id];
      const replayable = a.id in scores;
      const badge = !a.available
        ? '<span class="agent-badge na">not installed</span>'
        : hosted && state.setup.signed_in && !a.allowed
          ? '<span class="agent-badge na" data-tip="Your key\'s project has no model for this agent">no access</span>'
          : s?.roc_auc != null
          ? `<span class="agent-badge" data-tip="ROC AUC of its last run, on the real outcomes">${fmt(s.roc_auc, 3)}</span>`
          : "";
      const tools = a.tools.split(" · ")[0];
      const noShell = a.tools.includes("no shell") || a.tools.includes("no bash");
      const shell = noShell
        ? `<span class="chip-sm" data-tip="No shell: files and tabctx tools only">${svg('<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>', 11)}</span>`
        : `<span class="chip-sm" ${tipAttr(esc(a.tools))}>${svg('<path d="M5 8l4 4-4 4M11 16h7"/>', 11)}</span>`;
      return `<button class="agent" role="radio" type="button" data-id="${a.id}" aria-checked="${a.id === state.agent}" ${a.available || replayable ? "" : "disabled"} ${tipAttr(`${esc(a.llm)}<br>${esc(a.tools)}`)}>
        <div class="agent-top"><span class="agent-name">${esc(a.label)}</span>${badge}</div>
        <div class="agent-tools"><span class="chip-sm">${esc(tools.replace(" extension", ""))}</span>${shell}</div>
      </button>`;
    })
    .join("");
  document.querySelectorAll(".agent").forEach((b) => (b.onclick = () => selectAgent(b.dataset.id)));
  $("#replay").disabled = !(state.agent in scores) || isBusy();
  const agent = state.setup.agents.find((a) => a.id === state.agent);
  const signedOut = hosted && !state.setup.signed_in;
  $("#run").disabled = isBusy() || signedOut || !agent?.allowed;
  $("#run-label").textContent = signedOut ? "Sign in to run" : "Run agent";
}

function selectAgent(id) {
  state.agent = id;
  renderAgents();
}

const isBusy = () => ["live", "replay"].some((c) => $("#status").classList.contains(c));

async function start() {
  notice("");
  $("#run").disabled = true;
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agent: state.agent, prompt: $("#prompt").value }),
  });
  if (!res.ok) {
    notice((await res.json()).detail || "Could not start the agent");
    renderAgents();
    return;
  }
  watch(state.agent, false);
}

function notice(text) {
  $("#notice").hidden = !text;
  $("#notice").textContent = text;
}

// ---- a run ---------------------------------------------------------------

function watch(agent, replay) {
  state.source?.close();
  clearInterval(state.timer);
  Object.assign(state, { calls: new Map(), tabctxCalls: 0, used: new Set(), queue: [], replaying: replay, stats: null });
  const info = state.setup.agents.find((a) => a.id === agent);
  $("#empty").hidden = true;
  $("#run-view").hidden = false;
  $("#timeline").innerHTML = "";
  $("#outcome").hidden = true;
  $("#report-card").open = false;
  $("#run-title").textContent = info.label;
  renderSteps();
  setStatus(replay ? "replay" : "live");
  updateMeta(info);

  const source = new EventSource(`/api/runs/${agent}/events`);
  state.source = source;
  source.addEventListener("status", (e) => {
    const s = JSON.parse(e.data);
    state.started = replay ? Date.now() : s.started * 1000;
    if (!replay) state.timer = setInterval(() => updateMeta(info), 500);
  });
  source.addEventListener("event", (e) => enqueue(() => handle(JSON.parse(e.data), info)));
  source.addEventListener("outcome", (e) => {
    source.close();
    enqueue(() => finish(JSON.parse(e.data), info));
  });
  source.onerror = () => {
    source.close();
    if ($("#outcome").hidden && !state.queue.length) setStatus("failed", "Connection lost");
  };
  $("#run").disabled = true;
  $("#replay").disabled = true;
}

// A replay plays at a readable pace; a live run shows events as they come.
function enqueue(fn) {
  if (!state.replaying) return fn();
  state.queue.push(fn);
  if (state.queue.length === 1) drain();
}
function drain() {
  const fn = state.queue[0];
  if (!fn) return;
  const delay = fn() === "slow" ? 420 : 60;
  setTimeout(() => {
    state.queue.shift();
    drain();
  }, delay);
}

// An agent's progress note: bold and `code` as the agent wrote them,
// and nothing else (no images or links).
function inlineMd(text) {
  return DOMPurify.sanitize(marked.parseInline(text), { ALLOWED_TAGS: ["strong", "em", "code", "del"], ALLOWED_ATTR: [] });
}

function handle(ev, info) {
  const tl = $("#timeline");
  if (ev.type === "message") {
    const el = document.createElement("div");
    el.className = "msg";
    el.innerHTML = `<span class="avatar">${esc(info.label[0])}</span><div class="body">${inlineMd(ev.text.trim())}</div>`;
    el.onclick = () => el.classList.toggle("open");
    tl.appendChild(el);
    return;
  }
  if (ev.type === "tool_call") {
    if (ev.tabctx) state.tabctxCalls++;
    const el = document.createElement("div");
    if (ev.tabctx) {
      state.used.add(ev.tool);
      renderSteps();
      el.className = "card pending";
      el.innerHTML = cardHead(ev.tool, describeCall(ev.tool, ev.args), '<span class="spinner"></span>');
    } else {
      el.className = "minor";
      el.innerHTML = minorCall(ev.tool, ev.args);
    }
    tl.appendChild(el);
    state.calls.set(ev.id, { el, tool: ev.tool, args: ev.args, t0: performance.now() });
    updateMeta(info);
    scrollDown();
    return "slow";
  }
  if (ev.type === "tool_result") {
    const call = state.calls.get(ev.id);
    if (!call) return;
    if (!ev.tabctx) {
      if (ev.error) call.el.classList.add("err");
      return;
    }
    const secs = state.replaying ? null : (performance.now() - call.t0) / 1000;
    const right = secs == null ? "" : `<span class="card-time">${fmt(secs, 1)} s</span>`;
    call.el.classList.remove("pending");
    const failed = ev.error || !ev.result;
    if (failed) call.el.classList.add("error");
    call.el.innerHTML =
      cardHead(call.tool, describeCall(call.tool, call.args), right) +
      `<div class="card-body">${failed ? errorLine(ev.text) : renderResult(call.tool, ev.result, call.args)}</div>`;
    scrollDown();
    return "slow";
  }
  if (ev.type === "done") state.stats = ev.stats;
  if (ev.type === "error") tl.insertAdjacentHTML("beforeend", `<div class="minor err">Agent error: ${esc(ev.text)}</div>`);
}

function finish(outcome, info) {
  clearInterval(state.timer);
  state.replaying = false;
  const failed = outcome.log?.length && !outcome.report;
  setStatus(failed ? "failed" : "done", failed ? "Failed" : "Finished");
  updateMeta(info, true);
  if (failed) {
    $("#timeline").insertAdjacentHTML(
      "beforeend",
      `<div class="card error"><div class="card-body"><div class="error-text">${esc(outcome.log.join("\n"))}</div></div></div>`,
    );
  }
  renderOutcome(outcome);
  refreshRuns(); // re-enables Run as the viewer's access allows
}

async function refreshRuns() {
  state.setup.runs = (await (await fetch("/api/setup")).json()).runs;
  renderAgents();
}

function setStatus(kind, text) {
  const el = $("#status");
  el.className = `status ${kind}`;
  el.innerHTML =
    kind === "live" ? '<span class="pulse"></span>Running' : kind === "replay" ? '<span class="pulse"></span>Replaying' : esc(text);
}

function updateMeta(info, final = false) {
  const model = info.llm.replace(/\s*(via|\().*$/, "");
  const pills = [...(model ? [model] : []), `${state.tabctxCalls} tabctx calls`];
  const secs = state.stats?.seconds ?? (state.started && !state.replaying ? (Date.now() - state.started) / 1000 : null);
  if (secs != null && !(final && state.replaying)) pills.push(`${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, "0")}`);
  if (final && state.stats?.cost_usd) pills.push(`$${fmt(state.stats.cost_usd, 2)}`);
  $("#run-meta").innerHTML = pills.map((p) => `<span class="meta">${esc(p)}</span>`).join("");
}

function renderSteps() {
  $("#steps").innerHTML = STEPS.map(([tool, label]) => `<div class="step ${state.used.has(tool) ? "on" : ""}"><i></i>${label}</div>`).join("");
}

const scrollDown = () => {
  if (!state.replaying) window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
};

// ---- building blocks ---------------------------------------------------------

// A tool error's first clause (the rest, e.g. a column list, on hover).
function errorLine(text) {
  const full = String(text || "error").replace(/^Error executing tool \w+:\s*/, "");
  const short = full.split(/;\s|\n/)[0];
  return `<div class="error-text" ${short !== full ? tipAttr(esc(full)) : ""}>${esc(short)}${short !== full ? " …" : ""}</div>`;
}

function cardHead(tool, sub, right = "") {
  return `<div class="card-head"><span class="tool-icon">${svg(ICONS[tool] || ICONS.list_models)}</span>
    <div class="grow"><div class="card-title">${TITLES[tool] || esc(tool)}</div><div class="card-sub">${sub}</div></div>${right}</div>`;
}

function describeCall(tool, a = {}) {
  const file = (p) => (p ? `<code>${esc(base(p))}</code>` : "");
  switch (tool) {
    case "inspect_table":
      return `${file(a.path)}${a.target ? ` → <code>${esc(a.target)}</code>` : ""}`;
    case "evaluate":
      return `${file(a.path)} · held-out rows`;
    case "fit_model":
      return `${esc(a.model || "tabpfn-3.5")} on ${file(a.path)}`;
    case "predict":
      return `${file(a.path)}${a.where ? ` · ${esc(Object.entries(a.where).map(([k, v]) => `${k} = ${v}`).join(", "))}` : ""}`;
    case "what_if":
      return a.base_where ? `<code>${esc(Object.values(a.base_where).join(", "))}</code>` : `${(a.rows || []).length} cases`;
    case "explain":
      return `${esc(a.model || "tabpfn-3.5")} · permutation importance`;
    default:
      return "";
  }
}

function minorCall(tool, a = {}) {
  const path = a.file_path || a.path || "";
  const what = a.command
    ? `<code>${esc(String(a.command).replace(/^\/bin\/\w+ -lc /, "").slice(0, 80))}</code>`
    : path
      ? `<code>${esc(base(path))}</code>`
      : "";
  return `${svg('<path d="M9 6l6 6-6 6"/>', 12)}<span>${esc(tool)}</span>${what}`;
}

// A horizontal bar on a lane with a baseline. value in [0, 1].
const bar = (value, cls = "", tip = "") =>
  `<div class="lane" ${tip ? tipAttr(tip) : ""}><div class="bar ${cls}" style="width:${clamp01(value) * 100}%"></div></div>`;

// Part-to-whole, 2px gaps between parts. parts: [[class, share], ...]
function segBar(parts, classes, height = 12) {
  return `<div class="seg" style="height:${height}px">${parts
    .filter(([, s]) => s > 0)
    .map(([c, s]) => `<span ${tipAttr(`${esc(c)}: ${pct(s, 1)}`)} style="flex:${s};background:${classVar(c, classes)}"></span>`)
    .join("")}</div>`;
}
const legend = (parts, classes) =>
  `<div class="legend">${parts.map(([c, s]) => `<span><i style="background:${classVar(c, classes)}"></i>${esc(c)} ${pct(s)}</span>`).join("")}</div>`;
const fig = (v, k) => `<div class="fig"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`;
const raw = (r) => `<details class="more"><summary>Details</summary><pre class="json">${esc(JSON.stringify(r, null, 2))}</pre></details>`;

function renderResult(tool, r, args) {
  const body = { inspect_table: inspectCard, evaluate: evaluateCard, fit_model: fitCard, predict: predictCard, what_if: whatIfCard, explain: explainCard }[tool];
  return (body ? body(r, args) : "") + raw(r);
}

// ---- cards -------------------------------------------------------------------

function inspectCard(r) {
  const chips = r.features
    .map((f) => {
      const kind = f.note ? "id" : f.kind === "numeric" ? "numeric" : "text";
      const detail = f.kind === "numeric" ? `${fmt(f.min)}–${fmt(f.max)}` : `${fmt(f.distinct)} values`;
      return `<span class="col ${kind}" ${tipAttr(`<b>${esc(f.name)}</b><br>${esc(f.kind)} · ${detail}${f.missing ? ` · ${fmt(f.missing)} missing` : ""}${f.note ? "<br>likely an id" : ""}`)}></span>`;
    })
    .join("");
  const target = r.target ? `<span class="col target" ${tipAttr(`<b>${esc(r.target.name)}</b><br>target · ${esc(r.target.task)}`)}></span>` : "";
  let balance = "";
  if (r.target?.classes) {
    const total = Object.values(r.target.classes).reduce((a, b) => a + b, 0);
    const classes = Object.keys(r.target.classes);
    const parts = Object.entries(r.target.classes).map(([c, n]) => [c, n / total]);
    balance = `<div style="flex:1;min-width:220px"><div class="split-label">${esc(r.target.name)}</div>${segBar(parts, classes)}${legend(parts, classes)}</div>`;
  }
  const kinds = ["numeric", "text", "id"].filter((k) => r.features.some((f) => (k === "id" ? f.note : !f.note && (k === "numeric") === (f.kind === "numeric"))));
  return `<div class="row">${fig(fmt(r.rows), "rows")}${fig(fmt(r.columns), "columns")}${balance}</div>
    <div class="colmap">${chips}${target}</div>
    <div class="keys">${kinds.map((k) => `<span><i class="col ${k}"></i>${k === "id" ? "likely id" : k}</span>`).join("")}${target ? '<span><i class="col target"></i>target</span>' : ""}</div>`;
}

function evaluateCard(r) {
  const metric = r.best_by;
  const lo = metric === "roc_auc" ? 0.5 : 0;
  const scaled = (v) => (v - lo) / (1 - lo);
  const rows = r.results
    .map((m) => {
      const baseline = m.model === "gradient-boosting";
      const name = baseline ? "gradient boosting" : m.model;
      const tag = m.model === r.best ? '<span class="tag best">best</span>' : baseline ? '<span class="tag base">baseline</span>' : "";
      if (m.error) {
        return `<div class="bar-row"><div class="label">${esc(name)}</div>${bar(1, "ghost", `failed: ${esc(m.error)}`)}<div class="val faint">failed</div></div>`;
      }
      const v = m.metrics[metric];
      const tip = [`<b>${esc(name)}</b>`, ...Object.entries(m.metrics).map(([k, x]) => `${esc(metricName(k))} ${fmt(x, 3)}`), `fit ${fmt(m.fit_seconds, 1)} s`].join("<br>");
      return `<div class="bar-row"><div class="label">${esc(name)}${tag}</div>${bar(scaled(v ?? lo), baseline ? "neutral" : "", tip)}<div class="val">${fmt(v, 3)}</div></div>`;
    })
    .join("");
  const label = metricName(metric);
  return `<div class="split-label">${esc(label)} on ${fmt(r.test_rows)} held-out rows</div><div class="bars">${rows}</div>
    ${lo ? `<div class="axis-note"><span>0.5 chance</span><span>1.0 perfect</span></div>` : ""}`;
}

function fitCard(r) {
  return `<div class="figs">${fig(esc(r.model), "model")}${fig(fmt(r.train_rows), "rows")}${fig(r.features.length, "features")}${fig(`${fmt(r.fit_seconds, 1)} s`, "to fit")}</div>
    <div class="chips"><span class="chip">saved as <code>${esc(r.name)}</code></span>${(r.dropped || []).map((d) => `<span class="chip">dropped <code>${esc(d)}</code></span>`).join("")}</div>`;
}

function predictCard(r) {
  const key = r.ranked_by;
  const counts = r.summary.predicted_counts;
  let split = "";
  let classes = [];
  if (counts) {
    // The mean probability, not the 50% calls: for a rare outcome almost
    // no row crosses 50%, and "0% Yes" would say nothing.
    const mean = r.summary.mean_probability;
    classes = Object.keys(mean || counts);
    const parts = classes.map((c) => [c, mean ? mean[c] : (counts[c] || 0) / r.rows]);
    split = `<div style="flex:1;min-width:240px"><div class="split-label">${mean ? "expected" : "predicted"}</div>${segBar(parts, classes)}${legend(parts, classes)}</div>`;
  }
  const head = `<div class="row">${fig(fmt(r.rows), "rows")}${fig(`${fmt(r.predict_seconds, 1)} s`, "to predict")}${split}</div>`;
  const cols = r.top_rows.length ? Object.keys(r.top_rows[0]).filter((k) => !k.startsWith("p_") && !k.startsWith("predicted_")) : [];
  const isProba = key?.startsWith("p_");
  const cls = isProba ? key.split("=").slice(1).join("=") : null;
  const tone = cls && cls === positiveOf(classes) ? "yes" : "";
  const rows = r.top_rows
    .map(
      (row) => `<tr>${cols.map((k, i) => `<td class="${i === 0 ? "id" : ""}">${esc(row[k] ?? "–")}</td>`).join("")}
      <td>${isProba ? `<div class="pbar">${bar(row[key], tone, `P(${esc(cls)}) = ${fmt(row[key], 3)}`)}<span class="num">${fmt(row[key], 2)}</span></div>` : fmt(row[key], 2)}</td></tr>`,
    )
    .join("");
  const title = isProba ? `P(${esc(cls)})` : esc(key);
  const table = r.top_rows.length
    ? `<div class="table-wrap"><table><thead><tr>${cols.map((k) => `<th>${esc(k)}</th>`).join("")}<th>${title}</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : "";
  const where = r.rows_matching_where != null ? `<span class="chip">${fmt(r.rows_matching_where)} rows match</span>` : "";
  return `${head}${table}<div class="chips">${where}<span class="chip">→ <code>${esc(base(r.output))}</code></span></div>`;
}

function explainCard(r) {
  const top = r.importance.slice(0, 8);
  const max = Math.max(...top.map((i) => i.score_drop), 1e-9);
  return `<div class="split-label">${esc(metricName(r.metric))} lost when a column is shuffled</div><div class="bars">${top
    .map(
      (i) =>
        `<div class="bar-row"><div class="label mono">${esc(i.column)}</div>${bar(i.score_drop / max, "", `Shuffling <b>${esc(i.column)}</b> lowers ${esc(metricName(r.metric))} by ${fmt(i.score_drop, 3)} (±${fmt(i.spread, 3)})`)}<div class="val">${fmt(i.score_drop, 3)}</div></div>`,
    )
    .join("")}</div>
    <div class="chips"><span class="chip">${fmt(r.rows_predicted)} rows in one predict</span><span class="chip">${fmt(r.predict_seconds, 1)} s</span></div>`;
}

function whatIfCard(r, args) {
  const results = r.results;
  if (!results.length) return "";
  const probs = results[0].probabilities;
  const who = ""; // the card's header names the row
  if (!probs) {
    return `${who}<div class="bars">${results.map((x, i) => `<div class="bar-row"><div class="label">${caseLabel(x, args, i)}</div><div></div><div class="val">${fmt(x.prediction, 2)}</div></div>`).join("")}</div>`;
  }
  const classes = Object.keys(probs);
  const cls = positiveOf(classes);
  const first = probs[cls];
  return `${who}<div class="split-label">P(${esc(cls)})</div><div class="bars">${results
    .map((x, i) => {
      const p = x.probabilities[cls];
      const d = p - first;
      const delta =
        i === 0 || Math.abs(d) < 0.0005
          ? "<span></span>"
          : `<span class="delta ${d < 0 ? "down" : "up"}">${svg(d < 0 ? '<path d="M12 5v14M6 13l6 6 6-6"/>' : '<path d="M12 19V5M6 11l6-6 6 6"/>', 12)}${fmt(Math.abs(d) * 100, 1)} pts</span>`;
      return `<div class="wi-row"><div class="label">${caseLabel(x, args, i)}</div>${bar(p, "yes", `P(${esc(cls)}) = ${fmt(p, 3)}`)}<div class="num">${fmt(p, 2)}</div>${delta}</div>`;
    })
    .join("")}</div>`;
}

function caseLabel(x, args, i) {
  const changes = x.changes ?? args?.rows?.[i];
  if (!changes || !Object.keys(changes).length) return '<span class="muted">as it is</span>';
  // The new values on one line, the columns they go in under them.
  const entries = Object.entries(changes);
  const tip = entries.map(([k, v]) => `${esc(k)} = ${esc(v)}`).join("<br>");
  return `<span ${tipAttr(tip)}>${esc(entries.map(([, v]) => v).join(" + "))}<span class="faint">${esc(entries.map(([k]) => k).join(" + "))}</span></span>`;
}

// ---- outcome ---------------------------------------------------------------------

function renderOutcome(o) {
  $("#outcome").hidden = false;
  const s = o.score || {};
  if (s.error || s.roc_auc == null || s.positive_in_top_10 == null || s.positive_in_top_100 == null) {
    $("#scores").innerHTML = `<div class="score" style="grid-column:1/-1"><div class="k">Not scored</div><div class="faint">${esc(s.error || "no predictions file")}</div></div>`;
  } else {
    const rate = s.base_rate;
    const what = state.setup.dataset?.outcome || "positive";
    const top10 = s.positive_in_top_10;
    const top100 = s.positive_in_top_100;
    const hits = Math.round(top10 * 10);
    const auc = `<div class="score"><div class="v">${fmt(s.roc_auc, 3)}</div><div class="k">ROC AUC</div>
      <div class="viz"><div class="ref-lane" ${tipAttr(`0.5 is a coin flip; 1.0 ranks every ${esc(what)} case first`)}><div class="bar" style="width:${clamp01((s.roc_auc - 0.5) / 0.5) * 100}%"></div></div>
      <div class="scale"><span>0.5 chance</span><span>1.0</span></div></div></div>`;
    const ten = `<div class="score"><div class="v">${hits}<span class="faint" style="font-size:24px"> / 10</span></div><div class="k">top-10 risks ${esc(what)}</div>
      <div class="viz"><div class="dots" ${tipAttr(`${hits} of its 10 highest risks were ${esc(what)}; a random 10 would hold about ${fmt(rate * 10)}`)}>${Array.from({ length: 10 }, (_, i) => `<i class="${i < hits ? "hit" : ""}"></i>`).join("")}</div>
      <div class="scale"><span>random ≈ ${fmt(rate * 10)}</span><span></span></div></div></div>`;
    const hundred = `<div class="score"><div class="v">${pct(top100)}</div><div class="k">of top 100 ${esc(what)}</div>
      <div class="viz"><div class="ref-lane" ${tipAttr(`${pct(top100)} of its top 100, against ${pct(rate)} of all`)}><div class="bar yes" style="width:${top100 * 100}%"></div><div class="mark" style="left:${rate * 100}%"></div></div>
      <div class="scale"><span>base rate ${pct(rate)} ▲</span><span>${fmt(top100 / rate, 1)}×</span></div></div></div>`;
    $("#scores").innerHTML = auc + ten + hundred;
  }
  $("#report").innerHTML = o.report ? DOMPurify.sanitize(marked.parse(o.report)) : `<p class="faint">The agent wrote no report.</p>`;
  $("#report-card").hidden = false;
}

init();
