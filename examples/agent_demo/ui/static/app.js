// tabctx agent demo: one page, no framework. The server streams each
// agent's transcript as normalized events (../transcripts.py). Two views of
// the same events: the answer a business reader follows (answer.js), and,
// with "Show the work", every command and tabctx call as a card here.

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
// A task with a "clean" step (its agent cleans raw files first) shows it as
// the first progress step, done once the agent writes or runs something
// named for cleaning (clean.py, a clean_*.csv), not on its first `ls`.
const CLEAN = ["clean", "Clean"];
const CLEANING = /clean/i;
const hasClean = () => (state.setup?.dataset?.steps || []).some(([tool]) => tool === "clean");
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
  clean: '<path d="M14 4l6 6M4 20l6-1 9-9-5-5-9 9z" /><path d="M8 14l2 2" />',
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
  // Presenting: the work fills the screen, and a replay waits for the
  // presenter before each command (→, space or a clicker) unless on auto.
  present: false,
  auto: false,
  waiting: false,
  go: false,
  stepAt: 0,
  stepTotal: 0,
  gen: 0, // a new run or replay starts a new generation; old timers stand down
  pinned: true, // the work panel follows the newest tile until the viewer scrolls up
  // Open questions: the question being asked and the plan drafted for it
  // (null until drafted); ask: the question and approved plan of the run
  // being started.
  question: "",
  plan: null,
  planning: false,
  ask: null,
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
  initPanels();
  initPresent();
  const setup = await loadSetup();
  renderComposer(setup.dataset);
  renderAsk(setup.dataset || {});
  $("#prompt").value = setup.prompt;
  $("#prompt").readOnly = !setup.prompt_editable;
  $("#reset-prompt").hidden = !setup.prompt_editable;
  state.agent = setup.running || setup.runs.at(-1)?.agent || setup.agents.find((a) => a.available)?.id;
  renderAgents();
  $("#reset-prompt").onclick = () => ($("#prompt").value = setup.prompt);
  $("#run").onclick = (e) => {
    if (state.setup.hosted && !state.setup.signed_in) {
      e.stopPropagation(); // or the click, bubbling on, would close it again
      return openSettings(true);
    }
    if (state.setup.open && !state.plan) return draftPlan();
    start();
  };
  $("#replan").onclick = () => showPlan(null);
  $("#replay").onclick = () => watch(state.agent, true);
  $("#again-replay").onclick = () => watch(state.shown, true);
  $("#again-new").onclick = newQuestion;
  $("#stop").onclick = stopRun;
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

// The question as it would be asked, with the two files it is about.
function renderComposer(d) {
  if (!d) {
    $("#question").innerHTML = `<span class="muted">Run <code>prepare_data.py</code> first.</span>`;
    return;
  }
  $("#org").textContent = d.org || "";
  const file = (name, rows, what, note) =>
    `<div class="file" ${tipAttr(esc(name))}><span class="glyph" aria-hidden="true"></span><span class="file-name">${esc(what)}</span><span class="file-meta">${fmt(rows)} rows${note ? ` · ${note}` : ""}</span></div>`;
  $("#attachments").innerHTML = file(d.history_file, d.history_rows, cap(d.history)) + file(d.current_file, d.current_rows, cap(d.current), "outcome hidden");
  $("#you-files").textContent = `From ${d.history} (${fmt(d.history_rows)} rows) and ${d.current} (${fmt(d.current_rows)} rows, outcome hidden)`;
  $("#question").textContent = d.question || "";
  $("#you-text").textContent = d.question || "";
}
const cap = (t) => String(t || "").replace(/^\w/, (c) => c.toUpperCase());

// The settings popover and the "Show the work" switch.
function initPanels() {
  $("#settings-btn").onclick = (e) => {
    e.stopPropagation();
    openSettings($("#settings").hidden);
  };
  // A click inside (even on a control it has since re-rendered) keeps it open.
  document.addEventListener("click", (e) => {
    if (!$("#settings").hidden && e.target.isConnected && !$("#settings").contains(e.target)) openSettings(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#settings").hidden) {
      openSettings(false);
      $("#settings-btn").focus();
    }
  });
  let work = false;
  try {
    work = localStorage.getItem("work") === "1";
  } catch {}
  const toggle = $("#work-toggle");
  toggle.checked = work;
  document.body.classList.toggle("show-work", work);
  toggle.onchange = () => {
    document.body.classList.toggle("show-work", toggle.checked);
    try {
      localStorage.setItem("work", toggle.checked ? "1" : "0");
    } catch {}
  };
}
function openSettings(open) {
  $("#settings").hidden = !open;
  $("#settings-btn").setAttribute("aria-expanded", String(open));
  if (open) $("#settings").querySelector("input:not([hidden]), button:not(:disabled)")?.focus();
}
function newQuestion() {
  state.source?.close();
  $("#thread").hidden = true;
  $("#empty").hidden = false;
  if (state.setup.open) showPlan(null);
}

// ---- open questions: ask anything, see the plan, approve it ---------------

// The brief's heading: the question the shown run was asked.
function showAsk(ask) {
  $("#you-text").textContent = ask?.question || state.setup.dataset?.question || "";
}

// Each kind of plan step: its short name and icon (planner.py's KINDS).
const KINDS = {
  clean: ["Clean", ICONS.clean],
  evaluate: ["Check", ICONS.evaluate],
  predict: ["Score", ICONS.predict],
  explain: ["Explain", ICONS.explain],
  what_if: ["Test", ICONS.what_if],
  analyze: ["Count", '<path d="M17 5H7l6 7-6 7h10" />'],
  report: ["Write", '<path d="M6 3h9l4 4v14H6z" /><path d="M9 12h7M9 16h5" />'],
};

function renderAsk(d) {
  const open = !!state.setup.open;
  $("#question").hidden = open;
  $("#ask").hidden = !open;
  if (!open) return;
  $("#suggest").innerHTML = (d.suggestions || [])
    .map(([id, short], i) => `<button class="sug" type="button" data-i="${i}">${esc(short)}</button>`)
    .join("");
  $("#suggest").onclick = (e) => {
    const b = e.target.closest(".sug");
    if (!b) return;
    $("#ask-text").value = d.suggestions[b.dataset.i][2];
    $("#ask-text").focus();
    renderAgents();
  };
  $("#ask-text").oninput = () => renderAgents();
  $("#ask-text").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Enter asks; Shift+Enter is a new line
      if (!$("#run").disabled) $("#run").click();
    }
  };
}

async function draftPlan() {
  const question = $("#ask-text").value.trim();
  if (!question) return notice("Ask a question first, or pick one below.");
  notice("");
  state.planning = true;
  renderAgents();
  try {
    const res = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return notice(body.detail || "Could not draft a plan");
    state.question = body.question;
    showPlan(body.plan, true);
  } catch {
    notice("Could not reach the demo; try again.");
  } finally {
    state.planning = false;
    renderAgents();
  }
}

// The plan to approve (null: back to asking). Each step's words can be
// edited in place, and a step removed.
function showPlan(plan, fresh = false) {
  state.plan = plan;
  $("#plan").classList.toggle("fresh", fresh); // steps come in one by one, once
  const has = !!plan;
  $("#plan-box").hidden = !has;
  $("#suggest").hidden = has;
  $("#replan").hidden = !has;
  $("#replay").hidden = has;
  $("#ask-text").readOnly = has;
  $("#ask").classList.toggle("asked", has);
  if (has) {
    $("#plan").innerHTML = plan
      .map((step, i) => {
        const [name, icon] = KINDS[step.kind] || KINDS.analyze;
        return `<li class="pstep" data-kind="${esc(step.kind)}" style="--i:${i}">
          <span class="pk">${svg(icon, 17)}</span><span class="pk-name">${name}</span>
          <span class="pk-text" contenteditable="plaintext-only" spellcheck="false" data-i="${i}" aria-label="Step ${i + 1}">${esc(step.text)}</span>
          <button class="pk-x" type="button" data-i="${i}" aria-label="Remove this step" title="Remove this step">${svg('<path d="M6 6l12 12M18 6L6 18" />', 13)}</button>
        </li>`;
      })
      .join("");
    $("#plan").oninput = (e) => {
      const t = e.target.closest(".pk-text");
      if (t) state.plan[t.dataset.i].text = t.textContent.trim();
      renderAgents();
    };
    $("#plan").onkeydown = (e) => {
      if (e.key === "Enter" && e.target.closest(".pk-text")) {
        e.preventDefault(); // one line per step
        e.target.blur();
      }
    };
    $("#plan").onclick = (e) => {
      const x = e.target.closest(".pk-x");
      if (!x) return;
      state.plan.splice(Number(x.dataset.i), 1);
      showPlan(state.plan.length ? state.plan : null);
    };
  }
  renderAgents();
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
  const open = state.setup.open;
  const steps = (state.plan || []).filter((s) => s.text);
  $("#run").disabled =
    isBusy() || state.planning || (!signedOut && !agent?.allowed) || (open && !signedOut && (state.plan ? !steps.length : !$("#ask-text").value.trim()));
  $("#run-label").innerHTML = signedOut
    ? "Sign in to ask"
    : !open
      ? "Ask"
      : state.planning
        ? '<span class="pulse-dot"></span>Drafting a plan'
        : state.plan
          ? "Approve and run"
          : "Draft a plan";
  $("#agent-name").textContent = agent ? agent.label : "Agent";
}

function selectAgent(id) {
  state.agent = id;
  renderAgents();
}

const isBusy = () => ["live", "replay"].some((c) => $("#status").classList.contains(c));

async function start() {
  notice("");
  // An open question runs with the plan as approved (steps left empty dropped).
  state.ask = state.setup.open ? { question: state.question, plan: state.plan.filter((s) => s.text) } : null;
  $("#run").disabled = true;
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.ask ? { agent: state.agent, ...state.ask } : { agent: state.agent, prompt: $("#prompt").value }),
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
  Object.assign(state, { calls: new Map(), tabctxCalls: 0, used: new Set(), queue: [], replaying: replay, stats: null, shown: agent, waiting: false, go: false, stepAt: 0, stepTotal: 0, gen: state.gen + 1, pinned: true });
  const info = state.setup.agents.find((a) => a.id === agent);
  $("#empty").hidden = true;
  $("#thread").hidden = false;
  $("#again").hidden = true;
  $("#again-replay").hidden = false;
  openSettings(false);
  // A live run's question and plan are known now; a replay's (or a run
  // still going when the page loaded) come with its status, below.
  const ask = replay ? null : state.ask;
  showAsk(ask);
  Answer.reset(state.setup, agent, replay, ask);
  Actions.reset(state.setup);
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
    if (s.ask && JSON.stringify(s.ask) !== JSON.stringify(ask)) {
      showAsk(s.ask);
      Answer.reset(state.setup, agent, replay, s.ask); // before any of its events
    }
    if (!replay) state.timer = setInterval(() => updateMeta(info), 500);
  });
  source.addEventListener("event", (e) => {
    const ev = JSON.parse(e.data);
    enqueue(() => handle(ev, info), ev.type === "tool_call" ? "call" : "");
  });
  source.addEventListener("outcome", (e) => {
    source.close();
    enqueue(() => finish(JSON.parse(e.data), info));
  });
  source.onerror = () => {
    source.close();
    if (!$("#outcome").hidden) return;
    // A replay still draining its queue ends when the queue does.
    enqueue(() => {
      setStatus("failed", "Connection lost");
      Answer.fail("The connection to the agent was lost before it finished.");
      $("#again").hidden = false;
    });
  };
  $("#run").disabled = true;
  $("#replay").disabled = true;
}

// A replay plays at a readable pace; a live run shows events as they come.
// Presenting, a replay stops before each command until the presenter moves
// on, or plays at a pace to talk over (auto).
function enqueue(fn, kind = "") {
  if (!state.replaying) {
    const out = fn();
    follow();
    return out;
  }
  if (kind === "call") state.stepTotal++;
  state.queue.push({ fn, kind });
  renderStepper();
  if (state.queue.length === 1 && !state.waiting) drain();
}
function drain() {
  const item = state.queue[0];
  if (!item) return renderStepper();
  if (state.present && item.kind === "call" && state.stepAt > 0 && !state.auto && !state.go) {
    state.waiting = true;
    return renderStepper();
  }
  state.go = false;
  if (item.kind === "call") state.stepAt++;
  let slow = false;
  const gen = state.gen;
  try {
    slow = item.fn() === "slow";
    follow();
  } finally {
    renderStepper();
    // One event that fails to draw must not stop the rest of the replay.
    const pace = state.present ? (slow ? (state.auto ? 2400 : 500) : 160) : slow ? 420 : 60;
    setTimeout(() => {
      if (gen !== state.gen) return; // a new run started meanwhile: its queue isn't ours
      state.queue.shift();
      drain();
    }, pace);
  }
}
function advance() {
  if (!state.waiting) return;
  state.waiting = false;
  state.go = true;
  drain();
}

function initPresent() {
  const on = (present) => {
    state.present = present;
    document.body.classList.toggle("present", present);
    if (present) document.body.classList.add("show-work");
    else document.body.classList.toggle("show-work", $("#work-toggle").checked);
    if (!present && state.waiting) {
      state.waiting = false;
      drain();
    }
    renderStepper();
  };
  on(new URLSearchParams(location.search).has("present"));
  document.addEventListener("keydown", (e) => {
    const target = e.target instanceof Element ? e.target : document.body;
    if (target.closest("input, textarea, select, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.toLowerCase();
    if (key === "p") return on(!state.present);
    if (!state.present) return;
    // Space and Enter on a focused button or link keep their own meaning.
    const control = target.closest("button, a, summary, [role=button]");
    if (["arrowright", "pagedown"].includes(key) || (!control && [" ", "enter"].includes(key))) {
      if (!state.waiting) return; // nothing to show next: the key keeps its usual job
      e.preventDefault();
      advance();
    } else if (key === "a") {
      state.auto = !state.auto;
      if (state.auto) advance();
      renderStepper();
    }
  });
}

function renderStepper() {
  const el = $("#stepper");
  const show = state.present && state.replaying && state.stepTotal > 0;
  el.hidden = !show;
  if (!show) return;
  const done = !state.queue.length;
  el.innerHTML = done
    ? `<span>All ${state.stepTotal} steps shown</span>`
    : `<span class="n">${state.stepAt} / ${state.stepTotal}</span>${
        state.auto ? "<span>playing</span><kbd>A</kbd><span>to pause</span>" : `<kbd>→</kbd><span>${state.waiting ? "next step" : "…"}</span><kbd>A</kbd><span>auto</span>`
      }`;
  el.classList.toggle("ready", state.waiting);
}

// Keep the newest tile in view: in the work panel when it scrolls on its
// own, else the page (a live run only, so a replay leaves the reader be).
function follow() {
  const work = $("#work");
  if (!work.dataset.watched) {
    work.dataset.watched = "1";
    // Our own smooth scrolls pass through "far from the bottom": only the viewer's count.
    work.addEventListener(
      "scroll",
      () => {
        if (performance.now() - (state.autoScroll || 0) > 900) state.pinned = work.scrollHeight - work.scrollTop - work.clientHeight < 120;
      },
      { passive: true },
    );
  }
  if (state.pinned && document.body.classList.contains("show-work") && work.scrollHeight > work.clientHeight + 4 && getComputedStyle(work).overflowY === "auto") {
    state.autoScroll = performance.now();
    work.scrollTo({ top: work.scrollHeight, behavior: "smooth" });
  }
}

// An agent's progress note: bold and `code` as the agent wrote them,
// and nothing else (no images or links).
function inlineMd(text) {
  return DOMPurify.sanitize(marked.parseInline(text), { ALLOWED_TAGS: ["strong", "em", "code", "del"], ALLOWED_ATTR: [] });
}

function handle(ev, info) {
  try {
    Answer.event(ev);
  } catch (err) {
    console.error("the brief could not draw an event", ev, err); // the work panel still does
  }
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
      // The agent's own commands, as pictures of what they do (actions.js).
      const action = Actions.add(tl, ev.tool, ev.args);
      if (hasClean() && CLEANING.test(JSON.stringify(ev.args || {}))) {
        state.used.add(CLEAN[0]);
        renderSteps();
      }
      state.calls.set(ev.id, { action, tool: ev.tool, args: ev.args, t0: performance.now() });
      updateMeta(info);
      scrollDown();
      return "slow";
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
      Actions.result(call.action, call.tool, call.args, ev.text, ev.error);
      return "slow";
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
  Answer.outcome(outcome, state.stats);
  $("#again").hidden = false;
  refreshRuns(); // re-enables Run as the viewer's access allows
}

async function refreshRuns() {
  state.setup.runs = (await (await fetch("/api/setup")).json()).runs;
  renderAgents();
}

// Stop: a live run's agent is ended on the server (and waited for, so the
// next can start at once); a replay just stops drawing.
async function stopRun() {
  const { shown, replaying } = state;
  $("#stop").disabled = true;
  state.source?.close();
  clearInterval(state.timer);
  Object.assign(state, { queue: [], waiting: false, go: false, replaying: false, gen: state.gen + 1 });
  if (!replaying) {
    const res = await fetch(`/api/runs/${shown}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).catch(() => null);
    if (!res?.ok) {
      setStatus("failed", "Could not stop");
      $("#stop").hidden = false; // let them try again
      $("#again").hidden = false;
      return;
    }
  }
  setStatus("stopped", "Stopped");
  renderStepper();
  Answer.fail(replaying ? "The replay was stopped." : "The run was stopped before it finished.");
  $("#again").hidden = false;
  await refreshRuns(); // re-enables Ask and the replay
  // A stopped live run is thrown away: replay the recorded one, if any.
  $("#again-replay").hidden = !state.setup.runs.some((r) => r.agent === shown);
}

function setStatus(kind, text) {
  const busy = kind === "live" || kind === "replay";
  $("#stop").hidden = !busy;
  $("#stop").disabled = false;
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
  $("#steps").innerHTML = (hasClean() ? [CLEAN, ...STEPS] : STEPS).map(([tool, label]) => `<div class="step ${state.used.has(tool) ? "on" : ""}"><i></i>${label}</div>`).join("");
}

const scrollDown = () => {
  if (!state.replaying && !document.body.classList.contains("show-work")) window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
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
  if (!probs) {
    return `<div class="bars">${results.map((x, i) => `<div class="bar-row"><div class="label">${caseLabel(x, args, i)}</div><div></div><div class="val">${fmt(x.prediction, 2)}</div></div>`).join("")}</div>`;
  }
  const classes = Object.keys(probs);
  const cls = positiveOf(classes);
  const first = probs[cls];
  return `<div class="split-label">P(${esc(cls)})</div><div class="bars">${results
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
