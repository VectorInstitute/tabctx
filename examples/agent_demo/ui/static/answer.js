// The answer as a brief for whoever asked: one section per step of the
// task, each a plain question ("Is the data usable?"), a one-sentence
// answer written from the run's own numbers, and one quiet visual. Then
// the agent's memo, and a check against what really happened. Every
// command and tool call stays in the work panel (app.js), which reads the
// same events. Helpers (esc, fmt, pct, tipAttr, base) are app.js's.

const Answer = (() => {
  let S = null; // this run

  // Each step: its question, and its short name on the progress line.
  const STEPS = {
    clean: ["Is the data usable?", "Tidy"],
    examples: ["Which past rows will it learn from?", "Choose"],
    evaluate: ["Can it be predicted?", "Check"],
    predict: ["Who is most at risk?", "Score"],
    explain: ["What drives it?", "Explain"],
    what_if: ["What could change it?", "Test"],
    report: ["The memo", "Write"],
  };
  const STEP_OF = { inspect_table: "evaluate", list_models: "evaluate", evaluate: "evaluate", fit_model: "predict", predict: "predict", explain: "explain", what_if: "what_if" };
  const DOING = {
    inspect_table: "Reading the clean table",
    list_models: "Seeing which models it can use",
    evaluate: "Testing on last year's rows it hasn't seen",
    fit_model: "Learning from the past rows",
    predict: "Scoring every row",
    explain: "Scrambling one column at a time",
    what_if: "Changing one thing at a time",
  };

  // ---- words ------------------------------------------------------------------
  const cols = () => S.d.column_names || {};
  const label = (c) => cols()[c]?.[0] || String(c).replace(/_/g, " ").replace(/^\w/, (x) => x.toUpperCase());
  const lower = (t) => t.charAt(0).toLowerCase() + t.slice(1);
  const num = (v) => {
    const n = Number(v);
    if (v === "" || v == null || Number.isNaN(n)) return String(v ?? "");
    return Math.abs(n) >= 100 ? fmt(n) : fmt(n, Number.isInteger(n) ? 0 : 1);
  };
  const value = (c, v) => (v === "" || v == null ? "unknown" : (cols()[c]?.[1] || "{}").replace("{}", () => num(v)));
  const said = (c, v) => `${label(c)} ${value(c, v)}`;
  const changed = (c, v) => `${label(c)} → ${value(c, v)}`;
  const list = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);
  const LOWER_IS_BETTER = new Set(["rmse", "mae", "log_loss"]);
  const words = (n) => ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] ?? fmt(n);

  // ---- the run ----------------------------------------------------------------
  function reset(setup, agent, replaying) {
    const d = setup.dataset || {};
    S = { d, agent, replaying, steps: new Map(), calls: new Map(), top: new Map(), people: 0, predicted: false, cleaned: false };
    // With a cleaning step, choosing the examples follows it: the clean
    // past rows are what the tabular model learns from.
    const keys = (d.steps || []).map(([k]) => k).flatMap((k) => (k === "clean" ? [k, "examples"] : [k])).filter((k) => STEPS[k]);
    $("#answer").innerHTML = `
      <ol class="progress" id="progress">${keys.map((k) => `<li data-step="${k}">${STEPS[k][1]}</li>`).join("")}</ol>
      <p class="status-line" id="agent-status"><span class="pulse-dot"></span>Working on it</p>
      <div id="sections"></div><div id="final"></div>`;
    keys.forEach((key, i) => {
      const el = document.createElement("section");
      el.className = "sec";
      el.hidden = true;
      el.innerHTML = `<div class="sec-num">${String(i + 1).padStart(2, "0")}</div><div class="sec-main"><h2>${STEPS[key][0]}</h2><p class="lede"></p><div class="viz"></div></div>`;
      $("#sections").appendChild(el);
      S.steps.set(key, { el, state: "idle" });
    });
  }

  function setStep(key, state, doing) {
    const step = S.steps.get(key);
    if (!step) return;
    if (step.state === "done" && state === "working") state = "done"; // more of a finished step
    step.state = state;
    step.el.hidden = state === "idle";
    step.el.dataset.state = state;
    $(`#progress [data-step="${key}"]`)?.setAttribute("data-state", state);
    const lede = step.el.querySelector(".lede");
    if (state === "working" && doing && lede && !lede.dataset.final) {
      lede.textContent = `${doing}…`;
      lede.classList.add("doing");
    }
  }
  // A step's answer: its sentence, and its visual (replaced, or with
  // append: added below what is there, which keeps its animations still).
  function write(key, sentence, viz, append = false) {
    const step = S.steps.get(key);
    if (!step) return;
    const lede = step.el.querySelector(".lede");
    lede.innerHTML = sentence;
    lede.dataset.final = "1";
    lede.classList.remove("doing");
    const box = step.el.querySelector(".viz");
    if (viz != null) append ? box.insertAdjacentHTML("beforeend", viz) : (box.innerHTML = viz);
    setStep(key, "done");
  }
  const has = (key) => S.steps.has(key);

  function doingOf(tool, a = {}) {
    if (/^(write|edit)/i.test(tool)) return `Writing ${base(a.file_path || a.path || "a script")}`;
    if (/clean\.py/.test(a.command || "")) return "Running its cleaning script";
    return "Looking through the raw files";
  }

  function event(ev) {
    if (!S) return;
    if (ev.type === "tool_call") {
      S.calls.set(ev.id, ev);
      if (!ev.tabctx) {
        const path = ev.args?.file_path || ev.args?.path || "";
        if (/report\.md$/.test(path)) {
          for (const [k, s] of S.steps) if (k !== "report" && s.state === "working") setStep(k, "done");
          return setStep("report", "working", "Writing the memo");
        }
        if (has("clean") && !S.cleaned) setStep("clean", "working", doingOf(ev.tool, ev.args));
        return;
      }
      if (has("clean") && !S.cleaned) loadCleaning();
      // Later predicts (a program tried on many rows) belong to the what-ifs.
      const step = ev.tool === "predict" && S.predicted ? "what_if" : STEP_OF[ev.tool];
      if (step) setStep(step, "working", DOING[ev.tool]);
      return;
    }
    if (ev.type !== "tool_result") return;
    const call = S.calls.get(ev.id);
    if (!call) return;
    if (!call.tabctx) {
      if (has("clean") && !S.cleaned && !ev.error && /clean\.py/.test(call.args?.command || "")) loadCleaning();
      return;
    }
    const r = ev.result;
    if (ev.error || !r) return;
    if (call.tool === "evaluate") evaluate(r);
    else if (call.tool === "predict" && !S.predicted) predict(r, call.args);
    else if (call.tool === "explain") explain(r);
    else if (call.tool === "what_if") whatIf(r, call.args);
  }

  // ---- 01 · is the data usable? -------------------------------------------------
  async function loadCleaning() {
    const run = S;
    S.cleaned = true;
    setStep("clean", "working", "Setting the raw rows beside the clean ones");
    let v = null;
    try {
      const res = await fetch(`/api/runs/${S.agent}/cleaning`);
      if (res.ok) v = await res.json();
    } catch {}
    if (S !== run) return; // another run started meanwhile
    if (!v) {
      write("clean", "The agent cleaned both exports with a script of its own.", "");
      return write("examples", `It learns from the clean table of ${esc(S.d.history || "past rows")}, one row per ${esc(S.d.entity || "record")}.`, "");
    }
    const removed = (v.removed || []).map((x) => `${fmt(x.n)} ${x.what.replace(/ \(.*\)$/, "")}`);
    const top = (v.columns || []).filter((c) => c.kind === "category" && c.before > c.after).sort((a, b) => b.before - a.before)[0];
    const sentence =
      `Not as exported. The file had ${fmt(v.rows_before)} rows; <b>${fmt(v.rows_after)}</b> are distinct ${esc(S.d.entity || "records")}. ` +
      `The agent wrote a script that removed ${esc(list(removed))}` +
      (top ? `, and merged ${fmt(top.before)} spellings of ${esc(lower(label(top.clean)))} into ${fmt(top.after)}.` : ".");
    write("clean", sentence, cleanViz(v));
    const table = S.steps.get("clean").el.querySelector(".tidy-table");
    setTimeout(() => S === run && flip(table, true), 2200); // the mess first, then the fix
    examples(v);
  }

  // ---- 02 · which past rows will it learn from? -----------------------------------
  // The cohort: who counts once, what counts as the outcome (the export's
  // spellings, read one way or the other), and what each row says.
  function examples(v) {
    const e = v.examples;
    if (!e) return write("examples", `It learns from ${fmt(v.rows_after)} past ${esc(S.d.entity || "rows")}, each counted once.`, "");
    const classes = e.outcome?.classes || [];
    const pos = classes.find((c) => c.value === String(S.d.positive));
    const rest = classes.filter((c) => c !== pos);
    const said = classes.reduce((n, c) => n + c.spellings.length, 0);
    const outcome = S.d.outcome || "the outcome";
    const sentence =
      `It learns from <b>${fmt(e.rows)}</b> past ${esc(S.d.entity || "rows")}, each counted once` +
      (pos ? `: <b>${fmt(pos.n)}</b> ${esc(outcome)}, ${fmt(rest.reduce((n, c) => n + c.n, 0))} didn't.` : ".") +
      (pos && said > classes.length ? ` The export wrote the outcome ${words(said)} ways; the agent read ${esc(list(pos.spellings.map((x) => `“${x}”`)))} as ${esc(outcome)}.` : "");
    write("examples", sentence, examplesViz(v, e, pos, rest));
  }

  function examplesViz(v, e, pos, rest) {
    const chips = (xs) => xs.map((x) => `<span class="chip">${esc(x).replace(/^ | $/g, "␣")}</span>`).join("");
    const removed = (v.removed || []).map((x) => `<span class="chip">−${fmt(x.n)} ${esc(x.what.replace(/ \(.*\)$/, ""))}</span>`).join("");
    const total = (pos?.n || 0) + rest.reduce((n, c) => n + c.n, 0);
    const outcome = pos
      ? `<div class="ex-tile">
          <div class="ex-k">What counts as ${esc(S.d.outcome || "the outcome")}</div>
          <div class="ex-map"><span class="ex-from">${chips(pos.spellings)}</span><span class="arrow">→</span><span class="ex-to pos">${esc(S.d.outcome || pos.value)}</span>
            <span class="ex-from">${chips(rest.flatMap((c) => c.spellings))}</span><span class="arrow">→</span><span class="ex-to">not ${esc(S.d.outcome || "")}</span></div>
          <div class="ex-bar" ${tipAttr(`${fmt(pos.n)} of ${fmt(total)}`)}><i style="width:${(pos.n / total) * 100}%"></i></div>
          <div class="ex-legend"><span>${fmt(pos.n)} ${esc(S.d.outcome || "")}</span><span>${fmt(total - pos.n)} not ${esc(S.d.outcome || "")}</span></div>
        </div>`
      : "";
    return `<div class="ex">
        <div class="ex-tile">
          <div class="ex-k">Who counts</div>
          <div class="ex-big">${v.rows_before > e.rows ? `<s>${fmt(v.rows_before)}</s>` : ""}${fmt(e.rows)}</div>
          <div class="ex-chips">${removed}</div>
        </div>
        ${outcome}
        <div class="ex-tile">
          <div class="ex-k">What each row says</div>
          <div class="ex-big">${fmt(e.features)}</div>
          <div class="ex-sub">columns each, not counting the ID</div>
        </div>
      </div>`;
  }

  function flip(table, tidy) {
    if (!table) return;
    table.classList.toggle("tidy", tidy);
    table.closest(".viz").querySelector(".flip").textContent = tidy ? "Show as exported" : "Show cleaned";
  }

  function cleanViz(v) {
    const cell = (raw, clean) => {
      const fixed = raw !== clean && !(raw !== "" && clean !== "" && Number(raw) === Number(clean));
      const shownRaw = esc(raw).replace(/^ | $/g, "␣") || "&nbsp;";
      return `<td class="${fixed ? "fixed" : ""}"><span class="cell"><span class="raw">${shownRaw}</span><span class="clean">${esc(clean) || '<i class="blank">—</i>'}</span></span></td>`;
    };
    const head = v.sample.columns.map(([raw, clean]) => `<th><span class="cell"><span class="raw">${esc(raw)}</span><span class="clean">${esc(label(clean))}</span></span></th>`).join("");
    const rows = v.sample.rows.map((row) => `<tr>${row.map(([raw, clean]) => cell(raw, clean)).join("")}</tr>`).join("");
    const merges = v.columns
      .filter((c) => c.kind === "category" && c.before > c.after)
      .sort((a, b) => b.before - a.before)
      .slice(0, 3)
      .map(
        (c) => `<div class="merge"><span class="merge-name">${esc(label(c.clean))}</span>
          <span class="spellings">${c.example.spellings.map((s) => `<s>${esc(s).replace(/^ | $/g, "␣")}</s>`).join("")}</span>
          <span class="arrow">→</span><span class="merged">${esc(c.example.value)}</span></div>`,
      )
      .join("");
    return `<div class="tidy-scroll"><table class="tidy-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      <div class="viz-foot"><span>Five of the messiest rows</span><button class="link-btn flip" type="button">Show cleaned</button></div>
      <div class="merges">${merges}</div>`;
  }

  // ---- 03 · can it be predicted? -------------------------------------------------
  function evaluate(r) {
    const m = r.best_by;
    const ok = (r.results || []).filter((x) => !x.error && x.metrics?.[m] != null);
    const gbm = ok.find((x) => x.model === "gradient-boosting");
    const better = LOWER_IS_BETTER.has(m) ? (a, b) => a.metrics[m] - b.metrics[m] : (a, b) => b.metrics[m] - a.metrics[m];
    const tfm = ok.filter((x) => x.model !== "gradient-boosting").sort(better)[0];
    if (!tfm) return write("evaluate", "The models couldn't be compared on this table.", "");
    if (m !== "roc_auc") {
      return write("evaluate", `On last year's rows it hadn't seen, the tabular model scores ${fmt(tfm.metrics[m], 3)} (${esc(metricName(m))})${gbm ? `, against ${fmt(gbm.metrics[m], 3)} for the standard approach` : ""}.`, "");
    }
    const n = (x) => Math.round(x.metrics[m] * 100);
    const verdict = n(tfm) >= 75 ? "Yes." : n(tfm) >= 65 ? "Fairly well." : "Only somewhat.";
    const sentence =
      `${verdict} Shown one who ${esc(S.d.outcome)} and one who didn't, it picks right <b>${n(tfm)} times in 100</b>, on ${fmt(r.test_rows)} of last year's rows it hadn't seen.` +
      (gbm ? ` A model built for this table the standard way gets ${n(gbm)}.` : "");
    const x = (v) => `${Math.max(0, (v - 0.5) / 0.5) * 100}%`;
    const row = (res, name, sub, cls) =>
      `<div class="acc ${cls}"><div class="acc-name">${name}<small>${sub}</small></div><div class="acc-track"><div class="acc-fill" style="width:${x(res.metrics[m])}"></div></div><div class="acc-n">${n(res)}</div></div>`;
    const viz =
      row(tfm, "Tabular foundation model", `${esc(tfm.model)} · nothing trained on your data`, "main") +
      (gbm ? row(gbm, "Standard approach", "gradient boosting, built for this table", "base") : "") +
      `<div class="acc-axis"><span>50 · a coin flip</span><span>100 · always right</span></div>`;
    write("evaluate", sentence, viz);
  }

  // ---- 04 · who is most at risk? -------------------------------------------------
  function predict(r, args = {}) {
    S.predicted = true;
    const key = r.ranked_by || "";
    const proba = key.startsWith("p_"); // a chance; else a predicted value
    const pos = S.d.positive || "Yes";
    const mean = r.summary?.mean_probability?.[pos];
    const top = r.top_rows || [];
    const id = Object.keys(top[0] || {})[0]; // the id comes first
    const show = (args.show || []).filter((c) => c !== id).slice(0, 2);
    S.show = show;
    const sentence =
      (mean != null ? `About <b>${fmt(Math.round(mean * r.rows))}</b> of the ${fmt(r.rows)} ${esc(S.d.current)} are likely ${esc(S.d.outcome)}. ` : `The agent scored all ${fmt(r.rows)} ${esc(S.d.current)}. `) +
      (top.length ? `These ${words(top.length)} are ${proba ? "the most likely" : "at the top"}.` : "");
    const rows = top
      .map((row, i) => {
        const p = Number(row[key]);
        S.top.set(String(row[id]), row);
        const score = proba && Number.isFinite(p)
          ? `<span class="risk-track"><span class="risk-fill" style="width:${Math.max(0, Math.min(1, p)) * 100}%"></span></span><b>${pct(p)}</b>`
          : `<b>${esc(num(row[key]))}</b>`;
        return `<tr style="--i:${i}"><td class="rank">${i + 1}</td><td class="id">${esc(row[id])}</td>
          <td class="risk">${score}</td>
          <td class="why">${esc(show.filter((c) => c in row).map((c) => said(c, row[c])).join(" · "))}</td></tr>`;
      })
      .join("");
    write("predict", sentence, rows ? `<table class="ranked"><tbody>${rows}</tbody></table>` : "");
  }

  // ---- 05 · what drives it? -------------------------------------------------------
  function explain(r) {
    const top = (r.importance || []).filter((i) => i.score_drop > 0).slice(0, 6);
    if (!top.length) return write("explain", "No single column stands out.", "");
    const max = top[0].score_drop;
    const names = top.map((i) => lower(label(i.column)));
    const sentence = `Mostly <b>${esc(names[0])}</b>${names.length > 1 ? `, then ${esc(list(names.slice(1, 3)))}` : ""}. That is what the predictions lean on, not proof of what causes it.`;
    const viz = top
      .map(
        (i, k) => `<div class="drv ${k < 3 ? "main" : ""}"><span>${esc(label(i.column))}</span>
          <div class="drv-track" ${tipAttr(`Scrambling <b>${esc(label(i.column))}</b> makes the predictions worse by ${fmt(i.score_drop, 3)}`)}><div class="drv-fill" style="width:${(i.score_drop / max) * 100}%"></div></div></div>`,
      )
      .join("");
    write("explain", sentence, viz);
  }

  // ---- 06 · what could change it? --------------------------------------------------
  function whatIf(r, args = {}) {
    const pos = S.d.positive || "Yes";
    const res = r.results || [];
    // The first row is the case as it is; a regression what-if has no chance to move.
    if (!has("what_if") || res.length < 2 || res[0].probabilities?.[pos] == null) return;
    S.people++;
    const today = res[0].probabilities[pos];
    const id = args.base_where ? Object.values(args.base_where).join(", ") : "";
    const known = S.top.get(String(id));
    const facts = known ? (S.show || []).filter((c) => c in known).map((c) => said(c, known[c])).join(" · ") : "";
    const at = (p) => `${Math.max(0, Math.min(1, p)) * 100}%`;
    const tries = res
      .map((x, i) => [x, i])
      .slice(1)
      .filter(([x]) => x.probabilities?.[pos] != null)
      .map(([x, i]) => {
        const p = x.probabilities[pos];
        // The change itself; args.rows are whole rows unless based on one.
        const change = x.changes ?? (args.base_where ? args.rows?.[i] : null) ?? {};
        const what = Object.entries(change).map(([c, v]) => changed(c, v)).join(" + ") || "As it is";
        const dir = p < today ? "down" : "up";
        const lo = Math.min(p, today), hi = Math.max(p, today);
        return `<div class="try"><span class="try-what">${esc(what)}</span>
          <span class="try-track"><span class="try-span ${dir}" style="left:${at(lo)};width:calc(${at(hi)} - ${at(lo)})"></span><i class="now" style="left:${at(today)}"></i><i class="${dir}" style="left:${at(p)}" ${tipAttr(`${pct(today)} → ${pct(p)}`)}></i></span>
          <b class="${dir}">${pct(p)}</b></div>`;
      })
      .join("");
    const person = `<div class="person"><div class="person-head"><span class="id">${esc(id)}</span><span class="facts">${esc(facts)}</span><span class="today">today <b>${pct(today)}</b></span></div>${tries}</div>`;
    const who = S.people === 1 ? `one of the ${esc(S.d.entity || "rows")}` : `${words(S.people)} ${esc(S.d.entity || "rows")}`;
    write("what_if", `For ${who} most at risk, the agent changed one thing at a time and asked the model again. These are patterns the model learned, worth a pilot, not proven effects.`, person, true);
  }

  // ---- the memo and the check -------------------------------------------------------
  // Steps the agent started but never answered say so, rather than look done.
  function unanswered(note) {
    for (const [k, s] of S.steps) {
      const lede = s.el.querySelector(".lede");
      if (s.state === "idle" || (lede && lede.dataset.final)) continue;
      if (k === "report" && !lede) continue;
      if (lede) {
        lede.textContent = note;
        lede.classList.remove("doing");
      }
      setStep(k, "missing");
    }
  }

  function fail(text) {
    if (!S) return;
    unanswered("The agent didn't get to an answer here.");
    $("#agent-status").textContent = text;
  }

  function outcome(o, stats) {
    if (!S) return;
    if (o.report && has("report")) {
      const step = S.steps.get("report");
      step.el.querySelector(".lede")?.remove();
      step.el.querySelector(".viz").innerHTML = memo(o.report);
      setStep("report", "done");
    }
    unanswered("The agent didn't get to an answer here.");
    const secs = stats?.seconds;
    $("#agent-status").innerHTML = o.report
      ? `Answered${secs ? ` in ${Math.floor(secs / 60)} min ${Math.round(secs % 60)} s` : ""}, from one question and two raw files.`
      : "The agent stopped without an answer.";
    $("#final").innerHTML = check(o.score || {});
  }

  function memo(md) {
    // The memo's summary up front; the rest folded.
    const lines = md.split("\n").filter((l) => !l.startsWith("# "));
    const parts = lines.join("\n").split(/\n(?=## )/);
    const lead = parts.findIndex((p) => p.startsWith("## "));
    const head = lead >= 0 ? parts.slice(0, lead + 1).join("\n") : parts[0];
    const more = lead >= 0 ? parts.slice(lead + 1).join("\n") : "";
    const html = (m) => DOMPurify.sanitize(marked.parse(m));
    return `<article class="memo">${html(head)}
      ${more.trim() ? `<details class="memo-more"><summary>Read the whole memo</summary>${html(more)}</details>` : ""}</article>`;
  }

  function check(s) {
    if (s.error || s.positive_in_top_10 == null) return "";
    const what = S.d.outcome || "positive";
    const hits = Math.round(s.positive_in_top_10 * 10);
    const random = Math.max(1, Math.round(s.base_rate * 10));
    const dots = (n) => Array.from({ length: 10 }, (_, i) => `<i class="${i < n ? "hit" : ""}"></i>`).join("");
    return `<section class="sec check" data-state="done"><div class="sec-num">✓</div><div class="sec-main"><h2>Was it right?</h2>
      <p class="lede">The real outcomes were hidden from the agent. Of the ten it ranked highest, <b>${words(hits)} really ${esc(what)}</b>; ten picked at random would hold about ${words(random)}.</p>
      <div class="viz"><div class="tally"><span class="tally-name">Its top ten</span><span class="dots">${dots(hits)}</span><b>${hits}</b></div>
      <div class="tally base"><span class="tally-name">Ten at random</span><span class="dots">${dots(random)}</span><b>${random}</b></div>
      <p class="viz-note">Of its top hundred, ${pct(s.positive_in_top_100)} ${esc(what)}, against ${pct(s.base_rate)} of everyone.</p></div></div></section>`;
  }

  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".flip");
    if (!btn) return;
    const t = btn.closest(".viz").querySelector(".tidy-table");
    flip(t, !t.classList.contains("tidy"));
  });

  return { reset, event, outcome, fail };
})();
