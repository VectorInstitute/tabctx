// The agent's own commands (bash, file writes, edits) as pictures of what
// they do to the data, for a room watching on a big screen. Each command
// becomes a tile: a plain-English title, a small visual read from the
// command and its output, and the raw command one click away. Nothing here
// is guessed: a visual appears only when the command or its output says it.
// Consecutive checks of the data fold into one tile.

const Actions = (() => {
  const H = (html) => {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstChild;
  };
  const e = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const n = (v) => Number(v).toLocaleString();
  const fileOf = (p) => String(p || "").split("/").pop();
  const icon = (paths) =>
    `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;

  const ICON = {
    look: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    profile: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-5-5"/>',
    script: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4M9 13l-2 2 2 2M13 13l2 2-2 2"/>',
    edit: '<path d="M14 4l6 6M4 20l6-1 9-9-5-5-9 9z"/>',
    run: '<path d="M7 4l12 8-12 8z"/>',
    analyze: '<path d="M5 19V9M12 19V5M19 19v-7"/>',
    prepare: '<path d="M4 6h16v12H4zM4 10h16M10 10v8"/><path d="M17 3v4M15 5h4"/>',
    verify: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    report: '<path d="M6 3h9l4 4v14H6z"/><path d="M9 11h7M9 15h7M9 7h3"/>',
    other: '<path d="M4 17l6-5-6-5M12 19h8"/>',
  };

  // Columns of the raw export, so a check can show which ones it looked at.
  let columns = [];
  let last = null; // the tile a following check can fold into

  function commandOf(tool, a = {}) {
    const c = a.command ?? a.cmd ?? "";
    return String(Array.isArray(c) ? c.join(" ") : c).replace(/^\/bin\/\w+ -lc\s+/, "").replace(/^(['"])([\s\S]*)\1$/, "$2");
  }

  // What a command is, from the command alone.
  function classify(tool, a = {}) {
    const t = tool.toLowerCase();
    const path = a.file_path || a.path || "";
    if (/^(write|create|write_file)$/.test(t)) {
      if (/\.md$/i.test(path)) return { kind: "report", title: "Wrote the report" };
      return { kind: "script", title: /clean/i.test(path) ? "Wrote the cleaning script" : `Wrote ${fileOf(path)}` };
    }
    if (/^(edit|multiedit|str_replace|apply_patch|patch)$/.test(t)) return { kind: "edit", title: `Adjusted ${fileOf(path) || "a script"}` };
    if (/^(read|read_file|view|cat)$/.test(t)) return { kind: "look", title: `Read ${fileOf(path)}` };
    if (/^(ls|glob|grep|find|list)$/.test(t)) return { kind: "look", title: "Looked around the folder" };
    const cmd = commandOf(tool, a);
    if (!cmd) return { kind: "other", title: a.description || tool };
    if (/\bmd5|sha\d*sum|cmp\b|diff\b/.test(cmd)) return { kind: "verify", title: "Re-ran it to confirm nothing changed" };
    const script = cmd.match(/python3?\s+([\w./-]+\.py)/);
    if (script) return { kind: "run", title: /clean/i.test(script[1]) ? "Ran the cleaning" : `Ran ${fileOf(script[1])}`, file: script[1] };
    if (/python|pandas|read_csv|duckdb/.test(cmd)) {
      if (/to_csv|to_parquet/.test(cmd)) return { kind: "prepare", title: "Built test files for the what-ifs" };
      if (/_export\.|raw|\.xlsx/i.test(cmd)) return { kind: "profile", title: "Checked the raw data for problems" };
      return { kind: "analyze", title: "Checked a pattern in the clean data" };
    }
    if (/\b(head|wc|ls|cat|tail|file)\b/.test(cmd)) return { kind: "look", title: "Opened the files" };
    return { kind: "other", title: a.description || "Ran a command" };
  }

  // The columns a command names, in the export's own spelling.
  function columnsIn(text) {
    const found = [];
    for (const c of columns) if (text.includes(`"${c}"`) || text.includes(`'${c}'`)) found.push(c);
    return found;
  }

  const chip = (text, cls = "") => `<span class="a-chip ${cls}">${e(text)}</span>`;
  const sheet = (name, rows) =>
    `<span class="a-file"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 5h16v14H4zM4 10h16M10 10v9"/></svg><b>${e(name)}</b>${rows != null ? `<span>${n(rows)} rows</span>` : ""}</span>`;

  // Many spellings becoming a few: n grey dots, an arrow, k solid dots.
  function collapse(label, from, to) {
    const dots = (k, cls) => Array.from({ length: Math.min(k, 40) }, () => `<i class="${cls}"></i>`).join("");
    return `<div class="a-col"><span class="a-col-name">${e(label)}</span><span class="a-dots">${dots(from, "raw")}</span><span class="a-to">${from}&nbsp;&rarr;&nbsp;${to}</span><span class="a-dots">${dots(to, "neat")}</span></div>`;
  }

  // What a command did, from its output. Each piece only if the output says it.
  function picture(kind, call, text, a = {}) {
    const out = String(text || "");
    if (kind === "look") {
      const files = [...out.matchAll(/^\s*(\d+)\s+(\S+\.csv)\s*$/gm)].map(([, k, f]) => [f, Number(k) - 1]);
      if (files.length) return `<div class="a-files">${files.map(([f, k]) => sheet(f, k)).join("")}</div>`;
      const named = [...new Set(commandOf("", a).match(/[\w.-]+\.csv/g) || [])];
      return named.length ? `<div class="a-files">${named.map((f) => sheet(f)).join("")}</div>` : "";
    }
    if (kind === "profile") {
      const cols = columnsIn(commandOf("", a));
      const finds = [];
      const dup = out.match(/dup\w*[^\d\n]{0,24}(\d+)/i);
      if (dup && +dup[1]) finds.push(chip(`${n(dup[1])} duplicate rows`, "warn"));
      if (/\bTOTAL\b/.test(out)) finds.push(chip("a totals row", "warn"));
      if (/\bTEST-\d+/.test(out)) finds.push(chip("test policies", "warn"));
      // A column with a handful of values written many ways: one bar each.
      const spell = new Map();
      for (const [, c, k] of out.matchAll(/--\s*([\w ]+?)\s+nunique\s+(\d+)/g)) if (+k > 2 && +k < 60) spell.set(c, Math.max(+k, spell.get(c) || 0));
      const many = [...spell].sort((x, y) => y[1] - x[1]);
      const bars = many.length
        ? `<div class="a-spell"><span class="a-spell-k">${many.length} columns written many ways</span>${many
            .slice(0, 6)
            .map(([c, k]) => `<span class="a-spell-row"><span>${e(c)}</span><i style="width:${Math.min(k, 40) * 2.5}%"></i><b>${k}</b></span>`)
            .join("")}</div>`
        : "";
      const named = cols.length && !many.length ? `<div class="a-cols">${cols.map((c) => chip(c, "cname")).join("")}</div>` : "";
      return (finds.length ? `<div class="a-finds">${finds.join("")}</div>` : "") + bars + named;
    }
    if (kind === "script") {
      const src = String(a.content || "");
      const ops = [
        [/drop_duplicates/, "remove copies"],
        [/TEST|startswith\(["']TEST/, "drop test rows"],
        [/TOTAL/, "drop the totals row"],
        [/to_datetime|strptime|dateutil/, "read every date format"],
        [/to_numeric|float\(/, "numbers out of text"],
        [/spelling|\.map\(|replace\(/, "one spelling per value"],
        [/n\/a|placeholder|unknown/i, "placeholders to empty"],
        [/years_as_customer|tenure/, "years as customer"],
        [/snake|rename\(|\.lower\(\)/, "tidy column names"],
      ].filter(([re]) => re.test(src));
      const lines = src ? src.split("\n").length : 0;
      return `<div class="a-ops">${ops.map(([, l]) => chip(l, "op")).join("")}${lines ? chip(`${lines} lines`, "faint") : ""}</div>`;
    }
    if (kind === "edit") {
      const minus = String(a.old_string || "").split("\n").length;
      const plus = String(a.new_string || "").split("\n").length;
      return a.old_string != null ? `<div class="a-ops">${chip(`−${minus} line${minus > 1 ? "s" : ""}`, "minus")}${chip(`+${plus} line${plus > 1 ? "s" : ""}`, "plus")}</div>` : "";
    }
    if (kind === "run") {
      // One file's block: "N rows read" ... "M policies written".
      const first = out.split(/\n\s*\n/)[0];
      const read = first.match(/(\d[\d,]*)\s+rows read/);
      const wrote = first.match(/(\d[\d,]*)\s+\w+\s+written/);
      const rows = read && wrote ? `<div class="a-rows"><b>${n(read[1].replace(/,/g, ""))}</b><span>&rarr;</span><b>${n(wrote[1].replace(/,/g, ""))}</b><span class="faint">rows</span></div>` : "";
      const drops = [...first.matchAll(/dropped\s+(\d+)\s+([^(\n]+)/g)].map(([, k, what]) => chip(`−${k} ${what.replace(/exact |summary |['"]/g, "").trim()}`, "minus"));
      const cols = [...first.matchAll(/^\s*-\s*(\w+):\s*(\d+)\s+spellings\s*->\s*\[([^\]]*)\]/gm)].map(([, c, k, to]) => collapse(c, +k, to.split(",").length));
      const fixes = [...first.matchAll(/^\s*-\s*(?:(\w+):\s*)?(\d+)\s+(placeholder|impossible) values/gm)].map(([, col, k, w]) => chip(w === "impossible" ? `${k} impossible ${col || "value"}s emptied` : `${k} placeholders emptied`, "op"));
      const dates = /dates in (\d+) formats/.exec(first);
      if (dates) fixes.push(chip(`${dates[1]} date formats read`, "op"));
      if (!rows && !drops.length && !cols.length) return "";
      return `${rows}${drops.length ? `<div class="a-finds">${drops.join("")}</div>` : ""}${cols.length ? `<div class="a-collapse">${cols.slice(0, 6).join("")}</div>` : ""}${fixes.length ? `<div class="a-ops">${fixes.join("")}</div>` : ""}`;
    }
    if (kind === "analyze") {
      const cols = [...new Set([...commandOf("", a).matchAll(/["']([a-z]+(?:_[a-z0-9]+)+)["']|\.([a-z]+_[a-z0-9_]+)\b/g)].map((m) => m[1] || m[2]))].filter((c) => !/^(to_|read_|value_|drop_|reset_|set_|sort_|is_)/.test(c));
      return cols.length ? `<div class="a-cols">${cols.slice(0, 5).map((c) => chip(c, "cname")).join("")}</div>` : "";
    }
    if (kind === "verify") return /^[0-9a-f]{32}\s*$/m.test(out) ? `<div class="a-ops">${chip("same output", "plus")}</div>` : "";
    if (kind === "report") {
      const heads = (String(a.content || "").match(/^#{2}\s+.+$/gm) || []).map((h) => h.replace(/^#+\s+/, ""));
      return heads.length ? `<div class="a-ops">${heads.slice(0, 5).map((h) => chip(h, "faint")).join("")}</div>` : "";
    }
    return "";
  }

  function rawOf(tool, a = {}) {
    if (a.content != null) return `${a.file_path || a.path || ""}\n\n${a.content}`;
    if (a.old_string != null) return `${a.file_path || ""}\n- ${a.old_string}\n+ ${a.new_string}`;
    return commandOf(tool, a) || JSON.stringify(a, null, 2);
  }

  // A tile for one command, or the previous check's tile when this is another check.
  function add(timeline, tool, args) {
    const c = classify(tool, args || {});
    // The agent's notes in between don't break a run of checks; the
    // folded tile moves below them, where the eye is.
    let prev = timeline.lastElementChild;
    while (prev?.classList.contains("msg")) prev = prev.previousElementSibling;
    if (c.kind === "profile" && last?.dataset.kind === "profile" && last === prev) {
      const k = Number(last.dataset.count) + 1;
      last.dataset.count = k;
      last.querySelector(".a-count").textContent = `× ${k}`;
      last.querySelector(".a-cmd").textContent += `\n\n${rawOf(tool, args || {})}`;
      last.classList.remove("done");
      timeline.appendChild(last);
      return { el: last, kind: c.kind, merged: true };
    }
    const el = H(`<div class="act k-${c.kind}" data-kind="${c.kind}" data-count="1">
      <span class="a-icon">${icon(ICON[c.kind])}</span>
      <div class="a-main"><div class="a-title">${e(c.title)}<span class="a-count"></span><span class="a-spin"></span></div><div class="a-viz"></div></div>
      <button class="a-raw" type="button" aria-label="Show the command" title="Show the command">${icon('<path d="M9 6l6 6-6 6"/>')}</button>
      <pre class="a-cmd" hidden>${e(rawOf(tool, args || {}))}</pre></div>`);
    el.querySelector(".a-raw").onclick = () => {
      const pre = el.querySelector(".a-cmd");
      pre.hidden = !pre.hidden;
      el.classList.toggle("open", !pre.hidden);
    };
    // A write shows what it does at once; a command once its output is in.
    const now = picture(c.kind, null, "", args || {});
    if (now && ["script", "edit", "report"].includes(c.kind)) el.querySelector(".a-viz").innerHTML = now;
    timeline.appendChild(el);
    last = el;
    return { el, kind: c.kind, merged: false };
  }

  function result(entry, tool, args, text, error) {
    const el = entry.el;
    el.classList.add("done");
    if (error) {
      el.classList.add("err");
      return;
    }
    if (["script", "edit", "report"].includes(entry.kind)) return;
    // A folded tile draws once from everything its checks printed.
    el._out = `${el._out || ""}\n${text || ""}`;
    el._cmd = `${el._cmd || ""}\n${commandOf(tool, args || {})}`;
    const viz = picture(entry.kind, null, el._out, { ...(args || {}), command: el._cmd });
    if (viz) el.querySelector(".a-viz").innerHTML = viz;
  }

  return {
    reset(setup) {
      columns = [...(setup?.dataset?.columns || []), "Policy ID"];
      last = null;
    },
    add,
    result,
    classify,
    picture: (kind, text, args) => picture(kind, null, text, args),
  };
})();
if (typeof module !== "undefined") module.exports = Actions; // the unit tests run it under node
