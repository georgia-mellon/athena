// Athena dashboard: one WebSocket, every event is {topic, t, data}. No build step, no network beyond this server.
"use strict";
const $ = (id) => document.getElementById(id);
const MAXC = 20;               // characters shown per keyboard row (the threat engine's 20-keystroke window)
const SPARK_S = 60;            // sparkline window, seconds
const TL_MAX = 4000;           // timeline points kept
const LOG_MAX = 200;
const QUIET = new Set(["keys.stroke", "voice.window", "threat.update", "keys.readout", "voice.verdict", "secret.state",
                       "meet.state", "audio.level", "system.state", "keyguard.move"]);

const st = { voice: [], threat: [], recent: [], typed: 0, vclass: null, alarmTimer: 0, gate: -45 };

// ---------- theme ----------
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  $("theme").textContent = t === "dark" ? "Light" : "Dark";
  try { localStorage.setItem("cg-theme", t); } catch (e) {}
}
try { setTheme(localStorage.getItem("cg-theme") || "dark"); } catch (e) { setTheme("dark"); }
$("theme").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

// ---------- helpers ----------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pct = (x) => (x == null || isNaN(x) ? "–" : Math.round(100 * x) + "%");
const hhmmss = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
function log(t, text, cls) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="tm">${hhmmss(t)}</span>`;
  li.appendChild(document.createTextNode(text));
  if (cls) li.className = cls;
  const ol = $("log");
  ol.prepend(li);
  while (ol.children.length > LOG_MAX) ol.lastChild.remove();
}
function alarm(text) {
  const a = $("alarm");
  a.textContent = text; a.hidden = false;
  clearTimeout(st.alarmTimer);
  st.alarmTimer = setTimeout(() => (a.hidden = true), 15000);
}

// ---------- renderers ----------
const LEVEL_COLOR = { SAFE: "--safe", WATCH: "--watch", WARN: "--warn", CRITICAL: "--crit" };
function renderThreat(d, t) {
  const s = clamp(Number(d.score) || 0, 0, 100);
  const lv = LEVEL_COLOR[d.level] ? d.level : "SAFE";
  $("g-num").textContent = Math.round(s);
  const arc = $("g-arc");
  arc.setAttribute("stroke-dasharray", `${s} 100`);
  arc.style.stroke = `var(${LEVEL_COLOR[lv]})`;
  const el = $("level"); el.textContent = lv; el.className = "level lv-" + lv;
  const ul = $("reasons"); ul.textContent = "";
  const rs = d.reasons && d.reasons.length ? d.reasons : ["No active risk"];
  for (const r of rs) { const li = document.createElement("li"); li.textContent = r; ul.appendChild(li); }
  st.threat.push([t, s]);
  if (st.threat.length > TL_MAX) st.threat.splice(0, st.threat.length - TL_MAX);
  renderTimeline();
}
function renderTimeline() {
  const pts = st.threat;
  if (!pts.length) return;
  const t0 = pts[0][0], span = Math.max(60, pts[pts.length - 1][0] - t0);
  $("tl-line").setAttribute("points", pts.map(([t, s]) => `${(1000 * (t - t0) / span).toFixed(1)},${(100 - s).toFixed(1)}`).join(" "));
}

function voiceClass(p) { return p >= 0.5 ? "synthetic" : p >= 0.25 ? "unverified" : "real"; }
const VOICE_TEXT = { real: "human", unverified: "uncertain", synthetic: "AI voice" };
// The verdict and the curve are the mean of the last SMOOTH scored windows: one odd 4 s window doesn't flip the
// verdict or spike the graph. Silence doesn't move it (it only changes on speech); a new speaker starts over
// (voice.flush: the Flush button, the test room's clip switch, or a long gap).
const SMOOTH = 3;
function renderVoice(d, t) {
  const raw = clamp(Number(d.p_synthetic) || 0, 0, 1);
  st.recent = [...st.recent, [t, raw]].slice(-SMOOTH);
  const p = st.recent.reduce((a, [, x]) => a + x, 0) / st.recent.length;
  const c = voiceClass(p);
  $("vlight").className = "vlight " + c;
  $("vlabel").textContent = VOICE_TEXT[c];
  $("vp").textContent = p.toFixed(2);
  $("vp").title = `this window ${raw.toFixed(2)}, mean of the last ${st.recent.length}`;
  $("vlat").textContent = d.latency_ms == null ? "–" : Math.round(d.latency_ms) + " ms";
  if (c !== st.vclass) { log(t, `voice → ${VOICE_TEXT[c]} (AI score ${p.toFixed(2)})`, c === "synthetic" ? "alert-c" : ""); st.vclass = c; }
  st.voice.push([t, p]);
  renderSpark();
}
function renderSpark() {
  const now = Date.now() / 1000;
  st.voice = st.voice.filter(([t]) => t > now - SPARK_S - 5);
  $("spark-line").setAttribute("points",
    st.voice.map(([t, p]) => `${(600 * (t - (now - SPARK_S)) / SPARK_S).toFixed(1)},${(120 * (1 - p)).toFixed(1)}`).join(" "));
}
setInterval(renderSpark, 1000);  // keep the window sliding while the far end is silent
const MEET_EVENT = { joined: "joined the call (Meet)", left: "left the call (Meet)", meet_open: "Meet tab connected (not in a call)",
                     meet_closed: "Meet tab closed or disconnected", test_room: "test room connected" };
function renderFlush() {
  st.recent = []; st.voice = []; st.vclass = null;
  $("vlight").className = "vlight none"; $("vlabel").textContent = "listening…"; $("vp").textContent = "–";
  renderSpark();
}

// ---------- audio in: levels, the speech gate, what happened to the last window ----------
const DB_MIN = -80, DB_MAX = -10;
const dbPct = (db) => (100 * (clamp(db, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)).toFixed(1) + "%";
function renderLevels(d) {
  for (const k of ["mic", "far"]) {
    const db = d[k + "_db"];
    $("lv-" + k).style.width = db == null ? "0" : dbPct(db);
    $("lv-" + k).classList.toggle("hot", k === "far" && db != null && db > st.gate);
    $("lv-" + k + "-n").textContent = db == null ? "–" : Math.round(db) + " dB";
  }
  if (d.far_db != null && !st.recent.length && !st.vclass) $("vlabel").textContent = "hearing the caller…";
}
function renderGate(db) {
  st.gate = db;
  $("gate-line").style.left = dbPct(db);
  $("gate-n").textContent = Math.round(db) + " dB";
  if (!st.gateDragging) $("gate").value = db;
}
function renderWindow(d) {
  const sp = Math.round(100 * (Number(d.speech) || 0));
  $("vwin").textContent = d.scored ? "last window " + sp + "% speech: judged"
    : "last window " + sp + "% speech: below the gate, not judged (needs 50%)";
}
$("gate").oninput = () => { st.gateDragging = true; renderGate(Number($("gate").value)); };
$("gate").onchange = () => { st.gateDragging = false; post("/api/control/voice/threshold", { db: Number($("gate").value) }); };
function renderSystem(d) {
  st.ready = !!d.ready;
  if (d.voice_source) {
    for (const b of document.querySelectorAll("[data-src]")) b.classList.toggle("on", b.dataset.src === d.voice_source);
    $("voice-h").textContent = d.voice_source === "mic" ? "Your mic (test)" : "Caller voice";
  }
  if (d.speech_db != null) renderGate(Number(d.speech_db));
  renderConn();
}
function renderConn() {
  const c = $("conn");
  if (!st.live) { c.textContent = st.everLive ? "reconnecting…" : "connecting…"; c.className = "pill bad"; return; }
  c.textContent = st.ready ? "ready" : "starting…";
  c.className = "pill " + (st.ready ? "ok" : "wait");
  c.title = st.ready ? "models loaded and warmed up" : "loading models";
}
renderGate(-45);

function renderStroke() {
  st.typed = Math.min(st.typed + 1, MAXC);
  $("typed").textContent = "•".repeat(st.typed);
}
function readoutRow(el, guesses) {
  el.textContent = "";
  for (const g of (guesses || []).slice(-MAXC)) {
    const s = document.createElement("span");
    s.textContent = g.top1 == null || g.top1 === "" ? "?" : String(g.top1);
    if (g.exact != null) s.className = g.exact ? "ok-c" : g.hit ? "near-c" : "bad-c";
    if (g.hit && !g.exact) s.title = "true key in the attacker's top 3";
    if (g.p != null) s.title = `p=${Number(g.p).toFixed(2)}`;
    el.appendChild(s);
  }
}
function renderReadout(d) {
  readoutRow($("raw"), d.raw);
  readoutRow($("shd"), d.shielded);
  const ch = clamp(Number(d.chance) || 0, 0, 1);
  $("acc-raw").style.width = pct(clamp(d.acc_raw || 0, 0, 1));
  $("acc-shd").style.width = pct(clamp(d.acc_shielded || 0, 0, 1));
  $("acc-raw-n").textContent = pct(d.acc_raw);
  $("acc-shd-n").textContent = pct(d.acc_shielded);
  $("chance1").style.left = $("chance2").style.left = pct(ch);
  $("chance-n").textContent = pct(ch);
  const n = Math.max((d.raw || []).length, (d.shielded || []).length);
  if (n) { st.typed = Math.min(n, MAXC); $("typed").textContent = "•".repeat(st.typed); }
}
const ARMED_BY = { voice: "unverified caller", request: "caller asked for a code", manual: "armed by you" };
function renderSecretState(d) {
  const pill = $("sec-pill");
  if (!d.enabled) { pill.textContent = "offline"; pill.className = "pill off"; $("sec-why").textContent = d.error || "disabled"; }
  else if (d.armed) { pill.textContent = "armed"; pill.className = "pill armed"; $("sec-why").textContent = ARMED_BY[d.armed_by] || ""; }
  else { pill.textContent = "standing by"; pill.className = "pill ok"; $("sec-why").textContent = "real caller: nothing is cut"; }
  if (d.allowed) $("sec-why").textContent += ` · allowing for ${Math.round(d.allow_s || 30)} s`;
  $("sec-delay").textContent = d.enabled ? Math.round(d.delay_ms) + " ms" : "–";
  for (const b of document.querySelectorAll("[data-sec]"))
    b.classList.toggle("on", (b.dataset.sec === "arm" && d.manual === true) || (b.dataset.sec === "disarm" && d.manual === false)
      || (b.dataset.sec === "auto" && d.manual == null) || (b.dataset.sec === "allow" && !!d.allowed));
}
function renderSecretBlocked(d, t) {
  const ol = $("sec-log");
  if (ol.firstElementChild && ol.firstElementChild.classList.contains("muted")) ol.textContent = "";
  const what = d.category === "digits" ? `${d.length}-digit code` : d.category;
  const li = document.createElement("li");
  const dots = document.createElement("span"); dots.className = "dots"; dots.textContent = "•".repeat(Math.min(d.length || 4, 16));
  li.appendChild(dots);
  li.appendChild(document.createTextNode(`${what} ${d.allowed ? "allowed through" : "blocked"} · ${hhmmss(t)}`));
  ol.prepend(li);
  while (ol.children.length > 6) ol.lastChild.remove();
}
function renderShield(mode) {
  for (const b of document.querySelectorAll("[data-mode]")) b.classList.toggle("on", b.dataset.mode === mode);
}
// outgoing latency: shield.state.latency_ms is the whole lag (Keyguard lookahead + secret delay line)
function renderPipeline() {
  const tot = st.shieldLat, sec = st.secretDelay || 0;
  $("pipe-lat").textContent = tot == null ? "–" : Math.round(tot) + " ms";
  $("pipe-split").textContent = tot == null ? "" : `(shield ${Math.round(Math.max(0, tot - sec))} + secret ${Math.round(sec)})`;
}
// The meeting pill is what the extension in the Meet tab reports (/meet/status), not what the buttons did.
function renderMeet(d) {
  const pill = $("meet-pill");
  const ok = (b) => (b ? "✓" : "✗");
  if (!d.page) { pill.textContent = "no Meet tab"; pill.className = "pill off"; }
  else if (d.page === "meet" && !d.in_call) { pill.textContent = "Meet open · not in a call"; pill.className = "pill wait"; }
  else {
    pill.textContent = (d.page === "meet" ? "in call" : "test room") + " · mic " + ok(d.mic) + " caller " + ok(d.far);
    pill.className = "pill " + (d.mic && d.far ? "ok" : "wait");
  }
  $("meet-leave").disabled = d.page !== "meet";
  if (d.url && !$("meet-url").value) $("meet-url").value = d.url;
  pill.title = d.browser ? `${d.browser}${d.latency_ms != null ? ` · bridge ${Math.round(d.latency_ms)} ms` : ""}` : "";
}

// ---------- Ares vs Athena (Keyguard arms race); every field may be absent ----------
const KG_MOVES_MAX = 60;
const txt = (x) => (x == null || x === "" ? "–" : String(x));
const plain = (x) => txt(x).replace(/\*\*/g, "");  // Keyguard writes light markdown in results
function kgState(text, cls) { const p = $("kg-state"); p.textContent = text; p.className = "pill " + cls; }
function renderKgBurst(d) {
  kgState(`match running… (${txt(d.n_keys)} keys${d.shield ? ", shield " + d.shield : ""})`, "wait");
  $("kg-moves").textContent = "";
  $("kg-verdict").hidden = true;
}
function renderKgMove(d) {
  const ol = $("kg-moves");
  if (ol.firstElementChild && ol.firstElementChild.classList.contains("muted")) ol.textContent = "";
  const li = document.createElement("li");
  const who = document.createElement("span"); who.className = "who"; who.textContent = txt(d.agent);
  li.append(who, document.createTextNode(plain(d.title)));
  if (d.tag) { const tg = document.createElement("span"); tg.className = "tag"; tg.textContent = d.tag; li.appendChild(tg); }
  if (d.result) { const r = document.createElement("span"); r.className = "res"; r.textContent = plain(d.result); li.appendChild(r); }
  ol.prepend(li);
  while (ol.children.length > KG_MOVES_MAX) ol.lastChild.remove();
}
function fillRows(tbody, rows, ncol) {
  tbody.textContent = "";
  if (!rows.length) rows = [[{ v: "–", cls: "muted", span: ncol }]];
  for (const cells of rows) {
    const tr = document.createElement("tr");
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = txt(c.v);
      if (c.cls) td.className = c.cls;
      if (c.span) td.colSpan = c.span;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}
const readCls = (hitsTrue) => "mono " + (hitsTrue === true ? "bad-c" : hitsTrue === false ? "ok-c" : "");
function renderKgMatch(d) {
  $("kg-route").textContent = txt(d.route || d.backend);
  $("kg-secret").textContent = txt(d.secret);
  $("kg-decoy").textContent = txt(d.decoy);
  fillRows($("kg-rounds"), (d.rounds || []).map((r) => [
    { v: r.round }, { v: r.span_read, cls: readCls(r.reads_true_secret) },
    { v: r.stoi == null ? null : Number(r.stoi).toFixed(2) },
    { v: r.after_retrain_read, cls: readCls(r.after_retrain_reads_true) },
  ]), 4);
  fillRows($("kg-agents"), Object.entries(d.agents || {}).map(([name, a]) => [
    { v: name }, { v: a && a.before, cls: "mono" }, { v: a && a.after, cls: "mono" },
  ]), 3);
  const v = $("kg-verdict");
  v.hidden = d.protected == null;
  if (d.protected != null) {
    v.textContent = d.protected ? "🦉 secret protected" : "⚔️ secret leaked";
    v.className = "pill " + (d.protected ? "ok" : "off");
  }
  kgState(`match done${d.seconds != null ? ` in ${Number(d.seconds).toFixed(1)} s` : ""}`, "ok");
  const moves = d.moves || [];  // a page opened mid-call only saw the last move: rebuild the feed from the log
  if ($("kg-moves").querySelectorAll(".who").length < moves.length) { $("kg-moves").textContent = ""; moves.forEach(renderKgMove); }
}

// ---------- dispatch ----------
function handle(m) {
  const d = m.data || {}, t = m.t || Date.now() / 1000;
  switch (m.topic) {
    case "snapshot":
      Object.entries(d).filter(([, v]) => v && typeof v === "object" && "data" in v)
        .sort((a, b) => (a[1].t || 0) - (b[1].t || 0))
        .forEach(([topic, v]) => handle({ topic, t: v.t, data: v.data, replay: true }));
      return;
    case "threat.update": renderThreat(d, t); break;
    case "voice.verdict": renderVoice(d, t); break;
    case "voice.flush": renderFlush(); break;
    case "voice.window": renderWindow(d); break;
    case "audio.level": renderLevels(d); break;
    case "system.state": renderSystem(d); break;
    case "keys.stroke": renderStroke(); break;
    case "keys.readout": renderReadout(d); break;
    case "shield.state": renderShield(d.mode); st.shieldLat = d.latency_ms; renderPipeline(); break;
    case "secret.state": renderSecretState(d); st.secretDelay = d.enabled ? d.delay_ms : 0; renderPipeline(); break;
    case "meet.state": renderMeet(d); break;
    case "secret.blocked": renderSecretBlocked(d, t); break;
    case "keyguard.burst": renderKgBurst(d); break;
    case "keyguard.move": renderKgMove(d); break;
    case "keyguard.arms_race": renderKgMatch(d); break;
    case "driver.error": if (!m.replay) alarm(`Driver ${d.driver || "?"} failed: ${d.error || ""} (audio keeps flowing)`); break;
  }
  if (!QUIET.has(m.topic) && !m.replay) {
    let text = m.topic;
    if (m.topic === "threat.level_change") text = `threat → ${d.level || d.to || ""}`;
    else if (m.topic === "shield.state") text = `shield → ${d.mode}`;
    else if (m.topic === "secret.blocked") text = `${d.category === "digits" ? d.length + "-digit code" : d.category} ${d.allowed ? "allowed" : "blocked"} from your voice`;
    else if (m.topic === "secret.request") text = "caller asked for a code";
    else if (m.topic === "meet.call") text = MEET_EVENT[d.event] || "meeting: " + d.event;
    else if (m.topic === "voice.flush") text = d.reason === "gap" ? "new speaker (long silence): voice history cleared"
      : d.reason === "source" ? "now judging " + ($("voice-h").textContent || "the voice") : "voice history flushed";
    else if (m.topic === "control.meet") text = `meeting → ${d.action}${d.url ? " " + d.url : ""}`;
    else if (m.topic === "keyguard.burst") text = `keyguard: ${txt(d.n_keys)} keys typed, Ares vs Athena match started`;
    else if (m.topic === "keyguard.arms_race") text = `keyguard: match done, ${d.protected == null ? "no verdict" : d.protected ? "secret protected" : "secret leaked"}`;
    else if (m.topic === "driver.error") text = `driver ${d.driver} error: ${d.error}`;
    else text += " " + JSON.stringify(d);
    log(t, text, m.topic === "driver.error" || (m.topic === "threat.level_change" && (d.level || d.to) === "CRITICAL") ? "alert-c" : "");
  }
}

// ---------- socket with auto-reconnect ----------
let backoff = 500;
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { backoff = 500; st.live = st.everLive = true; renderConn(); };
  ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => {
    st.live = false; renderConn();
    setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 5000);
  };
  ws.onerror = () => ws.close();
}
connect();

// ---------- controls ----------
async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { alarm(`${path}: ${r.status} ${(await r.text()).slice(0, 200)}`); return false; }
    return true;
  } catch (e) { alarm(`${path}: ${e}`); return false; }
}
for (const b of document.querySelectorAll("[data-mode]")) b.onclick = () => post("/api/control/shield", { mode: b.dataset.mode });
for (const b of document.querySelectorAll("[data-sec]")) b.onclick = () => post("/api/control/secret", { action: b.dataset.sec });
for (const b of document.querySelectorAll("[data-src]")) b.onclick = () => post("/api/control/voice/source", { source: b.dataset.src });
$("voice-flush").onclick = () => post("/api/control/voice/flush", {});
$("meet-ext").onclick = () => post("/api/control/meet", { action: "extension" });

// ---------- meeting ----------
function meetUrl() {  // accept a full link, "meet.google.com/abc-defg-hij" or a bare code; blank = Meet's home page
  const v = $("meet-url").value.trim();
  if (!v) return null;
  if (/^https?:\/\//i.test(v)) return v;
  return v.includes("/") ? "https://" + v : "https://meet.google.com/" + v;
}
$("meet-join").onclick = () => post("/api/control/meet", { action: "join", url: meetUrl() });
$("meet-url").onkeydown = (e) => { if (e.key === "Enter") $("meet-join").click(); };
$("meet-leave").onclick = () => post("/api/control/meet", { action: "leave" });   // the extension clicks Meet's Leave
