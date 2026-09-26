// CallGuard dashboard: one WebSocket, every event is {topic, t, data}. No build step, no network beyond this server.
"use strict";
const $ = (id) => document.getElementById(id);
const MAXC = 20;               // characters shown per keyboard row (the threat engine's 20-keystroke window)
const SPARK_S = 60;            // sparkline window, seconds
const TL_MAX = 4000;           // timeline points kept
const LOG_MAX = 200;
const QUIET = new Set(["keys.stroke", "voice.window", "threat.update", "keys.readout", "voice.verdict"]);

const st = { voice: [], threat: [], typed: 0, vclass: null, alarmTimer: 0 };

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
function renderVoice(d, t) {
  const p = clamp(Number(d.p_synthetic) || 0, 0, 1);
  const c = voiceClass(p);
  $("vlight").className = "vlight " + c;
  $("vlabel").textContent = c;
  $("vp").textContent = p.toFixed(2);
  $("vlat").textContent = d.latency_ms == null ? "–" : Math.round(d.latency_ms) + " ms";
  if (c !== st.vclass) { log(t, `voice → ${c} (p=${p.toFixed(2)})`, c === "synthetic" ? "alert-c" : ""); st.vclass = c; }
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

function renderStroke() {
  st.typed = Math.min(st.typed + 1, MAXC);
  $("typed").textContent = "•".repeat(st.typed);
}
function readoutRow(el, guesses) {
  el.textContent = "";
  for (const g of (guesses || []).slice(-MAXC)) {
    const s = document.createElement("span");
    s.textContent = g.top1 == null || g.top1 === "" ? "?" : String(g.top1);
    if (g.truth != null) s.className = g.top1 === g.truth ? "ok-c" : g.hit ? "near-c" : "bad-c";
    if (g.truth != null && g.hit && g.top1 !== g.truth) s.title = "true key in the attacker's top 3";
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
function renderShield(mode) {
  for (const b of document.querySelectorAll(".shield button")) b.classList.toggle("on", b.dataset.mode === mode);
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
    case "keys.stroke": renderStroke(); break;
    case "keys.readout": renderReadout(d); break;
    case "shield.state": renderShield(d.mode); break;
    case "driver.error": if (!m.replay) alarm(`Driver ${d.driver || "?"} failed: ${d.error || ""} (audio keeps flowing)`); break;
  }
  if (!QUIET.has(m.topic) && !m.replay) {
    let text = m.topic;
    if (m.topic === "threat.level_change") text = `threat → ${d.level || d.to || ""}`;
    else if (m.topic === "shield.state") text = `shield → ${d.mode}`;
    else if (m.topic === "driver.error") text = `driver ${d.driver} error: ${d.error}`;
    else text += " " + JSON.stringify(d);
    log(t, text, m.topic === "driver.error" || (m.topic === "threat.level_change" && (d.level || d.to) === "CRITICAL") ? "alert-c" : "");
  }
}

// ---------- socket with auto-reconnect ----------
let backoff = 500;
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { backoff = 500; const c = $("conn"); c.textContent = "live"; c.className = "pill ok"; };
  ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => {
    const c = $("conn"); c.textContent = "reconnecting…"; c.className = "pill off";
    setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 5000);
  };
  ws.onerror = () => ws.close();
}
connect();

// ---------- controls ----------
async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) alarm(`${path}: ${r.status} ${(await r.text()).slice(0, 200)}`);
  } catch (e) { alarm(`${path}: ${e}`); }
}
for (const b of document.querySelectorAll(".shield button")) b.onclick = () => post("/api/control/shield", { mode: b.dataset.mode });
$("scn-start").onclick = () => post("/api/control/scenario", { action: "start", name: $("scn-name").value || "ai_caller" });
$("scn-stop").onclick = () => post("/api/control/scenario", { action: "stop", name: $("scn-name").value || "ai_caller" });
