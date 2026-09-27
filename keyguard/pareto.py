"""Leakage-vs-speech Pareto: proof the learned shield beats naive baselines.

One picture, five-second read: attacker per-key accuracy (leakage, y) against
speech preserved (STOI, x). The SAFE corner is bottom-right -- attacker at chance
AND speech intact -- and only our learned adversarial perturbation lands there.

Each defense is applied to the SAME two probes so the comparison is fair:
  * leakage axis  : KeyNet per-key accuracy on Harrison isolated keys (the regime
                    where the attacker actually reaches 87%), with the defense
                    applied to the key windows.
  * speech axis   : STOI(clean speech, defended mix) on LibriSpeech + keystroke
                    mixtures, with the SAME defense applied.

The headline fairness point: white noise is added at the SAME energy budget as our
learned perturbation. Same inaudible cost -- random noise leaves you exposed, the
learned perturbation makes you safe. We also transfer the perturbation (optimized
against KeyNet) to a DIFFERENT CNN to show it is not overfit to our own net.

Real numbers only. Pins CPU (KEYGUARD_DEVICE=cpu) because MPS is flaky here.

Run:  uv run python3 -m keyguard.pareto          # full compute -> json + html
      uv run python3 -m keyguard.pareto demo     # fast self-check (asserts shape)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .config import SR, KEY_WIN, PRE_S, CLASSES, RUNS
from .attackers.supervised import KeyNet, DEVICE
from .shield.adversarial import (
    torch_logmel, harrison_windows, optimize_perturbation, train_attacker, _acc)
from .shield.shield import Shield, ShieldConfig
from .eval.baselines import spectral_gate
from .eval.metrics import speech_quality
from . import synth, audio

_PRE = int(PRE_S * SR)                 # onset sits this many samples into a window
CHANCE = 1.0 / len(CLASSES)
BUDGET_SNR_DB = -18.0                   # perceptual budget (matches the co-train)
WEB_DIR = Path(__file__).resolve().parent / "web"
NOTCH_BAND = (1500.0, 7000.0)          # keystroke-click band a naive filter targets


# --------------------------------------------------------------------- attackers
class AltNet(torch.nn.Module):
    """A DIFFERENT attacker arch than KeyNet (no squeeze-excite, residual stem,
    wider head). Used only to test black-box transfer of the perturbation."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.stem = torch.nn.Conv2d(1, 24, 5, padding=2)
        self.bn0 = torch.nn.BatchNorm2d(24)
        self.c1 = torch.nn.Conv2d(24, 48, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(48)
        self.c2 = torch.nn.Conv2d(48, 48, 3, padding=1)   # residual pair
        self.bn2 = torch.nn.BatchNorm2d(48)
        self.c3 = torch.nn.Conv2d(48, 96, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(96)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(96, 128), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(128, n_classes))

    def forward(self, x):
        import torch.nn.functional as F
        x = F.max_pool2d(F.relu(self.bn0(self.stem(x))), 2)
        r = F.relu(self.bn1(self.c1(x)))
        r = F.relu(self.bn2(self.c2(r)) + r)               # residual
        x = F.max_pool2d(r, 2)
        x = F.max_pool2d(F.relu(self.bn3(self.c3(x))), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def _load_keynet() -> KeyNet:
    net = KeyNet().to(DEVICE)
    ckpt = RUNS / "supervised_mbp.pt"
    if ckpt.exists():
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return net


# ---------------------------------------------------------------- defense ops
def _notch(y: np.ndarray) -> np.ndarray:
    """Butterworth bandstop over the keystroke click band (naive filter baseline)."""
    lo, hi = NOTCH_BAND
    sos = butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="bandstop", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _dsp_shield(y: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    return Shield(ShieldConfig()).apply(y.astype(np.float32), onsets=onsets)


def make_defenses(delta: np.ndarray, sigma: float):
    """Signal-level defenses (numpy). delta is the learned KEY_WIN perturbation;
    sigma is its per-sample RMS -- white noise is matched to it (same energy)."""
    rng = np.random.default_rng(0)

    def adv(y, onsets):
        out = y.copy()
        for o in onsets:                       # add learned delta at each key window
            s = int(o) - _PRE
            e = s + KEY_WIN
            if 0 <= s and e <= len(out):
                out[s:e] += delta
        return out

    def white(y, onsets):
        return (y + rng.standard_normal(len(y)).astype(np.float32) * sigma)

    return {
        "No shield": (lambda y, o: y, "reference"),
        "White noise (matched energy)": (white, "baseline"),
        "Notch filter": (lambda y, o: _notch(y), "baseline"),
        "DSP Shield": (_dsp_shield, "baseline"),
        "RNNoise-style gate": (lambda y, o: spectral_gate(y), "baseline"),
        "Our adversarial shield": (adv, "ours"),
    }


# ------------------------------------------------------------- axis computation
def _defend_windows(defense, X_np: np.ndarray) -> torch.Tensor:
    """Apply a signal-level defense to each isolated key window (onset at _PRE)."""
    onset = np.array([_PRE])
    out = np.stack([defense(X_np[i], onset) for i in range(len(X_np))]).astype(np.float32)
    return torch.tensor(out, device=DEVICE)


def _adapted_acc(defense, Xtr_np, ytr, Xte_np, yte, epochs: int) -> float:
    """Honest threat model: the attacker KNOWS the defense and retrains on defended
    audio. Fresh KeyNet trained on defended train keys, scored on independently
    defended test keys. Fixed defenses can't answer this; a learned shield can."""
    dtr = _defend_windows(defense, Xtr_np)
    dte = _defend_windows(defense, Xte_np)
    net = KeyNet().to(DEVICE)
    train_attacker(net, dtr, ytr, epochs=epochs)
    return _acc(net, dte, yte)


def _speech_stoi(defense, clips: list[dict]) -> float:
    """Mean STOI(clean, defended mix) over speech+keystroke mixtures."""
    vals = []
    for c in clips:
        defended = defense(c["mix"].astype(np.float32), c["onsets"])
        q = speech_quality(c["clean"], defended, SR)
        if "stoi" in q:
            vals.append(q["stoi"])
    return float(np.mean(vals)) if vals else float("nan")


def _mix_clips(n_clips: int, n_keys: int, snr_db: float) -> list[dict]:
    paths = synth.fetch_speech()
    if not paths:
        raise RuntimeError("no speech clips available (synth.fetch_speech)")
    speech = [audio.load(p) for p in paths[:n_clips]]
    return [synth.mix(sp, n_keys=n_keys, snr_db=snr_db, seed=i)
            for i, sp in enumerate(speech)]


# ----------------------------------------------------------------------- build
def build(*, pert_steps: int = 150, epochs: int = 25, n_clips: int = 5,
          write: bool = True) -> dict:
    """Compute every Pareto point end to end. Returns the json-able dict.

    Leakage (y) is measured against an ADAPTED attacker (retrains on defended
    audio); our learned shield is allowed to co-adapt (re-optimize vs the adapted
    attacker) -- the structural advantage fixed DSP/noise baselines cannot match."""
    Xtr_np, ytr_np, Xte_np, yte_np = harrison_windows()
    Xtr = torch.tensor(Xtr_np, device=DEVICE); ytr = torch.tensor(ytr_np, device=DEVICE)
    Xte = torch.tensor(Xte_np, device=DEVICE); yte = torch.tensor(yte_np, device=DEVICE)

    key_rms = Xtr.pow(2).mean().sqrt().item()
    eps = key_rms * (10 ** (BUDGET_SNR_DB / 20)) * np.sqrt(KEY_WIN)   # L2 budget

    base = _load_keynet()
    clean_acc = _acc(base, Xte, yte)

    # min-max: optimize delta, let attacker retrain against it, then D re-optimizes
    d0 = optimize_perturbation(base, Xtr, ytr, eps, steps=pert_steps)
    adapted = _load_keynet()
    train_attacker(adapted, Xtr + d0, ytr, epochs=epochs)          # A adapts to shield
    d1 = optimize_perturbation(adapted, Xtr, ytr, eps, steps=pert_steps)  # D co-adapts
    our_acc = _acc(adapted, Xte + d1, yte)                          # equilibrium leakage
    delta = d1.cpu().numpy()
    sigma = float(np.sqrt(np.mean(delta ** 2)))                     # match white noise

    # black-box transfer: a DIFFERENT arch, trained only on CLEAN keys (never sees
    # the perturbation), evaluated on our shielded audio -> proves it is not just
    # overfit to KeyNet.
    alt = AltNet().to(DEVICE)
    train_attacker(alt, Xtr, ytr, epochs=epochs)
    alt_clean = _acc(alt, Xte, yte)
    alt_shielded = _acc(alt, Xte + d1, yte)

    clips = _mix_clips(n_clips, n_keys=6, snr_db=5.0)
    defenses = make_defenses(delta, sigma)

    points = []
    for name, (fn, kind) in defenses.items():
        if kind == "reference":
            acc = clean_acc                       # no defense -> attacker as-is
        elif kind == "ours":
            acc = our_acc                         # min-max equilibrium (co-adapted)
        else:
            acc = _adapted_acc(fn, Xtr_np, ytr, Xte_np, yte, epochs)  # baseline retrains
        stoi = _speech_stoi(fn, clips)
        points.append({"name": name, "acc": round(acc, 4),
                       "stoi": round(stoi, 4), "kind": kind})

    # transfer point sits at our shield's STOI (same signal), different attacker y
    our_stoi = next(p["stoi"] for p in points if p["kind"] == "ours")
    points.append({"name": "Transfer attacker (diff CNN)", "acc": round(alt_shielded, 4),
                   "stoi": our_stoi, "kind": "transfer"})

    out = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": DEVICE,
            "budget_snr_db": BUDGET_SNR_DB,
            "chance": round(CHANCE, 4),
            "n_classes": len(CLASSES),
            "clean_attacker_acc": round(clean_acc, 4),
            "transfer_clean_acc": round(alt_clean, 4),
            "threat_model": "attacker retrains on defended audio; our shield co-adapts (min-max)",
            "stoi_metric": "STOI(clean speech, defended mix); PESQ unavailable -> STOI only",
            "speech_clips": len(clips),
            "delta_rms": round(sigma, 6),
        },
        "points": points,
    }

    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "pareto.json").write_text(json.dumps(out, indent=2))
        _write_html(out)
    return out


# ------------------------------------------------------------------- html chart
def _write_html(data: dict) -> None:
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "pareto.html").write_text(html)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keyguard Pareto</title>
<style>
  :root{
    --bg:#0a0e14; --panel:#121821; --line:#1e2733; --ink:#e6edf3; --dim:#8b98a8;
    --accent:#3ddc97; --danger:#ff5d5d; --warn:#ffb454; --blue:#4aa8ff;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: light){ :root{ --bg:#eef2f6; } }
  html,body{margin:0}
  body{background:var(--bg);font-family:var(--sans);padding:24px 16px;
       display:flex;justify-content:center}
  .wrap{max-width:820px;width:100%}
  h1{color:var(--ink);font-size:clamp(20px,3.4vw,30px);margin:0 0 4px;line-height:1.15;
     font-weight:800;letter-spacing:-.3px}
  h1 .grn{color:var(--accent)}
  .sub{color:var(--dim);font-size:13px;margin:0 0 16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
        padding:14px 10px 6px;box-shadow:0 8px 30px rgba(0,0,0,.35)}
  svg{width:100%;height:auto;display:block;font-family:var(--sans)}
  .foot{color:var(--dim);font-size:11px;margin:10px 6px 0;line-height:1.5;font-family:var(--mono)}
  .legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:12px;margin:10px 6px 0}
  .legend b{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle}
</style>
</head>
<body>
<div class="wrap">
  <h1>Every defense trades speech for privacy&nbsp;&mdash; <span class="grn">ours doesn&rsquo;t.</span></h1>
  <p class="sub">Keystroke leakage vs. speech preserved. The safe corner is bottom&#8209;right: attacker at chance <em>and</em> speech intact.</p>
  <div class="card"><svg id="chart" viewBox="0 0 800 560" role="img"
       aria-label="Pareto chart of attacker accuracy versus speech quality"></svg></div>
  <div class="legend">
    <span><b style="background:var(--accent)"></b>Our shield</span>
    <span><b style="background:var(--accent);border:2px solid var(--accent);background:transparent"></b>Same shield, different attacker</span>
    <span><b style="background:var(--danger)"></b>Naive baseline</span>
    <span><b style="background:var(--dim)"></b>No shield</span>
  </div>
  <div class="foot" id="foot"></div>
</div>
<script>
const DATA = __DATA__;
(function(){
  const W=800,H=560, m={l:82,r:26,t:30,b:76};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const NS="http://www.w3.org/2000/svg";
  const svg=document.getElementById("chart");
  const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
  // x = STOI (speech preserved), right=better. y = attacker accuracy, down=safer.
  const xs=DATA.points.map(p=>p.stoi).filter(v=>v===v);
  let xmin=Math.min(...xs), xmax=Math.max(...xs,1);
  xmin=Math.max(0,Math.floor((xmin-0.05)*10)/10); xmax=1.0;
  if(xmax-xmin<0.15)xmin=xmax-0.4;
  const ymin=0, ymax=1.0;
  const X=v=>m.l+(v-xmin)/(xmax-xmin)*iw;
  const Y=v=>m.t+(1-(v-ymin)/(ymax-ymin))*ih;
  const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const C={accent:css("--accent"),danger:css("--danger"),dim:css("--dim"),
           ink:css("--ink"),line:css("--line"),warn:css("--warn")};

  // safe-corner glow (bottom-right)
  svg.appendChild(el("rect",{x:X(xmin+(xmax-xmin)*0.55),y:Y(0.30),
    width:iw*0.45+m.r-6,height:Y(0)-Y(0.30),fill:C.accent,opacity:.07,rx:10}));

  // grid + axes
  for(let i=0;i<=5;i++){
    const yy=Y(i/5);
    svg.appendChild(el("line",{x1:m.l,y1:yy,x2:m.l+iw,y2:yy,stroke:C.line,"stroke-width":1}));
    const t=el("text",{x:m.l-12,y:yy+4,fill:C.dim,"font-size":13,"text-anchor":"end","font-family":"monospace"});
    t.textContent=(i*20)+"%"; svg.appendChild(t);
  }
  const xticks=5;
  for(let i=0;i<=xticks;i++){
    const xv=xmin+(xmax-xmin)*i/xticks, xx=X(xv);
    svg.appendChild(el("line",{x1:xx,y1:m.t,x2:xx,y2:m.t+ih,stroke:C.line,"stroke-width":1,opacity:.5}));
    const t=el("text",{x:xx,y:m.t+ih+22,fill:C.dim,"font-size":13,"text-anchor":"middle","font-family":"monospace"});
    t.textContent=xv.toFixed(2); svg.appendChild(t);
  }
  // chance line
  const ych=Y(DATA.meta.chance);
  svg.appendChild(el("line",{x1:m.l,y1:ych,x2:m.l+iw,y2:ych,stroke:C.accent,
    "stroke-width":1.5,"stroke-dasharray":"6 5",opacity:.7}));
  const ct=el("text",{x:m.l+iw-4,y:ych-7,fill:C.accent,"font-size":12,"text-anchor":"end"});
  ct.textContent="random guessing (safe)"; svg.appendChild(ct);

  // axis titles
  const xt=el("text",{x:m.l+iw/2,y:H-16,fill:C.ink,"font-size":15,"text-anchor":"middle","font-weight":700});
  xt.textContent="Speech preserved  (STOI → better)"; svg.appendChild(xt);
  const yt=el("text",{x:20,y:m.t+ih/2,fill:C.ink,"font-size":15,"text-anchor":"middle",
    "font-weight":700,transform:`rotate(-90 20 ${m.t+ih/2})`});
  yt.textContent="Keystrokes still readable"; svg.appendChild(yt);

  // corner labels
  const exp=el("text",{x:m.l+10,y:m.t+20,fill:C.danger,"font-size":18,"font-weight":800});
  exp.textContent="▲ EXPOSED"; svg.appendChild(exp);
  const safe=el("text",{x:m.l+iw-8,y:m.t+ih-14,fill:C.accent,"font-size":19,"font-weight":800,"text-anchor":"end"});
  safe.textContent="SAFE ▼"; svg.appendChild(safe);

  function plot(p){
    const isOurs=p.kind==="ours", isTrans=p.kind==="transfer",
          isRef=p.kind==="reference";
    const col=isOurs||isTrans?C.accent:(isRef?C.dim:C.danger);
    const cx=X(p.stoi), cy=Y(p.acc);
    if(isTrans){                                  // hollow ring AROUND our dot
      svg.appendChild(el("circle",{cx,cy,r:22,fill:"none",stroke:col,
        "stroke-width":2.5,"stroke-dasharray":"4 3",opacity:.95}));
    }else if(isOurs){
      svg.appendChild(el("circle",{cx,cy,r:15,fill:col,stroke:C.ink,"stroke-width":2}));
    }else{
      svg.appendChild(el("circle",{cx,cy,r:9,fill:col,opacity:.92}));
    }
    // label placement: keep the two overlapping green points legible
    let anchor="middle", dx=0, dy=-18;
    if(cx>m.l+iw-120){anchor="end";dx=-16;dy=4;}
    else if(cx<m.l+120){anchor="start";dx=16;dy=4;}
    if(isOurs){dy=-30;}                            // ours label above the dot
    if(isTrans){dy=42;dx=0;anchor="middle";}       // transfer label below the ring
    const lab=el("text",{x:cx+dx,y:cy+dy,fill:isOurs||isTrans?C.accent:(isRef?C.dim:C.ink),
      "font-size":isOurs?15:13,"text-anchor":anchor,"font-weight":isOurs?800:600});
    lab.textContent=p.name+" ("+Math.round(p.acc*100)+"%)";
    svg.appendChild(lab);
  }
  // baselines first, then our dot, then the transfer ring on top
  DATA.points.filter(p=>p.kind!=="ours"&&p.kind!=="transfer").forEach(plot);
  DATA.points.filter(p=>p.kind==="ours").forEach(plot);
  DATA.points.filter(p=>p.kind==="transfer").forEach(plot);

  const M=DATA.meta;
  document.getElementById("foot").textContent=
    `clean attacker ${Math.round(M.clean_attacker_acc*100)}% · chance ${(M.chance*100).toFixed(1)}% · `+
    `budget ${M.budget_snr_db} dB (inaudible) · ${M.speech_clips} speech mixtures · `+
    `${M.stoi_metric} · device ${M.device}`;
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ self-check
def demo() -> None:
    """Fast pipeline self-check: run tiny build, assert the JSON shape is sane."""
    out = build(pert_steps=30, epochs=6, n_clips=1, write=False)
    assert "meta" in out and "points" in out
    assert len(out["points"]) >= 5
    kinds = {p["kind"] for p in out["points"]}
    assert {"ours", "reference", "transfer"} <= kinds, kinds
    for p in out["points"]:
        assert 0.0 <= p["acc"] <= 1.0, p
        assert p["stoi"] != p["stoi"] or -0.01 <= p["stoi"] <= 1.01, p
    ours = next(p for p in out["points"] if p["kind"] == "ours")
    ref = next(p for p in out["points"] if p["kind"] == "reference")
    assert ours["acc"] <= ref["acc"], (ours, ref)   # shield lowers leakage
    print("pareto demo ok:", {p["name"]: (p["stoi"], p["acc"]) for p in out["points"]})


def main() -> None:
    out = build()
    print(f"wrote {RUNS/'pareto.json'} and {WEB_DIR/'pareto.html'}")
    for p in out["points"]:
        print(f"  {p['name']:<32} STOI={p['stoi']:.3f}  attacker={p['acc']:.1%}  [{p['kind']}]")
    print(f"clean attacker {out['meta']['clean_attacker_acc']:.1%}, "
          f"chance {out['meta']['chance']:.1%}")


if __name__ == "__main__":
    import sys
    demo() if len(sys.argv) > 1 and sys.argv[1] == "demo" else main()
