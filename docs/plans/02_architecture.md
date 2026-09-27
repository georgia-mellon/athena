# 02: Architecture, driver contracts, threat score, hooks

## 1. Process layout
One Python process (`athena`), several threads, one asyncio server:

```
 physical mic ─► [mic stream 16k, 20 ms blocks] ─► shield driver ─► [virtual mic out: VB-CABLE Input] ─► Zoom mic
                        │ (raw copy)                     │ (shielded copy)
                        ▼                                ▼
                 raw ring buffer                  shielded ring buffer
                        └──────► attacker worker (per key event, OS timestamps) ──► keys.readout (raw vs shielded)
 pynput key listener ──► KeyClock (timestamps; identity kept in memory for the demo) ─┘

 Zoom speaker ─► [loopback capture of the output device] ─► far-end ring buffer ─► VAD ─► voice worker (Hearsay)
                                                                                       └─► voice.window / voice.verdict

 all events ─► EventBus ─► ThreatEngine ─► threat.update ─► hooks (console / jsonl / webhook) + WebSocket ─► dashboard
```
Replay mode swaps the three device endpoints for file sources/sinks with the same block timing. Everything
downstream is identical.

## 2. Package layout (file ownership = work packages in plan 05)
```
app/source/
  types.py          shared dataclasses + driver Protocols   (contracts; written first, changed only by review)
  config.py         dataclass config, TOML load, env (HEARSAY_ROOT, KEYGUARD_ROOT, ATHENA_*)
  bus.py            EventBus (thread-safe pub/sub; asyncio bridge)
  threat.py         ThreatEngine
  hooks.py          sinks: console, jsonl, webhook
  audio/            devices.py (list/pick devices), streams.py (mic in, virtual-mic out, loopback in), ring.py, vad.py, keys.py (pynput KeyClock), replay.py (file sources)
  drivers/          base.py (registry), hearsay_real.py, keyguard_real.py, mock.py
  pipeline.py       wires streams → drivers → bus (live + replay)
  server/           app.py (FastAPI + WebSocket), static/ (dashboard)
  cli.py            athena run|devices|bench|scenario
experiments/        attack_under_speech.py (+ results in docs/reports/)
demo/               scenarios/*.toml, render_agent.py (TTS lines), agent_caller.py (plays into a meeting)
tests/
```

## 3. Driver contracts (`app/source/types.py`)
```python
class VoiceAuthenticityDriver(Protocol):
    name: str
    sample_rate: int                      # 16000
    def score(self, audio: np.ndarray) -> VoiceScore: ...   # 1-D float32, >= 3 s, far-end speech
    # VoiceScore(p_synthetic: float in [0,1], margin: float, threshold: float, latency_ms: float, detail: dict)

class KeystrokeAttackerDriver(Protocol):
    name: str
    classes: list[str]                    # e.g. A-Z0-9
    def read(self, audio: np.ndarray, onsets: np.ndarray) -> list[KeyGuess]: ...
    # one KeyGuess per onset: top-k keys + probabilities. `onsets` = sample indices (from OS timestamps or detection)

class ShieldDriver(Protocol):
    name: str
    def process(self, block: np.ndarray, key_events: list[int]) -> np.ndarray: ...   # streaming, same length out
    def reset(self) -> None: ...
```
**Real drivers (read-only use of the upstream repos):**
- `hearsay_real`: `sys.path` += `HEARSAY_ROOT/src`, `HEARSAY_ROOT/scripts`; loads
  `data/models/r4ft_xlsr/R4ft_xlsr_light/{best.pth,config.json}` (sha256 checked against config.json), the R1
  booster `data/models/r1_lgbm_all_full.txt`, and fusion weights `data/scores/R5_r4ft_r1.json`. Every clip goes
  through `hearsay.preprocess.prep()`, as in Hearsay. `p_synthetic` = logistic of the fused margin, centred on the
  deployment threshold fixed on Hearsay's `val_testlike` (brief reading), so p = 0.5 is the decision boundary. Mode
  `r5` (default: the submitted fusion; adds classic features, ≈ +0.1 s) or `r4ft` (XLS-R alone, faster). See
  `docs/reports/hearsay_r5_in_athena.md`.
- `keyguard_real`: `sys.path` += `KEYGUARD_ROOT`. Attacker = `KeyNet` weights from `ATHENA_ATTACKER_WEIGHTS`, or a
  provisional Athena-trained KeyNet (plan 04) until the teammate ships theirs. Shield `dsp` =
  `keyguard.shield.shield.Shield` fed with OS key timestamps; `adversarial` = the teammate's D when available.
- `mock`: deterministic stand-ins with the same timing profile, used by tests, the UI and CI.

A driver that raises is quarantined. The pipeline keeps audio flowing, emits `driver.error`, and the dashboard shows it.

## 4. Threat score (`threat.py`, all constants in config)
Inputs, each in [0, 1]:
- `V` voice risk = EMA of `p_synthetic` over far-end *speech* windows only (half-life 3 windows, ~6 s of speech).
  Event-driven: it moves only when a speech window is scored, so silence and noise hold it flat (it never drifts to
  "human" without human-sounding speech). It resets to 0 on a new speaker: the dashboard's Flush, the test room's clip
  switch, or `new_speaker_gap_s` (20 s) of far-end audio without speech.
- `E` exposure = attacker accuracy on the **raw** mic over the last 20 keystrokes, rescaled above chance
  (`(acc - 1/K) / (1 - 1/K)`). This is what leaks if you don't shield.
- `L` residual leak = the same on the **shielded** stream. This is what still leaks.
- `T` typing activity = keystrokes in the last 10 s, saturating at 10.

Score:
```
threat = 100 * (1 - (1 - w_v*V) * (1 - w_l*L*T_on)) , boosted by the social-engineering rule:
if V >= 0.5 and T_on (typing while an unverified voice speaks): threat = max(threat, 60 + 40*leak), leak = L with the
shield on, E off (was 60 + 40*V*max(E, L); V dropped so a stricter Hearsay threshold, hearsay_ai_p, does not weaken it)
levels: SAFE < 25 <= WATCH < 50 <= WARN < 75 <= CRITICAL      (hysteresis 5 points)
```
Each update carries its `reasons` (human-readable). Exposure `E` doesn't raise the score when the shield is on
and `L` is low. The dashboard still shows it ("the shield is blocking an 85 % readable keyboard").

## 5. Hooks
`bus.subscribe(topic_glob, fn)`. Sinks from config:
`[[hooks]] kind = "console" | "jsonl" | "webhook"`, `topics = ["threat.level_change", ...]`, `url = ...`.
Webhook payload = the event dict (JSON), with retries and timeouts off the audio thread.

## 6. Dashboard (served by FastAPI at `/`, pushes over `/ws`)
Top: the threat gauge (0-100, level colour) and the reasons. Left: the inbound voice light, a p_synthetic
sparkline and "real / unverified / synthetic". Right: keyboard exposure, with typed (hidden as •••) | attacker
reads raw | attacker reads shielded, and accuracy. Bottom: timeline and event log. Controls: shield mode, replay
scenario start/stop. It's a static HTML/JS page with no build step.
