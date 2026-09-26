export const meta = {
  name: 'callguard-phase1',
  description: 'Build CallGuard phase 1: 7 parallel work packages, integration, independent review',
  phases: [
    { title: 'Build', detail: 'WP1-WP7 in parallel, disjoint file ownership' },
    { title: 'Integrate', detail: 'WP8: pipeline, CLI, demo scenario, e2e replay' },
    { title: 'Review', detail: 'independent acceptance check' },
  ],
}

const CG = 'C:\\Users\\danma\\Documents\\Dan\\Projects\\CallGuard'
const HS = 'C:\\Users\\danma\\Documents\\Dan\\Projects\\Hearsay'
const KG = 'C:\\Users\\danma\\Documents\\Dan\\Projects\\keyboard-acoustic-shield'
const PY = CG + '\\.venv\\Scripts\\python.exe'

const COMMON = `
You are one work package of the CallGuard build (HackGT 13 primary submission). Repo: ${CG} (private GitHub danmano411/callguard).
FIRST read ${CG}\\plans\\00_brief.md, 01_spec.md, 02_architecture.md, 05_workplan_and_merge.md (and 03/04 if relevant to you), and ${CG}\\callguard\\types.py (the contracts; do NOT edit types.py; if you truly need a contract change, say so in open_issues).
Rules:
- Write ONLY the files your package owns (listed below). Other packages are writing other files in the same checkout right now.
- Do NOT run git (no add/commit/checkout/stash). The coordinator commits.
- Upstream repos are READ-ONLY: Hearsay at ${HS} (HEARSAY_ROOT) and Keyguard at ${KG} (KEYGUARD_ROOT). Import them via sys.path; never write inside them (no caches, no pip installs into their venvs).
- Python env: ${PY} (Python 3.12; torch 2.14 cpu, transformers 5.17, lightgbm, fastapi, uvicorn, sounddevice, soundcard, pynput, librosa, soundfile, pystoi, pytest). Run tests with: cd "${CG}" && .venv/Scripts/python -m pytest -q <your tests>. Your shell cwd starts elsewhere: always cd to ${CG} or use absolute paths. If you need an extra dependency, use: cd "${CG}" && uv add <pkg> and report it.
- Tests must pass WITHOUT models or audio devices (use synthetic numpy signals / mocks); tests needing HEARSAY_ROOT/KEYGUARD_ROOT or devices must skip cleanly when unavailable.
- Match a lean style: small modules, docstrings that say why, no speculative abstractions. Windows paths; audio 16 kHz mono float32; BLOCK=320 (20 ms).
- Never block or crash the audio thread because of a model.
Return the structured result honestly: if something fails, say so.`

const WPS = [
 { key: 'wp1-core', prompt: `WP1 core. Own: callguard/bus.py, callguard/threat.py, callguard/hooks.py, callguard/config.py, callguard.example.toml, tests/test_bus.py, tests/test_threat.py, tests/test_hooks.py.
- bus.py: thread-safe EventBus(publish(Event), subscribe(glob_pattern, fn) -> unsubscribe fn), dispatch on a worker thread so publishers (audio threads) never block; an asyncio bridge (async queue per subscriber) the server can consume; bounded queues (drop oldest + count drops).
- threat.py: ThreatEngine implementing plan 02 section 4 exactly (V EMA half-life, E and L over last 20 keystrokes rescaled above chance, T typing activity, the social-engineering boost, level thresholds + hysteresis, human-readable reasons). Consumes voice.verdict / keys.readout / keys.stroke / shield.state events, publishes threat.update (>=2 Hz via tick()) and threat.level_change. All constants from config. Deterministic given a clock function (inject now()) for tests.
- hooks.py: sinks console / jsonl / webhook (httpx, timeout, retries, off-thread) configured from config [[hooks]] with topic globs.
- config.py: dataclasses + TOML load (tomllib) + env overrides (HEARSAY_ROOT default ../Hearsay relative to repo, KEYGUARD_ROOT default ../keyboard-acoustic-shield, CALLGUARD_* ), driver selection per slot (voice/attacker/shield: real|mock), device names, server port 8765, threat constants.
- Tests: threat scenarios (silence->SAFE; synthetic voice alone -> WARN; synthetic voice + typing with leaky raw and shield off -> CRITICAL; shield on with low L -> lower), hysteresis, bus ordering/drops, jsonl sink.` },
 { key: 'wp2-audio', prompt: `WP2 audio I/O. Own: callguard/audio/* (devices.py, streams.py, ring.py, vad.py, keys.py, replay.py), tests/test_audio.py, docs/zoom_setup.md.
- devices.py: list input/output devices (sounddevice) and loopback-capable speakers (soundcard include_loopback=True); find VB-CABLE ("CABLE Input"/"CABLE Output") and report routing status; if missing, a clear message with install steps (https://vb-audio.com/Cable/ , admin install, reboot). VB-CABLE is NOT installed on this laptop now; code must handle that.
- streams.py: MicShieldStream: duplex path physical mic -> callback(block)->block (the shield hook) -> output device (VB-CABLE Input), 16 kHz mono float32, 20 ms blocks, with resampling if the device rate differs; the callback must be O(1) and hand copies of raw + processed blocks to ring buffers for workers. LoopbackStream: capture the speaker device via soundcard loopback on a thread, resample to 16k mono, feed a ring buffer. Pass-through on any exception inside the processing hook (count errors).
- ring.py: lock-protected ring buffer with absolute sample counters (read last N seconds, read range by absolute index).
- vad.py: energy + zero-crossing VAD with hangover; speech_fraction(window).
- keys.py: KeyClock using pynput keyboard listener: record press timestamps mapped to the mic stream's absolute sample index (use a shared monotonic clock + stream start time); keep key identity only in memory (for the demo readout), expose recent events; a fake/scripted KeyClock for replay/tests.
- replay.py: FileSource that plays a WAV (resampled to 16k) as 20 ms blocks in real time or as-fast-as-possible, and a scripted key-event track; a mixer to layer sources.
- docs/zoom_setup.md: Windows setup: VB-CABLE install, Zoom mic = CABLE Output, disable Zoom noise suppression, speaker loopback; macOS BlackHole note; Meet/Teams equivalents; troubleshooting.
- Tests with synthetic signals only (no devices): ring buffer indexing, VAD on tone vs noise vs silence, replay timing, KeyClock sample mapping, stream callback pass-through on exception (test the callback function directly).` },
 { key: 'wp3-hearsay', prompt: `WP3 Hearsay driver. Own: callguard/drivers/hearsay_real.py, tests/test_hearsay_driver.py.
Implement VoiceAuthenticityDriver (types.py) using the frozen Hearsay models, READ-ONLY from HEARSAY_ROOT=${HS} (set os.environ['HEARSAY_ROOT'] before importing hearsay; add HEARSAY_ROOT/src and HEARSAY_ROOT/scripts to sys.path).
Reference code to read (do not modify): ${HS}\\scripts\\bench_score.py (functions r4ft(), fused(), threshold()), ${HS}\\scripts\\finetune_ssl.py (parse, ARCH, make_model, forward_windows, sha256), ${HS}\\scripts\\fuse.py (apply_transform), ${HS}\\scripts\\extract_classic.py (features(y, train, rng): prep + centre crop + spectral + bio), ${HS}\\src\\hearsay\\preprocess.py (prep), ${HS}\\src\\hearsay\\models\\ssl_e2e_data.py (how 64000-sample windows are cut).
Files: ${HS}\\data\\models\\r4ft_xlsr\\R4ft_xlsr_light\\{best.pth,config.json} (config has arch, sha256, inference, max_windows; verify best.pth sha256 == config sha256 at load, cache the verification by file size+mtime so startup is fast), R1 booster ${HS}\\data\\models\\r1_lgbm_all_full.txt (lightgbm Booster, features = hearsay.features.spectral.FEATURES + hearsay.features.bio.FEATURES), fusion weights ${HS}\\data\\scores\\R5_r4ft_r1.json.
Modes: 'r4ft' (fast) and 'r5' (the submitted fusion). Input: >=3 s float32 16 kHz of far-end speech -> prep() -> for live use 1 centre window (up to max_windows windows if the clip is long), R4ft logit (higher = fake).
Deployment threshold: compute once from Hearsay's val_testlike (hearsay.evaluate.manifest(), val_testlike rows; scores data/scores/R4ft_xlsr_light.parquet or R5_r4ft_r1.parquet) with the same threshold() rule as bench_score (brief weighting 4x real-flagged), cache to ${CG}\\runs\\hearsay_calibration.json keyed by model sha. p_synthetic = sigmoid((margin - thr)/s) with s = (median val_testlike fake margin - thr)/ln(19) (so a typical fake maps to 0.95). Record thr and s in VoiceScore.
Threads: torch.set_num_threads(configurable, default 4) so it doesn't starve the audio thread. device auto (cuda if available).
Tests: skip if HEARSAY_ROOT missing; otherwise load driver, score a real clip and a fake clip taken from Hearsay's test_internal_testlike rows (read paths from the manifest), assert real p<0.5 and fake p>0.5 for a handful of clips, and report per-call latency on CPU (print). Also a pure unit test of the p mapping.
In your summary include measured CPU latency for r4ft and r5 on a 4 s clip, and the thr/s values.` },
 { key: 'wp4-keyguard', prompt: `WP4 Keyguard drivers. Own: callguard/drivers/keyguard_real.py, tests/test_keyguard_driver.py.
Keyguard repo (teammate's, READ-ONLY) at KEYGUARD_ROOT=${KG}. It is under active development (latest commits add SKAID overlap attacker, calibration modes) so FIRST survey it: keyguard/config.py (SR, KEY_WIN=4800, PRE_S), keyguard/attackers/supervised.py (KeyNet, SupervisedAttacker), keyguard/shield/adversarial.py (torch_logmel, train_attacker, optimize_perturbation; it imports keyguard.memory which may need pymongo/dotenv: if import fails, insert a stub module into sys.modules rather than installing), keyguard/shield/shield.py (Shield, ShieldConfig: offline apply(y, onsets)), keyguard/realtime.py (_process_block, KeyClock: a block-wise real-time shield), keyguard/segment.py (onsets, windows), keyguard/features.py, data/pool/*.npz (wins (n,4800), labels A-Z0-9). Committed repo has NO trained weights (runs/ is gitignored).
Implement:
1) KeyguardAttacker (KeystrokeAttackerDriver): loads KeyNet weights from a path (config/env CALLGUARD_ATTACKER_WEIGHTS); if none, trains a PROVISIONAL KeyNet on data/pool/harrison.npz with Keyguard's own train_attacker (CPU, seeded) and caches it at ${CG}\\runs\\provisional_keynet.pt (log that it is provisional). read(audio, onsets): cut KEY_WIN windows at onset - PRE_S*SR, same features as training, softmax top-3.
2) KeyguardShield (ShieldDriver), streaming: process(block, key_events) for 20 ms blocks with OS key events (absolute sample indices). Prefer adapting realtime.py's block-wise approach; if you use the offline Shield.apply, buffer with a small lookahead (<=150 ms latency) and overlap-add; document the latency. Mode 'dsp' now; 'adversarial' raises NotImplementedError with a clear message (waits for teammate's D).
Tests (skip if KEYGUARD_ROOT missing): shield preserves length, is near-identity on blocks far from key events (speech-intact proxy: correlation > 0.95 on a sine/speech-like signal without keys), and attenuates/changes energy around a key event; attacker on held-out harrison presses gets top-1 well above chance (report the number). Keep the provisional training fast (< ~3 min CPU) and cached.
Summary: report attacker held-out top-1/top-3, shield latency, and exactly what in Keyguard you depend on (functions + commit hash from git -C ${KG} log -1 --oneline, read-only).` },
 { key: 'wp5-mocks', prompt: `WP5 mocks + registry. Own: callguard/drivers/base.py, callguard/drivers/mock.py, tests/test_drivers_mock.py.
- base.py: a registry/factory: make_voice(cfg), make_attacker(cfg), make_shield(cfg) selecting 'real' (import callguard.drivers.hearsay_real.HearsayDriver / callguard.drivers.keyguard_real.KeyguardAttacker, KeyguardShield lazily; those files are being written in parallel by other packages, so import lazily by name and document the expected class names: HearsayDriver(mode='r4ft'|'r5', threads=4), KeyguardAttacker(weights=None), KeyguardShield(mode='dsp')) or 'mock'. A Quarantine wrapper: runs driver calls with timing, catches exceptions, after N consecutive failures marks the driver quarantined and returns safe fallbacks (voice: None; attacker: []; shield: pass-through), and publishes driver.error through a callback.
- mock.py: MockVoice (deterministic p_synthetic from a scripted schedule or from a simple signal property; configurable latency), MockAttacker (returns the true key with configurable accuracy when truth provided, else uniform; accuracy drops to chance when the audio was shielded, detect via a marker/metadata the MockShield leaves), MockShield (identity or small deterministic perturbation around key events). They must satisfy the Protocols in types.py (isinstance with runtime_checkable).
- Tests: protocol conformance, quarantine behaviour, factory selection from a plain dict config (config.py is being written in parallel; accept a dict or an object with attributes).` },
 { key: 'wp6-server', prompt: `WP6 server + dashboard. Own: callguard/server/app.py, callguard/server/static/* (index.html, app.js, style.css), tests/test_server.py.
- app.py: FastAPI app factory create_app(bus, state_provider, controls) (bus = an object with subscribe(glob, fn) returning unsubscribe; for decoupling accept any object with that method since bus.py is written in parallel). GET / serves the dashboard; GET /api/state returns the latest snapshot; WS /ws pushes every event (JSON: {topic, t, data}) and the snapshot on connect; POST /api/control/shield {mode: off|dsp|adversarial}, POST /api/control/scenario {action: start|stop, name} calling the controls object's methods. Thread-safe bridge from bus callbacks (other threads) to asyncio via loop.call_soon_threadsafe.
- Dashboard (static, no build step, no external network needed at the expo: do NOT load fonts/scripts from CDNs; system fonts only; inline SVG): built for a projector. Top: big threat gauge 0-100 with level colour (SAFE/WATCH/WARN/CRITICAL) and reasons list. Left panel 'Who is speaking?': voice light (real / unverified / synthetic), p_synthetic sparkline (last 60 s), latency. Right panel 'Your keyboard': typed characters masked as dots, 'eavesdropper reads (no shield)' vs 'eavesdropper reads (shielded)' strings with per-char colouring right/wrong, rolling accuracy vs chance, shield mode toggle buttons. Bottom: timeline strip of threat over the call + event log. Scenario start/stop buttons. Reconnect automatically. Dark theme by default (projector) with a light toggle. Keep it clean and legible from the back of a room; large type.
- Event payload shapes to render (from plan 02 and types.py): threat.update {score, level, reasons[], V, E, L, T}; voice.verdict {p_synthetic, margin, latency_ms, speech_fraction}; keys.stroke {t, masked}; keys.readout {raw:[{truth, top1, p}], shielded:[...], acc_raw, acc_shielded, chance}; shield.state {mode}; driver.error {driver, error}.
- Tests: FastAPI TestClient: GET / 200 with the page, /api/state, a WebSocket receiving a published event, control endpoints calling a fake controls object.` },
 { key: 'wp7-attack', prompt: `WP7 attack proof (plan 04). Own: experiments/attack_under_speech.py, experiments/README.md, reports/attack_under_speech.md, reports/attack_under_speech.csv, reports/figures/attack_under_speech.png.
Read ${CG}\\plans\\04_attack_proof.md and follow it. Keyguard (READ-ONLY) at ${KG}: data/pool/harrison.npz (wins (n,4800) at 16 kHz, labels), keyguard.attackers.supervised.KeyNet, keyguard.shield.adversarial.torch_logmel / train_attacker (it imports keyguard.memory which may need pymongo/dotenv: if the import fails, stub the module in sys.modules; do not install into Keyguard), keyguard.segment.onsets, keyguard.shield.shield.Shield/ShieldConfig (use ShieldConfig(strength=1.0, randomize=1.0, decoys=6, key_frames=6), the dashboard settings). Set KEYGUARD_DEVICE=cpu.
Speech source: real speech clips, read-only, from Hearsay's processed data: use hearsay manifest ${HS}\\data\\processed\\manifest.parquet (pandas) rows with label=='bonafide' and source in ('librispeech','ljspeech') from split 'test_internal', paths relative to ${HS}; pick ~200 clips seeded. (Alternatively librosa.ex('libri1') etc.)
Protocol: per-key seeded 60/40 split of harrison presses; attackers 'clean' and 'speech-aug' (trained with random speech mixed at random levels +0..+20 dB speech-over-key; this is the adaptive attacker). For each test press, embed it at a random position in a 1.5 s speech excerpt at speech-to-key ratio in {keys only, -10, -5, 0, +5, +10, +20} dB (power ratio of the speech excerpt to the key window). Evaluate with (a) oracle onset and (b) detected onset (keyguard.segment.onsets on the mixture, nearest detection within 30 ms else miss = wrong). Then apply the DSP shield with the true onset and re-evaluate both attackers. Wilson 95% CIs. Speech quality of shielded vs unshielded mixture: STOI (pystoi) at each level; PESQ if 'pesq' installs cleanly via uv add (optional). Seeds fixed; CPU; keep total runtime under ~20 min.
Report (reports/attack_under_speech.md): the question, method, a table (level x attacker x shield: top-1, top-3, CI), the figure, and a verdict against the 3 pass criteria in plan 04 (criterion 3: cite Hearsay's measured result: on Keyguard-shielded clean speech the submitted Hearsay system flags 1.7% of real voices vs 0.7% clean, from Hearsay reports/generalization.md; you need not rerun it). State clearly the attacker is provisional (trained here on Keyguard's public bank) and in-domain (same keyboard).` },
]

const RESULT = {
  type: 'object',
  properties: {
    wp: { type: 'string' },
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests_command: { type: 'string' },
    tests_passed: { type: 'boolean' },
    tests_output_tail: { type: 'string' },
    key_numbers: { type: 'string' },
    open_issues: { type: 'array', items: { type: 'string' } },
    blocked_on_upstream: { type: 'array', items: { type: 'string' } },
  },
  required: ['wp', 'summary', 'files', 'tests_passed', 'open_issues'],
}

phase('Build')
const built = (await parallel(WPS.map(w => () =>
  agent(COMMON + '\n\n' + w.prompt, { label: w.key, phase: 'Build', schema: RESULT })
))).filter(Boolean)
log(`build done: ${built.map(b => b.wp + (b.tests_passed ? ' ok' : ' FAIL')).join(', ')}`)

phase('Integrate')
const integ = await agent(COMMON + `

WP8 integrate. Own: callguard/pipeline.py, callguard/cli.py, callguard/__main__.py, demo/* (scenarios/*.toml, build_scenario_audio.py, render_agent.py, agent_caller.py), tests/test_e2e_replay.py, README.md, docs/demo_runbook.md. You MAY make small fixes in other packages' files when integration requires it; list every such edit in open_issues.
The other packages just finished; their reports: ${JSON.stringify(built)}
Build:
- pipeline.py: Pipeline(cfg, bus) wiring: live mode = WP2 MicShieldStream (shield driver in the callback, raw+shielded rings) + LoopbackStream (far-end ring) + KeyClock; replay mode = WP2 replay sources with the same timing. Workers (threads): voice worker (every 2 s: if VAD speech_fraction of the last 4 s > 0.5, score via voice driver (quarantined) -> publish voice.window/voice.verdict), attacker worker (after key events, when the post-onset window is available: run attacker on raw ring and on shielded ring at the same onsets -> keys.readout with truth for the demo), threat tick (ThreatEngine.tick at 4 Hz). Controls: shield mode switch, scenario start/stop.
- cli.py: callguard run --mode live|replay [--scenario NAME] [--drivers real|mock] [--port 8765] [--no-browser]; callguard devices; callguard bench (latency of each driver on synthetic/real audio). Starts uvicorn with WP6 create_app.
- demo/: scenario 'ai_caller' (toml): timeline: 0-12 s far-end REAL colleague speech; 12-40 s far-end AI agent (synthetic) asks for the reset code; 22-34 s local user types a fake code (e.g. 'RESET4821' mapped to Keyguard classes A-Z0-9) with local mic speech underneath; shield toggles on at 28 s in the scripted run; 40-50 s agent hangs up, real colleague returns. build_scenario_audio.py assembles the audio into ${CG}\\demo\\audio (gitignored) from READ-ONLY sources: real far-end clips = Hearsay manifest bonafide clips from split test_internal (e.g. librispeech); synthetic agent = Hearsay manifest spoof clips from test_internal of a modern generator (prefer elevenlabs or xtts_v2 or playht; print which), concatenated; keystrokes = Keyguard harrison presses for the code characters placed at the scripted times into the local mic track with local real speech. render_agent.py: optional TTS rendering of custom agent lines with transformers VITS (facebook/mms-tts-eng) for live Zoom tests (document that this voice family is in Hearsay's training data). agent_caller.py: plays WAV lines to a chosen output device (for the second device in a Zoom call).
- tests/test_e2e_replay.py: replay the scenario with MOCK drivers as fast as possible; assert the threat level sequence reaches CRITICAL during the agent+typing segment and returns to SAFE/WATCH at the end, and that readout accuracy raw > shielded. Also a real-driver e2e (skip unless HEARSAY_ROOT and KEYGUARD_ROOT exist) that runs the full scenario once and records the level timeline.
- Actually RUN the real-driver replay once end to end (callguard run --mode replay --scenario ai_caller --drivers real --no-browser for the scenario length, with a flag to exit when the scenario ends) and report the observed threat/level timeline, voice verdicts per segment, and attacker readouts. Check the server serves the dashboard (curl / and /api/state while running).
- README.md (judge-facing: what it is, the threat, architecture diagram (text), quickstart, demo modes, results pointers incl. reports/attack_under_speech.md and Hearsay generalization numbers, credits: Hearsay (ours) + Keyguard (LordKarV) ) and docs/demo_runbook.md (step-by-step expo script incl. offline fallback).
Run the full test suite at the end.`, { label: 'wp8-integrate', phase: 'Integrate', schema: RESULT })

phase('Review')
const review = await agent(`You are an independent reviewer of the CallGuard phase-1 build at ${CG}. Do NOT edit any files and do NOT run git write commands. Read plans/00_brief.md, 01_spec.md, 02_architecture.md, 05_workplan_and_merge.md, then review the code.
Verify, with evidence (commands run and their output, file:line):
1. Full test suite: cd "${CG}" && .venv/Scripts/python -m pytest -q  (report counts).
2. Acceptance items in plan 01 section 6 and plan 05 section 3: which are met, partially met, not met. Actually run: .venv/Scripts/python -m callguard devices ; the mock replay e2e; and if time allows the real-driver replay (it can take a few minutes).
3. Hard rules: nothing written inside ${HS} or ${KG} (run git -C "${HS}" status --short and git -C "${KG}" status --short; report any changes, noting that Hearsay may already have unrelated pre-existing untracked files: list them rather than judging); no audio/weights/secrets staged in ${CG} (git -C "${CG}" status --short and check .gitignore coverage); audio path never blocked by model calls (inspect pipeline/streams: model calls off the audio callback, quarantine present).
4. Correctness bugs in threat.py (matches plan 02 section 4?), hearsay_real.py (sha check, prep, threshold calibration read-only), keyguard_real.py (window alignment with PRE_S, streaming shield latency), pipeline.py (sample-index alignment between key events and ring buffers; race conditions).
5. The attack-proof report: are the claims supported by its own table? Any leakage (test presses used in training)?
Integration report for context: ${JSON.stringify(integ)}
Return findings ranked by severity with concrete fixes.`, {
  label: 'review', phase: 'Review',
  schema: {
    type: 'object',
    properties: {
      tests: { type: 'string' },
      acceptance: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, status: { type: 'string' }, evidence: { type: 'string' } }, required: ['item', 'status'] } },
      findings: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string' }, file: { type: 'string' }, issue: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'issue'] } },
      upstream_untouched: { type: 'string' },
    },
    required: ['tests', 'acceptance', 'findings', 'upstream_untouched'],
  },
})

return { built, integ, review }
