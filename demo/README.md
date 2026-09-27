# demo/

Demo assets for the AI-caller story (plan 01 §2). Audio lands in `demo/audio/` (gitignored); fake code only: `RESET4821`.

| file | what it does |
|---|---|
| `scenarios/ai_caller.toml` | 60 s replay scenario: colleague (real, 0-12 s), ElevenLabs clone of that colleague asking for the reset code (12-40 s), colleague again (40-60 s). The user types RESET4821 at 20.5 s (shield off) and again at 29.5 s (shield `dsp` from 29 s). |
| `build_scenario_audio.py` | Builds `audio/ai_caller/{far_end.wav, mic.wav, keys.csv}` deterministically from read-only Hearsay test clips and Keyguard harrison TEST-split presses; prints sources, levels and per-window VAD speech fractions. |
| `agent_caller.py` | Plays WAV lines (files or a folder, in order) to an output device, for the second laptop/phone that joins the meeting as "IT Support". `--list` shows devices, `--device` picks one by substring. |
| `render_agent.py` | Renders custom agent lines to `audio/agent_lines/NN.wav` with VITS (`facebook/mms-tts-eng`). That voice family is in Hearsay's training data: an easy case. |

Build and check:

    .venv\Scripts\python demo\build_scenario_audio.py
    .venv\Scripts\python -c "from app.source.pipeline import load_scenario; print(load_scenario('ai_caller').seconds)"
    uv run athena run --mode replay --scenario ai_caller

Live Google Meet test: `python demo/render_agent.py`, then on the second device (in the same meeting)
`python demo/agent_caller.py demo/audio/agent_lines --device "<its speaker>"`. The clips in `demo/audio/` also show up
in the local test room's *Play as caller* list (`/meet/testroom`).
