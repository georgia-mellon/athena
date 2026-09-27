"""Shared constants. ponytail: one place for the knobs everything tunes against."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # Athena: the athena repo (vendored copy, keyguard/VENDORED.md)
DATA = ROOT / "data" / "keyguard"                # Athena: gitignored, filled by app.keystroke_guard.get_assets
RUNS = ROOT / "runs" / "keyguard"                # Athena: gitignored weights + agent memory

# Load .env so GEMINI_API_KEY / BACKBOARD_API_KEY reach os.environ everywhere.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SR = 16_000              # everything resampled to this
KEY_WIN_S = 0.30        # audio window around a keystroke onset (press+release)
KEY_WIN = int(SR * KEY_WIN_S)
PRE_S = 0.02            # grab a little before the onset
N_MELS = 64
N_FFT = 1024
HOP = 128               # ~8ms hops -> ~38 frames per 0.3s window

# onset detection
ONSET_MIN_GAP_S = 0.10  # keys can't be closer than this
ONSET_PROM = 0.10       # peak prominence as fraction of max envelope

# 26 letters + 10 digits + space (spacebar is acoustically distinct; recognizing it
# gives word boundaries that hugely help LM reconstruction). Harrison MBPWavs is A-Z0-9
# only, so space is learned from real continuous samples (labels.jsonl), not bank synth.
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)] + [" "]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
