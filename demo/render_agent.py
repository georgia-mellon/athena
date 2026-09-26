"""Render the AI agent's lines to WAVs with an open-source TTS (VITS, facebook/mms-tts-eng) for live Zoom tests.

Output: demo/audio/agent_lines/NN.wav (16 kHz), played into the call by demo/agent_caller.py.
Caveat: this voice family is likely in Hearsay's training data (its manifest has sim_mms_tts_eng and
mlaad_facebook_mms_tts_eng), so Hearsay flagging it is an EASY case, not evidence of generalisation. The replay
scenario (demo/build_scenario_audio.py) uses held-out ElevenLabs clones instead. Fake codes only.

    python demo/render_agent.py                       # default lines
    python demo/render_agent.py "Line one." "Line two."
"""
from __future__ import annotations

import argparse
from pathlib import Path

OUT = Path(__file__).resolve().parent / "audio" / "agent_lines"
LINES = [
    "Hi, this is IT support. We're seeing unusual sign-ins on your account.",
    "To lock it down, I've sent a reset code to your email.",
    "Please type the reset code while we're on the line, so I can confirm it on my side.",
    "Hmm, that code was rejected. Could you type it again for me?",
    "Great, that worked. Thanks for your help, have a good day.",
    "Just read me the verification code, please.",   # 06: the spoken-secret beat (keep it under 2 s)
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="*", help="lines to speak (default: the built-in IT-support script)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0, help="VITS is stochastic; seed it for repeatable audio")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
    sr = model.config.sampling_rate
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    for i, text in enumerate(a.lines or LINES, 1):
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        wav = resample_poly(wav, 16000, sr) if sr != 16000 else wav
        wav = wav * (10 ** (-23 / 20) / (np.sqrt(np.mean(wav ** 2)) + 1e-12))    # ~-23 dBFS RMS, like the scenario
        path = a.out / f"{i:02d}.wav"
        sf.write(path, wav.astype(np.float32), 16000, subtype="FLOAT")
        print(f"{path.name}: {len(wav) / 16000:.1f} s  {text}")


if __name__ == "__main__":
    main()
