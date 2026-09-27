"""Scripted scam calls in an AI voice -> demo/audio/testclips/scam_{1_bank,2_it_desk,3_login}.wav (+ .mp3 for a phone).

Each call asks the victim to TYPE a code (keystroke leak) and to READ one out loud ("read out your PIN", "read me
the code"): the reads are the Secret Shield's request triggers, so it arms and cuts the digits from your mic.
scam_3_login covers both cases in one call. Fake scripts, no real numbers. Voice: VITS (facebook/mms-tts-eng),
which Hearsay flags as AI.

    python demo/build_scam_calls.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "audio" / "testclips"
SR = 16000
# (line, seconds of silence after it)
SCRIPTS = {
    "scam_1_bank": [
        ("Hello, this is Sarah from the fraud prevention team at your bank.", 0.8),
        ("We noticed a suspicious charge of eight hundred dollars on your debit card this morning.", 0.8),
        ("Before I can block it, I need to verify that I am speaking with the account holder.", 0.8),
        ("I have just sent a six digit PIN to your phone in a text message.", 0.8),
        ("Can you type the PIN from that text message into the secure form I sent you?", 7.0),
        ("Thank you. Our system also needs a verbal confirmation.", 0.8),
        ("Can you read out your PIN for me, please?", 8.0),
        ("Sorry, the line cut out on my end. Please read me the code again, slowly.", 8.0),
        ("Now, to block the card itself, what is your card number?", 9.0),
        ("Perfect. The charge has been blocked. Thank you for your patience today.", 1.0),
    ],
    "scam_2_it_desk": [
        ("Hi, this is Mike from the IT help desk.", 0.8),
        ("We detected a sign in to your work account from another country about ten minutes ago.", 0.8),
        ("To secure the account, I am resetting your password right now.", 0.8),
        ("You should see a one time code in your authenticator app.", 0.8),
        ("Please type that code into the sign in page while I stay on the line.", 7.0),
        ("That did not go through on my side. I will need you to read me the verification code.", 8.0),
        ("Okay. And to confirm your identity, what is your employee PIN number?", 8.0),
        ("Last step. Can you read out the code from the new text message?", 8.0),
        ("Great, your account is secure now. Have a good day.", 1.0),
    ],
    # both cases in one call: the victim logs in (username + password) while the caller stays on the line (keystroke
    # shield), then the caller asks for the code and PIN out loud (request triggers: the Secret Shield arms).
    "scam_3_login": [
        ("Hello, this is Daniel from the account security team.", 0.8),
        ("Your account was flagged for unusual activity overnight, so we have locked it as a precaution.", 0.8),
        ("I can unlock it for you right now while we are on the call.", 0.8),
        ("Please open the sign in page on your computer and click log in.", 4.0),
        ("Now type your username or email address into the first box.", 10.0),
        ("Good. Next, type your password into the password field, and press enter.", 14.0),
        ("Hmm, it says the sign in failed. Please type your password one more time, slowly.", 14.0),
        ("That went through. The system just sent a verification code to your phone.", 0.8),
        ("Can you read me the code, please?", 8.0),
        ("Thank you. And to finish unlocking the account, what is your PIN number?", 8.0),
        ("Perfect. Your account is unlocked. Please stay logged in for the next few minutes.", 1.0),
    ],
}


def main() -> None:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in SCRIPTS.items():
        torch.manual_seed(0)  # VITS is stochastic: seed it for repeatable audio
        parts = []
        for text, pause in lines:
            with torch.no_grad():
                wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
            wav = resample_poly(wav, SR, model.config.sampling_rate)
            parts += [wav * (0.05 / (np.sqrt(np.mean(wav ** 2)) + 1e-9)), np.zeros(int(pause * SR))]
        y = np.concatenate(parts).astype(np.float32)
        sf.write(OUT / f"{name}.wav", y, SR)
        sf.write(OUT / f"{name}.mp3", y / max(1e-9, float(np.abs(y).max())) * 0.9, SR, format="MP3")
        print(OUT / f"{name}.wav", f"{len(y) / SR:.0f} s (+ .mp3)")


if __name__ == "__main__":
    main()
