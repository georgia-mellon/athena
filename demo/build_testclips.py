"""The test room's 10 test voices -> demo/audio/testclips (gitignored): 5 real + 5 AI clips from Hearsay's held-out
test_internal_testlike set, the ones E5 scored most clearly (every window) through the room's audio path.

    python demo/build_testclips.py        (needs HEARSAY_ROOT, default ../Hearsay)
"""
import librosa
import soundfile as sf

from app.hearsay.driver import HEARSAY_ROOT
from app.source.config import REPO

CLIPS = {
    "real_1_librispeech": "librispeech/dev-clean/5536/5536-43358-0001.wav",
    "real_2_librispeech": "librispeech/train-clean-360/3180/3180-138043-0043.wav",
    "real_3_ljspeech": "ljspeech/LJ002-0113.wav",
    "real_4_commonvoice": "cvoicefake_en/bonafide/common_voice_en_36752035.wav",
    "real_5_commonvoice": "cvoicefake_en/bonafide/common_voice_en_36800565.wav",
    "ai_1_unit_speech": "diffssd/unit_speech/speaker_5448/sentence_181.wav",
    "ai_2_wavefake_ljspeech_melgan_large": "wavefake/ljspeech_melgan_large/LJ005-0073_gen.wav",
    "ai_3_mlaad_microsoft_speecht5_tts": "mlaad_tiny/mlaad_microsoft_speecht5_tts/wives_and_daughters_50_f000174.wav",
    "ai_4_librisevoc_parallel_wave_gan": "librisevoc/librisevoc_parallel_wave_gan/3526_176653_000073_000002_gen.wav",
    "ai_5_asvspoof5_A26": "asvspoof5/asvspoof5_A26/E_0006661910.wav",
}

if __name__ == "__main__":
    out = REPO / "demo" / "audio" / "testclips"
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in CLIPS.items():
        y, sr = sf.read(HEARSAY_ROOT / "data" / "processed" / rel, dtype="float32", always_2d=True)
        y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=16000) if sr != 16000 else y.mean(1)
        sf.write(out / f"{name}.wav", y, 16000)
        print(out / f"{name}.wav")
