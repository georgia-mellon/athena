"""Fetch the Vosk small English model into runs/models/ (gitignored) for the spoken-secret shield.

Idempotent. Verifies the zip against a pinned sha256, then appends the low-latency decoder options (LOW_LATENCY)
to conf/model.conf; without them Vosk refreshes partial word timings only every ~2 s, too late for the 500 ms
delay line.
Run: .venv/Scripts/python app/secret_shield/get_model.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.secret_shield.spotter import LOW_LATENCY, MODEL_DIR  # noqa: E402

NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
MODELS = MODEL_DIR.parent


def patch_conf(dest: Path) -> None:
    conf = dest / "conf" / "model.conf"
    lines = conf.read_text().split()
    missing = [o for o in LOW_LATENCY if o not in lines]
    if missing:
        conf.write_text("\n".join(lines + missing) + "\n")


def main() -> int:
    dest = MODELS / NAME
    if (dest / "am" / "final.mdl").exists():
        patch_conf(dest)
        print(f"ok: {dest} already present")
        return 0
    MODELS.mkdir(parents=True, exist_ok=True)
    zpath = MODELS / f"{NAME}.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != SHA256:
        zpath.unlink()
        print(f"sha256 mismatch: got {digest}, pinned {SHA256}; zip deleted", file=sys.stderr)
        return 1
    with zipfile.ZipFile(zpath) as z:
        z.extractall(MODELS)
    zpath.unlink()
    patch_conf(dest)
    print(f"ok: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
