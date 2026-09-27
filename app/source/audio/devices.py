"""Device discovery and routing check (plan 03).

Outbound: physical mic -> Athena -> "CABLE Input" (VB-CABLE playback side); Zoom's mic = "CABLE Output".
Inbound: WASAPI loopback of the speaker Zoom plays to (soundcard, include_loopback=True).
"""
from __future__ import annotations

CABLE_IN = "cable input"    # playback device Athena writes the shielded mic into
CABLE_OUT = "cable output"  # recording device the meeting app selects as its microphone
PREFERRED_HOSTAPIS = ("Windows WASAPI", "MME", "Windows DirectSound")  # duplex needs both ends on one host API

INSTALL_STEPS = """VB-CABLE (virtual microphone) not found.
  1. Download VBCABLE_Driver_Pack from https://vb-audio.com/Cable/ and unzip it.
  2. Right-click VBCABLE_Setup_x64.exe -> Run as administrator -> Install Driver.
  3. Reboot, then run `athena devices` again.
  4. In Zoom: Settings > Audio > Microphone = "CABLE Output (VB-Audio Virtual Cable)"; Suppress background noise = Low.
  (macOS: install BlackHole 2ch instead, `brew install blackhole-2ch`.) Full guide: docs/meeting_setup.md"""


def list_devices() -> list[dict]:
    import sounddevice as sd
    apis = [a["name"] for a in sd.query_hostapis()]
    return [dict(index=i, name=d["name"], hostapi=apis[d["hostapi"]], inputs=d["max_input_channels"],
                 outputs=d["max_output_channels"], samplerate=d["default_samplerate"])
            for i, d in enumerate(sd.query_devices())]


def find_device(name: str | None, kind: str, hostapi: str | None = None, devices: list[dict] | None = None,
                exclude: str | None = None) -> dict | None:
    """First device whose name contains `name` (case-insensitive; None = any) and not `exclude`, with channels of
    `kind` ('input'|'output'), preferring host APIs in PREFERRED_HOSTAPIS order (or exactly `hostapi`)."""
    devices = list_devices() if devices is None else devices
    key = "inputs" if kind == "input" else "outputs"
    hits = [d for d in devices if d[key] > 0 and (name is None or name.lower() in d["name"].lower())
            and not (exclude and exclude.lower() in d["name"].lower())]
    for api in ([hostapi] if hostapi else PREFERRED_HOSTAPIS):
        for d in hits:
            if d["hostapi"] == api:
                return d
    return None if hostapi else (hits[0] if hits else None)


def pick_mic(devices: list[dict] | None = None, name: str | None = None) -> dict | None:
    """The physical mic: `name` if given, else the first input that isn't the virtual cable itself.
    ponytail: first-match heuristic; pass the mic name (CLI/config) when a laptop lists several."""
    return find_device(name, "input", devices=devices, exclude="cable")


def loopback_speakers() -> list[str]:
    """Names of devices whose output can be captured via WASAPI loopback."""
    try:
        import soundcard as sc
        return [m.name for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    except Exception as e:  # noqa: BLE001
        print(f"[devices] soundcard loopback unavailable: {e}")
        return []


def default_speaker() -> str | None:
    try:
        import soundcard as sc
        return sc.default_speaker().name
    except Exception:  # noqa: BLE001
        return None


def routing_status(devices: list[dict] | None = None, speakers: list[str] | None = None) -> tuple[bool, str]:
    """(ok, human-readable report) for `athena devices`. ok = VB-CABLE present and a loopback speaker exists."""
    devices = list_devices() if devices is None else devices
    speakers = loopback_speakers() if speakers is None else speakers
    cable_in = find_device(CABLE_IN, "output", devices=devices)
    cable_out = find_device(CABLE_OUT, "input", devices=devices)
    mic = pick_mic(devices)
    lines = [f"mic (default pick):     {mic['name'] + ' [' + mic['hostapi'] + ']' if mic else 'NONE'}",
             f"virtual mic out:        {cable_in['name'] + ' [' + cable_in['hostapi'] + ']' if cable_in else 'MISSING'}",
             f"meeting app mic:        {cable_out['name'] if cable_out else 'MISSING'}",
             f"loopback speakers:      {', '.join(speakers) or 'NONE'}"]
    ok = bool(cable_in and cable_out and speakers)
    if not (cable_in and cable_out):
        lines += ["", INSTALL_STEPS]
    if not speakers:
        lines += ["", "No loopback-capable speaker: inbound voice scoring needs WASAPI loopback (Windows) "
                      "or a second virtual cable as the meeting speaker (docs/meeting_setup.md)."]
    return ok, "\n".join(lines)


if __name__ == "__main__":
    print(routing_status()[1])
