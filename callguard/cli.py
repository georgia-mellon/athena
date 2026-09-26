"""callguard run | devices | bench (plan 01 §3).

  callguard run --mode replay --scenario ai_caller [--drivers real|mock] [--exit-at-end] [--mute] [--no-browser]
  callguard run --mode live                        (VB-CABLE + Zoom, plans/03)
  callguard devices                                (routing check; prints the VB-CABLE install steps if missing)
  callguard bench [--drivers real|mock]            (per-driver latency)
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import warnings
import webbrowser

import numpy as np

from callguard import config
from callguard.types import BLOCK, SR


def _cfg(args) -> config.Config:
    cfg = config.load(args.config)
    if args.drivers:
        cfg.drivers.voice = cfg.drivers.attacker = cfg.drivers.shield = args.drivers
    if getattr(args, "mic", None):
        cfg.devices.mic = args.mic
    if getattr(args, "speaker", None):
        cfg.devices.loopback = args.speaker
    return cfg


def cmd_run(args) -> int:
    import uvicorn

    from callguard import hooks
    from callguard.bus import EventBus
    from callguard.pipeline import Pipeline, load_scenario
    from callguard.server.app import create_app

    cfg = _cfg(args)
    port = args.port or cfg.server.port
    if args.mode == "replay":
        load_scenario(args.scenario)                    # fail fast if the audio isn't built
    bus = EventBus()
    hooks.install(bus, cfg.hooks)
    print(f"[callguard] loading drivers: voice={cfg.drivers.voice} ({cfg.drivers.hearsay_mode}) "
          f"attacker={cfg.drivers.attacker} shield={cfg.drivers.shield} ...", flush=True)
    t0 = time.perf_counter()
    pipe = Pipeline(cfg, bus)
    print(f"[callguard] drivers ready in {time.perf_counter() - t0:.1f} s: {pipe.voice.name}, {pipe.attacker.name}, "
          f"{pipe.shield.name}", flush=True)
    server = uvicorn.Server(uvicorn.Config(create_app(bus, controls=pipe), host=cfg.server.host, port=port,
                                           log_level="warning"))
    url = f"http://{cfg.server.host}:{port}/"

    def begin():
        while not server.started:
            time.sleep(0.1)
        print(f"[callguard] dashboard: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        if args.mode == "live":
            pipe.start_live()
            print("[callguard] live: mic -> shield -> virtual mic; scoring the meeting's output. Ctrl+C to stop.")
        else:
            time.sleep(args.delay)                      # let the browser connect before the story starts
            end = (lambda: setattr(server, "should_exit", True)) if args.exit_at_end else None
            print(f"[callguard] replay: {pipe.scenario('start', args.scenario, on_end=end, play=not args.mute)}",
                  flush=True)
    threading.Thread(target=begin, daemon=True).start()
    try:
        server.run()
    finally:
        pipe.stop()
        bus.flush()
        bus.close()
    return 0


def cmd_devices(args) -> int:
    from callguard.audio.devices import list_devices, routing_status
    for d in list_devices():
        print(f"{d['index']:3d}  in {d['inputs']} out {d['outputs']}  {d['hostapi']:<20} {d['name']}")
    ok, report = routing_status()
    print("\n" + report)
    return 0 if ok else 1


def _time(fn, n: int = 5) -> tuple[float, float]:
    fn()                                                # warm-up
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ms)), float(np.max(ms))


def cmd_bench(args) -> int:
    from callguard.drivers.base import make_attacker, make_shield, make_voice
    cfg = _cfg(args)
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32)
    key = np.zeros(SR, np.float32)
    key[SR // 2: SR // 2 + 200] = 0.5
    rows = []
    for label, make in (("voice", make_voice), ("attacker", make_attacker), ("shield", make_shield)):
        t0 = time.perf_counter()
        d = make(cfg)
        load = (time.perf_counter() - t0) * 1000
        if label == "voice":
            rows.append((d.name, "score 4 s window", load, *_time(lambda: d.score(speech))))
            rows.append((d.name, "score 2 s window", load, *_time(lambda: d.score(speech[:2 * SR]))))
        elif label == "attacker":
            rows.append((d.name, "read 1 keystroke", load, *_time(lambda: d.read(key, np.array([SR // 2])))))
        else:
            blk = speech[:BLOCK]
            d.reset()
            rows.append((d.name, "20 ms block, no key", load, *_time(lambda: d.process(blk, []), 50)))
            d.reset()
            n = [0]

            def keyed():  # a key event every block (absolute index = samples since reset): every block is touched
                n[0] += 1
                return d.process(blk, [n[0] * BLOCK])
            rows.append((d.name, "20 ms block, key active", load, *_time(keyed, 50)))
    print(f"{'driver':<34} {'op':<24} {'load ms':>8} {'median ms':>10} {'max ms':>8}")
    for name, op, load, med, mx in rows:
        print(f"{name:<34} {op:<24} {load:8.0f} {med:10.1f} {mx:8.1f}")
    print(f"(audio block budget: {BLOCK / SR * 1000:.0f} ms; voice runs every 2 s off the audio thread)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    warnings.filterwarnings("ignore", message="data discontinuity")  # soundcard loopback while nothing plays
    p = argparse.ArgumentParser(prog="callguard", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", help="TOML config (default: ./callguard.toml if present)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline and the dashboard")
    r.add_argument("--mode", choices=("live", "replay"), default="replay")
    r.add_argument("--scenario", default="ai_caller")
    r.add_argument("--drivers", choices=("real", "mock"), help="override every driver slot")
    r.add_argument("--port", type=int)
    r.add_argument("--mic", help="live: physical mic (substring of the device name)")
    r.add_argument("--speaker", help="live: the speaker the meeting plays to (loopback-captured)")
    r.add_argument("--no-browser", action="store_true")
    r.add_argument("--exit-at-end", action="store_true", help="replay: exit when the scenario ends")
    r.add_argument("--mute", action="store_true", help="replay: don't play the audio on the speakers")
    r.add_argument("--delay", type=float, default=3.0, help="replay: seconds before the scenario starts")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("devices", help="list audio devices and check the Zoom routing").set_defaults(fn=cmd_devices)
    b = sub.add_parser("bench", help="per-driver latency")
    b.add_argument("--drivers", choices=("real", "mock"), help="override every driver slot")
    b.set_defaults(fn=cmd_bench)
    args = p.parse_args(argv)
    if not hasattr(args, "drivers"):
        args.drivers = None
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
