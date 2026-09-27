"""Athena as a desktop app: engine + dashboard server in-process, dashboard in a native window.

  python -m app.source.desktop [--meet-url URL] [--port N] [--drivers real|mock] [--no-window]

The engine runs in meet mode; the server runs on a background thread and the pywebview window owns the main
thread. No pywebview/WebView2 -> the system browser opens the dashboard and we serve until Ctrl+C. Closing
the window stops everything.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.request
import webbrowser

from app.source.cli import _cfg

log = logging.getLogger("athena.desktop")
STOP = threading.Event()  # set it to shut a headless main() down (tests; window-less runs use Ctrl+C)


def _wait_healthy(url: str, server, thread: threading.Thread | None = None, timeout: float = 30.0) -> bool:
    """True once our server answers GET /api/health; False if it died or the timeout passed."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not server.should_exit and (thread is None or thread.is_alive()):
        if not server.started:
            time.sleep(0.1)
            continue
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.1)
    return False


LOADING = """<!doctype html><html><head><meta charset="utf-8"><title>Athena</title></head>
<body style="margin:0;height:100vh;display:grid;place-items:center;background:#000;color:#fafafa;
font:14px/1.5 system-ui,'Segoe UI',sans-serif"><div style="text-align:center">
<div style="font-size:22px;font-weight:800;color:#fdae17">athena</div>
<div id="s" style="color:#a3a3a3;margin-top:6px">Starting&hellip;</div></div></body></html>"""


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(prog="athena app", description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", help="TOML config (default: ./athena.toml if present)")
    p.add_argument("--drivers", choices=("real", "mock"), help="override every driver slot")
    p.add_argument("--port", type=int)
    p.add_argument("--meet-url", help="join this Meet as soon as the app is up")
    p.add_argument("--no-window", action="store_true", help="headless: serve only (tests, remote use)")
    args = p.parse_args(argv)
    STOP.clear()
    if args.no_window:
        return _run(args)
    try:                                                # window first, with a status line, while models load
        import webview
        win = webview.create_window("Athena", html=LOADING, width=1400, height=950, min_size=(900, 640))
    except Exception as e:                              # no webview / WebView2 / GUI backend
        log.warning("native window unavailable (%s); opening the system browser", e)
        return _run(args, on_ready=webbrowser.open)

    out: dict = {}

    def say(msg: str) -> None:
        try:
            win.evaluate_js(f"document.getElementById('s').textContent = {json.dumps(msg)}")
        except Exception:  # noqa: BLE001 - the window is gone or not ready yet
            pass

    def boot() -> None:
        out["rc"] = _run(args, say=say, on_ready=win.load_url)
        if out["rc"]:
            say("Athena could not start: see the terminal")
    win.events.closed += STOP.set                       # closing the window stops everything
    try:
        webview.start(boot)                             # boot runs on its own thread; the window owns this one
    except Exception as e:  # noqa: BLE001 - no GUI backend after all
        log.warning("native window unavailable (%s); opening the system browser", e)
        return _run(args, on_ready=webbrowser.open)
    STOP.set()
    return out.get("rc", 0)


def _run(args, say=lambda msg: None, on_ready=None) -> int:
    """Engine + server until STOP: load drivers, warm up the voice model, serve, then on_ready(url)."""
    import uvicorn

    from app.source import hooks
    from app.source.bus import EventBus
    from app.source.pipeline import Pipeline
    from dashboard.server import create_app

    cfg = _cfg(args)
    port = args.port or cfg.server.port
    url = f"http://{cfg.server.host}:{port}/"
    bus = EventBus()
    hooks.install(bus, cfg.hooks)
    print(f"[athena] loading drivers: voice={cfg.drivers.voice} attacker={cfg.drivers.attacker} "
          f"shield={cfg.drivers.shield} secret={cfg.drivers.secret} ...", flush=True)
    say("Loading the voice, keystroke and speech models (about 20 s)...")
    pipe = Pipeline(cfg, bus)
    pipe.meet_port = port                               # the test room's bridge dials this server
    if port != 8765:
        print(f"[athena] note: the Meet extension connects to port 8765; on port {port} only the test room is "
              "protected", file=sys.stderr, flush=True)
    say("Warming up the voice model...")
    pipe.warm_up()
    if hasattr(pipe, "start_meet"):
        pipe.start_meet()                               # before the server: no dashboard Arm can race its reset
    else:
        print("[athena] this build has no meet mode yet; the pipeline stays idle", flush=True)
    say("Starting the dashboard...")
    app = create_app(bus, controls=pipe)
    try:
        from app.source.connectors.meet.router import make_router
        app.include_router(make_router(pipe, port=port))
    except ImportError as e:
        log.warning("Meet connector not available (%s); dashboard only", e)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.server.host, port=port, log_level="warning"))
    server.install_signal_handlers = lambda: None       # not the main thread: Ctrl+C is handled below
    srv = threading.Thread(target=server.run, name="athena-server", daemon=True)
    srv.start()
    try:
        if not _wait_healthy(url, server, srv):
            print(f"[athena] server did not come up on {url}", file=sys.stderr, flush=True)
            return 1
        pipe.announce()                                 # the server wasn't listening when meet mode started
        print(f"[athena] dashboard: {url}", flush=True)
        if args.meet_url and hasattr(pipe, "meet"):
            try:
                pipe.meet("join", args.meet_url)
            except Exception as e:  # a bad URL or no browser must not kill the app
                print(f"[athena] could not join {args.meet_url}: {e}", file=sys.stderr, flush=True)
        if on_ready is not None:
            on_ready(url)
        print("[athena] serving; close the window or Ctrl+C to stop", flush=True)
        while srv.is_alive() and not server.should_exit and not STOP.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        server.should_exit = True
        srv.join(timeout=5)
        bus.flush()
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
