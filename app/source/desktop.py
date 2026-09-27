"""CallGuard as a desktop app: engine + dashboard server in-process, dashboard in a native window.

  python -m app.source.desktop [--meet-url URL] [--port N] [--drivers real|mock] [--no-window]

The engine runs in meet mode (the Google Meet bridge feeds it mic and far-end audio over /meet/*). The server runs
on a background thread; the window (pywebview: Edge WebView2 on Windows) owns the main thread. No pywebview or no
WebView2 -> the system browser opens the dashboard and we serve until Ctrl+C. Closing the window stops everything.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import urllib.request
import webbrowser

from app.source.cli import _cfg

log = logging.getLogger("callguard.desktop")
STOP = threading.Event()  # set it to shut a headless main() down (tests; window-less runs use Ctrl+C)


def _wait_healthy(url: str, server, thread: threading.Thread | None = None, timeout: float = 30.0) -> bool:
    """True once OUR server answers GET /api/health; False if it died (e.g. the port is taken by another CallGuard,
    which would answer the health check itself) or the timeout passed."""
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


def _open_window(url: str) -> bool:
    """Blocks until the native window closes. False if pywebview couldn't start a GUI backend."""
    try:
        import webview
        webview.create_window("CallGuard", url, width=1400, height=950, min_size=(900, 640))
        webview.start()
        return True
    except Exception as e:                              # ImportError, no WebView2 runtime, no GUI backend
        log.warning("native window unavailable (%s); opening the system browser", e)
        return False


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from app.source import hooks
    from app.source.bus import EventBus
    from app.source.pipeline import Pipeline
    from dashboard.server import create_app

    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(prog="callguard app", description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", help="TOML config (default: ./callguard.toml if present)")
    p.add_argument("--drivers", choices=("real", "mock"), help="override every driver slot")
    p.add_argument("--port", type=int)
    p.add_argument("--meet-url", help="join this Meet as soon as the app is up")
    p.add_argument("--no-window", action="store_true", help="headless: serve only (tests, remote use)")
    args = p.parse_args(argv)
    STOP.clear()

    cfg = _cfg(args)
    port = args.port or cfg.server.port
    url = f"http://{cfg.server.host}:{port}/"
    bus = EventBus()
    hooks.install(bus, cfg.hooks)
    print(f"[callguard] loading drivers: voice={cfg.drivers.voice} attacker={cfg.drivers.attacker} "
          f"shield={cfg.drivers.shield} secret={cfg.drivers.secret} ...", flush=True)
    pipe = Pipeline(cfg, bus)
    pipe.meet_port = port                               # the injected Meet bridge dials this server, not the default
    if hasattr(pipe, "start_meet"):
        pipe.start_meet()                               # before the server: no dashboard Arm can race its reset
    else:
        print("[callguard] this build has no meet mode yet; the pipeline stays idle", flush=True)
    app = create_app(bus, controls=pipe)
    try:
        from app.source.connectors.meet.router import make_router
        app.include_router(make_router(pipe, port=port))
    except ImportError as e:
        log.warning("Meet connector not available (%s); dashboard only", e)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.server.host, port=port, log_level="warning"))
    server.install_signal_handlers = lambda: None       # not the main thread: Ctrl+C is handled below
    srv = threading.Thread(target=server.run, name="callguard-server", daemon=True)
    srv.start()
    try:
        if not _wait_healthy(url, server, srv):
            print(f"[callguard] server did not come up on {url}", file=sys.stderr, flush=True)
            return 1
        print(f"[callguard] dashboard: {url}", flush=True)
        if args.meet_url and hasattr(pipe, "meet"):
            try:
                pipe.meet("join", args.meet_url)
            except Exception as e:                      # a bad URL or no browser must not kill the app
                print(f"[callguard] could not join {args.meet_url}: {e}", file=sys.stderr, flush=True)
        if args.no_window or not _open_window(url):
            if not args.no_window:
                webbrowser.open(url)
            print("[callguard] serving; Ctrl+C to stop", flush=True)
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
