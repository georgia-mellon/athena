# Dashboard

FastAPI server (`server.py`) + a static UI (`static/`, no CDNs). `create_app(bus, state_provider, controls)` serves
the page, pushes every bus event over `/ws`, and exposes the demo controls. The desktop app shows it in a native
window; `callguard run` opens it in the browser at <http://127.0.0.1:8765/>.

| route | what |
|---|---|
| `GET /` | the dashboard: threat gauge and reasons, voice light + p_synthetic, keystroke readouts (raw vs shielded), Secret Shield panel, Meeting bar, event log |
| `GET /api/state` | latest event per topic |
| `GET /api/health` | liveness (the desktop shell waits on it) |
| `POST /api/control/{shield,secret,meet,scenario}` | shield mode, Secret Shield arm/disarm/auto/allow, Meet join/leave, replay start/stop |
| `WS /ws` | the live event stream |

Local only: requests and WebSocket origins must be `127.0.0.1` / `localhost` / `::1` (the page shows what an
eavesdropper reads from your keyboard). Bus callbacks hand events to the server loop; each browser has a bounded
queue, so a slow client never stalls a driver. The Meet connector adds its `/meet/*` routes to the same app.
