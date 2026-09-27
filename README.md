# Athena

<!-- TEMPLATE: one-line pitch. Working line: "Athena protects what you hear, what you type, and what you say." -->

<!-- TEMPLATE: 2-3 sentences: what Athena is (a desktop app that guards a Google Meet call), who it's for, HackGT 13 / team GeorgiaMellon. -->

## The threat
<!-- TEMPLATE: the story in a few lines: an AI voice agent joins a Meet, asks you to type a reset code, then to read it out. What goes wrong on each channel (hear / type / say). -->

## How it works
<!-- TEMPLATE: one diagram (text or image) of Meet bridge -> pipeline -> three pillars -> threat score -> dashboard, and 3-5 bullets on the design choices (no driver install, fail-open audio, everything on-device). -->

## The three pillars

### Hearsay: what you hear
<!-- TEMPLATE: SHORT overview only (Hearsay is a subpart of Athena): what it detects, the one headline number, and a link to the Hearsay model repository for the full model documentation. -->

### Keystroke Guard: what you type
<!-- TEMPLATE: the keystroke-leak attack, the shield (DSP + adversarial), the headline numbers, credit to the Keystroke Guard author. -->

### Secret Shield: what you say
<!-- TEMPLATE: redacting codes/PINs/card numbers from your outgoing voice while the caller is unverified; the headline numbers and the known limit. -->

## Threat score
<!-- TEMPLATE: how the three signals combine into SAFE / WATCH / WARN / CRITICAL (link docs/plans/02_architecture.md). -->

## Quickstart
<!-- TEMPLATE: prerequisites (Windows, Python 3.12, uv, Chrome/Edge), install, get the Vosk model, run the desktop app, join a Meet, the test room, the offline replay demo. Commands in one code block. -->

## Demo
<!-- TEMPLATE: the replay story timeline and the live Meet demo; link docs/demo_runbook.md. Screenshot/GIF of the dashboard. -->

## Results
<!-- TEMPLATE: one small table per pillar, numbers only from docs/reports/*.md, with honest limits. -->

## Repository layout
<!-- TEMPLATE: app/source, app/hearsay, app/keystroke_guard, app/secret_shield, dashboard, demo, docs, tests: one line each. -->

## Team and credits
<!-- TEMPLATE: team GeorgiaMellon members; Hearsay (model repository link); Keystroke Guard author; third-party models (Vosk, XLS-R, ...) and licences. -->

## Ethics
<!-- TEMPLATE: fake codes only, consenting voices, nothing leaves the device, no recordings committed. -->
