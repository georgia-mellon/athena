// CallGuard <-> Google Meet audio bridge. Injected by the launcher (CDP, before Meet's own scripts) or loaded by the
// local test room. Self-contained, no dependencies.
//
// Mic:  getUserMedia -> AudioWorklet (downsample to 16 kHz, 320-sample float32 blocks) -> ws /meet/mic -> CallGuard
//       (Keyguard shield + spoken-secret delay line) -> processed blocks back -> jitter buffer (~60 ms) -> upsample ->
//       the audio track Meet sends. Video tracks are untouched.
// Far:  every remote audio track an RTCPeerConnection receives -> mixed in WebAudio -> 16 kHz blocks -> ws /meet/far
//       (Hearsay + the "read me the code" listener).
// FAIL OPEN: whenever CallGuard isn't answering (socket down, stalled, audio context blocked) the raw mic goes out
// unchanged; the socket reconnects with backoff. Only console.debug, never audio or text contents.
(() => {
  'use strict';
  const HOSTS = ['meet.google.com', '127.0.0.1', 'localhost', '[::1]'];
  if (window.__callguardBridge || !HOSTS.includes(location.hostname) || !window.AudioWorkletNode) return;
  const PORT = Number(window.__callguardPort) || 8765;
  const BASE = `ws://127.0.0.1:${PORT}/meet/`;
  const TARGET_MS = 60;                       // jitter buffer target
  const log = (...a) => console.debug('[callguard]', ...a);
  const stats = { mic: 'off', micSent: 0, micBack: 0, farSent: 0, farTracks: 0, rttMs: null };
  const NativePC = window.RTCPeerConnection;
  const md = navigator.mediaDevices;
  const nativeGUM = md && md.getUserMedia ? md.getUserMedia.bind(md) : null;
  window.__callguardBridge = { version: 1, port: PORT, source: window.__callguardBridgeSource || 'script',
                               stats, NativePC };

  // ---- the worklet: 16 kHz tap (both paths) + jitter buffer / fail-open switch (mic path) ------------------------
  const WORKLET = `
class Fifo {
  constructor(n) { this.b = new Float32Array(n); this.r = 0; this.n = 0; }
  push(x) {
    for (let i = 0; i < x.length; i++) {
      if (this.n === this.b.length) { this.r = (this.r + 1) % this.b.length; this.n--; }
      this.b[(this.r + this.n) % this.b.length] = x[i]; this.n++;
    }
  }
  shift() { if (!this.n) return null; const v = this.b[this.r]; this.r = (this.r + 1) % this.b.length; this.n--; return v; }
  drop(k) { k = Math.min(k, this.n); this.r = (this.r + k) % this.b.length; this.n -= k; }
}
class CallGuardTap extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const o = opts.processorOptions;
    this.mic = o.mode === 'mic';
    this.step = sampleRate / 16000;            // input samples per 16 kHz sample
    // ponytail: boxcar anti-alias + linear interpolation; fine for speech at 16 kHz, a polyphase filter if not.
    this.box = new Float32Array(Math.max(1, Math.round(this.step))); this.bi = 0; this.sum = 0;
    this.prev = 0; this.pos = 0; this.blk = new Float32Array(320); this.k = 0;
    this.q = new Fifo(16000); this.target = o.targetMs * 16; this.live = false; this.starve = 0;
    this.a = 0; this.b = 0; this.frac = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d instanceof Float32Array) {
        this.q.push(d);
        if (this.q.n > 3 * this.target) this.q.drop(this.q.n - this.target);   // drifted: drop, don't lag
      } else if (d && d.reset) { this.q.drop(this.q.n); this.setLive(false); }
    };
  }
  setLive(v) { if (v !== this.live) { this.live = v; this.port.postMessage({ live: v }); } }
  tap(x) {                                     // native rate -> 16 kHz blocks -> main thread
    for (let i = 0; i < x.length; i++) {
      this.sum += x[i] - this.box[this.bi]; this.box[this.bi] = x[i]; this.bi = (this.bi + 1) % this.box.length;
      const y = this.sum / this.box.length;
      while (this.pos <= 1) {
        this.blk[this.k++] = this.prev + (y - this.prev) * this.pos;
        if (this.k === 320) { this.port.postMessage(this.blk.slice()); this.k = 0; }
        this.pos += this.step;
      }
      this.pos -= 1; this.prev = y;
    }
  }
  process(inputs, outputs) {
    const inp = inputs[0], x = inp && inp.length ? inp[0] : null;
    if (x) this.tap(x);
    if (!this.mic) return true;
    const out = outputs[0][0];
    if (!this.live && this.q.n >= this.target) { this.setLive(true); this.starve = 0; }
    if (!this.live) { if (x) out.set(x); return true; }           // fail open: the raw mic
    const up = 16000 / sampleRate;
    for (let i = 0; i < out.length; i++) {
      this.frac += up;
      while (this.frac >= 1) {
        this.frac -= 1; this.a = this.b;
        const v = this.q.shift();
        if (v === null) { this.b = 0; this.starve++; } else { this.b = v; this.starve = 0; }
      }
      out[i] = this.a + (this.b - this.a) * this.frac;
    }
    if (this.starve > 200 * 16) this.setLive(false);              // 200 ms without answers: raw again
    return true;
  }
}
registerProcessor('callguard-tap', CallGuardTap);
`;
  let workletUrl = null;
  const addWorklet = (ctx) => {
    workletUrl = workletUrl || URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }));
    return ctx.audioWorklet.addModule(workletUrl);
  };
  const tapNode = (ctx, mode) => new AudioWorkletNode(ctx, 'callguard-tap', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1,
    channelCountMode: 'explicit', channelInterpretation: 'speakers', processorOptions: { mode, targetMs: TARGET_MS } });

  // ---- a WebSocket that keeps reconnecting (0.5 s doubling to 10 s) while wanted --------------------------------
  function socket(kind, onBlock) {
    const s = { ws: null, open: false, wanted: true, delay: 500, onopen: null, onclose: null };
    const retry = () => {
      if (!s.wanted) return;
      setTimeout(connect, s.delay);
      s.delay = Math.min(s.delay * 2, 10000);
    };
    function connect() {
      if (!s.wanted) return;
      let ws;
      try { ws = new WebSocket(BASE + kind); } catch (e) { retry(); return; }
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { s.open = true; s.delay = 500; log(kind, 'connected'); if (s.onopen) s.onopen(); };
      ws.onmessage = (e) => { if (onBlock && e.data instanceof ArrayBuffer) onBlock(new Float32Array(e.data)); };
      ws.onclose = (e) => {
        if (s.open) log(kind, 'disconnected', e.code);
        s.open = false; s.ws = null;
        if (s.onclose) s.onclose();
        retry();
      };
      s.ws = ws;
    }
    s.send = (buf) => {
      if (!s.open || s.ws.bufferedAmount > 64000) return false;   // never queue up seconds of audio
      s.ws.send(buf);
      return true;
    };
    s.close = () => { s.wanted = false; if (s.ws) s.ws.close(); };
    connect();
    return s;
  }

  // ---- mic -----------------------------------------------------------------------------------------------------
  let chains = [];                             // wrapped mics, newest last; only the newest is processed
  let micSock = null, stamps = [];
  const active = () => chains[chains.length - 1];
  const setMicState = () => { const c = active(); stats.mic = !c ? 'off' : c.live ? 'processed' : 'raw'; };

  function ensureMicSock() {
    if (micSock) return;
    micSock = socket('mic', (blk) => {
      const c = active();
      const t = stamps.shift();
      if (t !== undefined) {
        const rtt = performance.now() - t;
        stats.rttMs = stats.rttMs === null ? rtt : 0.9 * stats.rttMs + 0.1 * rtt;
      }
      stats.micBack++;
      if (c) c.node.port.postMessage(blk, [blk.buffer]);
    });
    micSock.onopen = () => { stamps = []; };
    micSock.onclose = () => { stamps = []; chains.forEach((c) => c.node.port.postMessage({ reset: true })); };
  }
  setInterval(() => {
    if (micSock && micSock.open && stats.rttMs !== null) micSock.ws.send(JSON.stringify({ rtt_ms: stats.rttMs }));
  }, 1000);

  async function wrapMic(stream) {
    const raw = stream.getAudioTracks()[0];
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    await addWorklet(ctx);
    if (ctx.state !== 'running') await Promise.race([ctx.resume(), new Promise((r) => setTimeout(r, 500))]);
    if (ctx.state !== 'running') {               // autoplay policy: a suspended context would send silence
      ctx.close();
      log('audio context blocked; mic passes through unprotected');
      return stream;
    }
    const node = tapNode(ctx, 'mic');
    const dest = ctx.createMediaStreamDestination();
    ctx.createMediaStreamSource(new MediaStream([raw])).connect(node).connect(dest);
    const track = dest.stream.getAudioTracks()[0];
    const chain = { node, live: false };
    chains.push(chain);
    stamps = [];
    ensureMicSock();
    setMicState();
    node.port.onmessage = (e) => {
      const d = e.data;
      if (d instanceof Float32Array) {
        if (chain === active() && micSock && micSock.send(d.buffer)) { stamps.push(performance.now()); stats.micSent++; }
      } else if (d && 'live' in d) {
        chain.live = d.live;
        setMicState();
        log('mic', d.live ? 'processed by CallGuard' : 'raw (CallGuard not answering)');
      }
    };
    let ended = false;
    const nativeStop = track.stop.bind(track);
    const end = () => {
      if (ended) return;
      ended = true;
      chains = chains.filter((c) => c !== chain);
      stamps = [];
      raw.stop();
      ctx.close();
      if (!chains.length && micSock) { micSock.close(); micSock = null; }
      setMicState();
    };
    track.stop = () => { nativeStop(); end(); };
    raw.addEventListener('ended', () => { nativeStop(); end(); track.dispatchEvent(new Event('ended')); });
    // Meet mutes with track.enabled: mute both ends at once so the delayed tail doesn't go out after mute.
    const en = Object.getOwnPropertyDescriptor(MediaStreamTrack.prototype, 'enabled');
    Object.defineProperty(track, 'enabled', { configurable: true, get: () => en.get.call(track),
                                              set: (v) => { en.set.call(track, v); raw.enabled = v; } });
    // Meet's device picker reads the track's label / settings: report the real mic's.
    Object.defineProperty(track, 'label', { configurable: true, get: () => raw.label });
    for (const m of ['getSettings', 'getCapabilities', 'getConstraints', 'applyConstraints']) {
      if (raw[m]) track[m] = raw[m].bind(raw);
    }
    return new MediaStream([track, ...stream.getVideoTracks()]);
  }

  if (nativeGUM) {
    md.getUserMedia = async function getUserMedia(constraints) {
      const stream = await nativeGUM(constraints);
      if (!constraints || !constraints.audio || !stream.getAudioTracks().length) return stream;
      try {
        return await wrapMic(stream);
      } catch (e) {
        log('mic tap failed; mic passes through', e && e.name);
        return stream;
      }
    };
  }

  // ---- far end: remote audio tracks -> one mix -> /meet/far -----------------------------------------------------
  let far = null;
  function farInit() {
    const ctx = new AudioContext();
    const f = { ctx, node: null, sock: socket('far', null) };
    f.ready = addWorklet(ctx).then(() => {
      f.node = tapNode(ctx, 'far');
      const mute = ctx.createGain();
      mute.gain.value = 0;
      f.node.connect(mute).connect(ctx.destination);             // keeps the tap pulled; plays nothing
      f.node.port.onmessage = (e) => {
        if (e.data instanceof Float32Array && f.sock.send(e.data.buffer)) stats.farSent++;
      };
    });
    return f;
  }
  async function farAdd(track) {
    if (!track || track.kind !== 'audio') return;
    try {
      far = far || farInit();
      await far.ready;
      if (far.ctx.state !== 'running') far.ctx.resume();
      const src = far.ctx.createMediaStreamSource(new MediaStream([track]));
      src.connect(far.node);
      stats.farTracks++;
      track.addEventListener('ended', () => { src.disconnect(); stats.farTracks--; });
    } catch (e) {
      log('far tap failed', e && e.name);
    }
  }
  if (NativePC) {
    const Hooked = class RTCPeerConnection extends NativePC {
      constructor(...args) {
        super(...args);
        this.addEventListener('track', (e) => farAdd(e.track));
      }
    };
    window.RTCPeerConnection = Hooked;
    if (window.webkitRTCPeerConnection) window.webkitRTCPeerConnection = Hooked;
  }
  log('bridge installed', window.__callguardBridge.source, BASE);
})();
