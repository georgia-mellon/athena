// CallGuard <-> Google Meet audio bridge. Injected by the launcher (CDP, before Meet's own scripts) or loaded by the
// local test room. Self-contained, no dependencies.
//
// Mic:  getUserMedia -> AudioWorklet (downsample to 16 kHz, 320-sample float32 blocks) -> ws /meet/mic -> CallGuard
//       (Keyguard shield + spoken-secret delay line) -> processed blocks back -> jitter buffer (60 ms, grows after
//       underflows) -> upsample -> the audio track Meet sends. Video tracks are untouched.
// Far:  every remote audio track an RTCPeerConnection receives -> mixed in WebAudio -> 16 kHz blocks -> ws /meet/far
//       (Hearsay + the "read me the code" listener).
// FAIL OPEN: whenever CallGuard isn't answering (socket down, > 15 % of the last 500 ms missing, audio context
// blocked or suspended) the raw mic goes out unchanged, crossfaded over 10 ms; the socket reconnects with backoff.
// Only console.debug, never audio or text contents.
(() => {
  'use strict';
  const HOSTS = ['meet.google.com', '127.0.0.1', 'localhost', '[::1]'];
  if (window.__callguardBridge || !HOSTS.includes(location.hostname) || !window.AudioWorkletNode) return;
  const PORT = Number(window.__callguardPort) || 8765;
  const BASE = `ws://127.0.0.1:${PORT}/meet/`;
  const TARGET_MS = 60;                       // jitter buffer target
  const log = (...a) => console.debug('[callguard]', ...a);
  const stats = { mic: 'off', micSent: 0, micBack: 0, farSent: 0, farTracks: 0, rttMs: null, ctxSuspended: 0,
                  rawSwaps: 0 };
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
    // jitter buffer target (16 kHz samples): +20 ms after each underflow up to 150 ms, -10 ms per 10 s without one
    this.q = new Fifo(16000); this.base = this.target = o.targetMs * 16; this.live = false; this.dry = false;
    this.clean = 0;
    // starved 16 kHz samples per render quantum over the last 500 ms: more than 15 % missing -> raw mic
    this.win = new Float32Array(Math.ceil(0.5 * sampleRate / 128)); this.wi = 0; this.wsum = 0;
    this.g = 0; this.dg = 1 / (0.01 * sampleRate);   // raw (0) <-> processed (1): a 10 ms crossfade on every switch
    this.a = 0; this.b = 0; this.frac = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d instanceof Float32Array) {
        this.q.push(d);
        if (this.q.n > 3 * this.target) this.q.drop(this.q.n - this.target);   // drifted: drop, don't lag
      } else if (d && d.reset) { this.q.drop(this.q.n); this.setLive(false); }
    };
  }
  setLive(v) {
    if (v === this.live) return;
    this.live = v; this.win.fill(0); this.wsum = 0;
    this.port.postMessage({ live: v });
  }
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
    if (!this.live && this.q.n >= this.target) this.setLive(true);
    if (!this.live && this.g === 0) { if (x) out.set(x); return true; }   // fail open: the raw mic
    const up = 16000 / sampleRate;
    let starved = 0;
    for (let i = 0; i < out.length; i++) {
      let p = 0;
      if (this.live) {
        this.frac += up;
        while (this.frac >= 1) {
          this.frac -= 1; this.a = this.b;
          const v = this.q.shift();
          if (v !== null) { this.b = v; this.dry = false; continue; }
          this.b = 0; starved++;
          if (!this.dry) { this.dry = true; this.target = Math.min(this.target + 320, 2400); }   // underflow
        }
        p = this.a + (this.b - this.a) * this.frac;
      }
      this.g = this.live ? Math.min(1, this.g + this.dg) : Math.max(0, this.g - this.dg);
      out[i] = this.g * p + (1 - this.g) * (x ? x[i] : 0);
    }
    if (!this.live) return true;
    this.wsum += starved - this.win[this.wi]; this.win[this.wi] = starved; this.wi = (this.wi + 1) % this.win.length;
    if (this.wsum > 0.15 * this.win.length * out.length * up) { this.setLive(false); return true; }   // raw again
    this.clean = starved ? 0 : this.clean + out.length;
    if (this.clean > 10 * sampleRate) { this.clean = 0; this.target = Math.max(this.base, this.target - 160); }
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
      ws.onopen = () => { s.open = true; log(kind, 'connected'); if (s.onopen) s.onopen(); };
      ws.onmessage = (e) => { if (onBlock && e.data instanceof ArrayBuffer) onBlock(new Float32Array(e.data)); };
      ws.onclose = (e) => {
        if (s.open) log(kind, 'disconnected', e.code);
        // a session that ran and ended: back in 0.5 s. 1013 (another tab owns the stream) keeps backing off.
        if (s.open && e.code !== 1013) s.delay = 500;
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

  // ---- the peer connections (hooked below): which of our tracks is actually being sent -------------------------
  const pcs = new Set();
  const senderOf = (tracks) => {
    const found = [];
    for (const pc of pcs) {
      if (pc.signalingState === 'closed') { pcs.delete(pc); continue; }
      for (const s of pc.getSenders()) if (s.track && tracks.includes(s.track)) found.push(s);
    }
    return found;
  };

  // ---- mic -----------------------------------------------------------------------------------------------------
  // Wrapped mics, newest last. One is processed: the newest whose track a connection sends (the call, not Meet's
  // mic preview), else the newest.
  let chains = [];
  let micSock = null, stamps = [];
  const active = () => {
    const sent = chains.filter((c) => senderOf([c.track, c.raw]).length);
    return sent.length ? sent[sent.length - 1] : chains[chains.length - 1];
  };
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
    const chain = { node, track, raw, live: false, swapped: [] };
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
    // The context gets suspended / interrupted mid-call (device change, OS audio session): the processed track goes
    // silent and the worklet can't fail open. Resume it; if that doesn't take, the call sends the raw mic meanwhile.
    ctx.onstatechange = () => {
      if (ended || ctx.state === 'closed') return;
      if (ctx.state === 'running') {
        for (const s of chain.swapped) if (s.track === raw) s.replaceTrack(track).catch(() => {});
        if (chain.swapped.length) log('audio context running again; call sends the processed mic');
        chain.swapped = [];
        return;
      }
      stats.ctxSuspended++;
      ctx.resume().catch(() => {});
      setTimeout(() => {
        if (ended || ctx.state === 'running' || ctx.state === 'closed') return;
        for (const s of senderOf([track])) {
          s.replaceTrack(raw).catch(() => {});
          chain.swapped.push(s);
          stats.rawSwaps++;
        }
        log('audio context', ctx.state, '; call sends the raw mic');
      }, 300);
    };
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
    const P = NativePC.prototype, addTrack = P.addTrack, addTransceiver = P.addTransceiver;
    P.addTrack = function (...a) { pcs.add(this); return addTrack.apply(this, a); };
    P.addTransceiver = function (...a) { pcs.add(this); return addTransceiver.apply(this, a); };
    const Hooked = class RTCPeerConnection extends NativePC {
      constructor(...args) {
        super(...args);
        pcs.add(this);
        this.addEventListener('track', (e) => farAdd(e.track));
      }
    };
    window.RTCPeerConnection = Hooked;
    if (window.webkitRTCPeerConnection) window.webkitRTCPeerConnection = Hooked;
  }
  log('bridge installed', window.__callguardBridge.source, BASE);
})();
