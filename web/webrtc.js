export function h264Codecs(capabilities) {
  const video = capabilities.codecs.filter((codec) => codec.mimeType.toLowerCase() === "video/h264")
    .sort((a, b) => Number(b.sdpFmtpLine?.includes("profile-level-id=42e0")) - Number(a.sdpFmtpLine?.includes("profile-level-id=42e0")));
  // Keep retransmission codecs so packet loss can recover without a new keyframe.
  return video.length ? [...video, ...capabilities.codecs.filter((codec) => codec.mimeType.toLowerCase() === "video/rtx")] : [];
}

export function resultIsStale(lastFrameAt, now, timeout = 2000) {
  return lastFrameAt !== null && now - lastFrameAt > timeout;
}

export async function waitForIce(pc, timeout = 12000) {
  if (pc.iceGatheringState === "complete") return;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", changed);
    };
    const changed = () => {
      if (pc.iceGatheringState === "complete") { cleanup(); resolve(); }
      else if (pc.connectionState === "closed") { cleanup(); reject(new Error("연결이 취소됐습니다.")); }
    };
    const timer = setTimeout(() => { cleanup(); reject(new Error("영상 연결 주소 확인 시간이 초과됐습니다.")); }, timeout);
    pc.addEventListener("icegatheringstatechange", changed);
    changed();
  });
}

export class WebRtcDemo {
  constructor(document) {
    this.document = document;
    this.el = Object.fromEntries(["status", "session", "new-session", "start", "stop", "video-file", "clear-file", "source", "result", "placeholder", "fps", "server-ms", "rtt", "detail"].map((id) => [id, document.getElementById(id)]));
    this.pc = null;
    this.stream = null;
    this.peerId = null;
    this.generation = 0;
    this.busy = false;
    this.lastFrameAt = null;
    this.callbackId = null;
    this.videoUrl = null;
    this.canvasRequest = null;
    this.el.start.onclick = () => {
      const attempt = this.generation + 1;
      this.start().catch((error) => { if (attempt === this.generation) this.fail(error.message); });
    };
    this.el.stop.onclick = () => this.stop();
    this.el["new-session"].onclick = () => this.newSession().catch((error) => this.status(error.message));
    this.el["video-file"].onchange = () => this.sourceChanged();
    this.el["clear-file"].onclick = () => { this.el["video-file"].value = ""; this.sourceChanged(); };
    this.el.source.onended = () => { if (this.videoUrl) this.stop("동영상 재생이 끝났습니다."); };
    globalThis.addEventListener("pagehide", () => this.stop());
  }

  async api(path, options) {
    const response = await fetch(path, { ...options, cache: "no-store" });
    const body = response.status === 204 ? null : await response.json();
    if (!response.ok) throw new Error(body?.error?.message || "서버 요청이 실패했습니다.");
    return body;
  }

  status(message, online = false) {
    this.el.status.textContent = message;
    this.el.status.dataset.online = String(online);
  }

  async init() {
    const health = await this.api("/healthz");
    if (health.status !== "ok") throw new Error("합성 서버가 준비되지 않았습니다.");
    this.config = await this.api("/api/webrtc/config");
    if (!globalThis.isSecureContext) throw new Error("카메라를 사용하려면 HTTPS 주소로 접속해 주세요.");
    if (!globalThis.RTCPeerConnection) throw new Error("WebRTC를 지원하는 브라우저가 필요합니다.");
    await this.sessions();
    this.buttons(false);
    this.status("준비 완료", true);
  }

  async sessions(selected) {
    const payload = await this.api("/api/sessions");
    const sessions = payload.sessions || [];
    this.el.session.replaceChildren();
    for (const session of sessions) {
      const option = this.document.createElement("option");
      option.value = session.session_id;
      option.textContent = `${session.session_id} · 등록 ${session.entry_count}명`;
      this.el.session.append(option);
    }
    if (!sessions.length) return this.newSession();
    if (selected) this.el.session.value = selected;
  }

  async newSession() {
    const session = await this.api("/api/sessions", { method: "POST" });
    await this.sessions(session.session_id);
  }

  sourceChanged() {
    const file = this.el["video-file"].files[0];
    this.el.start.textContent = file ? "동영상 시작" : "카메라 시작";
    this.el["clear-file"].hidden = !file;
  }

  buttons(active) {
    for (const name of ["start", "session", "new-session", "video-file", "clear-file"]) this.el[name].disabled = active;
    this.el.stop.disabled = !active;
  }

  async inputStream() {
    const file = this.el["video-file"].files[0];
    if (!file) return navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
    this.videoUrl = URL.createObjectURL(file);
    this.el.source.src = this.videoUrl;
    await this.el.source.play();
    if (this.el.source.captureStream) return this.el.source.captureStream();
    const canvas = this.document.createElement("canvas");
    const scale = Math.min(1, 640 / Math.max(this.el.source.videoWidth, this.el.source.videoHeight));
    canvas.width = Math.floor(this.el.source.videoWidth * scale / 2) * 2;
    canvas.height = Math.floor(this.el.source.videoHeight * scale / 2) * 2;
    const context = canvas.getContext("2d", { alpha: false });
    const draw = () => {
      if (!this.videoUrl) return;
      context.drawImage(this.el.source, 0, 0, canvas.width, canvas.height);
      this.canvasRequest = requestAnimationFrame(draw);
    };
    draw();
    return canvas.captureStream(30);
  }

  async start() {
    if (this.busy) return;
    this.busy = true;
    const generation = ++this.generation;
    this.buttons(true);
    this.status("영상 연결 중");
    const stream = await this.inputStream();
    if (generation !== this.generation) { stream.getTracks().forEach((track) => track.stop()); return; }
    this.stream = stream;
    if (!this.videoUrl) { this.el.source.srcObject = stream; await this.el.source.play(); }
    if (generation !== this.generation) return;
    const pc = new RTCPeerConnection({ iceServers: this.config.ice_servers, bundlePolicy: "max-bundle" });
    this.pc = pc;
    const stats = pc.createDataChannel("stats", { ordered: false, maxRetransmits: 0 });
    stats.onmessage = (event) => {
      if (generation !== this.generation) return;
      let payload;
      try { payload = JSON.parse(event.data); } catch { return; }
      if (payload.type === "error") return this.fail(payload.message);
      if (payload.type === "frame") {
        this.el["server-ms"].textContent = `${payload.server_ms.toFixed(1)} ms`;
        const swapped = payload.faces.swapped || 0;
        const blurred = payload.faces.blur_fallback || 0;
        this.el.detail.textContent = `합성 ${swapped}명 · blur ${blurred}명 · 최신 입력 우선 처리`;
      }
    };
    pc.onconnectionstatechange = () => {
      if (generation !== this.generation) return;
      if (["failed", "closed", "disconnected"].includes(pc.connectionState)) this.fail("영상 연결이 끊겼습니다. 다시 시작해 주세요.");
    };
    pc.ontrack = (event) => {
      if (generation !== this.generation || event.track.kind !== "video") return;
      // This is a latency preference; the browser still adapts to network jitter.
      if ("jitterBufferTarget" in event.receiver) event.receiver.jitterBufferTarget = 0;
      this.el.result.srcObject = new MediaStream([event.track]);
      this.watchFrames(generation);
      this.el.result.play().catch((error) => { if (generation === this.generation) this.fail(error.message); });
    };
    const track = stream.getVideoTracks()[0];
    if (!track) throw new Error("영상 트랙을 읽을 수 없습니다.");
    const transceiver = pc.addTransceiver(track, {
      direction: "sendrecv", streams: [stream],
      sendEncodings: [{ maxBitrate: this.config.max_bitrate, maxFramerate: this.config.target_fps }],
    });
    const sender = transceiver.sender;
    const parameters = sender.getParameters();
    parameters.degradationPreference = "maintain-framerate";
    await sender.setParameters(parameters);
    const codecs = h264Codecs(RTCRtpSender.getCapabilities("video"));
    if (!codecs.length) throw new Error("이 브라우저는 H.264 영상 전송을 지원하지 않습니다.");
    transceiver.setCodecPreferences(codecs);
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIce(pc);
    if (generation !== this.generation) return;
    const answer = await this.api("/api/webrtc/offer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: "offer", session_id: this.el.session.value }),
    });
    if (generation !== this.generation) { await this.api(`/api/webrtc/${answer.peer_id}`, { method: "DELETE" }); return; }
    this.peerId = answer.peer_id;
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    this.connectedAt = performance.now();
    this.timer = setInterval(() => this.updateStats(generation), 1000);
  }

  watchFrames(generation) {
    const video = this.el.result;
    this.frameTimes = [];
    // Reveal only a decoded remote frame. A display:none video can otherwise
    // wait forever for its first compositor/requestVideoFrameCallback event.
    video.onloadeddata = () => {
      if (generation !== this.generation || !video.videoWidth) return;
      this.lastFrameAt = performance.now();
      video.hidden = false;
      this.el.placeholder.hidden = true;
      this.status("합성 중", true);
    };
    const arrived = (now) => {
      if (generation !== this.generation) return;
      this.lastFrameAt = now;
      this.frameTimes.push(now);
      this.frameTimes = this.frameTimes.filter((value) => now - value < 2000);
      video.hidden = false;
      this.el.placeholder.hidden = true;
      this.status("합성 중", true);
      if (video.requestVideoFrameCallback) this.callbackId = video.requestVideoFrameCallback(arrived);
    };
    if (video.requestVideoFrameCallback) this.callbackId = video.requestVideoFrameCallback(arrived);
  }

  async updateStats(generation) {
    if (generation !== this.generation || !this.pc) return;
    const now = performance.now();
    if (resultIsStale(this.lastFrameAt, now) || (this.lastFrameAt === null && now - this.connectedAt > 60000)) return this.fail("합성 결과 수신이 멈췄습니다. 다시 시작해 주세요.");
    const times = (this.frameTimes || []).filter((value) => now - value < 2000);
    if (times.length > 1) this.el.fps.textContent = ((times.length - 1) * 1000 / (times.at(-1) - times[0])).toFixed(1);
    try {
      const stats = await this.pc.getStats();
      if (generation !== this.generation) return;
      for (const item of stats.values()) {
        if (item.type === "candidate-pair" && item.state === "succeeded" && item.nominated && Number.isFinite(item.currentRoundTripTime)) this.el.rtt.textContent = `${(item.currentRoundTripTime * 1000).toFixed(0)} ms`;
        if (!this.el.result.requestVideoFrameCallback && item.type === "inbound-rtp" && item.kind === "video" && item.framesDecoded !== this.decodedFrames) {
          this.decodedFrames = item.framesDecoded; this.lastFrameAt = now;
          this.el.result.hidden = false; this.el.placeholder.hidden = true;
          this.el.fps.textContent = Number.isFinite(item.framesPerSecond) ? item.framesPerSecond.toFixed(1) : "—";
        }
      }
    } catch { /* Connection teardown can race a pending stats request. */ }
  }

  fail(message) { this.stop(message || "영상 처리가 실패했습니다."); }

  stop(message = "중지됨") {
    ++this.generation;
    this.busy = false;
    clearInterval(this.timer);
    if (this.callbackId !== null && this.el.result.cancelVideoFrameCallback) this.el.result.cancelVideoFrameCallback(this.callbackId);
    if (this.canvasRequest !== null) cancelAnimationFrame(this.canvasRequest);
    if (this.pc) { this.pc.onconnectionstatechange = null; this.pc.close(); }
    this.pc = null;
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.el.source.pause(); this.el.source.srcObject = null; this.el.source.removeAttribute("src"); this.el.source.load();
    this.el.result.srcObject = null; this.el.result.hidden = true;
    this.el.placeholder.hidden = false; this.el.placeholder.textContent = message;
    this.el.result.onloadeddata = null;
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    this.videoUrl = null;
    this.lastFrameAt = null;
    this.buttons(false);
    this.status(message);
    if (this.peerId) fetch(`/api/webrtc/${this.peerId}`, { method: "DELETE", keepalive: true }).catch(() => {});
    this.peerId = null;
    for (const name of ["fps", "server-ms", "rtt"]) this.el[name].textContent = "—";
  }
}

if (typeof document !== "undefined") {
  const demo = new WebRtcDemo(document);
  demo.init().catch((error) => demo.status(error.message));
}
