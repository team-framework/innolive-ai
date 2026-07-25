"use strict";

export const PROFILE = Object.freeze({
  protocolVersion: 1,
  longEdge: 640,
  jpegQuality: 0.90,
  targetFps: 30,
  requestWindow: 5,
  timeoutMs: 1500,
  upscaleSmallInputs: false,
});

const MAGIC = [0x49, 0x4c, 0x46, 0x31]; // ILF1
const MAX_SEQUENCE = 0xffffffff;
const FRAME_INTERVAL_MS = 1000 / PROFILE.targetFps;

export function fitLongEdge(width, height, limit = PROFILE.longEdge) {
  const sourceWidth = Math.max(1, Number(width));
  const sourceHeight = Math.max(1, Number(height));
  const scale = Math.min(1, limit / Math.max(sourceWidth, sourceHeight));
  return [
    Math.max(32, Math.round(sourceWidth * scale)),
    Math.max(32, Math.round(sourceHeight * scale)),
  ];
}

export function framePacket(sequence, jpeg) {
  if (!Number.isInteger(sequence) || sequence < 0 || sequence > MAX_SEQUENCE) {
    throw new Error("sequence must be an unsigned 32-bit integer");
  }
  const header = new ArrayBuffer(8);
  const bytes = new Uint8Array(header);
  bytes.set(MAGIC, 0);
  new DataView(header).setUint32(4, sequence, false);
  return new Blob([header, jpeg], { type: "application/octet-stream" });
}

export function parseTerminal(text) {
  const metadata = JSON.parse(text);
  if (
    metadata?.v !== PROFILE.protocolVersion
    || !["result", "error"].includes(metadata?.type)
    || !Number.isInteger(metadata?.seq)
    || metadata.seq < 0
    || metadata.seq > MAX_SEQUENCE
  ) {
    throw new Error("invalid ILF1 terminal response");
  }
  return metadata;
}

export function percentile(samples, quantile) {
  if (!samples.length) return null;
  const ordered = [...samples].sort((left, right) => left - right);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(ordered.length * quantile) - 1),
  );
  return ordered[index];
}

class App {
  constructor() {
    this.elements = Object.fromEntries(
      [
        "start", "stop", "status", "source", "output", "capture",
        "capture-fps", "result-fps", "display-fps", "round-trip",
        "server-time", "queue", "dropped", "diagnostics",
      ].map((id) => [id, document.getElementById(id)]),
    );
    this.captureContext = this.elements.capture.getContext("2d", { alpha: false });
    this.outputContext = this.elements.output.getContext("2d", { alpha: false });
    this.blurCanvas = document.createElement("canvas");
    this.blurContext = this.blurCanvas.getContext("2d", { alpha: false });
    this.elements.start.addEventListener("click", () => this.start());
    this.elements.stop.addEventListener("click", () => this.stop());
    this.resetState();
    this.blackout("보호 결과 대기 중");
  }

  resetState() {
    this.running = false;
    this.generation = (this.generation || 0) + 1;
    this.stream = null;
    this.socket = null;
    this.videoRequest = null;
    this.frameRequest = null;
    this.encoding = false;
    this.pendingLatest = null;
    this.inFlight = new Map();
    this.nextSequence = 1;
    this.lastCaptureAt = -Infinity;
    this.lastTerminalSequence = 0;
    this.lastDisplayedSequence = 0;
    this.latestRender = null;
    this.rendering = false;
    this.counters = {
      captureDropped: 0,
      encodeBusyDropped: 0,
      pendingReplaced: 0,
      staleResults: 0,
      renderDropped: 0,
      errors: 0,
    };
    this.rates = {
      capture: [],
      encoded: [],
      sent: [],
      result: [],
      displayed: [],
    };
    this.samples = { rtt: [], captureToResult: [], captureToDisplay: [] };
  }

  async start() {
    if (this.running || this.socket) return;
    if (
      !window.isSecureContext
      && !["localhost", "127.0.0.1"].includes(location.hostname)
    ) {
      this.blackout("원격 카메라는 HTTPS가 필요합니다");
      return;
    }
    this.elements.start.disabled = true;
    this.blackout("서버 readiness 확인 중");
    const generation = ++this.generation;
    try {
      const response = await fetch("/healthz", { cache: "no-store" });
      if (!response.ok) throw new Error(`server is not ready (${response.status})`);
      this.validateHealth(await response.json());
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: PROFILE.targetFps, max: PROFILE.targetFps },
        },
        audio: false,
      });
      if (generation !== this.generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.stream = stream;
      this.elements.source.srcObject = stream;
      await this.elements.source.play();

      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${scheme}://${location.host}/ws`);
      this.socket = socket;
      socket.onopen = () => {
        if (generation !== this.generation || socket !== this.socket) return;
        this.running = true;
        this.elements.stop.disabled = false;
        this.setStatus("연결됨 · ILF1 B1-640-Q90-W5", true);
        this.blackout("첫 보호 결과 대기 중");
        this.scheduleCapture();
      };
      socket.onmessage = (event) => {
        if (generation !== this.generation || socket !== this.socket) return;
        this.receive(event);
      };
      socket.onerror = () => this.failClosed("WebSocket transport error");
      socket.onclose = (event) => {
        if (socket === this.socket) {
          this.failClosed(`WebSocket closed (${event.code})`, false);
        }
      };
    } catch (error) {
      this.failClosed(error.message);
    }
  }

  validateHealth(health) {
    const profile = health.serving_profile || {};
    if (
      health.status !== "ok"
      || health.profile !== "B1-640-Q90-W5"
      || health.protocol?.name !== "ILF1"
      || Number(health.protocol?.version) !== PROFILE.protocolVersion
      || Number(profile.engine_batch) !== 1
      || Number(profile.max_long_edge) !== PROFILE.longEdge
      || Number(profile.jpeg_quality) !== 90
      || Number(profile.client_window) !== PROFILE.requestWindow
      || Number(profile.target_fps) !== PROFILE.targetFps
      || Number(profile.max_streams) !== 1
      || health.runtime?.scheduler !== "serialized_b1"
      || Number(health.runtime?.runtime_instances) !== 1
    ) {
      throw new Error("server does not satisfy the B1-640-Q90-W5 contract");
    }
  }

  scheduleCapture() {
    this.clearCaptureSchedule();
    if (!this.running) return;
    if (typeof this.elements.source.requestVideoFrameCallback === "function") {
      this.videoRequest = this.elements.source.requestVideoFrameCallback((now) => {
        this.videoRequest = null;
        this.capture(now);
      });
    } else {
      this.frameRequest = requestAnimationFrame((now) => {
        this.frameRequest = null;
        this.capture(now);
      });
    }
  }

  clearCaptureSchedule() {
    if (
      this.videoRequest !== null
      && typeof this.elements.source.cancelVideoFrameCallback === "function"
    ) {
      this.elements.source.cancelVideoFrameCallback(this.videoRequest);
    }
    if (this.frameRequest !== null) cancelAnimationFrame(this.frameRequest);
    this.videoRequest = null;
    this.frameRequest = null;
  }

  capture(now) {
    this.scheduleCapture();
    if (!this.running || now - this.lastCaptureAt + 0.5 < FRAME_INTERVAL_MS) return;
    this.lastCaptureAt = now;
    this.mark(this.rates.capture, now);
    if (this.encoding || !this.elements.source.videoWidth) {
      this.counters.captureDropped += 1;
      this.counters.encodeBusyDropped += 1;
      return;
    }

    const [width, height] = fitLongEdge(
      this.elements.source.videoWidth,
      this.elements.source.videoHeight,
    );
    this.elements.capture.width = width;
    this.elements.capture.height = height;
    this.captureContext.drawImage(this.elements.source, 0, 0, width, height);
    const generation = this.generation;
    const capturedAt = performance.now();
    this.encoding = true;
    this.elements.capture.toBlob((jpeg) => {
      this.encoding = false;
      if (!this.running || generation !== this.generation) return;
      if (!jpeg) return this.failClosed("JPEG encoding failed");
      this.mark(this.rates.encoded, performance.now());
      if (this.pendingLatest) {
        this.counters.captureDropped += 1;
        this.counters.pendingReplaced += 1;
      }
      this.pendingLatest = { jpeg, capturedAt, width, height };
      this.pump();
    }, "image/jpeg", PROFILE.jpegQuality);
  }

  pump() {
    if (
      !this.running
      || !this.pendingLatest
      || !this.socket
      || this.socket.readyState !== WebSocket.OPEN
      || this.inFlight.size >= PROFILE.requestWindow
    ) return;
    if (this.nextSequence > MAX_SEQUENCE) {
      this.failClosed("sequence exhausted; reconnect required");
      return;
    }
    const sequence = this.nextSequence++;
    const frame = this.pendingLatest;
    this.pendingLatest = null;
    const sentAt = performance.now();
    const timeout = setTimeout(
      () => this.failClosed(`response timeout for seq ${sequence}`),
      PROFILE.timeoutMs,
    );
    this.inFlight.set(sequence, { ...frame, sentAt, timeout, generation: this.generation });
    try {
      this.socket.send(framePacket(sequence, frame.jpeg));
      this.mark(this.rates.sent, sentAt);
    } catch (error) {
      this.failClosed(error.message);
    }
  }

  receive(event) {
    try {
      if (typeof event.data !== "string") throw new Error("response must be JSON text");
      const metadata = parseTerminal(event.data);
      const sequence = metadata.seq;
      const request = this.inFlight.get(sequence);
      if (!request) throw new Error(`terminal seq ${sequence} is not in flight`);
      if (sequence <= this.lastTerminalSequence) {
        throw new Error(`terminal sequence regressed at ${sequence}`);
      }
      this.lastTerminalSequence = sequence;
      clearTimeout(request.timeout);
      this.inFlight.delete(sequence);

      if (metadata.type === "error") {
        this.counters.errors += 1;
        throw new Error(`${metadata.code}: ${metadata.message}`);
      }
      if (
        metadata.width !== request.width
        || metadata.height !== request.height
        || !Array.isArray(metadata.objects)
      ) {
        throw new Error("result metadata does not match the retained local frame");
      }
      this.pump();
      const receivedAt = performance.now();
      const rtt = receivedAt - request.sentAt;
      const captureToResult = receivedAt - request.capturedAt;
      this.mark(this.rates.result, receivedAt);
      this.addSample(this.samples.rtt, rtt);
      this.addSample(this.samples.captureToResult, captureToResult);
      if (sequence <= this.lastDisplayedSequence) {
        this.counters.staleResults += 1;
        return;
      }
      if (this.latestRender) {
        this.counters.renderDropped += 1;
        this.counters.staleResults += 1;
      }
      this.latestRender = {
        sequence,
        metadata,
        request,
        receivedAt,
        rtt,
        captureToResult,
        metadataBytes: new TextEncoder().encode(event.data).byteLength,
      };
      this.scheduleRender();
    } catch (error) {
      this.failClosed(error.message);
    }
  }

  scheduleRender() {
    if (!this.running || this.rendering || !this.latestRender) return;
    requestAnimationFrame(() => this.renderLatest());
  }

  async renderLatest() {
    if (!this.running || this.rendering || !this.latestRender) return;
    const item = this.latestRender;
    this.latestRender = null;
    this.rendering = true;
    let bitmap;
    try {
      bitmap = await createImageBitmap(item.request.jpeg);
      if (!this.running || item.sequence <= this.lastDisplayedSequence) {
        this.counters.staleResults += 1;
        return;
      }
      this.drawProtected(bitmap, item.metadata);
      this.lastDisplayedSequence = item.sequence;
      const displayedAt = performance.now();
      const captureToDisplay = displayedAt - item.request.capturedAt;
      this.mark(this.rates.displayed, displayedAt);
      this.addSample(this.samples.captureToDisplay, captureToDisplay);
      this.updateMetrics(item, captureToDisplay, displayedAt);
      this.setStatus(`보호 결과 seq ${item.sequence}`, true);
    } catch (error) {
      this.counters.renderDropped += 1;
      this.failClosed(`render failed: ${error.message}`);
    } finally {
      bitmap?.close();
      this.rendering = false;
      this.scheduleRender();
    }
  }

  drawProtected(bitmap, metadata) {
    const { width, height } = metadata;
    this.elements.output.width = width;
    this.elements.output.height = height;
    this.blurCanvas.width = width;
    this.blurCanvas.height = height;
    this.outputContext.fillStyle = "#000";
    this.outputContext.fillRect(0, 0, width, height);
    this.outputContext.drawImage(bitmap, 0, 0, width, height);

    const objects = metadata.objects;
    if (objects.length > 100) throw new Error("result exceeds the object limit");
    if (objects.length) {
      this.blurContext.fillStyle = "#000";
      this.blurContext.fillRect(0, 0, width, height);
      this.blurContext.filter = "blur(24px)";
      if (this.blurContext.filter !== "blur(24px)") {
        throw new Error("browser does not support the required blur filter");
      }
      this.blurContext.drawImage(bitmap, 0, 0, width, height);
      this.blurContext.filter = "none";
      this.outputContext.save();
      this.outputContext.beginPath();
      for (const object of objects) {
        const polygon = object.mask_polygon;
        if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 64) {
          throw new Error("tracked object has no bounded mask polygon");
        }
        polygon.forEach(([x, y], index) => {
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            throw new Error("mask polygon contains a non-finite point");
          }
          if (index) this.outputContext.lineTo(x, y);
          else this.outputContext.moveTo(x, y);
        });
        this.outputContext.closePath();
      }
      this.outputContext.clip();
      this.outputContext.drawImage(this.blurCanvas, 0, 0);
      this.outputContext.restore();
    }

    for (const object of objects) {
      const [x1, y1, x2, y2] = object.bbox || [];
      if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
      this.outputContext.strokeStyle = object.source === "held" ? "#ffd166" : "#3ee6a8";
      this.outputContext.lineWidth = 2;
      this.outputContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
      this.outputContext.fillStyle = this.outputContext.strokeStyle;
      this.outputContext.font = "13px ui-monospace, monospace";
      this.outputContext.fillText(
        `#${object.track_id} ${object.source} ${(object.confidence * 100).toFixed(0)}%`,
        x1,
        Math.max(14, y1 - 4),
      );
    }
  }

  updateMetrics(item, captureToDisplay, now) {
    const timing = item.metadata.timing_ms || {};
    const metrics = {
      profile: PROFILE,
      seq: item.sequence,
      capture_fps: this.rate(this.rates.capture, now),
      encoded_fps: this.rate(this.rates.encoded, now),
      sent_fps: this.rate(this.rates.sent, now),
      result_fps: this.rate(this.rates.result, now),
      displayed_fps: this.rate(this.rates.displayed, now),
      pending_frames: this.pendingLatest ? 1 : 0,
      inflight_requests: this.inFlight.size,
      capture_dropped: this.counters.captureDropped,
      encode_busy_dropped: this.counters.encodeBusyDropped,
      pending_replaced: this.counters.pendingReplaced,
      stale_results: this.counters.staleResults,
      render_dropped: this.counters.renderDropped,
      jpeg_bytes: item.request.jpeg.size,
      metadata_bytes: item.metadataBytes,
      round_trip_ms: Number(item.rtt.toFixed(2)),
      round_trip_p50_ms: this.roundedPercentile(this.samples.rtt, 0.50),
      round_trip_p95_ms: this.roundedPercentile(this.samples.rtt, 0.95),
      capture_to_result_ms: Number(item.captureToResult.toFixed(2)),
      capture_to_display_ms: Number(captureToDisplay.toFixed(2)),
      capture_to_display_p50_ms: this.roundedPercentile(this.samples.captureToDisplay, 0.50),
      capture_to_display_p95_ms: this.roundedPercentile(this.samples.captureToDisplay, 0.95),
      upscale_small_inputs: PROFILE.upscaleSmallInputs,
      server: item.metadata,
    };
    this.elements["capture-fps"].textContent = `${metrics.capture_fps} / ${metrics.encoded_fps}`;
    this.elements["result-fps"].textContent = String(metrics.result_fps);
    this.elements["display-fps"].textContent = String(metrics.displayed_fps);
    this.elements["round-trip"].textContent = `${item.rtt.toFixed(1)} / ${metrics.round_trip_p95_ms} ms`;
    this.elements["server-time"].textContent = `${Number(timing.server_total || 0).toFixed(1)} ms`;
    this.elements.queue.textContent = `${metrics.pending_frames} / ${metrics.inflight_requests}`;
    this.elements.dropped.textContent = `${metrics.capture_dropped} / ${metrics.stale_results + metrics.render_dropped}`;
    this.elements.diagnostics.textContent = JSON.stringify(metrics, null, 2);
  }

  roundedPercentile(samples, quantile) {
    const value = percentile(samples, quantile);
    return value === null ? null : Number(value.toFixed(2));
  }

  mark(series, now) {
    series.push(now);
    while (series.length && now - series[0] > 1000) series.shift();
  }

  rate(series, now) {
    while (series.length && now - series[0] > 1000) series.shift();
    return series.length;
  }

  addSample(series, value) {
    series.push(value);
    if (series.length > 300) series.shift();
  }

  setStatus(message, online = false) {
    this.elements.status.textContent = message;
    this.elements.status.dataset.online = String(online);
  }

  blackout(message) {
    if (!this.elements.output.width) this.elements.output.width = 640;
    if (!this.elements.output.height) this.elements.output.height = 360;
    this.outputContext.fillStyle = "#000";
    this.outputContext.fillRect(0, 0, this.elements.output.width, this.elements.output.height);
    this.setStatus(message, false);
  }

  clearInFlight() {
    for (const request of this.inFlight.values()) clearTimeout(request.timeout);
    this.inFlight.clear();
    this.pendingLatest = null;
    this.latestRender = null;
  }

  failClosed(message, closeSocket = true) {
    this.running = false;
    this.generation += 1;
    this.clearCaptureSchedule();
    this.clearInFlight();
    this.blackout(message);
    const socket = this.socket;
    this.socket = null;
    if (closeSocket && socket?.readyState === WebSocket.OPEN) {
      socket.close(1002, "fail closed");
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.elements.source.srcObject = null;
    this.elements.start.disabled = false;
    this.elements.stop.disabled = true;
  }

  stop() {
    this.failClosed("중지됨");
  }
}

if (typeof document !== "undefined") new App();
