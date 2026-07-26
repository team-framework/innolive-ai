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
const CAPTURE_EARLY_TOLERANCE_MS = 2;

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

export function captureDeadline(
  currentDeadline,
  now,
  intervalMs = FRAME_INTERVAL_MS,
  earlyToleranceMs = CAPTURE_EARLY_TOLERANCE_MS,
) {
  if (!Number.isFinite(now) || intervalMs <= 0 || earlyToleranceMs < 0) {
    throw new Error("invalid capture deadline input");
  }
  if (!Number.isFinite(currentDeadline)) {
    return { due: true, nextDeadline: now + intervalMs };
  }
  if (now + earlyToleranceMs < currentDeadline) {
    return { due: false, nextDeadline: currentDeadline };
  }
  const elapsedIntervals = Math.floor(
    (now + earlyToleranceMs - currentDeadline) / intervalMs,
  ) + 1;
  const nextDeadline = currentDeadline + elapsedIntervals * intervalMs;
  return { due: true, nextDeadline };
}

export class BitmapLease {
  constructor(jpeg, decode, clock, onRelease = () => {}) {
    this.bitmap = null;
    this.decodeMs = null;
    this.released = false;
    this.onRelease = onRelease;
    const startedAt = clock();
    let decoding;
    try {
      decoding = Promise.resolve(decode(jpeg));
    } catch (error) {
      decoding = Promise.reject(error);
    }
    this.promise = decoding.then(
      (bitmap) => {
        this.decodeMs = clock() - startedAt;
        if (this.released) {
          bitmap.close();
          return null;
        }
        this.bitmap = bitmap;
        return bitmap;
      },
      (error) => {
        this.decodeMs = clock() - startedAt;
        if (this.released) return null;
        throw error;
      },
    );
    // Decode can finish before its terminal result arrives. Mark the promise handled
    // while retaining the rejection for the renderer's fail-closed path.
    this.promise.catch(() => {});
  }

  release() {
    if (this.released) return false;
    this.released = true;
    this.onRelease();
    if (this.bitmap) {
      this.bitmap.close();
      this.bitmap = null;
    }
    return true;
  }
}

export class LatestRenderQueue {
  constructor() {
    this.active = null;
    this.waiting = null;
    this.lastCommittedSequence = 0;
  }

  enqueue(item) {
    if (item.sequence <= this.lastCommittedSequence) {
      return { accepted: false, dropped: item };
    }
    const replaced = this.waiting;
    this.waiting = item;
    return { accepted: true, dropped: replaced };
  }

  begin() {
    if (this.active || !this.waiting) return null;
    this.active = this.waiting;
    this.waiting = null;
    return this.active;
  }

  canCommit(item) {
    return this.active === item && item.sequence > this.lastCommittedSequence;
  }

  commit(item) {
    if (!this.canCommit(item)) return false;
    this.lastCommittedSequence = item.sequence;
    return true;
  }

  finish(item) {
    if (this.active === item) this.active = null;
  }

  drain() {
    const items = [this.active, this.waiting].filter(Boolean);
    this.active = null;
    this.waiting = null;
    return items;
  }
}

class App {
  constructor() {
    this.elements = Object.fromEntries(
      [
        "start", "stop", "status", "source", "output", "capture",
        "capture-fps", "sent-fps", "result-fps", "display-fps", "round-trip",
        "grpc-round-trip", "server-time", "queue", "dropped", "diagnostics",
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
    this.renderRequest = null;
    this.encoding = false;
    this.pendingLatest = null;
    this.inFlight = new Map();
    this.nextSequence = 1;
    this.nextCaptureDeadline = null;
    this.lastTerminalSequence = 0;
    this.renderQueue = new LatestRenderQueue();
    this.measurementStartedAt = null;
    this.lastMetricSampleAt = null;
    this.metricHistory = [];
    this.frameCounts = { capture: 0, encoded: 0, sent: 0, result: 0, display: 0 };
    this.counters = {
      captureDropped: 0,
      encodeBusyDropped: 0,
      pendingReplaced: 0,
      staleResults: 0,
      renderDropped: 0,
      bitmapOwners: 0,
      bitmapOwnerPeak: 0,
      errors: 0,
    };
    this.rates = {
      capture: [],
      encoded: [],
      sent: [],
      result: [],
      display: [],
    };
    this.samples = {
      rtt: [],
      localJpegDecode: [],
      resultToDisplay: [],
      captureToDisplay: [],
    };
    window.__INNOLIVE_BITMAP_OWNERS__ = 0;
    window.__INNOLIVE_METRIC_HISTORY__ = this.metricHistory;
  }

  async start() {
    if (this.running || this.socket) return;
    this.resetState();
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
      if (typeof this.elements.source.requestVideoFrameCallback !== "function") {
        throw new Error("requestVideoFrameCallback is required for real-frame capture");
      }

      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${scheme}://${location.host}/ws`);
      this.socket = socket;
      socket.onopen = () => {
        if (generation !== this.generation || socket !== this.socket) return;
        this.running = true;
        this.measurementStartedAt = performance.now();
        this.lastMetricSampleAt = this.measurementStartedAt;
        this.elements.stop.disabled = false;
        this.setStatus("연결됨 · WebSocket → gRPC ProcessVideo", true);
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
      || health.grpc?.service !== "AiProcessor"
      || health.grpc?.serving !== true
      || health.transport_path?.[1] !== "grpc-bidi-ProcessVideo"
    ) {
      throw new Error("server does not satisfy the B1-640-Q90-W5 contract");
    }
  }

  scheduleCapture() {
    this.clearCaptureSchedule();
    if (!this.running) return;
    this.videoRequest = this.elements.source.requestVideoFrameCallback((now) => {
      this.videoRequest = null;
      this.capture(now);
    });
  }

  clearCaptureSchedule() {
    if (
      this.videoRequest !== null
      && typeof this.elements.source.cancelVideoFrameCallback === "function"
    ) {
      this.elements.source.cancelVideoFrameCallback(this.videoRequest);
    }
    this.videoRequest = null;
  }

  capture(now) {
    this.scheduleCapture();
    if (!this.running) return;
    const deadline = captureDeadline(this.nextCaptureDeadline, now);
    this.nextCaptureDeadline = deadline.nextDeadline;
    if (!deadline.due) return;
    this.mark(this.rates.capture, now);
    this.frameCounts.capture += 1;
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
      this.frameCounts.encoded += 1;
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
    const request = {
      ...frame,
      sentAt,
      timeout,
      generation: this.generation,
      bitmapLease: null,
    };
    this.inFlight.set(sequence, request);
    try {
      this.socket.send(framePacket(sequence, frame.jpeg));
      request.bitmapLease = this.createBitmapLease(frame.jpeg);
      this.mark(this.rates.sent, sentAt);
      this.frameCounts.sent += 1;
    } catch (error) {
      this.failClosed(error.message);
    }
  }

  receive(event) {
    let request = null;
    try {
      if (typeof event.data !== "string") throw new Error("response must be JSON text");
      const metadata = parseTerminal(event.data);
      const sequence = metadata.seq;
      request = this.inFlight.get(sequence);
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
      this.mark(this.rates.result, receivedAt);
      this.frameCounts.result += 1;
      this.addSample(this.samples.rtt, rtt);
      const item = {
        sequence,
        generation: request.generation,
        metadata,
        request,
        receivedAt,
        rtt,
        metadataBytes: new TextEncoder().encode(event.data).byteLength,
      };
      const queued = this.renderQueue.enqueue(item);
      if (!queued.accepted) {
        request.bitmapLease?.release();
        request = null;
        this.counters.staleResults += 1;
        return;
      }
      if (queued.dropped) {
        queued.dropped.request.bitmapLease?.release();
        this.counters.renderDropped += 1;
        this.counters.staleResults += 1;
      }
      request = null; // Ownership moved from in-flight to the render queue.
      this.scheduleRender();
    } catch (error) {
      request?.bitmapLease?.release();
      this.failClosed(error.message);
    }
  }

  createBitmapLease(jpeg) {
    this.counters.bitmapOwners += 1;
    this.counters.bitmapOwnerPeak = Math.max(
      this.counters.bitmapOwnerPeak,
      this.counters.bitmapOwners,
    );
    if (this.counters.bitmapOwners > PROFILE.requestWindow + 2) {
      this.counters.bitmapOwners -= 1;
      throw new Error("bitmap ownership exceeded the W5 + active + waiting bound");
    }
    window.__INNOLIVE_BITMAP_OWNERS__ = this.counters.bitmapOwners;
    return new BitmapLease(
      jpeg,
      (blob) => createImageBitmap(blob),
      () => performance.now(),
      () => {
        this.counters.bitmapOwners -= 1;
        window.__INNOLIVE_BITMAP_OWNERS__ = this.counters.bitmapOwners;
      },
    );
  }

  scheduleRender() {
    if (
      !this.running
      || this.renderRequest !== null
      || this.renderQueue.active
      || !this.renderQueue.waiting
    ) return;
    this.renderRequest = requestAnimationFrame(() => {
      this.renderRequest = null;
      this.renderNext();
    });
  }

  async renderNext() {
    if (!this.running) return;
    const item = this.renderQueue.begin();
    if (!item) return;
    try {
      const bitmap = await item.request.bitmapLease.promise;
      if (
        !bitmap
        || !this.running
        || item.generation !== this.generation
        || !this.renderQueue.canCommit(item)
      ) {
        this.counters.staleResults += 1;
        return;
      }
      this.drawProtected(bitmap, item.metadata);
      if (!this.renderQueue.commit(item)) throw new Error("render commit regressed");
      const displayedAt = performance.now();
      const localJpegDecode = item.request.bitmapLease.decodeMs;
      const resultToDisplay = displayedAt - item.receivedAt;
      const captureToDisplay = displayedAt - item.request.capturedAt;
      this.mark(this.rates.display, displayedAt);
      this.frameCounts.display += 1;
      this.addSample(this.samples.localJpegDecode, localJpegDecode);
      this.addSample(this.samples.resultToDisplay, resultToDisplay);
      this.addSample(this.samples.captureToDisplay, captureToDisplay);
      this.updateMetrics(
        item,
        { localJpegDecode, resultToDisplay, captureToDisplay },
        displayedAt,
      );
      this.setStatus(`보호 결과 seq ${item.sequence}`, true);
    } catch (error) {
      if (item.generation === this.generation) {
        this.counters.renderDropped += 1;
        this.failClosed(`render failed: ${error.message}`);
      }
    } finally {
      item.request.bitmapLease?.release();
      this.renderQueue.finish(item);
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

  updateMetrics(item, latency, now) {
    const timing = item.metadata.timing_ms || {};
    const elapsedMs = Math.max(0, now - this.measurementStartedAt);
    const elapsedSeconds = Math.max(elapsedMs / 1000, Number.EPSILON);
    const metrics = {
      profile: PROFILE,
      seq: item.sequence,
      capture_fps: this.rate(this.rates.capture, now),
      encoded_fps: this.rate(this.rates.encoded, now),
      sent_fps: this.rate(this.rates.sent, now),
      result_fps: this.rate(this.rates.result, now),
      display_fps: this.rate(this.rates.display, now),
      capture_frames: this.frameCounts.capture,
      encoded_frames: this.frameCounts.encoded,
      sent_frames: this.frameCounts.sent,
      result_frames: this.frameCounts.result,
      display_frames: this.frameCounts.display,
      run_elapsed_ms: Number(elapsedMs.toFixed(3)),
      exact_capture_fps: Number((this.frameCounts.capture / elapsedSeconds).toFixed(6)),
      exact_result_fps: Number((this.frameCounts.result / elapsedSeconds).toFixed(6)),
      exact_display_fps: Number((this.frameCounts.display / elapsedSeconds).toFixed(6)),
      exact_display_capture_ratio: this.frameCounts.capture
        ? Number((this.frameCounts.display / this.frameCounts.capture).toFixed(6))
        : 0,
      pending_frames: this.pendingLatest ? 1 : 0,
      inflight_requests: this.inFlight.size,
      websocket_buffered_bytes: this.socket?.bufferedAmount || 0,
      capture_dropped: this.counters.captureDropped,
      encode_busy_dropped: this.counters.encodeBusyDropped,
      pending_replaced: this.counters.pendingReplaced,
      stale_results: this.counters.staleResults,
      render_dropped: this.counters.renderDropped,
      bitmap_owners: this.counters.bitmapOwners,
      bitmap_owner_peak: this.counters.bitmapOwnerPeak,
      jpeg_bytes: item.request.jpeg.size,
      metadata_bytes: item.metadataBytes,
      round_trip_ms: Number(item.rtt.toFixed(2)),
      round_trip_p50_ms: this.roundedPercentile(this.samples.rtt, 0.50),
      round_trip_p95_ms: this.roundedPercentile(this.samples.rtt, 0.95),
      local_jpeg_decode_ms: Number(latency.localJpegDecode.toFixed(2)),
      local_jpeg_decode_p95_ms: this.roundedPercentile(this.samples.localJpegDecode, 0.95),
      result_to_display_ms: Number(latency.resultToDisplay.toFixed(2)),
      result_to_display_p95_ms: this.roundedPercentile(this.samples.resultToDisplay, 0.95),
      capture_to_display_ms: Number(latency.captureToDisplay.toFixed(2)),
      capture_to_display_p50_ms: this.roundedPercentile(this.samples.captureToDisplay, 0.50),
      capture_to_display_p95_ms: this.roundedPercentile(this.samples.captureToDisplay, 0.95),
      upscale_small_inputs: PROFILE.upscaleSmallInputs,
      server: item.metadata,
    };
    this.elements["capture-fps"].textContent = `${metrics.capture_fps} / ${metrics.encoded_fps}`;
    this.elements["sent-fps"].textContent = String(metrics.sent_fps);
    this.elements["result-fps"].textContent = String(metrics.result_fps);
    this.elements["display-fps"].textContent = String(metrics.display_fps);
    this.elements["round-trip"].textContent = `${item.rtt.toFixed(1)} / ${metrics.round_trip_p95_ms} ms`;
    this.elements["grpc-round-trip"].textContent = `${Number(timing.grpc_round_trip || 0).toFixed(1)} ms`;
    this.elements["server-time"].textContent = `${Number(timing.server_total || 0).toFixed(1)} ms`;
    this.elements.queue.textContent = `${metrics.pending_frames} / ${metrics.inflight_requests}`;
    this.elements.dropped.textContent = `${metrics.capture_dropped} / ${metrics.stale_results + metrics.render_dropped}`;
    this.elements.diagnostics.textContent = JSON.stringify(metrics, null, 2);
    window.__INNOLIVE_METRICS__ = metrics;
    if (now - this.lastMetricSampleAt >= 1000) {
      this.lastMetricSampleAt = now;
      const browserSample = { ...metrics };
      delete browserSample.server;
      this.metricHistory.push(browserSample);
      if (this.metricHistory.length > 300) this.metricHistory.shift();
    }
    window.dispatchEvent(new CustomEvent("innolive:metrics", { detail: metrics }));
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
    for (const request of this.inFlight.values()) {
      clearTimeout(request.timeout);
      request.bitmapLease?.release();
    }
    this.inFlight.clear();
    this.pendingLatest = null;
    if (this.renderRequest !== null) cancelAnimationFrame(this.renderRequest);
    this.renderRequest = null;
    for (const item of this.renderQueue.drain()) item.request.bitmapLease?.release();
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
