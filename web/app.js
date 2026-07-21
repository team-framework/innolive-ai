const CAPTURE_FPS = 30;
const PIPELINE_DEPTH = 2;

class BoundedQueue {
  constructor(capacity, onDrop) {
    this.capacity = capacity;
    this.onDrop = onDrop;
    this.items = [];
  }

  push(...items) {
    this.items.push(...items);
    while (this.items.length > this.capacity) {
      this.items.shift();
      this.onDrop();
    }
  }

  shift() {
    return this.items.shift();
  }

  takeLatest(count) {
    while (this.items.length > count) {
      this.items.shift();
      this.onDrop();
    }
    return this.items.splice(0, count);
  }

  clear() {
    this.items.length = 0;
  }

  get length() {
    return this.items.length;
  }
}

class BinaryPacket {
  static encode(frames) {
    const metadata = {
      v: 1,
      frames: frames.map((frame) => ({
        id: frame.id,
        capturedAt: frame.capturedAt,
        size: frame.jpeg.size,
      })),
    };
    const header = new TextEncoder().encode(JSON.stringify(metadata));
    const prefix = new ArrayBuffer(4);
    new DataView(prefix).setUint32(0, header.byteLength);
    return new Blob([prefix, header, ...frames.map((frame) => frame.jpeg)]);
  }

  static decode(buffer) {
    const view = new DataView(buffer);
    const headerSize = view.getUint32(0);
    const header = JSON.parse(
      new TextDecoder().decode(buffer.slice(4, 4 + headerSize)),
    );
    let offset = 4 + headerSize;
    return {
      processingMs: header.processingMs,
      frames: header.frames.map((frame) => {
        const jpeg = buffer.slice(offset, offset + frame.size);
        offset += frame.size;
        return { ...frame, jpeg };
      }),
    };
  }
}

class CameraCapture {
  constructor(video, onFrame, onDrop) {
    this.video = video;
    this.onFrame = onFrame;
    this.onDrop = onDrop;
    this.canvas = document.createElement("canvas");
    this.context = this.canvas.getContext("2d", { alpha: false });
    this.frameId = 0;
    this.timer = null;
    this.videoRequest = null;
    this.stream = null;
    this.generation = 0;
    this.lastCaptureAt = -Infinity;
    this.encoding = false;
  }

  async start(generation) {
    this.generation = generation;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: CAPTURE_FPS, max: CAPTURE_FPS },
      },
      audio: false,
    });
    if (generation !== this.generation) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    this.stream = stream;
    this.video.srcObject = stream;
    await this.video.play();
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    this.lastCaptureAt = -Infinity;
    this.schedule(generation, 0);
  }

  stop() {
    this.generation++;
    this.encoding = false;
    this.clearSchedule();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  clearSchedule() {
    clearTimeout(this.timer);
    this.timer = null;
    if (
      this.videoRequest !== null &&
      typeof this.video.cancelVideoFrameCallback === "function"
    ) {
      this.video.cancelVideoFrameCallback(this.videoRequest);
    }
    this.videoRequest = null;
  }

  schedule(generation, delay) {
    this.clearSchedule();
    if (generation !== this.generation) return;
    if (typeof this.video.requestVideoFrameCallback === "function") {
      this.videoRequest = this.video.requestVideoFrameCallback((now) => {
        this.videoRequest = null;
        this.capture(now, generation);
      });
      return;
    }
    this.timer = setTimeout(() => {
      this.timer = null;
      this.capture(performance.now(), generation);
    }, delay);
  }

  capture(now, generation) {
    const interval = 1000 / CAPTURE_FPS;
    this.schedule(generation, interval);
    if (now - this.lastCaptureAt + 0.5 < interval) return;
    this.lastCaptureAt = now;
    if (this.encoding) {
      this.onDrop();
      return;
    }

    this.encoding = true;
    const capturedAt = Date.now();
    const encodeStartedAt = performance.now();
    const frameId = this.frameId++;
    this.context.drawImage(
      this.video,
      0,
      0,
      this.canvas.width,
      this.canvas.height,
    );
    this.canvas.toBlob(
      (jpeg) => {
        this.encoding = false;
        if (generation !== this.generation || !jpeg) return;
        const readyAt = performance.now();
        this.onFrame(
          {
            id: frameId,
            capturedAt,
            jpeg,
            readyAt,
            captureEncodeMs: readyAt - encodeStartedAt,
          },
          generation,
        );
      },
      "image/jpeg",
      0.82,
    );
  }
}

class StreamConnection {
  constructor(queue, onFrame, onStatus) {
    this.queue = queue;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.socket = null;
    this._inFlight = new Map();
    this.generation = 0;
    this.lastFrameId = -Infinity;
  }

  connect(generation) {
    this.generation = generation;
    this.lastFrameId = -Infinity;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${scheme}://${location.host}/ws`);
    this.socket.binaryType = "arraybuffer";
    this.socket.onopen = () => {
      if (generation !== this.generation) return;
      this.onStatus("Streaming");
      this.pump(generation);
    };
    this.socket.onmessage = (event) => this.receive(event, generation);
    this.socket.onerror = () => {
      if (generation === this.generation) this.onStatus("Connection error");
    };
    this.socket.onclose = () => {
      if (generation === this.generation) this.onStatus("Disconnected");
    };
  }

  close() {
    this.generation++;
    this.socket?.close();
    this.socket = null;
    this.clearInFlight();
  }

  pump(generation) {
    if (
      generation !== this.generation ||
      !this.queue.length ||
      this.socket?.readyState !== WebSocket.OPEN
    ) {
      return;
    }

    while (
      this._inFlight.size < PIPELINE_DEPTH &&
      this.queue.length &&
      generation === this.generation &&
      this.socket?.readyState === WebSocket.OPEN
    ) {
      const frame = this.queue.takeLatest(1)[0];
      const sentAt = performance.now();
      this._inFlight.set(frame.id, { frame, sentAt });
      try {
        this.socket.send(BinaryPacket.encode([frame]));
      } catch (error) {
        this._inFlight.delete(frame.id);
        this.onStatus(error.message);
      }
    }
  }

  receive(event, generation) {
    if (generation !== this.generation) return;
    if (typeof event.data === "string") {
      this.clearInFlight();
      this.onStatus(JSON.parse(event.data).error ?? "Server error");
      return;
    }

    const receivedAt = performance.now();
    const batch = BinaryPacket.decode(event.data);
    for (const frame of batch.frames) {
      const source = this._inFlight.get(frame.id);
      if (!source) continue;
      this._inFlight.delete(frame.id);
      if (frame.id < this.lastFrameId) continue;
      this.lastFrameId = frame.id;
      frame.captureEncodeMs = source.frame.captureEncodeMs;
      frame.captureQueueMs = source.sentAt - source.frame.readyAt;
      frame.roundTripMs = receivedAt - source.sentAt;
      frame.serverProcessingMs = batch.processingMs;
      this.onFrame(frame, generation);
    }
    this.pump(generation);
  }

  clearInFlight() {
    this._inFlight.clear();
  }

  get inFlight() {
    return this._inFlight.size > 0;
  }

  get inFlightCount() {
    return this._inFlight.size;
  }
}

class Playout {
  constructor(canvas, queue, onDisplay) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d", { alpha: false });
    this.queue = queue;
    this.onDisplay = onDisplay;
    this.generation = 0;
    this.decodeToken = 0;
    this.decoding = false;
    this.ready = null;
    this.frameRequest = null;
  }

  reset(generation) {
    this.generation = generation;
    this.decodeToken++;
    this.decoding = false;
    this.ready?.bitmap.close();
    this.ready = null;
    if (this.frameRequest !== null) cancelAnimationFrame(this.frameRequest);
    this.frameRequest = null;
    this.queue.clear();
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  schedule() {
    if (this.decoding || this.ready || !this.queue.length) return;
    const frame = this.queue.takeLatest(1)[0];
    const generation = this.generation;
    const token = ++this.decodeToken;
    this.decoding = true;
    const decodeStartedAt = performance.now();
    createImageBitmap(new Blob([frame.jpeg], { type: "image/jpeg" }))
      .then((bitmap) => {
        if (generation !== this.generation || token !== this.decodeToken) {
          bitmap.close();
          return;
        }
        frame.browserDecodeMs = performance.now() - decodeStartedAt;
        this.ready = { frame, bitmap };
        this.frameRequest = requestAnimationFrame(() => this.draw());
      })
      .catch(() => {})
      .finally(() => {
        if (token !== this.decodeToken) return;
        this.decoding = false;
        if (generation === this.generation && !this.ready) this.schedule();
      });
  }

  draw() {
    this.frameRequest = null;
    const ready = this.ready;
    this.ready = null;
    if (!ready) return;
    const { frame, bitmap } = ready;
    this.resize(bitmap.width, bitmap.height);
    this.context.drawImage(bitmap, 0, 0);
    this.drawMetadata(frame.faces);
    bitmap.close();
    this.onDisplay(frame, this.queue.length);
    this.schedule();
  }

  resize(width, height) {
    if (this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
  }

  drawMetadata(faces) {
    this.context.lineWidth = 2;
    this.context.font = "600 14px system-ui";
    for (const face of faces) {
      const [x1, y1, x2, y2] = face.bbox;
      const label = `#${face.trackId ?? "–"} ${(face.confidence * 100).toFixed(0)}%`;
      this.context.strokeStyle = "#73a7ff";
      this.context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      this.context.fillStyle = "#73a7ff";
      this.context.fillRect(
        x1,
        Math.max(0, y1 - 22),
        this.context.measureText(label).width + 12,
        22,
      );
      this.context.fillStyle = "#07101f";
      this.context.fillText(label, x1 + 6, Math.max(15, y1 - 6));
    }
  }
}

class App {
  constructor() {
    this.elements = Object.fromEntries(
      [
        "camera",
        "output",
        "empty",
        "toggle",
        "status",
        "latency",
        "capture-time",
        "transport-time",
        "decode-time",
        "ai-time",
        "encode-time",
        "browser-decode-time",
        "dropped",
      ].map((id) => [id, document.querySelector(`#${id}`)]),
    );
    this.generation = 0;
    this.running = false;
    this.droppedCapture = 0;
    this.droppedPlayout = 0;
    this.captureQueue = new BoundedQueue(2, () => this.dropCapture());
    this.playoutQueue = new BoundedQueue(2, () => this.dropPlayout());
    this.capture = new CameraCapture(
      this.elements.camera,
      (frame, generation) => {
        if (generation !== this.generation) return;
        this.captureQueue.push(frame);
        this.connection.pump(generation);
      },
      () => this.dropCapture(),
    );
    this.connection = new StreamConnection(
      this.captureQueue,
      (frame, generation) => this.receive(frame, generation),
      (status) => {
        this.elements.status.textContent = status;
      },
    );
    this.playout = new Playout(
      this.elements.output,
      this.playoutQueue,
      (frame, depth) => this.updateMetrics(frame, depth),
    );
    this.elements.toggle.addEventListener("click", () => this.toggle());
  }

  async toggle() {
    if (this.running) {
      this.stop();
      return;
    }

    this.running = true;
    const generation = ++this.generation;
    this.elements.toggle.textContent = "Stop";
    this.elements.toggle.dataset.running = "true";
    this.elements.status.textContent = "Starting camera";
    this.playout.reset(generation);
    this.connection.connect(generation);
    try {
      await this.capture.start(generation);
    } catch (error) {
      this.elements.status.textContent = error.message;
      this.stop();
    }
  }

  stop() {
    this.running = false;
    this.generation++;
    this.capture.stop();
    this.connection.close();
    this.captureQueue.clear();
    this.playout.reset(this.generation);
    this.elements.toggle.textContent = "Start camera";
    this.elements.toggle.dataset.running = "false";
    this.elements.status.textContent = "Idle";
    this.elements.empty.hidden = false;
  }

  receive(frame, generation) {
    if (generation !== this.generation) return;
    this.playoutQueue.push(frame);
    this.playout.schedule();
    this.elements.empty.hidden = true;
  }

  updateMetrics(frame, depth) {
    const timing = frame.timing ?? {};
    const transportMs = Math.max(
      0,
      (frame.roundTripMs ?? 0) - (frame.serverProcessingMs ?? 0),
    );
    this.elements.latency.textContent = `${Math.round(Date.now() - frame.capturedAt)} ms / ${depth}`;
    this.elements["capture-time"].textContent = `${this.ms(frame.captureEncodeMs)} / ${this.ms(frame.captureQueueMs)}`;
    const gateway = timing.gateway ?? {};
    this.elements["transport-time"].textContent = [
      this.ms(transportMs),
      `q${this.ms(gateway.ingressQueueMs)}`,
      `w${this.ms(gateway.grpcWriteMs)}`,
      `g${this.ms(gateway.grpcWaitMs)}`,
      `r${this.ms(gateway.grpcResidualMs)}`,
      `o${this.ms(gateway.responseQueueMs)}`,
    ].join(" · ");
    this.elements["decode-time"].textContent = this.ms(timing.decodeMs);
    this.elements["ai-time"].textContent = `${this.ms(timing.inferenceMs)} + ${this.ms(timing.trackingMs)} · B${timing.inferenceBatchSize ?? "–"}`;
    this.elements["encode-time"].textContent = this.ms(timing.blurEncodeMs);
    this.elements["browser-decode-time"].textContent = this.ms(
      frame.browserDecodeMs,
    );
  }

  ms(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "—";
  }

  dropCapture() {
    this.droppedCapture++;
    this.updateDropped();
  }

  dropPlayout() {
    this.droppedPlayout++;
    this.updateDropped();
  }

  updateDropped() {
    this.elements.dropped.textContent = `${this.droppedCapture} / ${this.droppedPlayout}`;
  }
}

if (typeof document !== "undefined") new App();
