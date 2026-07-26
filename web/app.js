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
const MAX_SESSION_ID_BYTES = 256;
export const TAB_SESSION_STORAGE_KEY = "innolive.demo.session-id";
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

export function objectsToBlur(objects) {
  return objects.filter((object) => object.whitelisted !== true);
}

export function validateSessionId(value) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("session ID must not be empty");
  }
  if (new TextEncoder().encode(value).byteLength > MAX_SESSION_ID_BYTES) {
    throw new Error(`session ID exceeds ${MAX_SESSION_ID_BYTES} UTF-8 bytes`);
  }
  return value;
}

export function whitelistUrl(sessionId) {
  return `/api/whitelist?session_id=${encodeURIComponent(validateSessionId(sessionId))}`;
}

export function websocketUrl(locationValue, sessionId) {
  const scheme = locationValue.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${locationValue.host}/ws?session_id=${encodeURIComponent(validateSessionId(sessionId))}`;
}

export function requireResultSession(metadata, sessionId) {
  if (metadata?.session_id !== sessionId) {
    throw new Error("result session does not match the active stream");
  }
  return metadata;
}

export async function getWhitelistStatus(sessionId, request = fetch) {
  const response = await request(whitelistUrl(sessionId), {
    cache: "no-store",
  });
  return readApiResponse(response);
}

export async function listSessions(request = fetch) {
  const response = await request("/api/sessions", { cache: "no-store" });
  const payload = await readApiResponse(response);
  if (!Array.isArray(payload?.sessions)) {
    throw new Error("gateway returned an invalid session list");
  }
  const sessions = payload.sessions.map(normalizeSessionInfo);
  if (new Set(sessions.map((session) => session.session_id)).size !== sessions.length) {
    throw new Error("gateway returned duplicate session IDs");
  }
  return sessions;
}

export async function createSession(request = fetch) {
  const response = await request("/api/sessions", { method: "POST" });
  return normalizeSessionInfo(await readApiResponse(response));
}

export async function ensureTabSession(
  sessions,
  storage,
  create = () => createSession(),
) {
  const knownSessions = Array.from(sessions, normalizeSessionInfo);
  let storedSessionId = null;
  try {
    storedSessionId = storage?.getItem(TAB_SESSION_STORAGE_KEY) || null;
  } catch {
    storedSessionId = null;
  }
  const reused = knownSessions.find((session) => session.session_id === storedSessionId);
  if (reused) return { active: reused, sessions: knownSessions, created: false };

  const active = normalizeSessionInfo(await create());
  if (knownSessions.some((session) => session.session_id === active.session_id)) {
    throw new Error("server returned an existing session ID for CreateSession");
  }
  try {
    storage?.setItem(TAB_SESSION_STORAGE_KEY, active.session_id);
  } catch {
    // Session creation still works when browser storage is unavailable.
  }
  return { active, sessions: [active, ...knownSessions], created: true };
}

export function enrollmentSize(width, height, maxLongEdge = PROFILE.longEdge) {
  const sourceWidth = Number(width);
  const sourceHeight = Number(height);
  if (
    !Number.isFinite(sourceWidth)
    || !Number.isFinite(sourceHeight)
    || sourceWidth <= 0
    || sourceHeight <= 0
    || !Number.isFinite(maxLongEdge)
    || maxLongEdge <= 0
  ) {
    throw new Error("image has invalid dimensions");
  }
  const scale = Math.min(1, maxLongEdge / Math.max(sourceWidth, sourceHeight));
  return [
    Math.max(1, Math.round(sourceWidth * scale)),
    Math.max(1, Math.round(sourceHeight * scale)),
  ];
}

export async function normalizeWhitelistImage(
  file,
  {
    maxLongEdge = PROFILE.longEdge,
    jpegQuality = PROFILE.jpegQuality,
    decode = decodeEnrollmentImage,
    createCanvas = () => document.createElement("canvas"),
  } = {},
) {
  if (
    !file
    || (file.type && !file.type.startsWith("image/"))
    || file.type === "image/svg+xml"
  ) {
    throw new Error("a browser-decodable image is required");
  }
  if (!Number.isFinite(jpegQuality) || jpegQuality <= 0 || jpegQuality > 1) {
    throw new Error("JPEG quality must be in 0..1");
  }

  let bitmap;
  try {
    bitmap = await decode(file);
    const [width, height] = enrollmentSize(bitmap.width, bitmap.height, maxLongEdge);
    const canvas = createCanvas();
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("browser could not create an image canvas");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.drawImage(bitmap.drawable || bitmap, 0, 0, width, height);
    const jpeg = await encodeCanvasJpeg(canvas, jpegQuality);
    if (!(jpeg instanceof Blob) || jpeg.type !== "image/jpeg" || jpeg.size === 0) {
      throw new Error("browser could not encode the image as JPEG");
    }
    return jpeg;
  } finally {
    bitmap?.close?.();
  }
}

export async function decodeEnrollmentImage(source) {
  if (typeof createImageBitmap === "function") {
    try {
      return await createImageBitmap(source, { imageOrientation: "from-image" });
    } catch {
      return createImageBitmap(source);
    }
  }
  if (
    typeof Image !== "function"
    || typeof URL === "undefined"
    || typeof URL.createObjectURL !== "function"
    || typeof URL.revokeObjectURL !== "function"
  ) {
    throw new Error("this browser cannot decode local images");
  }

  const objectUrl = URL.createObjectURL(source);
  try {
    const image = new Image();
    image.decoding = "async";
    if (typeof image.decode === "function") {
      image.src = objectUrl;
      await image.decode();
    } else {
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error("browser could not decode the image"));
        image.src = objectUrl;
      });
    }
    return {
      width: image.naturalWidth,
      height: image.naturalHeight,
      drawable: image,
      close: () => URL.revokeObjectURL(objectUrl),
    };
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  }
}

export async function addWhitelistFiles(
  files,
  sessionId,
  request = fetch,
  normalize = normalizeWhitelistImage,
) {
  const url = whitelistUrl(sessionId);
  const results = [];
  for (const [index, file] of Array.from(files).entries()) {
    const name = file.name || `face-${index + 1}.jpg`;
    try {
      const jpeg = await normalize(file);
      const response = await request(url, {
        method: "POST",
        headers: { "content-type": "image/jpeg" },
        body: jpeg,
      });
      results.push({ name, ok: true, response: await readApiResponse(response) });
    } catch (error) {
      results.push({ name, ok: false, error: error.message });
    }
  }
  return results;
}

function normalizeSessionInfo(value) {
  const sessionId = validateSessionId(value?.session_id);
  const entryCount = Number(value?.entry_count);
  const whitelistVersion = Number(value?.whitelist_version);
  const createdAt = Number(value?.created_at_unix_ms || 0);
  if (!Number.isSafeInteger(entryCount) || entryCount < 0) {
    throw new Error("session entry_count must be a non-negative integer");
  }
  if (!Number.isSafeInteger(whitelistVersion) || whitelistVersion < 0) {
    throw new Error("session whitelist_version must be a non-negative integer");
  }
  if (!Number.isSafeInteger(createdAt) || createdAt < 0) {
    throw new Error("session created_at_unix_ms must be a non-negative integer");
  }
  return {
    session_id: sessionId,
    entry_count: entryCount,
    whitelist_version: whitelistVersion,
    created_at_unix_ms: createdAt,
  };
}

function encodeCanvasJpeg(canvas, quality) {
  if (typeof canvas.convertToBlob === "function") {
    return canvas.convertToBlob({ type: "image/jpeg", quality });
  }
  return new Promise((resolve, reject) => {
    if (typeof canvas.toBlob !== "function") {
      reject(new Error("browser cannot encode canvas images"));
      return;
    }
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("browser could not encode the image as JPEG"));
    }, "image/jpeg", quality);
  });
}

async function readApiResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`gateway returned HTTP ${response.status} without JSON`);
  }
  if (!response.ok) {
    const code = payload?.error?.code || `HTTP_${response.status}`;
    const message = payload?.error?.message || "request failed";
    throw new Error(`${code}: ${message}`);
  }
  return payload;
}

export function negotiateServerProfile(health, preferred = PROFILE) {
  const profile = health?.serving_profile || {};
  const violations = [];
  if (health?.status !== "ok") violations.push("server is not ready");
  if (health?.protocol?.name !== "ILF1") violations.push("ILF1 is required");
  if (Number(health?.protocol?.version) !== preferred.protocolVersion) {
    violations.push(`ILF1 v${preferred.protocolVersion} is required`);
  }
  if (health?.grpc?.serving !== true) violations.push("gRPC backend is not serving");
  if (
    !Array.isArray(health?.transport_path)
    || !health.transport_path.includes("grpc-bidi-ProcessVideo")
  ) {
    violations.push("gRPC ProcessVideo transport is required");
  }

  positiveInteger(profile.engine_batch, "engine_batch", violations);
  const maxLongEdge = positiveInteger(
    profile.max_long_edge ?? profile.image_size,
    "max_long_edge",
    violations,
  );
  const jpegQuality = finiteRange(
    profile.jpeg_quality,
    "jpeg_quality",
    1,
    100,
    violations,
  );
  const requestWindow = positiveInteger(
    profile.client_window,
    "client_window",
    violations,
  );
  const targetFps = finiteRange(
    profile.target_fps,
    "target_fps",
    Number.MIN_VALUE,
    240,
    violations,
  );
  const maxStreams = profile.max_streams === undefined || Number(profile.max_streams) === 0
    ? null
    : positiveInteger(profile.max_streams, "max_streams", violations);

  if (maxLongEdge !== null && maxLongEdge < 32) {
    violations.push("max_long_edge must be at least 32");
  }
  if (violations.length) {
    throw new Error(`incompatible server: ${violations.join("; ")}`);
  }

  return Object.freeze({
    protocolVersion: preferred.protocolVersion,
    longEdge: Math.min(preferred.longEdge, maxLongEdge),
    jpegQuality: Math.min(preferred.jpegQuality, jpegQuality / 100),
    targetFps: Math.min(preferred.targetFps, targetFps),
    requestWindow: Math.min(preferred.requestWindow, requestWindow),
    timeoutMs: preferred.timeoutMs,
    upscaleSmallInputs: preferred.upscaleSmallInputs,
    maxStreams,
  });
}

function positiveInteger(value, name, violations) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) {
    violations.push(`${name} must be a positive integer`);
    return null;
  }
  return number;
}

function finiteRange(value, name, minimum, maximum, violations) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < minimum || number > maximum) {
    violations.push(`${name} must be in ${minimum}..${maximum}`);
    return null;
  }
  return number;
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
        "session-id", "comparison-session-id", "whitelist-files", "add-whitelist",
        "new-session", "refresh-whitelist", "compare-sessions", "whitelist-status",
        "session-list", "enrollment-results",
      ].map((id) => [id, document.getElementById(id)]),
    );
    this.captureContext = this.elements.capture.getContext("2d", { alpha: false });
    this.outputContext = this.elements.output.getContext("2d", { alpha: false });
    this.blurCanvas = document.createElement("canvas");
    this.blurContext = this.blurCanvas.getContext("2d", { alpha: false });
    this.elements.start.addEventListener("click", () => this.start());
    this.elements.stop.addEventListener("click", () => this.stop());
    this.elements["add-whitelist"].addEventListener("click", () => this.enrollWhitelist());
    this.elements["new-session"].addEventListener("click", () => this.createNewSession());
    this.elements["refresh-whitelist"].addEventListener("click", () => {
      this.refreshSessions();
    });
    this.elements["compare-sessions"].addEventListener("click", () => this.compareSessions());
    this.elements["session-id"].addEventListener("change", () => {
      this.selectActiveSession();
    });
    this.elements["comparison-session-id"].addEventListener("change", () => {
      this.renderSessions(
        this.elements["session-id"].value,
        this.elements["comparison-session-id"].value,
      );
    });
    this.sessionStatuses = [];
    this.sessionsReady = false;
    this.resetState();
    this.blackout("보호 결과 대기 중");
    void this.initializeSessions();
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
    this.streamSessionId = null;
    this.activeProfile = PROFILE;
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
      this.streamSessionId = validateSessionId(this.elements["session-id"].value);
      this.elements["session-id"].disabled = true;
      this.elements["new-session"].disabled = true;
      this.elements["refresh-whitelist"].disabled = true;
      const response = await fetch("/healthz", { cache: "no-store" });
      if (!response.ok) throw new Error(`server is not ready (${response.status})`);
      this.activeProfile = negotiateServerProfile(await response.json());
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: {
            ideal: this.activeProfile.targetFps,
            max: this.activeProfile.targetFps,
          },
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

      const socket = new WebSocket(websocketUrl(location, this.streamSessionId));
      this.socket = socket;
      socket.onopen = () => {
        if (generation !== this.generation || socket !== this.socket) return;
        this.running = true;
        this.measurementStartedAt = performance.now();
        this.lastMetricSampleAt = this.measurementStartedAt;
        this.elements.stop.disabled = false;
        this.setStatus(`연결됨 · ${this.streamSessionId} · gRPC ProcessVideo`, true);
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

  async enrollWhitelist() {
    const button = this.elements["add-whitelist"];
    try {
      const sessionId = validateSessionId(this.elements["session-id"].value);
      const files = Array.from(this.elements["whitelist-files"].files || []);
      if (!files.length) throw new Error("등록할 이미지를 한 장 이상 선택하세요");
      button.disabled = true;
      this.elements["whitelist-status"].textContent = `${files.length}장 JPEG 변환·등록 중`;
      const results = await addWhitelistFiles(files, sessionId);
      this.elements["enrollment-results"].textContent = results.map((result) => (
        result.ok
          ? `✓ ${result.name} → ${result.response.entry_count} entries · v${result.response.whitelist_version}`
          : `✗ ${result.name} → ${result.error}`
      )).join("\n");
      await this.refreshWhitelistStatus();
    } catch (error) {
      this.elements["whitelist-status"].textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  async refreshWhitelistStatus() {
    try {
      const sessionId = validateSessionId(this.elements["session-id"].value);
      const status = await getWhitelistStatus(sessionId);
      this.upsertSession(status);
      this.renderSessions(sessionId, this.elements["comparison-session-id"].value);
      this.elements["whitelist-status"].textContent = this.formatWhitelistStatus(status);
      return status;
    } catch (error) {
      this.elements["whitelist-status"].textContent = error.message;
      return null;
    }
  }

  async compareSessions() {
    try {
      const firstId = validateSessionId(this.elements["session-id"].value);
      const secondId = validateSessionId(this.elements["comparison-session-id"].value);
      if (firstId === secondId) throw new Error("서로 다른 session ID를 입력하세요");
      const [first, second] = await Promise.all([
        getWhitelistStatus(firstId),
        getWhitelistStatus(secondId),
      ]);
      this.upsertSession(first);
      this.upsertSession(second);
      this.renderSessions(firstId, secondId);
      this.elements["enrollment-results"].textContent = [
        `A · ${this.formatWhitelistStatus(first)}`,
        `B · ${this.formatWhitelistStatus(second)}`,
        "각 session_id의 count/version을 서버에서 독립 조회했습니다.",
      ].join("\n");
    } catch (error) {
      this.elements["enrollment-results"].textContent = error.message;
    }
  }

  formatWhitelistStatus(status) {
    return `${status.session_id}: ${status.entry_count} entries · v${status.whitelist_version}`;
  }

  async initializeSessions() {
    this.elements.start.disabled = true;
    this.elements["new-session"].disabled = true;
    this.elements["refresh-whitelist"].disabled = true;
    this.elements["whitelist-status"].textContent = "서버 세션 목록 확인 중";
    try {
      const listed = await listSessions();
      const resolved = await ensureTabSession(
        listed,
        this.tabStorage(),
        () => createSession(),
      );
      this.sessionStatuses = resolved.sessions;
      this.sessionsReady = true;
      this.renderSessions(resolved.active.session_id);
      this.elements["whitelist-status"].textContent = resolved.created
        ? `새 탭 전용 세션 생성됨 · ${resolved.active.session_id}`
        : `탭 세션 재사용 · ${this.formatWhitelistStatus(resolved.active)}`;
    } catch (error) {
      this.sessionsReady = false;
      this.elements["whitelist-status"].textContent = error.message;
      this.elements["session-list"].textContent = "세션 목록을 불러오지 못했습니다.";
    } finally {
      this.elements["new-session"].disabled = false;
      this.elements["refresh-whitelist"].disabled = false;
    }
  }

  async refreshSessions() {
    const button = this.elements["refresh-whitelist"];
    button.disabled = true;
    try {
      const resolved = await ensureTabSession(
        await listSessions(),
        this.tabStorage(),
        () => createSession(),
      );
      const comparisonId = this.elements["comparison-session-id"].value;
      this.sessionStatuses = resolved.sessions;
      this.sessionsReady = true;
      this.renderSessions(resolved.active.session_id, comparisonId);
      this.elements["whitelist-status"].textContent = resolved.created
        ? `저장된 탭 세션이 없어 새로 생성했습니다 · ${resolved.active.session_id}`
        : this.formatWhitelistStatus(resolved.active);
    } catch (error) {
      this.elements["whitelist-status"].textContent = error.message;
    } finally {
      button.disabled = this.running;
    }
  }

  async createNewSession() {
    const button = this.elements["new-session"];
    button.disabled = true;
    try {
      const previousActive = this.elements["session-id"].value;
      const created = await createSession();
      this.upsertSession(created, true);
      this.storeTabSession(created.session_id);
      this.sessionsReady = true;
      this.renderSessions(created.session_id, previousActive);
      this.elements["whitelist-status"].textContent = `새 세션 생성됨 · ${created.session_id}`;
    } catch (error) {
      this.elements["whitelist-status"].textContent = error.message;
    } finally {
      button.disabled = this.running;
    }
  }

  selectActiveSession() {
    const sessionId = validateSessionId(this.elements["session-id"].value);
    this.storeTabSession(sessionId);
    let comparisonId = this.elements["comparison-session-id"].value;
    if (comparisonId === sessionId) {
      comparisonId = this.sessionStatuses.find(
        (session) => session.session_id !== sessionId,
      )?.session_id || "";
    }
    this.renderSessions(sessionId, comparisonId);
    const status = this.sessionStatuses.find((session) => session.session_id === sessionId);
    if (status) this.elements["whitelist-status"].textContent = this.formatWhitelistStatus(status);
  }

  upsertSession(status, newest = false) {
    const normalized = normalizeSessionInfo(status);
    const existing = this.sessionStatuses.find(
      (session) => session.session_id === normalized.session_id,
    );
    if (existing && normalized.created_at_unix_ms === 0) {
      normalized.created_at_unix_ms = existing.created_at_unix_ms;
    }
    this.sessionStatuses = this.sessionStatuses.filter(
      (session) => session.session_id !== normalized.session_id,
    );
    if (newest) this.sessionStatuses.unshift(normalized);
    else this.sessionStatuses.push(normalized);
  }

  renderSessions(activeId = "", comparisonId = "") {
    const sessions = [...this.sessionStatuses].sort((left, right) => (
      right.created_at_unix_ms - left.created_at_unix_ms
      || left.session_id.localeCompare(right.session_id)
    ));
    this.sessionStatuses = sessions;
    const active = sessions.some((session) => session.session_id === activeId)
      ? activeId
      : sessions[0]?.session_id || "";
    let comparison = sessions.some((session) => session.session_id === comparisonId)
      ? comparisonId
      : "";
    if (!comparison || comparison === active) {
      comparison = sessions.find((session) => session.session_id !== active)?.session_id || "";
    }

    this.replaceSessionOptions(this.elements["session-id"], sessions, active, false);
    this.replaceSessionOptions(
      this.elements["comparison-session-id"],
      sessions,
      comparison,
      true,
    );
    this.elements["session-id"].disabled = this.running || !active;
    this.elements["comparison-session-id"].disabled = sessions.length < 2;
    this.elements.start.disabled = this.running || !this.sessionsReady || !active;
    this.elements["add-whitelist"].disabled = !active;
    this.elements["compare-sessions"].disabled = sessions.length < 2;
    this.elements["new-session"].disabled = this.running;
    this.elements["refresh-whitelist"].disabled = this.running;
    this.elements["session-list"].textContent = sessions.length
      ? sessions.map((session) => {
        const roles = [
          session.session_id === active ? "영상·등록" : null,
          session.session_id === comparison ? "비교" : null,
        ].filter(Boolean).join(" / ");
        return `${roles ? `[${roles}] ` : ""}${this.formatWhitelistStatus(session)}`;
      }).join("\n")
      : "서버에 등록된 세션이 없습니다.";
  }

  replaceSessionOptions(select, sessions, selectedId, allowEmpty) {
    select.replaceChildren();
    if (allowEmpty && sessions.length < 2) {
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "비교할 다른 세션 없음";
      select.append(empty);
    }
    for (const session of sessions) {
      const option = document.createElement("option");
      option.value = session.session_id;
      option.textContent = this.formatWhitelistStatus(session);
      option.selected = session.session_id === selectedId;
      select.append(option);
    }
  }

  tabStorage() {
    try {
      return window.sessionStorage;
    } catch {
      return null;
    }
  }

  storeTabSession(sessionId) {
    try {
      this.tabStorage()?.setItem(TAB_SESSION_STORAGE_KEY, sessionId);
    } catch {
      // The selected session remains valid for this page without storage.
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
    const deadline = captureDeadline(
      this.nextCaptureDeadline,
      now,
      1000 / this.activeProfile.targetFps,
    );
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
      this.activeProfile.longEdge,
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
    }, "image/jpeg", this.activeProfile.jpegQuality);
  }

  pump() {
    if (
      !this.running
      || !this.pendingLatest
      || !this.socket
      || this.socket.readyState !== WebSocket.OPEN
      || this.inFlight.size >= this.activeProfile.requestWindow
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
      this.activeProfile.timeoutMs,
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
      requireResultSession(metadata, this.streamSessionId);
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
    if (this.counters.bitmapOwners > this.activeProfile.requestWindow + 2) {
      this.counters.bitmapOwners -= 1;
      throw new Error("bitmap ownership exceeded the negotiated window bound");
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
    for (const object of objects) {
      const polygon = object.mask_polygon;
      if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 64) {
        throw new Error("tracked object has no bounded mask polygon");
      }
      for (const [x, y] of polygon) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          throw new Error("mask polygon contains a non-finite point");
        }
      }
    }
    const blurredObjects = objectsToBlur(objects);
    if (blurredObjects.length) {
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
      for (const object of blurredObjects) {
        const polygon = object.mask_polygon;
        polygon.forEach(([x, y], index) => {
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
      this.outputContext.strokeStyle = object.whitelisted
        ? "#5ab0ff"
        : object.source === "held" ? "#ffd166" : "#3ee6a8";
      this.outputContext.lineWidth = 2;
      this.outputContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
      this.outputContext.fillStyle = this.outputContext.strokeStyle;
      this.outputContext.font = "13px ui-monospace, monospace";
      this.outputContext.fillText(
        `#${object.track_id} ${object.whitelisted ? "whitelist" : object.source} ${(object.confidence * 100).toFixed(0)}%`,
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
      profile: this.activeProfile,
      session_id: this.streamSessionId,
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
      upscale_small_inputs: this.activeProfile.upscaleSmallInputs,
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
    this.elements["session-id"].disabled = !this.sessionsReady;
    this.elements["new-session"].disabled = false;
    this.elements["refresh-whitelist"].disabled = false;
    this.streamSessionId = null;
    this.elements.start.disabled = !this.sessionsReady;
    this.elements.stop.disabled = true;
  }

  stop() {
    this.failClosed("중지됨");
  }
}

if (typeof document !== "undefined") new App();
