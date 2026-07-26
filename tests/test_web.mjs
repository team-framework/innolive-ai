import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  BitmapLease,
  LatestRenderQueue,
  PROFILE,
  TAB_SESSION_STORAGE_KEY,
  addWhitelistFiles,
  captureDeadline,
  createSession,
  deleteSession,
  deleteWhitelistEntry,
  enrollmentSize,
  ensureTabSession,
  fitLongEdge,
  framePacket,
  getWhitelistStatus,
  listSessions,
  negotiateServerProfile,
  normalizeWhitelistImage,
  objectsToBlur,
  parseTerminal,
  requireResultSession,
  validateSessionId,
  websocketUrl,
  whitelistUrl,
} from "../web/app.js";

const browserHtml = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
const browserScript = await readFile(new URL("../web/app.js", import.meta.url), "utf8");
assert.doesNotMatch(browserHtml, /comparison-session|compare-sessions|비교할 서버 세션/);
assert.match(browserHtml, /id="session-list"/);
assert.match(browserHtml, /id="whitelist-dropzone"/);
assert.match(browserHtml, /id="whitelist-entries"/);
assert.doesNotMatch(browserHtml, /id="diagnostics"|sent-fps|round-trip|server-time/);
assert.doesNotMatch(
  browserHtml,
  /현재 terminal metadata|WHITELIST|SESSIONS|클릭해 여러 장|JPEG · PNG · WebP/,
);
assert.equal([...browserHtml.matchAll(/id="[^"]*-fps"/g)].length, 3);
assert.doesNotMatch(
  browserScript,
  /__INNOLIVE_METRICS__|__INNOLIVE_METRIC_HISTORY__|__INNOLIVE_BITMAP_OWNERS__|보호 결과 seq|fillText\(/,
);

assert.deepEqual(PROFILE, {
  protocolVersion: 1,
  longEdge: 640,
  jpegQuality: 0.90,
  targetFps: 30,
  requestWindow: 5,
  timeoutMs: 1500,
  upscaleSmallInputs: false,
});

assert.deepEqual(fitLongEdge(1920, 1080), [640, 360]);
assert.deepEqual(fitLongEdge(1080, 1920), [360, 640]);
assert.deepEqual(fitLongEdge(320, 240), [320, 240]);

assert.equal(validateSessionId(" Session-A "), " Session-A ");
assert.throws(() => validateSessionId("   "), /must not be empty/);
assert.throws(() => validateSessionId("가".repeat(86)), /256 UTF-8 bytes/);
assert.equal(
  whitelistUrl(" Session-A "),
  "/api/whitelist?session_id=%20Session-A%20",
);
assert.equal(
  websocketUrl({ protocol: "https:", host: "demo.test" }, "session-b"),
  "wss://demo.test/ws?session_id=session-b",
);
assert.equal(
  requireResultSession({ session_id: "session-a" }, "session-a").session_id,
  "session-a",
);
assert.throws(
  () => requireResultSession({ session_id: "session-b" }, "session-a"),
  /does not match/,
);

const serverSessions = [
  {
    session_id: "session-existing",
    entry_count: 2,
    whitelist_version: 3,
    created_at_unix_ms: 100,
    active_stream_count: 2,
  },
];
const listedSessions = await listSessions(async (url, options) => {
  assert.equal(url, "/api/sessions");
  assert.equal(options.cache, "no-store");
  return {
    ok: true,
    status: 200,
    json: async () => ({ sessions: serverSessions }),
  };
});
assert.deepEqual(listedSessions, serverSessions);

const createdSession = await createSession(async (url, options) => {
  assert.equal(url, "/api/sessions");
  assert.equal(options.method, "POST");
  return {
    ok: true,
    status: 201,
    json: async () => ({
      session_id: "session-created",
      entry_count: 0,
      whitelist_version: 0,
      created_at_unix_ms: 200,
      active_stream_count: 0,
    }),
  };
});
assert.equal(createdSession.session_id, "session-created");

let deletedSessionRequest = null;
await deleteSession(" session/to-delete ", async (url, options) => {
  deletedSessionRequest = { url, options };
  return { ok: true, status: 204 };
});
assert.deepEqual(deletedSessionRequest, {
  url: "/api/sessions/%20session%2Fto-delete%20",
  options: { method: "DELETE" },
});
await assert.rejects(
  deleteSession("session-streaming", async () => ({
    ok: false,
    status: 409,
    json: async () => ({
      error: { code: "FAILED_PRECONDITION", message: "session has active streams" },
    }),
  })),
  /FAILED_PRECONDITION: session has active streams/,
);

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
}

let createdTabs = 0;
const createTabSession = async () => ({
  session_id: `session-tab-${++createdTabs}`,
  entry_count: 0,
  whitelist_version: 0,
  created_at_unix_ms: 200 + createdTabs,
  active_stream_count: 0,
});
const firstTabStorage = memoryStorage();
const secondTabStorage = memoryStorage();
const firstTab = await ensureTabSession(serverSessions, firstTabStorage, createTabSession);
const secondTab = await ensureTabSession(serverSessions, secondTabStorage, createTabSession);
assert.equal(firstTab.created, true);
assert.equal(secondTab.created, true);
assert.notEqual(firstTab.active.session_id, secondTab.active.session_id);
assert.equal(
  firstTabStorage.getItem(TAB_SESSION_STORAGE_KEY),
  firstTab.active.session_id,
);
const reusedTab = await ensureTabSession(
  firstTab.sessions,
  firstTabStorage,
  async () => { throw new Error("a refreshed tab must not create another session"); },
);
assert.equal(reusedTab.created, false);
assert.equal(reusedTab.active.session_id, firstTab.active.session_id);

assert.deepEqual(enrollmentSize(1600, 800), [640, 320]);
assert.deepEqual(enrollmentSize(320, 240), [320, 240]);
assert.throws(() => enrollmentSize(0, 100), /invalid dimensions/);

const imageOperations = [];
let enrollmentBitmapCloses = 0;
let encodedQuality = null;
const canvas = {
  width: 0,
  height: 0,
  getContext: () => ({
    fillStyle: null,
    fillRect(x, y, width, height) {
      imageOperations.push(["fill", this.fillStyle, x, y, width, height]);
    },
    drawImage(_bitmap, x, y, width, height) {
      imageOperations.push(["draw", x, y, width, height]);
    },
  }),
  toBlob(callback, type, quality) {
    encodedQuality = quality;
    callback(new Blob(["jpeg"], { type }));
  },
};
const normalizedPng = await normalizeWhitelistImage(
  { type: "image/png" },
  {
    decode: async () => ({
      width: 1600,
      height: 800,
      close: () => { enrollmentBitmapCloses += 1; },
    }),
    createCanvas: () => canvas,
  },
);
assert.equal(normalizedPng.type, "image/jpeg");
assert.deepEqual([canvas.width, canvas.height], [640, 320]);
assert.deepEqual(imageOperations[0], ["fill", "#fff", 0, 0, 640, 320]);
assert.deepEqual(imageOperations[1], ["draw", 0, 0, 640, 320]);
assert.equal(encodedQuality, PROFILE.jpegQuality);
assert.equal(enrollmentBitmapCloses, 1, "normalized image bitmap must close once");

let failedBitmapCloses = 0;
await assert.rejects(
  normalizeWhitelistImage(
    { type: "image/webp" },
    {
      decode: async () => ({
        width: 100,
        height: 100,
        close: () => { failedBitmapCloses += 1; },
      }),
      createCanvas: () => ({ getContext: () => null }),
    },
  ),
  /could not create an image canvas/,
);
assert.equal(failedBitmapCloses, 1, "failed normalization must close its bitmap");

let enrollmentInFlight = 0;
let maxEnrollmentInFlight = 0;
const enrollmentCalls = [];
const normalizedTypes = [];
const enrollmentProgress = [];
const enrollmentRequest = async (url, options) => {
  enrollmentInFlight += 1;
  maxEnrollmentInFlight = Math.max(maxEnrollmentInFlight, enrollmentInFlight);
  enrollmentCalls.push({ url, options });
  await Promise.resolve();
  enrollmentInFlight -= 1;
  const count = enrollmentCalls.length;
  return {
    ok: count !== 2,
    status: count === 2 ? 400 : 201,
    json: async () => count === 2
      ? { error: { code: "INVALID_ARGUMENT", message: "expected one face" } }
      : { session_id: "session-a", entry_count: count, whitelist_version: count },
  };
};
const enrollmentResults = await addWhitelistFiles(
  [
    { name: "first.jpg", type: "image/jpeg" },
    { name: "invalid.png", type: "image/png" },
    { name: "third.webp", type: "image/webp" },
  ],
  "session-a",
  enrollmentRequest,
  async (file) => {
    normalizedTypes.push(file.type);
    return new Blob([file.name], { type: "image/jpeg" });
  },
  (progress) => enrollmentProgress.push(`${progress.index}:${progress.state}`),
);
assert.equal(maxEnrollmentInFlight, 1, "enrollments must be submitted sequentially");
assert.deepEqual(enrollmentResults.map((result) => result.ok), [true, false, true]);
assert.equal(enrollmentCalls.length, 3, "one invalid face must not skip later files");
assert.deepEqual(normalizedTypes, ["image/jpeg", "image/png", "image/webp"]);
assert.ok(enrollmentCalls.every((call) => call.options.body.type === "image/jpeg"));
assert.deepEqual(enrollmentProgress, [
  "0:preparing", "0:uploading", "0:success",
  "1:preparing", "1:uploading", "1:error",
  "2:preparing", "2:uploading", "2:success",
]);

let requestsAfterNormalizationFailure = 0;
const isolatedResults = await addWhitelistFiles(
  [
    { name: "unsupported.heic", type: "image/heic" },
    { name: "valid.png", type: "image/png" },
  ],
  "session-a",
  async () => {
    requestsAfterNormalizationFailure += 1;
    return {
      ok: true,
      status: 201,
      json: async () => ({
        session_id: "session-a",
        entry_count: 1,
        whitelist_version: 1,
      }),
    };
  },
  async (file) => {
    if (file.type === "image/heic") throw new Error("browser could not decode image");
    return new Blob(["jpeg"], { type: "image/jpeg" });
  },
);
assert.deepEqual(isolatedResults.map((result) => result.ok), [false, true]);
assert.equal(requestsAfterNormalizationFailure, 1);

const queriedStatus = await getWhitelistStatus("session-b", async (url) => ({
  ok: true,
  status: 200,
  json: async () => ({
    session_id: new URL(url, "http://local").searchParams.get("session_id"),
    entry_count: 2,
    whitelist_version: 3,
    entry_ids: ["entry-a", "entry/b"],
  }),
}));
assert.deepEqual(queriedStatus, {
  session_id: "session-b",
  entry_count: 2,
  whitelist_version: 3,
  entry_ids: ["entry-a", "entry/b"],
});

let deletedWhitelistRequest = null;
await deleteWhitelistEntry("session b", "entry/b", async (url, options) => {
  deletedWhitelistRequest = { url, options };
  return { ok: true, status: 204 };
});
assert.deepEqual(deletedWhitelistRequest, {
  url: "/api/whitelist?session_id=session%20b&entry_id=entry%2Fb",
  options: { method: "DELETE" },
});
await assert.rejects(
  getWhitelistStatus("session-b", async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      session_id: "session-b",
      entry_count: 2,
      whitelist_version: 3,
      entry_ids: ["duplicate", "duplicate"],
    }),
  })),
  /duplicate whitelist entry IDs/,
);
await assert.rejects(
  getWhitelistStatus("session-b", async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      session_id: "different-session",
      entry_count: 0,
      whitelist_version: 0,
      entry_ids: [],
    }),
  })),
  /does not match the requested session/,
);
await assert.rejects(
  getWhitelistStatus("session-b", async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      session_id: "session-b",
      entry_count: 1,
      whitelist_version: 1,
      entry_ids: [],
    }),
  })),
  /inconsistent whitelist entry count/,
);

const health = {
  status: "ok",
  profile: "a-newer-profile-name",
  protocol: { name: "ILF1", version: 1 },
  transport_path: ["browser-websocket", "metrics", "grpc-bidi-ProcessVideo"],
  grpc: { service: "AiProcessor", serving: true },
  serving_profile: {
    engine_batch: 4,
    max_long_edge: 1280,
    jpeg_quality: 95,
    client_window: 12,
    target_fps: 60,
    max_streams: 32,
  },
};
assert.deepEqual(negotiateServerProfile(health), {
  ...PROFILE,
  maxStreams: 32,
});
assert.deepEqual(negotiateServerProfile({
  ...health,
  serving_profile: {
    engine_batch: 1,
    image_size: 320,
    jpeg_quality: 75,
    client_window: 2,
    target_fps: 15,
  },
}), {
  ...PROFILE,
  longEdge: 320,
  jpegQuality: 0.75,
  targetFps: 15,
  requestWindow: 2,
  maxStreams: null,
});
assert.throws(
  () => negotiateServerProfile({ ...health, protocol: { name: "ILF1", version: 2 } }),
  /ILF1 v1 is required/,
);
assert.deepEqual(
  negotiateServerProfile({
    ...health,
    serving_profile: { ...health.serving_profile, max_streams: 0 },
  }).maxStreams,
  null,
);
assert.throws(
  () => negotiateServerProfile({
    ...health,
    serving_profile: { ...health.serving_profile, max_streams: -1 },
  }),
  /max_streams must be a positive integer/,
);

const jpeg = new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])]);
const packet = new Uint8Array(await framePacket(0x01020304, jpeg).arrayBuffer());
assert.deepEqual([...packet.slice(0, 8)], [0x49, 0x4c, 0x46, 0x31, 1, 2, 3, 4]);
assert.deepEqual([...packet.slice(8)], [0xff, 0xd8, 0xff, 0xd9]);

assert.deepEqual(
  parseTerminal('{"v":1,"type":"result","seq":7,"objects":[]}'),
  { v: 1, type: "result", seq: 7, objects: [] },
);
assert.throws(() => parseTerminal('{"v":1,"type":"frame","seq":7}'));
assert.throws(() => parseTerminal('{"v":1,"type":"result"}'));

const whitelistedFace = {
  whitelisted: true,
  mask_polygon: [[10, 10], [50, 10], [50, 50], [10, 50]],
};
const protectedFace = {
  whitelisted: false,
  mask_polygon: [[30, 30], [70, 30], [70, 70], [30, 70]],
};
assert.deepEqual(
  objectsToBlur([whitelistedFace, protectedFace]),
  [protectedFace],
  "the non-whitelisted mask must remain in the blur union across overlap",
);

let deadline = null;
let dueFrames = 0;
for (let index = 0; index < 60; index += 1) {
  const jitter = index % 2 ? -0.9 : 0.7;
  const decision = captureDeadline(deadline, index * (1000 / 30) + jitter);
  deadline = decision.nextDeadline;
  if (decision.due) dueFrames += 1;
}
assert.equal(dueFrames, 60, "normal 32.x ms jitter must not halve capture cadence");
const delayed = captureDeadline(100, 240);
assert.equal(delayed.due, true);
assert.ok(delayed.nextDeadline > 242, "missed deadlines must not create a catch-up burst");

let normalCloses = 0;
let normalReleases = 0;
const normalLease = new BitmapLease(
  jpeg,
  () => Promise.resolve({ close: () => { normalCloses += 1; } }),
  () => { normalReleases += 1; },
);
assert.ok(await normalLease.promise);
assert.equal(normalLease.release(), true);
assert.equal(normalLease.release(), false);
assert.equal(normalCloses, 1, "a rendered bitmap must close exactly once");
assert.equal(normalReleases, 1, "a bitmap ownership lease must release exactly once");

let resolveLateBitmap;
let lateCloses = 0;
const lateLease = new BitmapLease(
  jpeg,
  () => new Promise((resolve) => { resolveLateBitmap = resolve; }),
);
assert.equal(lateLease.release(), true);
resolveLateBitmap({ close: () => { lateCloses += 1; } });
assert.equal(await lateLease.promise, null);
assert.equal(lateCloses, 1, "a decode-late-resolve bitmap must close exactly once");

const renderQueue = new LatestRenderQueue();
const frame1 = { sequence: 1 };
const frame2 = { sequence: 2 };
const frame3 = { sequence: 3 };
assert.equal(renderQueue.enqueue(frame1).accepted, true);
assert.equal(renderQueue.begin(), frame1);
assert.equal(renderQueue.enqueue(frame2).dropped, null);
assert.equal(renderQueue.enqueue(frame3).dropped, frame2);
assert.equal(renderQueue.active, frame1, "a newer result must not cancel active render");
assert.equal(renderQueue.commit(frame1), true);
renderQueue.finish(frame1);
assert.equal(renderQueue.begin(), frame3, "only the latest not-started render may survive");
assert.equal(renderQueue.commit(frame3), true);
renderQueue.finish(frame3);
assert.equal(renderQueue.lastCommittedSequence, 3);
assert.equal(renderQueue.enqueue(frame2).accepted, false, "commit sequence must be monotonic");

console.log("ILF1 browser compatibility contract passed");
