import assert from "node:assert/strict";

import {
  BitmapLease,
  LatestRenderQueue,
  PROFILE,
  addWhitelistFiles,
  captureDeadline,
  fitLongEdge,
  framePacket,
  getWhitelistStatus,
  negotiateServerProfile,
  objectsToBlur,
  parseTerminal,
  percentile,
  requireResultSession,
  validateSessionId,
  websocketUrl,
  whitelistUrl,
} from "../web/app.js";

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

let enrollmentInFlight = 0;
let maxEnrollmentInFlight = 0;
const enrollmentCalls = [];
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
    { name: "invalid.jpg", type: "image/jpeg" },
    { name: "third.jpg", type: "image/jpeg" },
  ],
  "session-a",
  enrollmentRequest,
);
assert.equal(maxEnrollmentInFlight, 1, "enrollments must be submitted sequentially");
assert.deepEqual(enrollmentResults.map((result) => result.ok), [true, false, true]);
assert.equal(enrollmentCalls.length, 3, "one invalid face must not skip later files");

const queriedStatus = await getWhitelistStatus("session-b", async (url) => ({
  ok: true,
  status: 200,
  json: async () => ({ session_id: new URL(url, "http://local").searchParams.get("session_id"), entry_count: 0, whitelist_version: 0 }),
}));
assert.deepEqual(queriedStatus, {
  session_id: "session-b",
  entry_count: 0,
  whitelist_version: 0,
});

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

assert.equal(percentile([1, 2, 3, 4], 0.50), 2);
assert.equal(percentile([1, 2, 3, 4], 0.95), 4);
assert.equal(percentile([], 0.95), null);

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

let clock = 10;
let normalCloses = 0;
let normalReleases = 0;
const normalLease = new BitmapLease(
  jpeg,
  () => Promise.resolve({ close: () => { normalCloses += 1; } }),
  () => clock,
  () => { normalReleases += 1; },
);
clock = 14;
assert.ok(await normalLease.promise);
assert.equal(normalLease.decodeMs, 4);
assert.equal(normalLease.release(), true);
assert.equal(normalLease.release(), false);
assert.equal(normalCloses, 1, "a rendered bitmap must close exactly once");
assert.equal(normalReleases, 1, "a bitmap ownership lease must release exactly once");

let resolveLateBitmap;
let lateCloses = 0;
const lateLease = new BitmapLease(
  jpeg,
  () => new Promise((resolve) => { resolveLateBitmap = resolve; }),
  () => 0,
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
