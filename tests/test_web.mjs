import assert from "node:assert/strict";

import {
  BitmapLease,
  LatestRenderQueue,
  PROFILE,
  captureDeadline,
  fitLongEdge,
  framePacket,
  objectsToBlur,
  parseTerminal,
  percentile,
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

console.log("ILF1 B1-640-Q90-W5 browser contract passed");
