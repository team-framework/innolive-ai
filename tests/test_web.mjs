import assert from "node:assert/strict";

import {
  PROFILE,
  fitLongEdge,
  framePacket,
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

console.log("ILF1 B1-640-Q90-W5 browser contract passed");
