import assert from "node:assert/strict";
import { h264Codecs, resultIsStale, waitForIce, WebRtcDemo } from "../web/webrtc.js";

assert.equal(h264Codecs({ codecs: [{mimeType:"video/VP8"}, {mimeType:"video/H264"}] }).length, 1);
assert.equal(resultIsStale(null, 10000), false);
assert.equal(resultIsStale(1000, 3001), true);

const pc = new EventTarget();
pc.iceGatheringState = "gathering";
await assert.rejects(waitForIce(pc, 5), /초과/);
const ready = waitForIce(pc, 50);
pc.iceGatheringState = "complete";
pc.dispatchEvent(new Event("icegatheringstatechange"));
await ready;

globalThis.addEventListener = () => {};
const elements = {};
const document = { getElementById: (id) => elements[id] ||= {
  dataset: {}, files: [], pause() {}, load() {}, removeAttribute() {},
} };
const demo = new WebRtcDemo(document);
let closed = 0, stopped = 0;
demo.pc = { close() { closed++; } };
demo.stream = { getTracks: () => [{ stop() { stopped++; } }] };
demo.el.result.srcObject = { unprocessed: false };
demo.fail("inference failed");
assert.equal(closed, 1);
assert.equal(stopped, 1);
assert.equal(demo.el.result.srcObject, null);
assert.equal(demo.el.result.hidden, true);
assert.equal(demo.el.placeholder.hidden, false);

let resolveCamera;
demo.inputStream = () => new Promise((resolve) => { resolveCamera = resolve; });
const starting = demo.start();
demo.stop();
resolveCamera({ getTracks: () => [{ stop() { stopped++; } }] });
await starting;
assert.equal(stopped, 2);
assert.equal(demo.pc, null);
assert.equal(demo.busy, false);
