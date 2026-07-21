import assert from "node:assert/strict";
import fs from "node:fs";

globalThis.WebSocket = { OPEN: 1 };
globalThis.location = { protocol: "http:", host: "localhost" };
globalThis.createImageBitmap = async () => ({
  width: 640,
  height: 360,
  close() {},
});
const animationFrames = [];
globalThis.requestAnimationFrame = (callback) => {
  animationFrames.push(callback);
  return animationFrames.length;
};
globalThis.cancelAnimationFrame = () => {};

const source = `${fs.readFileSync(new URL("../web/app.js", import.meta.url), "utf8")}
export { StreamConnection, BinaryPacket, BoundedQueue, Playout };`;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { StreamConnection, BinaryPacket, BoundedQueue, Playout } = await import(
  moduleUrl
);

function response(frameId) {
  const jpeg = new Uint8Array([0xff, 0xd8]);
  const metadata = new TextEncoder().encode(
    JSON.stringify({
      v: 1,
      processingMs: 12,
      frames: [
        {
          id: frameId,
          capturedAt: frameId,
          size: jpeg.byteLength,
          width: 640,
          height: 360,
          faces: [],
          timing: {
            decodeMs: 2,
            inferenceMs: 4,
            trackingMs: 1,
            blurEncodeMs: 3,
            inferenceBatchSize: 1,
          },
        },
      ],
    }),
  );
  const packet = new Uint8Array(4 + metadata.byteLength + jpeg.byteLength);
  new DataView(packet.buffer).setUint32(0, metadata.byteLength);
  packet.set(metadata, 4);
  packet.set(jpeg, 4 + metadata.byteLength);
  return packet.buffer;
}

const dropped = [];
const queue = new BoundedQueue(1, () => dropped.push(true));
queue.push(
  ...Array.from({ length: 3 }, (_, id) => ({
    id,
    capturedAt: id,
    readyAt: performance.now(),
    captureEncodeMs: 1,
    jpeg: new Blob([[0xff, 0xd8]], { type: "image/jpeg" }),
  })),
);
assert.equal(queue.length, 1);
assert.equal(dropped.length, 2);

const received = [];
const sent = [];
const connection = new StreamConnection(
  queue,
  (frame) => received.push(frame),
  () => {},
);
connection.generation = 1;
connection.socket = {
  readyState: WebSocket.OPEN,
  send: (packet) => sent.push(packet),
};
connection.pump(1);

assert.equal(sent.length, 1);
assert.ok(sent[0] instanceof Blob);
assert.equal(BinaryPacket.decode(await sent[0].arrayBuffer()).frames.length, 1);
assert.equal(connection.inFlight, true);

connection.receive({ data: response(2) }, 1);
assert.equal(received.length, 1);
assert.equal(received[0].timing.inferenceBatchSize, 1);
assert.equal(received[0].serverProcessingMs, 12);
assert.equal(connection.inFlight, false);

const pipelineQueue = new BoundedQueue(2, () => {});
const pipelineFrames = [3, 4].map((id) => ({
    id,
    capturedAt: id,
    readyAt: performance.now(),
    captureEncodeMs: 1,
    jpeg: new Blob([[0xff, 0xd8]], { type: "image/jpeg" }),
  }));
const pipelineReceived = [];
const pipeline = new StreamConnection(
  pipelineQueue,
  (frame) => pipelineReceived.push(frame.id),
  () => {},
);
pipeline.generation = 1;
pipeline.socket = {
  readyState: WebSocket.OPEN,
  send: (packet) => sent.push(packet),
};
pipelineQueue.push(pipelineFrames[0]);
pipeline.pump(1);
assert.equal(pipeline.inFlightCount, 1);
pipelineQueue.push(pipelineFrames[1]);
pipeline.pump(1);
assert.equal(pipeline.inFlightCount, 2);
pipeline.receive({ data: response(3) }, 1);
assert.equal(pipelineReceived[0], 3);
assert.equal(pipeline.inFlightCount, 1);

const context = {
  clearRect() {},
  drawImage() {},
  strokeRect() {},
  fillRect() {},
  fillText() {},
  measureText: () => ({ width: 10 }),
};
const canvas = { width: 640, height: 360, getContext: () => context };
const playoutQueue = new BoundedQueue(1, () => {});
let displayed = 0;
const playout = new Playout(canvas, playoutQueue, () => {
  displayed += 1;
});
playout.generation = 1;
playoutQueue.push({ jpeg: new ArrayBuffer(2), faces: [] });
playout.schedule();

await new Promise(setImmediate);
assert.equal(displayed, 0);
assert.equal(animationFrames.length, 1);
animationFrames.shift()();
assert.equal(displayed, 1);

console.log("latest-frame streaming and immediate-decode playout passed");
