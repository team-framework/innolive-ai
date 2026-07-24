# gRPC performance tests

The benchmark suite uses [ghz](https://ghz.sh/) so load generation, gRPC
stream handling, latency measurement, and raw result generation are delegated to
a dedicated tool. The Python runner adds reproducible configuration, local
server lifecycle management, process resource sampling, and report packaging.

## Setup

Install the pinned ghz release inside this repository (the binary is ignored by
Git):

```bash
./benchmarks/scripts/install_ghz.sh
```

The installer verifies the checksum published with the ghz release. Override
`GHZ_VERSION` or `GHZ_INSTALL_DIR` only when intentionally testing another
toolchain.

## Run

For real-time video communication latency, use the per-frame probe. It timestamps
each protobuf message immediately before sending and matches the echoed response;
the managed server uses `passthrough` so image processing is excluded:

```bash
.venv/bin/python benchmarks/run_video_latency.py \
  --video /path/to/demo.mp4 \
  --fps 30 \
  --max-frames 300 \
  --concurrency 2 \
  --start-server
```

This command writes only `summary.md`, `summary.png`, and `result.json` directly
under `benchmark-results/`. Use the ghz runner below for saturation/load tests;
ghz stream-call latency is not a per-frame RTT measurement.

Run a short smoke profile and let the runner manage the local server:

```bash
.venv/bin/python benchmarks/run_benchmarks.py --profile smoke --start-server
```

Run the baseline against an already running server and sample that process:

```bash
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile baseline \
  --target 127.0.0.1:50051 \
  --server-pid "$(pgrep -f 'ai_processor_server.py' | head -1)"
```

Stream a demo video at 10 sampled FPS through the real bidirectional RPC while
the managed server echoes bytes in `passthrough` mode:

```bash
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile video_realtime \
  --start-server \
  --scenario process_video \
  --video /path/to/demo.mp4
```

Find the maximum unpaced communication throughput with the same frames:

```bash
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile video_transport \
  --start-server \
  --scenario process_video \
  --video /path/to/demo.mp4
```

`video_realtime` spaces messages according to the sampled FPS.
`video_transport` sends them without an interval. Both managed-server profiles
select `--processing-mode passthrough`, excluding grayscale decode/encode work.
To measure end-to-end processing instead, start the server normally (the default
is `grayscale`) with the managed `video_e2e` profile:

```bash
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile video_e2e \
  --start-server \
  --scenario process_video \
  --video /path/to/demo.mp4
```

Video extraction defaults live in `[video]` in `config.toml`. They can be
overridden per run with `--video-sample-fps`, `--video-max-frames`, and
`--video-jpeg-quality`. Frames are extracted and JPEG-encoded before measurement
starts. The report records source video hash and dimensions, source/sample FPS,
sampled frame count, JPEG quality, and min/average/max encoded frame bytes.

Useful options:

```bash
# Run only the bidirectional streaming scenario
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile baseline --start-server --scenario process_video

# Benchmark a remote target (resource sampling is omitted)
.venv/bin/python benchmarks/run_benchmarks.py \
  --profile baseline --target grpc.example.com:50051
```

## Profiles and extension points

Edit `benchmarks/config.toml` to add profiles or scenarios. Profiles define test
intensity; scenarios define the RPC and payload. Both accept `extra_args`, an
array of ghz command-line arguments, so new ghz features can be used without
changing the runner. Scenario values override profile values.

Each scenario may define these thresholds:

```toml
[scenarios.thresholds]
max_error_rate = 0.0
max_p95_ms = 250.0
```

For stable comparisons, run the load generator on a separate machine, keep the
server configuration and payload hash identical, avoid other heavy workloads,
and repeat the baseline several times. A local run includes both client and
server contention and is best treated as a development signal.

## Recorded output

Every run creates `benchmark-results/<timestamp>-<profile>/`. The directory is
ignored by Git and contains only:

- `summary.md`: concise human-readable metrics.
- `summary.png`: throughput and latency graph.
- `result.json`: structured summary, input hash/configuration, environment, and
  server resources for future comparison or automation.

Raw ghz output, request payloads, resource samples, and server logs are created
in a temporary working directory and removed after the compact report is built.

ghz reports latency for a completed stream call, not for each message inside the
stream. The summary therefore labels call latency separately and derives
messages/s as `calls/s × messages per stream`.
