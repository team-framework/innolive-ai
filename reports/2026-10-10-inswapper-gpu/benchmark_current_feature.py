"""Network-free benchmark of the unmodified #37 server processing path."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

ROOT = Path('/home/gorani/apps/new-inno-live-server-python')
sys.path.insert(0, str(ROOT))
import service.face_anonymization as composition
from ai_processor_server import GrpcServerSettings, StreamState, build_grpc_server
from grpc_client import VideoFrame, VideoProcessorClient
from protos import ai_processor_pb2 as pb
from scripts.benchmark_grpc import run_stream, summarize
from service.face_anonymization import FaceAnonymizationConfig
from service.recognition import SessionRegistry, StreamRecognition
from service.runtime import RuntimeConfig, RuntimeManager, sha256_file
from service.tracking import StreamTracker


def distribution(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(array.mean()), 'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)), 'max': float(array.max())}


def canvas(source, faces, width, height):
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    cols, rows = math.ceil(math.sqrt(faces)), math.ceil(faces / math.ceil(math.sqrt(faces)))
    cell_w, cell_h = width // cols, height // rows
    scale = min(cell_w / source.shape[1], cell_h / source.shape[0])
    portrait = cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_LINEAR)
    for i in range(faces):
        x = i % cols * cell_w + (cell_w - portrait.shape[1]) // 2
        y = i // cols * cell_h + (cell_h - portrait.shape[0]) // 2
        image[y:y + portrait.shape[0], x:x + portrait.shape[1]] = portrait
    return image


class Instrumentation:
    def __init__(self, runtime):
        self.tls = threading.local()
        self.frames = {}
        self.runtime = runtime
        self._wrap(runtime.extractor, 'predict', 'metadata_predict_ms')
        self._wrap(runtime.renderer, 'detect', 'yunet_ms')
        self._wrap(runtime.renderer, 'source_latent', 'source_latent_ms')
        self._wrap(runtime.renderer, 'swap', 'face_swap_total_ms')
        self._wrap(composition, '_encode_jpeg', 'jpeg_encode_ms')
        self._wrap(runtime.renderer.swapper, 'run', 'inswapper_ort_run_ms')
        self._wrap(runtime.renderer.arcface, 'run', 'arcface_ort_run_ms')
        original = runtime.process

        def process(*args, **kwargs):
            frame_trace = {}
            self.tls.current = frame_trace
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                frame_trace['composition_direct_ms'] = [(time.perf_counter() - started) * 1000]
                self.frames[(id(args[2]), args[3])] = frame_trace
                self.tls.current = None
        runtime.process = process

    def _wrap(self, owner, name, metric):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                current = getattr(self.tls, 'current', None)
                if current is not None:
                    current.setdefault(metric, []).append((time.perf_counter() - started) * 1000)
        setattr(owner, name, wrapped)


async def run_config(bundle, source, faces, width, height, sessions, warmup, count, instrumentation):
    servicer = bundle.servicer
    image = canvas(source, faces, width, height)
    success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert success
    jpeg = encoded.tobytes()
    states = []
    records = []

    async def drive(stream, tracker, first, n, collect):
        for seq in range(first, first + n):
            request = pb.VideoChunk(data=jpeg, timestamp=seq, frame_id=seq, session_id=stream.session_id,
                                    output_mode=2, anonymization_mode=1)
            started = time.perf_counter()
            outcome = await servicer._process_request(request, tracker, stream, seq)
            elapsed_ms = (time.perf_counter() - started) * 1000
            response = outcome.response
            if response.error_code:
                raise RuntimeError(f'{response.error_code}: {response.error_message}')
            infos = [f.anonymization for f in response.faces if f.class_name in {'', 'face'}]
            trace = instrumentation.frames.pop((id(stream.identities), seq), {})
            if collect:
                timing = response.timing
                stage = {name: getattr(timing, name) for name in ['queue_ms', 'decode_ms', 'inference_ms',
                        'tracking_ms', 'blur_encode_ms', 'runtime_total_ms', 'serialize_ms', 'server_total_ms']}
                composition_ms = sum(trace.get('composition_direct_ms', []))
                encode_ms = sum(trace.get('jpeg_encode_ms', []))
                records.append({'session_id': stream.session_id, 'frame': seq, 'elapsed_ms': elapsed_ms,
                                'timing': stage, 'trace': trace, 'compute_excluding_codec_and_queue_ms':
                                timing.runtime_total_ms + composition_ms - encode_ms,
                                'statuses': [info.status for info in infos],
                                'fallback_reasons': [info.fallback_reason for info in infos if info.fallback_reason],
                                'identities': [info.identity_key for info in infos],
                                'adaface_calls': response.stats.adaface_calls})

    try:
        for index in range(sessions):
            session_id = f'bench-{faces}-{width}-{sessions}-{index}'
            stream = StreamState(session_id=session_id, output_mode=2, anonymization_mode=1,
                                 recognition=StreamRecognition(None, servicer.settings.recognition, owner=session_id))
            tracker = StreamTracker(servicer.settings.tracker_config, servicer.runtime.device)
            states.append((stream, tracker))
        for stream, tracker in states:
            await drive(stream, tracker, 1, warmup, False)
        free_before, total_memory = torch.cuda.mem_get_info()
        started = time.perf_counter()
        await asyncio.gather(*(drive(stream, tracker, warmup + 1, count, True) for stream, tracker in states))
        elapsed = time.perf_counter() - started
        free_after, _ = torch.cuda.mem_get_info()
        statuses = Counter(status for record in records for status in record['statuses'])
        reasons = Counter(reason for record in records for reason in record['fallback_reasons'])
        trace_names = sorted({name for record in records for name in record['trace']})
        result = {'faces_requested': faces, 'input_width': width, 'input_height': height, 'sessions': sessions,
                  'frames_per_session': count, 'warmup_frames_per_session': warmup, 'frames_completed': len(records),
                  'wall_s': elapsed, 'fps_aggregate_network_free': len(records) / elapsed,
                  'fps_per_session_network_free': len(records) / elapsed / sessions,
                  'successful_swaps_per_second': statuses['swapped'] / elapsed,
                  'successful_faces_per_frame': statuses['swapped'] / len(records),
                  'statuses': dict(statuses), 'fallback_reasons': dict(reasons),
                  'adaface_calls': sum(record['adaface_calls'] for record in records),
                  'elapsed_ms': distribution([record['elapsed_ms'] for record in records]),
                  'compute_excluding_codec_and_queue_ms': distribution([record['compute_excluding_codec_and_queue_ms'] for record in records]),
                  'timing_ms': {name: distribution([record['timing'][name] for record in records]) for name in records[0]['timing']},
                  'per_frame_trace_ms': {name: distribution([sum(record['trace'].get(name, [])) for record in records]) for name in trace_names},
                  'per_call_trace_ms': {name: distribution([v for record in records for v in record['trace'].get(name, [])]) for name in trace_names},
                  'gpu_memory': {'total_bytes': total_memory, 'free_before': free_before, 'free_after': free_after},
                  'records': records}
        result['full_swap_coverage'] = statuses['swapped'] == faces * len(records) and not reasons
        print(json.dumps({key: value for key, value in result.items() if key not in {'records', 'timing_ms', 'per_frame_trace_ms', 'per_call_trace_ms'}}), flush=True)
        return result
    finally:
        for stream, tracker in states:
            stream.recognition.close()
            stream.identities.clear()
            if stream.session is not None:
                servicer.sessions.release_stream(stream.session)
            tracker.reset()


async def main(args):
    import tensorrt, onnxruntime, ultralytics
    source = cv2.imread(str(args.source))
    assert source is not None
    config = RuntimeConfig(checkpoint=ROOT / 'models/best.pt', engine=args.scratch / 'best_b1.engine',
                           backend='tensorrt', device='0', warmup_runs=10)
    settings = GrpcServerSettings(runtime=config, port=0, inference_timeout_seconds=30,
                    face_anonymization=FaceAnonymizationConfig(preset_manifest=args.scratch / 'face_presets.json',
                    arcface_model=Path.home() / '.insightface/models/buffalo_l/w600k_r50.onnx',
                    metadata_device='cuda:0', swap_provider='cuda'))
    runtime = await asyncio.to_thread(RuntimeManager, config)
    bundle = build_grpc_server(settings, runtime, adaface=None)
    report = {'measured_at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'method': 'AiProcessorServicer._process_request invoked directly; no sockets or RPC in timed configurations',
              'limitations': ['Repeated static tiled synthetic portrait, not live video',
              'Empty whitelist: AdaFace inference is absent', 'NVML version mismatch: GPU utilization not collected',
              'Other pre-existing host services kept running', 'Composition includes CPU alignment/blend and JPEG encoding; codec-excluded compute reported separately'],
              'environment': {'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'tensorrt': tensorrt.__version__,
              'onnxruntime': onnxruntime.__version__, 'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__,
              'opencv_threads': cv2.getNumThreads(), 'torch_threads': torch.get_num_threads(),
              'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()},
              'artifacts': {str(p): sha256_file(p) for p in [config.checkpoint, config.engine,
              args.scratch / 'face_presets.json', ROOT / 'models/face_metadata_extracter.pt',
              ROOT / 'models/face_swap/inswapper_128.onnx', settings.face_anonymization.arcface_model, args.source]},
              'engine_manifest': runtime.manifest, 'configs': {}}
    try:
        await bundle.server.start()
        await bundle.set_serving(True)
        face_runtime = bundle.servicer.face_anonymization
        started = time.perf_counter()
        await asyncio.to_thread(face_runtime._metadata)
        await asyncio.to_thread(face_runtime._renderer)
        report['experimental_model_load_ms'] = (time.perf_counter() - started) * 1000
        report['active_ort_providers'] = {'swapper': face_runtime.renderer.swapper.get_providers(),
                                        'arcface': face_runtime.renderer.arcface.get_providers()}
        if any('CUDAExecutionProvider' not in providers for providers in report['active_ort_providers'].values()):
            raise RuntimeError('Actual ONNX CUDA session unavailable; refusing CPU fallback benchmark')
        instrumentation = Instrumentation(face_runtime)
        test_image = canvas(source, 1, 640, 360)
        _, test_jpeg = cv2.imencode('.jpg', test_image)
        async with VideoProcessorClient(f'127.0.0.1:{bundle.bound_port}') as client:
            async for result in client.process_video([VideoFrame(test_jpeg.tobytes(), 1, 1, anonymization_mode='face_swap')],
                                                     session_id='bench-loopback-swap-smoke', timeout=60):
                report['loopback_swap_smoke'] = {'statuses': [f.anonymization.status for f in result.response.faces],
                                               'output_jpeg_bytes': len(result.processed_jpeg)}
        for faces, width, height, sessions in [(1, 640, 360, 1), (1, 1280, 720, 1), (1, 1920, 1080, 1),
                                              (2, 1280, 720, 1), (4, 1280, 720, 1), (1, 1280, 720, 4), (1, 1280, 720, 8)]:
            key = f'faces{faces}-{width}x{height}-sessions{sessions}'
            print(f'RUN {key}', flush=True)
            report['configs'][key] = await run_config(bundle, source, faces, width, height, sessions,
                                                     args.warmup, args.frames, instrumentation)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        gate_samples, inflight = await run_stream(f'127.0.0.1:{bundle.bound_port}', [test_jpeg.tobytes()] * 120,
                                                 120, 'bench-standard-b1-gate')
        report['standard_blur_loopback_gate'] = summarize(gate_samples, inflight, args.source,
                                                        f'127.0.0.1:{bundle.bound_port}')
        print('STANDARD_BLUR_GATE', report['standard_blur_loopback_gate']['passed'], flush=True)
    finally:
        await bundle.set_serving(False)
        await bundle.server.stop(0)
        await bundle.servicer.close()
        runtime.close()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--frames', type=int, default=120)
    asyncio.run(main(parser.parse_args()))
