"""Synchronized, prepared-input model timings and same-input TRT output checks."""
from __future__ import annotations
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import tensorrt as trt

ROOT = Path('/home/gorani/apps/new-inno-live-server-python')
SCRATCH = Path('/tmp/innolive-bench-20261010-IbBR9b')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))
from benchmark_current_feature import distribution
from service.inswapper import InSwapperRenderer, alignment
from service.face_presets import PresetCatalog
from service.face_metadata import MetadataExtractor, crop_metadata_face, FaceAttributes
from service.runtime import sha256_file
from experiments.trt_swap_client.app import TensorRtInSwapperGenerator, _require_current_swapper_engine


def cuda_bench(fn, count=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(count):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        gpu.append(start.elapsed_time(end))
    return {'gpu_event_ms': distribution(gpu), 'synchronized_wall_ms': distribution(wall),
            'warmup_iterations': warmup, 'measured_iterations': count}


def wall_bench(fn, count=200, warmup=30):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {'synchronized_wall_ms': distribution(values), 'warmup_iterations': warmup, 'measured_iterations': count}


def compare(reference, candidate):
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {'shape': list(candidate.shape), 'finite': bool(np.isfinite(candidate).all()),
            'mean_absolute_error_raw_0_1': float(error.mean()), 'max_absolute_error_raw_0_1': float(error.max()),
            'mean_absolute_error_times_255': float(error.mean() * 255)}


def main():
    report = {'method': 'Prepared inputs; no network, image decode/encode, tracking, alignment, or paste-back in forward timings',
              'note': 'ORT CUDA I/O binding uses synchronized wall time; TensorRT and PyTorch use CUDA events on their execution stream',
              'components': {}, 'same_input_output_checks': {}, 'artifacts': {}}
    renderer = InSwapperRenderer(ROOT / 'models/face_swap/inswapper_128.onnx',
                                Path.home() / '.insightface/models/buffalo_l/w600k_r50.onnx',
                                ROOT / 'models/face_detection_yunet_2023mar.onnx', 'cuda')
    catalog = PresetCatalog(SCRATCH / 'face_presets.json')
    preset = catalog.match(FaceAttributes('female', '20s', 'off', 'none'), 1)
    image = cv2.imread(str(preset.image))
    landmarks = renderer.detect(image)[0][4:14].reshape(5, 2)
    crop = cv2.warpAffine(image, alignment(landmarks, 128), (128, 128))
    blob = cv2.dnn.blobFromImage(crop, 1 / 255, (128, 128), swapRB=True)
    latent = renderer.source_latent(preset)
    feed = {renderer.inputs[0].name: blob, renderer.inputs[1].name: latent}
    reference = renderer.swapper.run(None, feed)[0]
    report['components']['inswapper_onnx_cuda_host_io'] = wall_bench(lambda: renderer.swapper.run(None, feed))
    io = renderer.swapper.io_binding()
    gpu_blob, gpu_latent = torch.from_numpy(blob).cuda(), torch.from_numpy(latent).cuda()
    gpu_output = torch.empty(reference.shape, dtype=torch.float32, device='cuda')
    io.bind_input(renderer.inputs[0].name, 'cuda', 0, np.float32, tuple(gpu_blob.shape), gpu_blob.data_ptr())
    io.bind_input(renderer.inputs[1].name, 'cuda', 0, np.float32, tuple(gpu_latent.shape), gpu_latent.data_ptr())
    io.bind_output(renderer.swapper.get_outputs()[0].name, 'cuda', 0, np.float32, tuple(gpu_output.shape), gpu_output.data_ptr())
    torch.cuda.synchronize()
    report['components']['inswapper_onnx_cuda_gpu_io'] = wall_bench(lambda: renderer.swapper.run_with_iobinding(io))
    report['same_input_output_checks']['onnx_gpu_io_vs_host_io'] = compare(reference, gpu_output.cpu().numpy())
    del io, gpu_blob, gpu_latent, gpu_output

    for name in ['fp32', 'mixed']:
        engine = ROOT / f'models/face_swap/inswapper_128_trt11_{name}.engine'
        _require_current_swapper_engine(engine, ROOT / 'models/face_swap/inswapper_128.onnx')
        generator = TensorRtInSwapperGenerator(SimpleNamespace(input_size=(128, 128)), engine, '0')
        output = generator.forward(blob, latent)
        report['same_input_output_checks'][f'trt_{name}_vs_onnx_cuda'] = compare(reference, output)
        report['components'][f'inswapper_trt_{name}_host_io'] = wall_bench(lambda: generator.forward(blob, latent))
        stream = torch.cuda.current_stream(0)

        def forward_gpu():
            if not generator.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TRT enqueue failed')
        report['components'][f'inswapper_trt_{name}_gpu_forward'] = cuda_bench(forward_gpu)
        report['artifacts'][str(engine)] = {'sha256': sha256_file(engine),
                    'manifest': json.loads(engine.with_suffix(engine.suffix + '.json').read_text()),
                    'bindings': generator.provider_summary()}
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    engine_path = SCRATCH / 'best_b1.engine'
    data = engine_path.read_bytes()
    metadata_size = int.from_bytes(data[:4], 'little')
    json.loads(data[4:4 + metadata_size])
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = trt_runtime.deserialize_cuda_engine(data[4 + metadata_size:])
    assert engine is not None
    context = engine.create_execution_context()
    tensors = {}
    type_map = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}
    bindings = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape, dtype = tuple(engine.get_tensor_shape(name)), engine.get_tensor_dtype(name)
        tensors[name] = torch.zeros(shape, dtype=type_map[dtype], device='cuda')
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            rgb = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
            prepared_yolo = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to('cuda').float().div_(255)
            tensors[name].copy_(prepared_yolo)
        context.set_tensor_address(name, tensors[name].data_ptr())
        bindings.append({'name': name, 'shape': list(shape), 'dtype': str(dtype)})
    stream = torch.cuda.current_stream(0)
    report['components']['yolo_b1_640_trt_fp16_gpu_forward'] = cuda_bench(lambda: context.execute_async_v3(stream.cuda_stream))
    report['artifacts'][str(engine_path)] = {'sha256': sha256_file(engine_path), 'bindings': bindings}
    del context, engine, trt_runtime, tensors, data
    gc.collect()
    torch.cuda.empty_cache()

    extractor = MetadataExtractor(ROOT / 'models/face_metadata_extracter.pt', 'cuda:0')
    face = renderer.detect(image)[0]
    bbox = np.array([face[0], face[1], face[0] + face[2], face[1] + face[3]])
    metadata_crop = crop_metadata_face(image, bbox)
    with torch.inference_mode():
        for batch in [1, 2, 4]:
            prepared = extractor.prepare_inputs([metadata_crop] * batch)
            report['components'][f'metadata_cuda_fp32_batch{batch}_gpu_forward'] = cuda_bench(lambda: extractor.model(prepared))
    print(json.dumps(report, indent=2), flush=True)
    (SCRATCH / 'model-components.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
