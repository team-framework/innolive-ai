# Runtime model files

The TensorRT, AdaFace, and YuNet binaries are deployment artifacts and are not
tracked by Git. The server expects these defaults:

| File | Source | SHA-256 used for smoke testing |
| --- | --- | --- |
| `adaface_ir18_casia.ckpt` | [Official AdaFace IR-18 CASIA checkpoint](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view) | `2be3042e2266b745824ba95dad4d09a0e4d9f67bd08fb0b79f5b7a605c48fd30` |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

Download YuNet with:

```bash
curl -L --fail \
  -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The AdaFace checkpoint is hosted on Google Drive. Download it from the linked
official checkpoint page and save it as `models/adaface_ir18_casia.ckpt`.
Alternative paths can be supplied with `--adaface-weights` and
`--adaface-detector`.

The server stores only normalized embeddings in memory. Enrollment images and
aligned face crops are not written to this directory.
