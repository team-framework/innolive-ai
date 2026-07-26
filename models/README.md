# Runtime model files

The TensorRT, AdaFace, and YuNet binaries are deployment artifacts and are not
tracked by Git. The server expects these defaults:

| File | Source | SHA-256 used for smoke testing |
| --- | --- | --- |
| `adaface_vit_base_kprpe_webface12m.ckpt` | [Official CVLFace AdaFace ViT-Base KP-RPE WebFace12M checkpoint](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m) | `04b4bee1de7cefa9e97900f8449fca906d8afbab2029bd39cc5049d33e927ed9` |
| `adaface_ir18_casia.ckpt` | [Official AdaFace IR-18 CASIA checkpoint](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view) | `2be3042e2266b745824ba95dad4d09a0e4d9f67bd08fb0b79f5b7a605c48fd30` |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

Download YuNet with:

```bash
curl -L --fail \
  -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The default recognizer is pinned to CVLFace revision
`daefd5012d369588bd214fbaf4cc6b1d286e7066`. Download its backbone-only state
dictionary with:

```bash
curl -L --fail \
  -o models/adaface_vit_base_kprpe_webface12m.ckpt \
  https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m/resolve/daefd5012d369588bd214fbaf4cc6b1d286e7066/pretrained_model/model.pt
shasum -a 256 models/adaface_vit_base_kprpe_webface12m.ckpt
```

Alternative architecture and artifact paths can be supplied with
`--adaface-architecture`, `--adaface-weights`, and `--adaface-detector`.

The legacy IR-18 CASIA checkpoint remains supported through
`--adaface-architecture ir18`; it is the smallest backbone and training-set
combination in the official model table. An official
[IR-18 WebFace4M checkpoint](https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view)
is architecture-compatible and can be evaluated through `--adaface-weights`.
Recalibrate the cosine threshold on deployment data before changing the
production default; do not lower it solely to improve apparent recall.

CVLFace source code is MIT-licensed. The publisher does not declare an SPDX
license for the WebFace12M weights and asks users to follow the training data
license. Review those rights separately before commercial deployment.

The server stores only normalized embeddings in memory. Enrollment images and
aligned face crops are not written to this directory.
