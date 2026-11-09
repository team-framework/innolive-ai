# Third-party notices

This file records third-party software, model artifacts, and training-data
provenance that are bundled with or material to this repository. It does not
replace the applicable upstream license terms. The license in `LICENSE` covers
only rights that the project copyright holders are able to grant; it does not
relicense third-party material.

## RF-DETR

The runtime pins `rfdetr==1.11.2` from [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).
The supplied RFDETRSegSmall checkpoint records `license=Apache-2.0` and is installed
separately as `models/rfdetr_seg_s.pth`. This records the software/checkpoint metadata;
it does not establish the rights to the training images. The serving adapter and
TensorRT exporter use the upstream model and preprocessing contract.

## Independent BoT-SORT

`service/_vendor/botsort/kalman_filter.py` is from
[NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT), MIT license (2022 Nir Aharon).
The full license is retained in `service/_vendor/botsort/LICENSE`.
`service/botsort.py` implements the motion/IoU association and camera compensation
for the existing no-ReID serving configuration, with stream-local IDs, class-safe
association, bounded removed history, and empty-frame handling. It does not import
or vendor Ultralytics code.

## Historical detector

Earlier revisions used Ultralytics 8.4.104 and `models/best.pt` (YOLO26n-seg).
Their original AGPL-3.0 notices remain applicable to those historical revisions.
The current runtime, exporter and lab have removed that package and artifact.
The repository's own `LICENSE` is unchanged by the dependency migration.

## MobileFaceSwap

`service/_vendor/mobilefaceswap/model.py` and `arcface.py` are unmodified copies
from [Seanseattle/MobileFaceSwap](https://github.com/Seanseattle/MobileFaceSwap),
commit `29552474c2f621c27818ff7fc32447e1a60a96fb`. Its Apache-2.0 license is
included in `service/_vendor/mobilefaceswap/LICENSE`.
`service/mobilefaceswap_torch.py` adapts the conditioned synthesis network to
PyTorch, with strict tensor conversion, BatchNorm fusion and device caching.
Model checkpoints, SCRFD weights and preset images are installed separately;
this code license does not grant rights in those artifacts.

## Training data and external annotation tools

The historical `models/best.pt` checkpoint was trained using WIDER FACE images
and bounding-box annotations after masks were generated in an external data
pipeline with SAM 3.1 and reviewed with CVAT. The WIDER FACE dataset, generated
masks, SAM code, and SAM weights are not included in this repository.

### WIDER FACE

The [official WIDER FACE project page](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
labels the dataset as Creative Commons BY-NC-ND, without specifying a version
on that page. Those terms include attribution, non-commercial-use, and
no-derivatives restrictions.

Dataset citation: Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang,
“WIDER FACE: A Face Detection Benchmark,” CVPR 2016.

The repository's AGPL-3.0 license does not grant rights in the WIDER FACE
images or annotations, and it should not be treated as confirmation that the
WIDER-FACE-trained checkpoint is cleared for commercial use. Before commercial
distribution or deployment of `models/best.pt` or an engine exported from it,
obtain appropriate permission or legal review, or replace it with a checkpoint
trained only on data cleared for that use.

### SAM 3.1

SAM 3.1 is a set of updated checkpoints for Meta's SAM 3 project. Meta
distributes the SAM 3 code and checkpoints under the custom
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE), not
under this repository's AGPL-3.0 license. Because no SAM Materials are
distributed here and SAM was used only as an external annotation tool, its
license is not added as a license for this repository. Anyone obtaining or
using SAM code or weights must comply with the upstream SAM License separately.

## AdaFace

`service/adaface_backbones.py` contains inference-only adaptations of the IR
backbones from [AdaFace](https://github.com/mk-minchul/AdaFace), commit
`c60eaa786a42c03444f3df7096dbaf9d57ae010d`, and the fixed ViT-Base KP-RPE
backbone from [CVLFace](https://github.com/mk-minchul/CVLface), revision
`daefd5012d369588bd214fbaf4cc6b1d286e7066` of the published model bundle.

MIT License

Copyright (c) 2022 Minchul Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## YuNet

The optional `face_detection_yunet_2023mar.onnx` runtime artifact is published
by the [OpenCV Zoo YuNet project](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
