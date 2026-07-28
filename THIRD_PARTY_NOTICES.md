# Third-party notices

This file records third-party software, model artifacts, and training-data
provenance that are bundled with or material to this repository. It does not
replace the applicable upstream license terms. The license in `LICENSE` covers
only rights that the project copyright holders are able to grant; it does not
relicense third-party material.

## Ultralytics YOLO and BoT-SORT

The runtime pins `ultralytics==8.4.104` and directly imports Ultralytics YOLO
and BoT-SORT components. The tracked `models/best.pt` YOLO26n-seg checkpoint
also embeds this license notice in its metadata:

> AGPL-3.0 (https://ultralytics.com/license)

[Ultralytics](https://github.com/ultralytics/ultralytics) publishes its
software and YOLO-trained models under GNU Affero General Public License v3.0
by default, with an Enterprise License offered as an alternative. This
repository follows the AGPL-3.0 path; the complete license text is in
`LICENSE`. An Enterprise License holder must follow the terms of that separate
agreement for the Ultralytics components.

The Python dependency is installed separately and is not vendored in this
repository. Exporting the checkpoint to TensorRT does not by itself remove the
licenses that apply to the source checkpoint or exporter.

## Training data and external annotation tools

The tracked `models/best.pt` checkpoint was trained using WIDER FACE images
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
