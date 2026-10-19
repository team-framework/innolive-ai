# MobileFaceSwap inference definitions

`model.py` and `arcface.py` are unmodified copies from
[Seanseattle/MobileFaceSwap](https://github.com/Seanseattle/MobileFaceSwap/tree/29552474c2f621c27818ff7fc32447e1a60a96fb/models),
commit `29552474c2f621c27818ff7fc32447e1a60a96fb` (Apache-2.0; see LICENSE).
They are imported lazily only when the MobileFaceSwap renderer is selected.

Preprocessing, SCRFD decoding, tracked-face association and mask composition live
in `service/mobilefaceswap.py`. Checkpoints remain user-supplied local assets.
