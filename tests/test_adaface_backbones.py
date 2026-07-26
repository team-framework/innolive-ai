from __future__ import annotations

import unittest

import torch

from service.adaface_backbones import (
    ADAFACE_ARCHITECTURES,
    _product_bucket_ids,
    build_adaface_backbone,
    checkpoint_backbone_state,
)


class AdaFaceBackboneTests(unittest.TestCase):
    def test_ir_architectures_keep_the_512_dimension_contract(self):
        self.assertEqual(ADAFACE_ARCHITECTURES, ("ir18", "ir50", "ir101", "vit_base_kprpe"))
        model = build_adaface_backbone("ir18").eval()
        with torch.inference_mode():
            embeddings, norms = model(torch.zeros((2, 3, 112, 112)))
        self.assertEqual(tuple(embeddings.shape), (2, 512))
        self.assertEqual(tuple(norms.shape), (2, 1))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2)))

    def test_checkpoint_prefixes_are_architecture_specific(self):
        tensor = torch.ones(1)
        legacy = checkpoint_backbone_state(
            "ir18",
            {"model.body.0.weight": tensor, "head.weight": tensor},
        )
        cvlface = checkpoint_backbone_state(
            "vit_base_kprpe",
            {"net.pos_embed": tensor, "classifier.weight": tensor},
        )
        self.assertEqual(legacy, {"body.0.weight": tensor})
        self.assertEqual(cvlface, {"pos_embed": tensor})

    def test_fixed_kprpe_bucket_map_matches_the_published_shape(self):
        buckets = _product_bucket_ids()
        self.assertEqual(tuple(buckets.shape), (196, 196))
        self.assertEqual(int(buckets.min()), 0)
        self.assertEqual(int(buckets.max()), 48)
        self.assertTrue(torch.equal(buckets.diag(), torch.full((196,), 24)))

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported AdaFace architecture"):
            build_adaface_backbone("unknown")


if __name__ == "__main__":
    unittest.main()
