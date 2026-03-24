from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from spatialdino.models.ssl import SSL


def make_ssl_config():
    return OmegaConf.create(
        {
            "global_crop_size": [8, 8, 8],
            "patch_size": [4, 4, 4],
            "in_chans": 1,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "ffn_layer": "mlp",
            "drop_path_rate": 0.0,
            "drop_path_uniform": False,
            "layerscale": None,
            "interpolate_offset": 0.1,
            "dino_loss_weight": 1.0,
            "ibot_loss_weight": 1.0,
            "n_prototypes": 8,
        }
    )


class SSLTests(unittest.TestCase):
    def test_teacher_starts_from_student_weights_and_is_frozen(self) -> None:
        model = SSL(make_ssl_config())

        student_state = model.student.state_dict()
        teacher_state = model.teacher.state_dict()

        self.assertListEqual(list(student_state.keys()), list(teacher_state.keys()))

        for key in student_state:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(student_state[key], teacher_state[key]))

        self.assertTrue(all(not param.requires_grad for param in model.teacher.parameters()))
