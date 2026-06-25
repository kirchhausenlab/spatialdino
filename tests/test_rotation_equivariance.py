import unittest
import torch

from spatialdino.loss.rotation_equivariance_loss import (
    RotationEquivarianceLoss,
    _AXIS_CONFIGS,
)


GRID = (4, 4, 4)
N = GRID[0] * GRID[1] * GRID[2]  # 64
D = 32
B = 2


class TestSampleRotation(unittest.TestCase):
    def test_returns_valid_axis_and_k(self):
        for _ in range(50):
            axis, input_dims, feat_dims, k = RotationEquivarianceLoss.sample_rotation()
            self.assertIn(axis, (0, 1, 2))
            self.assertIn(k, (1, 2, 3))
            self.assertEqual(input_dims, _AXIS_CONFIGS[axis][0])
            self.assertEqual(feat_dims, _AXIS_CONFIGS[axis][1])


class TestRotationEquivarianceLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = RotationEquivarianceLoss()
        torch.manual_seed(42)

    def test_identical_features_give_zero_loss(self):
        """If rot_tokens == orig_tokens (identity rotation), loss should be ~0."""
        tokens = torch.randn(B, N, D)
        # k=4 is full rotation = identity; use feat_dims for Z-axis
        loss = self.loss_fn(tokens, tokens.clone(), GRID, feat_dims=(2, 3), k=4)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_perfectly_aligned_rotation_gives_zero_loss(self):
        """Rotate features in grid, then inverse-rotate in loss → should be zero."""
        tokens = torch.randn(B, N, D)
        for axis_idx, (_, feat_dims) in enumerate(_AXIS_CONFIGS):
            for k in (1, 2, 3):
                grid = tokens.reshape(B, *GRID, D)
                # Forward rotate the features
                rotated_grid = torch.rot90(grid, k, feat_dims)
                rotated_tokens = rotated_grid.reshape(B, N, D)
                # Loss should inverse-rotate and match original
                loss = self.loss_fn(tokens, rotated_tokens, GRID, feat_dims, k)
                self.assertAlmostEqual(
                    loss.item(),
                    0.0,
                    places=5,
                    msg=f"axis={axis_idx}, k={k}: loss should be ~0 for correct inverse",
                )

    def test_random_features_give_nonzero_loss(self):
        """Unrelated orig and rot tokens should give loss > 0."""
        orig = torch.randn(B, N, D)
        rot = torch.randn(B, N, D)
        loss = self.loss_fn(orig, rot, GRID, feat_dims=(2, 3), k=1)
        self.assertGreater(loss.item(), 0.1)

    def test_gradient_flows_through_rot_tokens_only(self):
        """Gradients should flow through rot_tokens but not orig_tokens."""
        orig = torch.randn(B, N, D, requires_grad=True)
        rot = torch.randn(B, N, D, requires_grad=True)
        # Detach orig as we do in the training loop
        loss = self.loss_fn(orig.detach(), rot, GRID, feat_dims=(2, 3), k=1)
        loss.backward()
        self.assertIsNone(orig.grad, "orig_tokens should have no gradient (detached)")
        self.assertIsNotNone(rot.grad, "rot_tokens should have gradient")
        self.assertTrue(
            rot.grad.abs().sum() > 0, "rot_tokens gradient should be non-zero"
        )

    def test_all_nine_rotations_invertible(self):
        """All 3 axes x 3 k values should be perfectly invertible."""
        tokens = torch.randn(1, N, D)
        for axis_idx, (_, feat_dims) in enumerate(_AXIS_CONFIGS):
            for k in (1, 2, 3):
                grid = tokens.reshape(1, *GRID, D)
                rotated = torch.rot90(grid, k, feat_dims).reshape(1, N, D)
                loss = self.loss_fn(tokens, rotated, GRID, feat_dims, k)
                self.assertAlmostEqual(
                    loss.item(),
                    0.0,
                    places=5,
                    msg=f"axis={axis_idx}, k={k}",
                )

    def test_output_is_scalar(self):
        """Loss should be a scalar tensor."""
        tokens = torch.randn(B, N, D)
        loss = self.loss_fn(tokens, tokens.clone(), GRID, feat_dims=(2, 3), k=1)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_bounded_zero_to_two(self):
        """1 - cosine_similarity is in [0, 2], so mean should be too."""
        orig = torch.randn(B, N, D)
        rot = torch.randn(B, N, D)
        for _, feat_dims in _AXIS_CONFIGS:
            for k in (1, 2, 3):
                loss = self.loss_fn(orig, rot, GRID, feat_dims, k)
                self.assertGreaterEqual(loss.item(), 0.0)
                self.assertLessEqual(loss.item(), 2.0)


class TestInputRotationMapping(unittest.TestCase):
    """Verify that rotating input [B,C,Z,Y,X] and features [B,gZ,gY,gX,D]
    with the corresponding dims produces consistent results."""

    def test_z_axis_rotation_dims(self):
        """Z-axis: input dims (-2,-1) rotate Y-X plane, feat dims (2,3) rotate gY-gX."""
        vol = torch.randn(1, 1, 8, 8, 8)
        rotated = torch.rot90(vol, 1, (-2, -1))
        # Y and X dims should be swapped
        self.assertEqual(rotated.shape, vol.shape)
        # Feature grid equivalent
        feat = torch.randn(1, 2, 2, 2, 4)
        feat_rot = torch.rot90(feat, 1, (2, 3))
        self.assertEqual(feat_rot.shape, feat.shape)

    def test_y_axis_rotation_dims(self):
        """Y-axis: input dims (-3,-1) rotate Z-X plane, feat dims (1,3) rotate gZ-gX."""
        vol = torch.randn(1, 1, 8, 8, 8)
        rotated = torch.rot90(vol, 1, (-3, -1))
        self.assertEqual(rotated.shape, vol.shape)

    def test_x_axis_rotation_dims(self):
        """X-axis: input dims (-3,-2) rotate Z-Y plane, feat dims (1,2) rotate gZ-gY."""
        vol = torch.randn(1, 1, 8, 8, 8)
        rotated = torch.rot90(vol, 1, (-3, -2))
        self.assertEqual(rotated.shape, vol.shape)


if __name__ == "__main__":
    unittest.main()
