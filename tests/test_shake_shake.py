"""CPU checks for independent gradient mixing, spatial skip, and inference."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.shake_shake import ShakeShakeBlock, ShakeShakeFunction, ShakeShakeNet, ShiftedProjection


class ShakeShakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_budget_shape_and_both_branches_receive_gradients(self):
        model = ShakeShakeNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4_561_354)
        self.assertLess(sum(p.numel() for p in model.parameters()), 5_000_000)
        self.assertEqual(len(model.stages), 12)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        nn.functional.cross_entropy(logits, torch.tensor([1, 7])).backward()
        for block in model.stages:
            for branch in (block.branch1, block.branch2):
                for index in (1, 4):
                    self.assertGreater(branch[index].weight.grad.abs().max().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_forward_alpha_backward_beta_are_distinct_and_per_image(self):
        first = torch.full((2, 3, 2, 2), 2.0, requires_grad=True)
        second = torch.full_like(first, 10.0, requires_grad=True)
        alpha = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1)
        beta = torch.tensor([0.625, 0.125]).view(2, 1, 1, 1)
        output = ShakeShakeFunction.apply(first, second, alpha, beta)
        self.assertTrue(torch.equal(output[:, 0, 0, 0], torch.tensor([8.0, 4.0])))
        # Nonuniform upstream gradients verify correct scaling (including AMP's
        # external GradScaler multiplier), rather than merely gradient existence.
        upstream = torch.arange(output.numel(), dtype=torch.float32).view_as(output) * 128
        output.backward(upstream)
        self.assertTrue(torch.equal(first.grad, upstream * beta))
        self.assertTrue(torch.equal(second.grad, upstream * (1 - beta)))

    def test_coefficients_do_not_promote_amp_activations(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                first = torch.ones((2, 2, 2, 2), dtype=dtype, requires_grad=True)
                second = torch.zeros_like(first, requires_grad=True)
                alpha = torch.full((2, 1, 1, 1), 0.25, dtype=torch.float32)
                beta = torch.full_like(alpha, 0.75)
                output = ShakeShakeFunction.apply(first, second, alpha, beta)
                self.assertEqual(output.dtype, dtype)
                output.sum().backward()
                self.assertEqual(first.grad.dtype, dtype)
                self.assertTrue(torch.equal(first.grad, torch.full_like(first, 0.75)))
                self.assertTrue(torch.equal(second.grad, torch.full_like(second, 0.25)))

    def test_train_block_samples_two_independent_per_image_coefficients(self):
        block = ShakeShakeBlock(4, 4).train()
        alpha, beta = torch.full((3, 1, 1, 1), .2), torch.full((3, 1, 1, 1), .8)
        with patch("src.shake_shake.torch.rand", side_effect=[alpha, beta]) as draw:
            block(torch.randn(3, 4, 8, 8)).sum().backward()
        self.assertEqual(draw.call_count, 2)
        self.assertTrue(all(call.args[0] == (3, 1, 1, 1) for call in draw.call_args_list))

    def test_projection_matches_author_right_down_shift(self):
        projection = ShiftedProjection(1, 2, 2)
        projection.bn = nn.Identity()
        with torch.no_grad():
            projection.path1.weight.fill_(1)
            projection.path2.weight.fill_(1)
        x = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
        result = projection(x)
        self.assertTrue(torch.equal(result[0, 0], torch.tensor([[1., 3.], [9., 11.]])))
        self.assertTrue(torch.equal(result[0, 1], torch.tensor([[0., 0.], [0., 6.]])))

    def test_eval_is_deterministic_mirror_is_explicit_and_checkpoint_roundtrips(self):
        model = ShakeShakeNet(width=8, blocks=[1, 1, 1]).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad(), patch("src.shake_shake.torch.rand", side_effect=AssertionError("eval RNG")):
            standard = model.classify(x)
            self.assertTrue(torch.equal(standard, model.classify(x)))
            expected = (standard + model.classify(x.flip(-1))) * 0.5
            self.assertTrue(torch.equal(model(x), expected))
            restored = ShakeShakeNet(width=8, blocks=[1, 1, 1]).eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            self.assertTrue(torch.equal(restored(x), expected))

    def test_cpu_autocast_full_backward(self):
        model = ShakeShakeNet(width=8, blocks=[1, 1, 1]).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(torch.randn(2, 3, 32, 32))
            loss = nn.functional.cross_entropy(logits, torch.tensor([2, 5]))
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_invalid_configuration(self):
        for kwargs in ({"blocks": [1, 1]}, {"blocks": [1, 0, 1]}, {"blocks": [True, 1, 1]},
                       {"width": 39}, {"width": 0}, {"width": True},
                       {"num_classes": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ShakeShakeNet(**kwargs)


if __name__ == "__main__":
    unittest.main()
