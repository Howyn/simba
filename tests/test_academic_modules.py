"""CPU smoke tests for the optional CTR-GC and FR-Head modules.

Run from the repository root with the experiment's PyTorch environment:
    python -m unittest discover -s tests -p test_academic_modules.py
No GPU, dataset, full Simba model, or pytest is required.
"""

import io
import unittest

import torch

from model.ctr_gc import CTRGCUnit
from model.fr_head import MultiLevelFRHead


class CTRGCTest(unittest.TestCase):
    def test_shape_and_gradients_for_first_and_later_layers(self):
        torch.manual_seed(7)
        adjacency = torch.eye(5).repeat(3, 1, 1)
        for in_channels, out_channels in ((3, 8), (8, 8)):
            with self.subTest(in_channels=in_channels, out_channels=out_channels):
                model = CTRGCUnit(in_channels, out_channels, adjacency)
                x = torch.randn(2, in_channels, 6, 5, requires_grad=True)
                output = model(x)
                self.assertEqual(tuple(output.shape), (2, out_channels, 6, 5))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertIsNotNone(model.topology.grad)
                self.assertTrue(torch.isfinite(model.topology.grad).all())


class FRHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        # Two stride-2 stages map 16 input frames to 4 at level 10.
        self.model = MultiLevelFRHead(
            in_channels=8, num_classes=4, num_person=2, levels=(10,),
            level_weights=(1.0,), hidden_channels=16, num_frames=16,
            num_joints=5,
        )
        self.labels = torch.tensor([0, 1, 2, 3])

    def run_case(self, predictions):
        features = torch.randn(8, 8, 4, 5, requires_grad=True)
        logits = torch.full((4, 4), -2.0)
        logits[torch.arange(4), torch.tensor(predictions)] = 2.0
        loss = self.model([features], logits, self.labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        return loss

    def test_mixed_correct_and_misclassified_samples(self):
        before = self.model.heads[0].spatial_loss.prototypes.clone()
        self.run_case([0, 0, 2, 1])
        after = self.model.heads[0].spatial_loss.prototypes
        # Classes 0 and 2 have true positives; classes 1 and 3 do not.
        self.assertFalse(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[2], after[2]))
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[3], after[3]))

    def test_all_misclassified_has_no_true_positive_update(self):
        before = self.model.heads[0].spatial_loss.prototypes.clone()
        self.run_case([1, 2, 3, 0])
        self.assertTrue(torch.equal(before, self.model.heads[0].spatial_loss.prototypes))

    def test_all_correct_has_no_fn_or_fp(self):
        self.run_case([0, 1, 2, 3])

    def test_prototype_checkpoint_and_eval_does_not_update(self):
        self.run_case([0, 0, 2, 1])
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = MultiLevelFRHead(
            in_channels=8, num_classes=4, num_person=2, levels=(10,),
            level_weights=(1.0,), hidden_channels=16, num_frames=16,
            num_joints=5,
        )
        restored.load_state_dict(torch.load(buffer, map_location="cpu"))
        self.assertTrue(torch.equal(self.model.heads[0].spatial_loss.prototypes,
                                    restored.heads[0].spatial_loss.prototypes))
        self.assertTrue(torch.equal(self.model.heads[0].temporal_loss.prototypes,
                                    restored.heads[0].temporal_loss.prototypes))
        restored.eval()
        before = {key: value.clone() for key, value in restored.state_dict().items()}
        loss = restored([torch.randn(8, 8, 4, 5)], torch.randn(4, 4), self.labels)
        self.assertEqual(loss.item(), 0.0)
        for key, value in restored.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)


if __name__ == "__main__":
    unittest.main()
