"""Quantization correctness without external files or invented EEG evidence."""
import unittest
import torch
from topobrain.tokenizer import FrozenTokenizer

class QuantizationTests(unittest.TestCase):
    def test_nearest_uses_normalized_features_and_actual_book_norm(self):
        z = torch.tensor([[[3.,4.],[2.,0.],[-2.,1.]]])
        book = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[-0.8,0.4]])
        expected = ((torch.nn.functional.normalize(z,dim=-1)[...,None,:]-book)**2).sum(-1).argmin(-1)
        self.assertTrue(torch.equal(FrozenTokenizer.nearest(z,book),expected))

    def test_quantization_is_scale_invariant_for_positive_input_scale(self):
        z=torch.tensor([[2.,1.],[-1.,3.]])
        book=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        self.assertTrue(torch.equal(FrozenTokenizer.nearest(z,book),FrozenTokenizer.nearest(z*100,book)))
