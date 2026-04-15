import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util.util import (
    tensor2im,
    tensor2im_co,
    tensor2im_sa,
    tensor2array,
    tensor2array_labels,
)


class TestTensor2Im:
    def _make_tensor(self, shape=(1, 1, 128, 128, 128)):
        return torch.randn(*shape)

    def test_tensor2im_output_shape(self):
        t = self._make_tensor()
        result = tensor2im(t)
        assert result.shape == (6, 128, 128)

    def test_tensor2im_dtype(self):
        t = self._make_tensor()
        result = tensor2im(t)
        assert result.dtype == np.uint8

    def test_tensor2im_value_range(self):
        t = torch.ones(1, 1, 128, 128, 128)
        result = tensor2im(t)
        assert result.max() == 255
        assert result.min() >= 0

    def test_tensor2im_co_output_shape(self):
        t = self._make_tensor()
        result = tensor2im_co(t)
        assert result.shape == (6, 128, 128)

    def test_tensor2im_sa_output_shape(self):
        t = self._make_tensor()
        result = tensor2im_sa(t)
        assert result.shape == (6, 128, 128)

    def test_tensor2im_custom_dtype(self):
        t = self._make_tensor()
        result = tensor2im(t, imtype=np.float32)
        assert result.dtype == np.float32


class TestTensor2Array:
    def test_tensor2array_output_shape(self):
        t = torch.randn(1, 1, 64, 64, 64)
        result = tensor2array(t)
        assert result.shape == (64, 64, 64)

    def test_tensor2array_preserves_values(self):
        data = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)
        t = torch.from_numpy(data)
        result = tensor2array(t)
        np.testing.assert_array_almost_equal(result, data[0, 0, :, :, :])

    def test_tensor2array_labels_output_shape(self):
        t = torch.randn(1, 4, 64, 64, 64)
        result = tensor2array_labels(t)
        assert result.shape == (4, 64, 64, 64)

    def test_tensor2array_labels_preserves_values(self):
        data = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        t = torch.from_numpy(data)
        result = tensor2array_labels(t)
        np.testing.assert_array_almost_equal(result, data[0, :, :, :, :])
