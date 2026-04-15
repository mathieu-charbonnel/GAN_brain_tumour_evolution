import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util.image_pool import ImagePool


class TestImagePool:
    def test_pool_size_zero_returns_input(self):
        pool = ImagePool(pool_size=0)
        images = torch.randn(2, 3, 8, 8, 8)
        result = pool.query(images)
        assert torch.equal(result, images)

    def test_pool_fills_up(self):
        pool = ImagePool(pool_size=5)
        images = torch.randn(1, 3, 8, 8, 8)
        for _ in range(5):
            pool.query(images)
        assert pool.num_imgs == 5

    def test_pool_returns_correct_shape(self):
        pool = ImagePool(pool_size=10)
        images = torch.randn(2, 3, 8, 8, 8)
        result = pool.query(images)
        assert result.shape == images.shape

    def test_pool_output_is_tensor(self):
        pool = ImagePool(pool_size=5)
        images = torch.randn(1, 3, 8, 8, 8)
        result = pool.query(images)
        assert isinstance(result, torch.Tensor)

    def test_pool_with_more_queries_than_size(self):
        pool = ImagePool(pool_size=3)
        for _ in range(10):
            images = torch.randn(1, 3, 8, 8, 8)
            result = pool.query(images)
            assert result.shape == (1, 3, 8, 8, 8)
        assert pool.num_imgs == 3
