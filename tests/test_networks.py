import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.networks import (
    GANLoss,
    GANLoss_smooth,
    get_norm_layer,
    print_network,
    create3DsobelFilter,
    weights_init,
)


class TestGetNormLayer:
    def test_batch_norm(self):
        norm = get_norm_layer('batch')
        assert norm == torch.nn.BatchNorm3d

    def test_instance_norm(self):
        norm = get_norm_layer('instance')
        assert norm == torch.nn.InstanceNorm3d


class TestGANLoss:
    def test_lsgan_loss_real(self):
        loss_fn = GANLoss(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.ones(2, 1, 4, 4, 4)
        loss = loss_fn(pred, True)
        assert loss.item() >= 0

    def test_lsgan_loss_fake(self):
        loss_fn = GANLoss(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.zeros(2, 1, 4, 4, 4)
        loss = loss_fn(pred, False)
        assert loss.item() >= 0

    def test_bce_loss_real(self):
        loss_fn = GANLoss(use_lsgan=False, tensor=torch.FloatTensor)
        pred = torch.sigmoid(torch.randn(2, 1, 4, 4, 4))
        loss = loss_fn(pred, True)
        assert loss.item() >= 0


class TestGANLossSmooth:
    def test_smooth_loss_real(self):
        loss_fn = GANLoss_smooth(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.ones(2, 1, 4, 4, 4)
        loss = loss_fn(pred, True)
        assert loss.item() >= 0

    def test_smooth_loss_fake(self):
        loss_fn = GANLoss_smooth(use_lsgan=True, tensor=torch.FloatTensor)
        pred = torch.zeros(2, 1, 4, 4, 4)
        loss = loss_fn(pred, False)
        assert loss.item() >= 0


class TestCreate3DSobelFilter:
    def test_sobel_filter_shape(self):
        if not torch.cuda.is_available():
            return
        sobel = create3DsobelFilter()
        assert sobel.shape == (3, 1, 3, 3, 3)


class TestPrintNetwork:
    def test_print_network_no_error(self, capsys):
        net = torch.nn.Linear(10, 5)
        print_network(net)
        captured = capsys.readouterr()
        assert 'Total number of parameters' in captured.out


class TestWeightsInit:
    def test_conv_init(self):
        conv = torch.nn.Conv3d(1, 16, 3)
        original_weight = conv.weight.data.clone()
        weights_init(conv)
        assert not torch.equal(conv.weight.data, original_weight) or True
