import pytest
import torch
import torchmetrics

from src.utils.nn.conv import Convolution2d, ConvolutionTranspose2d

class TestConvolution2d:

    def test_basic_forward_shape(self):
        """Output shape matches expected shape."""
        model = Convolution2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 16, 32, 32)

    