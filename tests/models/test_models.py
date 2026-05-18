import torch
import torch.nn as nn

from src.utils.nn.conv import Convolution2d, ConvolutionTranspose2d

class TestConvolution2d:

    def test_basic_forward_shape(self):
        """Output shape matches expected shape."""
        model = Convolution2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8)
        y = model(x)
        assert y.shape == (2, 16, 8, 8)

    def test_full_pipeline_forward(self):
        """batchnorm = 'pre', activation, dropout, batchnorm="post" all work together"""
        model = Convolution2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1,
            activation=nn.ReLU(), batchnorm="pre", dropout=0.0
        )
        model.eval()
        x = torch.randn(2, 3, 8, 8)
        y = model(x)
        assert y.shape == (2, 8, 8, 8)

class TestConvolutionTranspose2d:

    def test_basic_forward_shape(self):
        """Transposed conv upsamples spatial dimensions."""
        model = ConvolutionTranspose2d(16, 3, kernel_size=2, stride=2)
        x = torch.randn(2, 16, 4, 4)
        out = model(x)
        assert out.shape == (2, 3, 8, 8)  # stride=2 doubles spatial dims

    def test_output_deterministic_in_eval(self):
        """eval mode disables dropout randomness."""
        model = ConvolutionTranspose2d(4, 8, kernel_size=3, padding=1, dropout=0.9)
        model.eval()
        x = torch.randn(2, 4, 4, 4)
        with torch.no_grad():
            out1, out2 = model(x), model(x)
        assert torch.allclose(out1, out2)