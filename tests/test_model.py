import torch
from model.simple_model import FFN_Network

def test_relu_forward_shape():
    model = FFN_Network(1, 2, 16)
    x = torch.randn(32, 1)
    y = model(x)
    assert y.shape == (32, 1)

def test_relu_forward_method():
    model = FFN_Network(1, 1, 8)
    x = torch.randn(4, 1)
    y = model.forward(x)
    assert y.shape == (4, 1)
