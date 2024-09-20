from tinygrad import Tensor, dtypes, nn
from tinygrad.nn.state import get_parameters
from extra.models.unet import CrossAttention, BasicTransformerBlock # type: ignore
from typing import List, Callable
import numpy as np
import unittest

def make_input(*shape:int, count:int=1, dtype=dtypes.float16) -> List[Tensor]:
  return [Tensor.rand(*shape, dtype=dtype) for _ in range(count)]

def randomize_weights(model):
  for w in get_parameters(model):
    w.replace(Tensor.rand(*w.shape)).realize()
  return model

def make_calls(func:Callable[[Tensor,Tensor],Tensor], b1_i1:Tensor, b2_i1:Tensor, b1_i2:Tensor, b2_i2:Tensor):
  x1, x2 = func(Tensor.cat(b1_i1, b2_i1), Tensor.cat(b1_i2, b2_i2)).chunk(2)
  y1, y2 = func(b1_i1, b1_i2), func(b2_i1, b2_i2)

  np.testing.assert_allclose(x1.numpy(), y1.numpy(), atol=1e-6, rtol=1e-3)
  np.testing.assert_allclose(x2.numpy(), y2.numpy(), atol=1e-6, rtol=1e-3)

class Dual_Call_Tests(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(42)

  def test_add(self):
    make_calls(Tensor.add, *make_input(1, 32, count=4))

  def test_linear(self):
    l = randomize_weights(nn.Linear(32, 8))
    make_calls((lambda x,y: l(x)+l(y)), *make_input(1, 32, count=4))

  def test_cross_attention(self):
    ca = randomize_weights(CrossAttention(32, 64, 4, 8))
    make_calls(ca, *make_input(1, 32, count=2), *make_input(1, 64, count=2))

  def test_self_attention(self):
    ca = randomize_weights(CrossAttention(32, 32, 4, 8))
    make_calls((lambda x,y: ca(x)+ca(y)), *make_input(1, 32, count=4))

  def test_basic_transformer_block(self):
    btb = randomize_weights(BasicTransformerBlock(32, 64, 4, 8))
    make_calls(btb, *make_input(1, 4, 32, count=2), *make_input(1, 4, 64, count=2))

if __name__ == "__main__":
  unittest.main()
