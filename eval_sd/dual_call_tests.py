from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from extra.models.unet import CrossAttention # type: ignore
from typing import List, Callable
import numpy as np
import unittest

def make_input(*shape:int, count:int=1) -> List[Tensor]:
  return [Tensor.rand(*shape) for _ in range(count)]

def make_calls(func:Callable[[Tensor,Tensor],Tensor], b1_i1:Tensor, b2_i1:Tensor, b1_i2:Tensor, b2_i2:Tensor):
  x1, x2 = func(Tensor.cat(b1_i1, b2_i1), Tensor.cat(b1_i2, b2_i2)).chunk(2)
  y1, y2 = func(b1_i1, b1_i2), func(b2_i1, b2_i2)

  np.testing.assert_allclose(x1.numpy(), y1.numpy())
  np.testing.assert_allclose(x2.numpy(), y2.numpy())

class Dual_Call_Tests(unittest.TestCase):
  def test_1(self):
    make_calls(Tensor.add, *make_input(1, 32, count=4))

  def test_2(self):
    ca = CrossAttention(32, 64, 4, 8)
    for p in get_parameters(ca):
      p.replace(Tensor.rand(*p.shape)).realize()
    
    make_calls(ca, *make_input(1, 32, count=2), *make_input(1, 64, count=2))

if __name__ == "__main__":
  unittest.main()
