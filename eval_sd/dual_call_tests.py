from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from extra.models.unet import CrossAttention # type: ignore
import numpy as np

def test_1():
  a, b, c, d = [Tensor.rand(1, 32) for _ in range(4)]

  x1, x2 = (Tensor.cat(a,b) + Tensor.cat(c,d)).chunk(2)
  y1, y2 = a + c, b + d

  np.testing.assert_allclose(x1.numpy(), y1.numpy())
  np.testing.assert_allclose(x2.numpy(), y2.numpy())

def test_2():
  ca = CrossAttention(32, 64, 4, 8)
  for p in get_parameters(ca):
    p.replace(Tensor.rand(*p.shape)).realize()
  
  a, b = [Tensor.rand(1, 32) for _ in range(2)]
  c, d = [Tensor.rand(1, 64) for _ in range(2)]

  x1, x2 = ca(Tensor.cat(a, b), Tensor.cat(c, d)).chunk(2)
  y1, y2 = ca(a, c), ca(b, d)

  np.testing.assert_allclose(x1.numpy(), y1.numpy())
  np.testing.assert_allclose(x2.numpy(), y2.numpy())

if __name__ == "__main__":
  test_1()
  test_2()
