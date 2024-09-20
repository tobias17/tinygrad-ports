from tinygrad import Tensor
import numpy as np

def test_1():
  a, b, c, d = [Tensor.rand(1, 32) for _ in range(4)]

  x1, x2 = (Tensor.cat(a,b) + Tensor.cat(c,d)).chunk(2)
  y1, y2 = a + c, b + d

  np.testing.assert_allclose(x1.numpy(), y1.numpy())
  np.testing.assert_allclose(x2.numpy(), y2.numpy())

if __name__ == "__main__":
  test_1()
