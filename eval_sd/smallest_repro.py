from tinygrad import Tensor, TinyJit
import numpy as np

@TinyJit
def run(x:Tensor, y:Tensor):
  return (x * y).realize()

Tensor.manual_seed(1234)
sigmas = Tensor.rand(500).realize()

def make_call() -> Tensor:
  Tensor.manual_seed(1234)
  sigma = Tensor.rand(1)
  return run(Tensor.rand(64).realize(), Tensor.rand(64).realize()) * sigma

def create_block(realize:bool=True):
  values = [] # type: ignore
  for _ in range(10):
    o1, o2 = make_call(), make_call()
    if realize:
      o1.realize()
      o2.realize()
    h = o1 + o2
    values.append(h.numpy())
  return values

for v1, v2 in zip(create_block(True), create_block(False)):
  np.testing.assert_allclose(v1, v2)
