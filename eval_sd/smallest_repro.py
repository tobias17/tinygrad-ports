from tinygrad import Tensor, TinyJit
import numpy as np

@TinyJit
def run(x:Tensor, y:Tensor):
  return (x * y).realize()

Tensor.manual_seed(1234)
sigmas = Tensor.rand(500).realize()

def sigma_to_idx(s:Tensor) -> Tensor:
  dists = s - sigmas
  return dists.abs().argmin(axis=0).view(*s.shape)

def make_call() -> Tensor:
  Tensor.manual_seed(1234)
  sigma = Tensor.rand(1)
  sigma = sigmas[sigma_to_idx(sigma)]
  c_out = -sigma

  x = Tensor.rand(64)
  return run((x * Tensor.rand(64)).realize(), Tensor.rand(64).realize()) * c_out

def create_block(realize:bool=True):
  values = [] # type: ignore
  for _ in range(10):
    o1, o2 = make_call(), make_call()
    if realize:
      o1.realize()
      o2.realize()
    h = o1 + 8.0*(o2 - o1)
    values.append(h.numpy())
  return values

for v1, v2 in zip(create_block(True), create_block(False)):
  np.testing.assert_allclose(v1, v2)
