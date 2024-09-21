from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.state import get_parameters
from examples.sdxl import append_dims
from extra.models.unet import UNetModel # type: ignore
import numpy as np

params = {"adm_in_ch": 32, "in_ch": 4, "out_ch": 4, "model_ch": 128, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 16, "transformer_depth": [1, 2, 10], "ctx_dim": 88, "use_linear": True}

model = UNetModel(**params)
for w in get_parameters(model):
  w.replace(Tensor.rand(*w.shape)).realize()

@TinyJit
def run(x, tms, ctx, y):
  return (model(x, tms, ctx, y)).realize()

Tensor.manual_seed(1234)
sigmas = Tensor.rand(500).realize()

def sigma_to_idx(s:Tensor) -> Tensor:
  dists = s - sigmas
  return dists.abs().argmin(axis=0).view(*s.shape)

def make_call() -> Tensor:
  Tensor.manual_seed(1234)
  x   = Tensor.rand(1, 4, 32, 32)
  y   = Tensor.rand(1, 32)
  ctx = Tensor.rand(1, 17, 88)

  sigma = Tensor.rand(1)
  sigma_shape = sigma.shape
  sigma = sigmas[sigma_to_idx(sigma)]
  sigma = append_dims(sigma, x)

  c_out = -sigma
  c_in  = 1 / (sigma**2 + 1.0) ** 0.5
  tms   = sigma_to_idx(sigma.reshape(sigma_shape))

  def prep(*tensors:Tensor):
    return tuple(t.cast(dtypes.float16).realize() for t in tensors)
  return run(*prep(x*c_in, tms, ctx, y)) * c_out + x

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
  np.testing.assert_allclose(v1, v2, atol=1e-5, rtol=1e-5)
