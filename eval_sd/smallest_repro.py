from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.state import get_parameters
from examples.sdxl import append_dims # type: ignore
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

def make_call(x, y, ctx) -> Tensor:
  Tensor.manual_seed(1234)
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

def create_entry(x, y, ctx, realize):
  o1, o2 = make_call(x, y, ctx), make_call(x, y, ctx)
  if realize:
    o1.realize()
    o2.realize()
  h = o1 + 8.0*(o2 - o1)
  return h.numpy()

dual_realize = False

Tensor.manual_seed(1234)
y_1   = Tensor.rand(1, 32)
ctx_1 = Tensor.rand(1, 17, 88)
y_2   = Tensor.rand(1, 32)
ctx_2 = Tensor.rand(1, 17, 88)

values_1 = []
values_2 = []

for _ in range(10):
  x = Tensor.rand(1, 4, 32, 32)
  values_1.append(create_entry(x, y_1, ctx_1, True))
  values_2.append(create_entry(x, y_2, ctx_2, dual_realize))

for v1, v2 in zip(values_1, values_2):
  np.testing.assert_allclose(v1, v2)
