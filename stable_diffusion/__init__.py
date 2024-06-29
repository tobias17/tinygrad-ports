# This module incorporates code from the following:
# Github Name                    | License | Link
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad import Tensor, dtypes # type: ignore
import numpy as np
import math

def tensor_identity(x:Tensor) -> Tensor:
  return x

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/util.py#L207
def timestep_embedding(timesteps:Tensor, dim:int, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
  args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
  return Tensor.cat(args.cos(), args.sin(), dim=-1).cast(dtypes.float16)

def append_dims(x:Tensor, t:Tensor) -> Tensor:
  dims_to_append = len(t.shape) - len(x.shape)
  assert dims_to_append >= 0
  return x.reshape(x.shape + (1,)*dims_to_append)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/discretizer.py#L42
class LegacyDDPMDiscretization:
  def __init__(self, linear_start:float=0.00085, linear_end:float=0.0120, num_timesteps:int=1000):
    self.num_timesteps = num_timesteps
    betas = np.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)

  def __call__(self, n:int, flip:bool=False) -> Tensor:
    if n < self.num_timesteps:
      timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
      alphas_cumprod = self.alphas_cumprod[timesteps]
    elif n == self.num_timesteps:
      alphas_cumprod = self.alphas_cumprod
    sigmas = Tensor((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigmas = Tensor.cat(Tensor.zeros((1,)), sigmas)
    return sigmas if flip else sigmas.flip(axis=0) # sigmas is "pre-flipped", need to do oposite of flag

import embedders # type: ignore
import first_stage # type: ignore
import unet # type: ignore
