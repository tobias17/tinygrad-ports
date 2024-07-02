from tinygrad import Tensor # type: ignore
from tinygrad.helpers import tqdm # type: ignore
from examples.stable_diffusion import AutoencoderKL, get_alphas_cumprod # type: ignore
from examples.sdxl import append_dims # type: ignore
from extra.models.unet import UNetModel # type: ignore
from extra.models.clip import FrozenOpenClipEmbedder # type: ignore

from typing import Dict, Tuple
import numpy as np

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L521
class StableDiffusionV2:
  def __init__(self, unet_config:Dict, cond_stage_config:Dict, parameterization:str="v"):
    self.model             = UNetModel(**unet_config)
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model  = FrozenOpenClipEmbedder(**cond_stage_config)
    self.alphas_cumprod    = get_alphas_cumprod()
    self.parameterization  = parameterization
  
  def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor):
    return self.model(x, tms, ctx)

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L53
def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps) -> np.ndarray:
  # assume uniform
  c = num_ddpm_timesteps // num_ddim_timesteps
  ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

  # add one to get the final alpha values right (the ones from first scale to data during sampling)
  steps_out = ddim_timesteps + 1
  return steps_out

# TODO TF: figure out what this is actually does
# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L103
def extract_into_tensor(a:Tensor, t:Tensor, x_shape:Tuple[int,...]) -> Tensor:
  return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

class DdimSampler:
  def __init__(self, ddpm_num_timesteps:int, scale:float):
    self.ddpm_num_timesteps = ddpm_num_timesteps
    self.scale = scale

  def sample(self, steps:int, model:StableDiffusionV2, x:Tensor, c:Tensor, uc:Tensor, eta:float=0.0):
    ddim_timesteps = make_ddim_timesteps(steps, self.ddpm_num_timesteps)
    assert model.alphas_cumprod.shape[0] == self.ddpm_num_timesteps

    alphas = model.alphas_cumprod[ddim_timesteps]
    alphas_prev = model.alphas_cumprod[0].cat(alphas[:-1])
    sqrt_one_minus_alphas = Tensor.sqrt(1.0 - alphas)
    sqrt_alphas_cumprod = model.alphas_cumprod.sqrt()

    sigmas = eta * Tensor.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    c_in = Tensor.cat(uc, c)

    time_range = np.flip(ddim_timesteps)
    for i, timestep in enumerate(t := tqdm(time_range)):
      x_t = Tensor.cat(x,x)
      t   = Tensor.full((2,),timestep)
      x_uc, x_c = model(x_t, t, c_in).chunk(2)
      output = x_uc + self.scale * (x_c - x_uc)

      if model.parameterization == "v":
        # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L296
        e_t = extract_into_tensor(sqrt_alphas_cumprod, t, x_t.shape)   * output \
            + extract_into_tensor(sqrt_one_minus_alphas, t, x_t.shape) * x_t
      else:
        e_t = output

      index   = ddim_timesteps.shape[0] - i - i
      a_t     = alphas[index]     .reshape(x.shape[0], 1, 1, 1)
      a_prev  = alphas_prev[index].reshape(x.shape[0], 1, 1, 1)
      sigma_t = sigmas[index]     .reshape(x.shape[0], 1, 1, 1)
      sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape(x.shape[0], 1, 1, 1)

      if model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * eta) / a_t.sqrt()
      else:
        # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L288
        pred_x0 = extract_into_tensor(sqrt_alphas_cumprod,   t, x_t.shape) * x_t \
                - extract_into_tensor(sqrt_one_minus_alphas, t, x_t.shape) * output

      dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
      x = a_prev.sqrt() * pred_x0 + dir_xt

params = {
  "unet_config": {
    "adm_in_ch": None,
    "in_ch": 4,
    "out_ch": 4,
    "model_ch": 320,
    "attention_resolutions": [4, 2, 1],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 4, 4],
    "d_head": 64,
    "transformer_depth": 1,
    "ctx_dim": 1024,
    "use_linear": True,
  },
  "cond_stage_config": {
    "dims": 1024,
    "n_heads": 16,
    "layers": 24,
    "return_pooled": False,
  }
}

if __name__ == "__main__":
  pass

