from tinygrad import Tensor, dtypes # type: ignore
from examples.stable_diffusion import AutoencoderKL # type: ignore
from extra.models.unet import UNetModel # type: ignore
from extra.models.clip import FrozenOpenClipEmbedder # type: ignore

from typing import Dict
import numpy as np

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L521
class StableDiffusionV2:
  def __init__(self, unet_config:Dict, cond_stage_config:Dict):
    self.model = UNetModel(**unet_config)
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model  = FrozenOpenClipEmbedder(**cond_stage_config)

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L53
def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps):
  # assume uniform
  c = num_ddpm_timesteps // num_ddim_timesteps
  ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

  # add one to get the final alpha values right (the ones from first scale to data during sampling)
  steps_out = ddim_timesteps + 1
  return steps_out

class DdimSampler:
  def __init__(self):
    self.ddim_timesteps = make_ddim_timesteps()
  
  def sample(self, x:Tensor, c:Tensor, uc:Tensor):
    pass

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

