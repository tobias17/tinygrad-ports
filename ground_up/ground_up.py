from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, safe_load
from extra.models.unet import UNetModel
import numpy as np

class DiffusionModel:
  def __init__(self, *args, **kwargs):
    self.diffusion_model = UNetModel(*args, **kwargs)

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

class EulerDiscreteScheduler:
  inference_steps: int = -1

  def __init__(self, num_timesteps:int=1000, beta_start:float=0.0001, beta_end:float=0.02, beta_schedule:str="scaled_linear"):
    self.num_timesteps = num_timesteps

    if beta_schedule == "scaled_linear":
      self.betas = Tensor.linspace(beta_start**0.5, beta_end**0.5, num_timesteps).square()
    else:
      raise NotImplementedError(f"Beta schedule '{beta_schedule}' not implemented for {self.__class__.__name__}")
    
    self.alphas         = 1.0 - self.betas
    self.alphas_cumprod = self.alphas.cumprod(axis=0)

    # FIXME: try other way around
    self.all_timesteps = Tensor(np.linspace(0, num_timesteps - 1, num_timesteps, dtype=np.float32)[::-1].copy())
    # self.timesteps = Tensor(np.linspace(num_timesteps - 1, 0, num_timesteps, dtype=np.float32))

    sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
    self.all_sigmas = sigmas.cat(Tensor.zeros(1))

  def config_timesteps(self, inference_steps:int) -> 'EulerDiscreteScheduler':
    if self.inference_steps == inference_steps:
      return self

    step_ratio = self.num_timesteps // inference_steps
    timesteps = (np.arange(0, inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32) + 1
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0]]).astype(np.float32)
    self.sigmas = Tensor(sigmas)
    self.timesteps = Tensor(timesteps, dtype=dtypes.float32)
    self.inference_steps = inference_steps

    return self
  
  def scale_model_input(self, sample:Tensor, i:int) -> Tensor:
    sigma = self.sigmas[i]
    sample = sample / ((sigma.square() + 1).sqrt())
    return sample

class SDXL:
  def __init__(self):
    self.model = DiffusionModel(**{"adm_in_ch": 2816, "in_ch": 4, "out_ch": 4, "model_ch": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048, "use_linear": True})
    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(1000, flip=True)

def main():
  ROOT = "../compare"
  pos_prompt_embeds = Tensor(np.load(f"{ROOT}/pos_prompt_embeds.npy")).realize()
  neg_prompt_embeds = Tensor(np.load(f"{ROOT}/neg_prompt_embeds.npy")).realize()
  pos_pooled_embeds = Tensor(np.load(f"{ROOT}/pos_pooled_embeds.npy")).realize()
  neg_pooled_embeds = Tensor(np.load(f"{ROOT}/neg_pooled_embeds.npy")).realize()
  latents           = Tensor(np.load(f"{ROOT}/latents.npy"          )).realize()
  add_time_ids      = Tensor(np.load(f"{ROOT}/add_time_ids.npy"     )).realize()

  prompt_embeds   = Tensor.cat(neg_prompt_embeds, pos_prompt_embeds, dim=0)
  add_text_embeds = Tensor.cat(neg_pooled_embeds, pos_pooled_embeds, dim=0)
  add_time_ids    = Tensor.cat(add_time_ids,      add_time_ids,      dim=0)

  scheduler = EulerDiscreteScheduler().config_timesteps(20)
  model = SDXL()
  weights_url = 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors'
  load_state_dict(model, safe_load(str(fetch(weights_url, 'sd_xl_base_1.0.safetensors'))), strict=False)

  timesteps = list(range(1, 1000, 50))[::-1]
  for i, t in enumerate(timesteps):
    latent_model_input = scheduler.scale_model_input(Tensor.cat(latents, latents), i)
    added_cond_kwargs = {"text_embeds":add_text_embeds, "time_ids":add_time_ids}

    

if __name__ == "__main__":
  main()
