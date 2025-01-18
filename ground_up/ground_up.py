from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, tqdm
from tinygrad.nn import Conv2d
from tinygrad.nn.state import load_state_dict, safe_load
from examples.sdxl import FirstStage
from local_unet import UNetModel, log_difference
import numpy as np
from PIL import Image



#######################################
#   Overwrite functions from stdlib   #
from tinygrad.ops import Ops, identity_element
def _cumalu(self, axis:int, op:Ops, _include_initial=False) -> Tensor:
  assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX, Ops.MUL)
  pl_sz = self.shape[axis] - int(not _include_initial)
  pooled = self.transpose(axis,-1).pad((pl_sz, -int(_include_initial)), value=identity_element(op, self.dtype))._pool((self.shape[axis],))
  return (pooled.sum(-1) if op is Ops.ADD else pooled.max(-1)).transpose(axis,-1)
Tensor._cumalu = _cumalu # type: ignore
def cumprod(self, axis:int=0) -> Tensor:
  """
  Computes the cumulative prod of the tensor along the specified `axis`.

  ```python exec="true" source="above" session="tensor" result="python"
  t = Tensor.ones(2, 3) * 2
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  print(t.cumprod(1).numpy())
  ```
  """
  return self._split_cumalu(axis, Ops.MUL)
Tensor.cumprod = cumprod # type: ignore
#######################################



class FirstStageModel:
  def __init__(self, embed_dim:int=4, **kwargs):
    self.encoder = FirstStage.Encoder(**kwargs)
    self.decoder = FirstStage.Decoder(**kwargs)
    self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
    self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

  def decode(self, z:Tensor) -> Tensor:
    return z.sequential([self.post_quant_conv, self.decoder])

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

  def __init__(self, num_timesteps:int=1000, beta_start:float=0.00085, beta_end:float=0.012, beta_schedule:str="scaled_linear"):
    self.num_timesteps = num_timesteps

    if beta_schedule == "scaled_linear":
      self.betas = Tensor.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=dtypes.float32).square()
      y = self.betas.numpy()
    else:
      raise NotImplementedError(f"Beta schedule '{beta_schedule}' not implemented for {self.__class__.__name__}")
    
    self.alphas         = 1.0 - self.betas
    x = self.alphas.numpy()
    z = np.cumprod(self.alphas.numpy(), axis=0)
    self.alphas_cumprod = Tensor(z)

    # FIXME: try other way around
    self.all_timesteps = Tensor(np.linspace(0, num_timesteps - 1, num_timesteps, dtype=np.float32)[::-1].copy())
    # self.timesteps = Tensor(np.linspace(num_timesteps - 1, 0, num_timesteps, dtype=np.float32))

    sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
    self.all_sigmas = sigmas.cat(Tensor.zeros(1))

  def config_timesteps(self, inference_steps:int) -> 'EulerDiscreteScheduler':
    if self.inference_steps == inference_steps:
      return self

    step_ratio = self.num_timesteps // inference_steps
    timesteps = (np.arange(0, inference_steps, dtype=np.float32) * step_ratio).round()[::-1].copy() + 1
    acp_np = self.alphas_cumprod.numpy()
    sigmas = np.array(((1 - acp_np) / acp_np) ** 0.5, dtype=np.float32)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas), dtype=np.float32), sigmas)
    sigmas = np.concatenate([sigmas, [0]]).astype(np.float32)
    self.sigmas = Tensor(sigmas)
    self.timesteps = Tensor(timesteps, dtype=dtypes.float32)
    self.inference_steps = inference_steps

    return self
  
  def scale_model_input(self, sample:Tensor, i:int) -> Tensor:
    sigma = self.sigmas[i]
    sample = sample / ((sigma.square() + 1).sqrt())
    return sample


  def step(self, sample:Tensor, i:int, model_output:Tensor, pred_type:str="epsilon") -> Tensor:
    sigma = self.sigmas[i]

    if pred_type == "epsilon":
      pred = sample - sigma * model_output
    else:
      raise NotImplementedError(f"{self.__class__.__name__} does not support pred_type '{pred_type}'")

    derivative = (sample - pred) / sigma
    dt = self.sigmas[i + 1] - sigma
    prev = sample + derivative * dt

    return prev

class SDXL:
  def __init__(self):
    self.model = DiffusionModel(**{"adm_in_ch": 2816, "in_ch": 4, "out_ch": 4, "model_ch": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048, "use_linear": True})
    self.first_stage_model = FirstStageModel(**{"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256})
    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(1000, flip=True)

  def decode(self, z:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0/0.13025 * z)


GUIDANCE_SCALE = 8.0

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
    print(f"Running step {i}")
    latent_model_input = scheduler.scale_model_input(Tensor.cat(latents, latents), i)

    noise_pred = model.model.diffusion_model(latent_model_input, Tensor(t).expand(2), prompt_embeds, add_text_embeds, add_time_ids, i).realize()
    log_difference("noise_p", noise_pred, Tensor(np.load(f"../compare/stages/{i:02d}/noise_pred.npy")))
    noise_pred_u, noise_pred_c = noise_pred.chunk(2)
    noise_pred = noise_pred_u + GUIDANCE_SCALE * (noise_pred_c - noise_pred_u)

    latents = scheduler.step(latents, i, noise_pred).realize()
    log_difference("latents", latents, Tensor(np.load(f"../compare/stages/{i:02d}/end_latents.npy")))
    print()
  
  x = model.decode(latents)
  x = (x + 1.0) / 2.0
  x = x.reshape(3,1024,1024).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)

  im = Image.fromarray(x.numpy())
  im.save("gen.png")

if __name__ == "__main__":
  main()
