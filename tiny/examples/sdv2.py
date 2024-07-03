from tinygrad import Tensor, dtypes, TinyJit # type: ignore
from tinygrad.helpers import tqdm, fetch # type: ignore
from tinygrad.nn.state import safe_load, load_state_dict # type: ignore
from examples.stable_diffusion import AutoencoderKL, get_alphas_cumprod # type: ignore
from extra.models.unet import UNetModel # type: ignore
from extra.models.clip import FrozenOpenClipEmbedder # type: ignore

from typing import Dict, Tuple
import argparse, tempfile, os
from pathlib import Path
from PIL import Image
import numpy as np

class DiffusionModel:
  def __init__(self, model:UNetModel):
    self.diffusion_model = model

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L521
class StableDiffusionV2:
  def __init__(self, unet_config:Dict, cond_stage_config:Dict, parameterization:str="v"):
    self.model             = DiffusionModel(UNetModel(**unet_config))
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model  = FrozenOpenClipEmbedder(**cond_stage_config)
    self.alphas_cumprod    = get_alphas_cumprod()
    self.parameterization  = parameterization
  
  def denoise(self, x:Tensor, tms:Tensor, ctx:Tensor) -> Tensor:
    return self.model.diffusion_model(x, tms, ctx)

  def decode(self, x:Tensor, height:int, width:int) -> Tensor:
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,height,width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)
    return x

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L53
def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps) -> np.ndarray:
  # assume uniform
  c = num_ddpm_timesteps // num_ddim_timesteps
  ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

  # add one to get the final alpha values right (the ones from first scale to data during sampling)
  steps_out = ddim_timesteps + 1
  return steps_out

# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L103
def extract_into_tensor(a:Tensor, t:Tensor, x_shape:Tuple[int,...]) -> Tensor:
  return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

class DdimSampler:
  def __init__(self, ddpm_num_timesteps:int, scale:float):
    self.ddpm_num_timesteps = ddpm_num_timesteps
    self.scale = scale

  def __call__(self, model:StableDiffusionV2, x:Tensor, c:Tensor, uc:Tensor, steps:int, eta:float=0.0) -> Tensor:
    ddim_timesteps = make_ddim_timesteps(steps, self.ddpm_num_timesteps)
    assert model.alphas_cumprod.shape[0] == self.ddpm_num_timesteps

    alphas = model.alphas_cumprod[Tensor(ddim_timesteps)]
    alphas_prev = Tensor.cat(model.alphas_cumprod[:1], alphas[:-1])
    sqrt_one_minus_alphas_cumprod = Tensor.sqrt(1.0 - model.alphas_cumprod)
    sqrt_alphas_cumprod = model.alphas_cumprod.sqrt()

    sigmas = eta * Tensor.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    c_in = Tensor.cat(uc, c)

    # @TinyJit
    def run(denoiser, x:Tensor, tms:Tensor, ctx:Tensor) -> Tensor:
      return denoiser(x, tms, ctx).realize()

    def prep_for_jit(*tensors:Tensor) -> Tuple[Tensor,...]:
      return tuple(t.cast(dtypes.float16).realize() for t in tensors)

    time_range = np.flip(ddim_timesteps).tolist()
    for i, timestep in enumerate(tqdm(time_range)):
      x_in = Tensor.cat(x,x)
      tms  = Tensor([timestep])
      x_uc, x_c = run(model.denoise, *prep_for_jit(x_in, tms.cat(tms), c_in)).chunk(2)
      output = x_uc + self.scale * (x_c - x_uc)

      if model.parameterization == "v":
        # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L296
        e_t = extract_into_tensor(sqrt_alphas_cumprod,           tms, x_in.shape) * output \
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, tms, x_in.shape) * output
      else:
        e_t = output

      index   = ddim_timesteps.shape[0] - i - 1
      a_t     = alphas[index]     .reshape(x.shape[0], 1, 1, 1)
      a_prev  = alphas_prev[index].reshape(x.shape[0], 1, 1, 1)
      sigma_t = sigmas[index]     .reshape(x.shape[0], 1, 1, 1)
      sqrt_one_minus_at = sqrt_one_minus_alphas_cumprod[index].reshape(x.shape[0], 1, 1, 1)

      if model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * eta) / a_t.sqrt()
      else:
        # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L288
        pred_x0 = extract_into_tensor(sqrt_alphas_cumprod,           tms, x_in.shape) * output \
                - extract_into_tensor(sqrt_one_minus_alphas_cumprod, tms, x_in.shape) * output

      dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
      x = a_prev.sqrt() * pred_x0 + dir_xt
    
    return x

params: Dict = {
  "unet_config": {
    "adm_in_ch": None,
    "in_ch": 4,
    "out_ch": 4,
    "model_ch": 320,
    "attention_resolutions": [4, 2, 1],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 4, 4],
    "d_head": 64,
    "transformer_depth": [1, 1, 1, 1],
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
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion v2.X', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps',       type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',      type=str,   default=default_prompt, help="Description of image to generate")
  parser.add_argument('--out',         type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--seed',        type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance',    type=float, default=6.0, help="Prompt strength")
  parser.add_argument('--width',       type=int,   default=768, help="The output image width")
  parser.add_argument('--height',      type=int,   default=768, help="The output image height")
  parser.add_argument('--weights-fn',  type=str,   help="Filename of weights to use")
  parser.add_argument('--weights-url', type=str,   help="Custom URL to download weights from")
  parser.add_argument('--noshow',      action='store_true', help="Don't show the image")
  args = parser.parse_args()

  N = 1
  C = 4
  F = 8
  assert args.width  % F == 0, f"img_width must be multiple of {F}, got {args.width}"
  assert args.height % F == 0, f"img_height must be multiple of {F}, got {args.height}"

  Tensor.no_grad = True
  model = StableDiffusionV2(**params)

  default_weights_url = 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors'
  weights_fn = args.weights_fn
  if not weights_fn:
    weights_url = args.weights_url if args.weights_url else default_weights_url
    weights_fn  = fetch(weights_url, os.path.basename(str(weights_url)))
  load_state_dict(model, safe_load(weights_fn), strict=False)

  c  = model.cond_stage_model(args.prompt)
  uc = model.cond_stage_model("")
  del model.cond_stage_model
  print("created conditioning")
  
  shape = (N, C, args.height // F, args.width // F)
  randn = Tensor.randn(shape)

  sampler = DdimSampler(1000, args.guidance)
  z = sampler(model, randn, c, uc, args.steps)
  print("created samples")
  x = model.decode(z, args.height, args.width).realize()
  print("decoded samples")
  print(x.shape)

  im = Image.fromarray(x.numpy())
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()
