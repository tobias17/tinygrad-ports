from tinygrad import Tensor, TinyJit, dtypes # type: ignore
from tinygrad.helpers import fetch # type: ignore
from tinygrad.nn.state import safe_load, load_state_dict # type: ignore

from stable_diffusion import append_dims, LegacyDDPMDiscretization # type: ignore
from stable_diffusion.embedders import Embedder, FrozenClosedClipEmbedder, FrozenOpenClipEmbedder, ConcatTimestepEmbedderND
from stable_diffusion.first_stage import FirstStageModel # type: ignore
from stable_diffusion.unet import DiffusionModel
from stable_diffusion.samplers import DPMPP2MSampler

from typing import Dict, Tuple, List, Set, Optional
from abc import ABC, abstractmethod
import os, argparse, tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class Conditioner:
  OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
  KEY2CATDIM      = {"vector": 1, "crossattn": 2, "concat": 1}
  embedders: List[Embedder]

  def __init__(self, embedders:List[Dict]):
    self.embedders = []
    for emb in embedders:
      self.embedders.append(emb["class"](**emb["args"]))

  def get_keys(self) -> Set[str]:
    return set(e.input_key for e in self.embedders)

  def __call__(self, batch:Dict, force_zero_embeddings:List=[]) -> Dict[str,Tensor]:
    output: Dict[str,Tensor] = {}

    for embedder in self.embedders:
      emb_out = embedder(batch[embedder.input_key])

      if isinstance(emb_out, Tensor):
        emb_out = [emb_out]
      else:
        assert isinstance(emb_out, (list, tuple))

      for emb in emb_out:
        if embedder.input_key in force_zero_embeddings:
          emb = Tensor.zeros_like(emb)

        out_key = self.OUTPUT_DIM2KEYS[len(emb.shape)]
        if out_key in output:
          output[out_key] = Tensor.cat(output[out_key], emb, dim=self.KEY2CATDIM[out_key])
        else:
          output[out_key] = emb

    return output

class StableDiffusion(ABC):
  @abstractmethod
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    pass
  @abstractmethod
  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:
    pass
  @abstractmethod
  def decode(self, x:Tensor) -> Tensor:
    pass
  @abstractmethod
  def delete_conditioner(self) -> None:
    pass

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL(StableDiffusion):
  def __init__(self, conditioner:Dict, first_stage_model:Dict, model:Dict, **kwargs):
    self.conditioner = Conditioner(**conditioner)
    self.first_stage_model = FirstStageModel(**first_stage_model)
    self.model = DiffusionModel(**model)

    self.sigmas = LegacyDDPMDiscretization()(kwargs["denoiser"]["num_idx"], flip=True)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L173
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    N = 1
    batch_c : Dict = {
      "txt": pos_prompt,
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    batch_uc: Dict = {
      "txt": "",
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    return self.conditioner(batch_c), self.conditioner(batch_uc, force_zero_embeddings=["txt"])

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/denoiser.py#L42
  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_out   = -sigma
    c_in    = 1 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    def prep(*tensors:Tensor):
      return tuple(t.cast(dtypes.float16).realize() for t in tensors)

    @TinyJit
    def run(model, x, tms, ctx, y, c_out, add):
      return (model(x, tms, ctx, y)*c_out + add).realize()

    return run(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0 / 0.13025 * x)

  def delete_conditioner(self) -> None:
    del self.conditioner

configs: Dict = {
  "StableDiffusion_1x": {
    "class": ""
  },
  "SDXL_Base": {
    "default_weights_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "class": SDXL,
    "args": {
      "model": {
        "adm_in_channels": 2816,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [4, 2],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4],
        "d_head": 64,
        "transformer_depth": [1, 2, 10],
        "ctx_dim": 2048,
      },
      "conditioner": {
        "embedders": [
          { "class": FrozenClosedClipEmbedder, "args": {} },
          { "class": FrozenOpenClipEmbedder,   "args": {} },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "original_size_as_tuple" } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "crop_coords_top_left"   } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "target_size_as_tuple"   } },
        ],
      },
      "first_stage_model": {
        "ch": 128,
        "in_ch": 3,
        "out_ch": 3,
        "z_ch": 4,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "resolution": 256,
      },
      "denoiser": {
        "num_idx": 1000
      },
    },
  },
  "SDXL_Refiner": {
    "class": SDXL,
    "args": {
      "model": {
        "adm_in_channels": 2560,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 384,
        "attention_resolutions": [4, 2],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "d_head": 64,
        "transformer_depth": [4, 4, 4, 4],
        "ctx_dim": [1280, 1280, 1280, 1280],
      },
      "conditioner": {
        "embedders": [
          { "class": FrozenClosedClipEmbedder, "args": {} },
          { "class": FrozenOpenClipEmbedder,   "args": {} },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "original_size_as_tuple" } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "crop_coords_top_left"   } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "target_size_as_tuple"   } },
        ],
      },
      "first_stage_model": {
        "ch": 128,
        "in_ch": 3,
        "out_ch": 3,
        "z_ch": 4,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "resolution": 256,
      },
      "denoiser": {
        "num_idx": 1000,
      },
    }
  }
}

def from_pretrained(config_key:str, weights_fn:Optional[str]=None, weights_url:Optional[str]=None) -> StableDiffusion:
  if weights_fn is not None:
    assert weights_url is None, "Got passed both a weights_fn and weights_url, options are mutually exclusive"
  else:
    weights_url = weights_url if weights_url is not None else config["default_weights_url"]
    weights_fn  = fetch(weights_url, os.path.basename(weights_url))

  model: StableDiffusion = config["class"](**config["args"])
  load_state_dict(model, safe_load(weights_fn), strict=False)

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run SDXL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--arch',        type=str,   default="SDXL_Base", choices=list(configs.keys()), help="What architecture to use")
  parser.add_argument('--steps',       type=int,   default=5, help="The number of diffusion steps")
  parser.add_argument('--prompt',      type=str,   default="a horse sized cat eating a bagel", help="Description of image to generate")
  parser.add_argument('--out',         type=str,   default=Path(tempfile.gettempdir())/"rendered.png", help="Output filename")
  parser.add_argument('--seed',        type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance',    type=float, default=6.0, help="Prompt strength")
  parser.add_argument('--width',       type=int,   default=1024, help="The output image width")
  parser.add_argument('--height',      type=int,   default=1024, help="The output image height")
  parser.add_argument('--aesthetic',   type=float, default=5.0, help="Aesthetic store for conditioning, only for SDXL_Refiner")
  parser.add_argument('--weights-fn',  type=str,   help="Filename of weights to load")
  parser.add_argument('--weights-url', type=str,   help="Url to download weights from")
  parser.add_argument('--noshow',      action='store_true', help="Don't show the image")
  args = parser.parse_args()

  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)
  
  config = configs.get(args.arch, None)
  assert config is not None, f"Somehow got passed invalid architecture '{args.arch}', expected value in {list(configs.keys())}"
  model = from_pretrained(config, args.weights_fn, args.weights_url)

  N = 1
  C = 4
  F = 8

  assert args.width  % F == 0, f"img_width must be multiple of {F}, got {args.width}"
  assert args.height % F == 0, f"img_height must be multiple of {F}, got {args.height}"

  c, uc = model.create_conditioning(args.prompt, args.width, args.height, args.aesthetic)
  model.delete_conditioner()
  for v in c .values(): v.realize()
  for v in uc.values(): v.realize()
  print("created conditioning")

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L101
  shape = (N, C, args.height // F, args.width // F)
  randn = Tensor.randn(shape)

  sampler = DPMPP2MSampler(args.guidance)
  z = sampler(model.denoise, randn, c, uc, args.steps)
  print("created samples")
  x = model.decode(z).realize()
  print("decoded samples")

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,args.height,args.width).permute(1,2,0).clip(0,1)*255
  x = x.cast(dtypes.float32).realize().cast(dtypes.uint8)
  print(x.shape)

  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()
