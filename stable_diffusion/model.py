import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tinygrad import Tensor, TinyJit, dtypes # type: ignore
from tinygrad.helpers import fetch, Context, getenv # type: ignore
from tinygrad.nn.state import safe_load, load_state_dict, torch_load # type: ignore

from stable_diffusion import append_dims, get_alphas_cumprod, LegacyDDPMDiscretization # type: ignore
from stable_diffusion.embedders import Embedder, FrozenClosedClipEmbedder, FrozenOpenClipEmbedder, ConcatTimestepEmbedderND
from stable_diffusion.first_stage import FirstStageModel # type: ignore
from stable_diffusion.unet import DiffusionModel
from stable_diffusion.samplers import Sampler, SDv1Sampler, DPMPP2MSampler

from typing import Dict, Tuple, List, Set, Optional, Type
from abc import ABC, abstractmethod
import os, argparse, tempfile, re
from pathlib import Path
from PIL import Image
import numpy as np

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class GeneralConditioner:
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

def prep_for_jit(*tensors:Tensor) -> Tuple[Tensor,...]:
  return tuple(t.cast(dtypes.float16).realize() for t in tensors)

class StableDiffusion(ABC):
  samplers: Dict[str,Type[Sampler]] # the first entry in the dict is considered default
  @abstractmethod
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    pass
  @abstractmethod
  def denoise(self, x:Tensor, sigmas_or_tms:Tensor, cond:Dict) -> Tensor:
    pass
  @abstractmethod
  def decode(self, x:Tensor) -> Tensor:
    pass
  @abstractmethod
  def delete_conditioner(self) -> None:
    pass

class StableDiffusionV1(StableDiffusion):
  samplers = {
    "basic": SDv1Sampler,
  }

  def __init__(self, first_stage_model:Dict, model:Dict, num_timesteps:int):
    self.cond_stage_model = FrozenClosedClipEmbedder()
    self.first_stage_model = FirstStageModel(**first_stage_model)
    self.model = DiffusionModel(**model)

    disc = LegacyDDPMDiscretization()
    self.sigmas = disc(num_timesteps, flip=True)
    self.alphas_cumprod = disc.alphas_cumprod

  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    return {"crossattn": self.cond_stage_model(pos_prompt)}, {"crossattn": self.cond_stage_model("")}

  def denoise(self, x:Tensor, tms:Tensor, cond:Dict) -> Tensor:
    @TinyJit
    def run(x, tms, ctx):
      return self.model.diffusion_model(x, tms, ctx, None).realize()

    return run(*prep_for_jit(x, tms, cond["crossattn"]))

  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1 / 0.18215 * x)

  def delete_conditioner(self) -> None:
    del self.cond_stage_model

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL(StableDiffusion):
  samplers = {
    "dpmpp2m": DPMPP2MSampler,
  }

  def __init__(self, conditioner:Dict, first_stage_model:Dict, model:Dict, num_timesteps:int):
    self.conditioner = GeneralConditioner(**conditioner)
    self.first_stage_model = FirstStageModel(**first_stage_model)
    self.model = DiffusionModel(**model)

    self.sigmas = LegacyDDPMDiscretization()(num_timesteps, flip=True)

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

    @TinyJit
    def run(x, tms, ctx, y, c_out, add):
      return (self.model.diffusion_model(x, tms, ctx, y)*c_out + add).realize()

    return run(*prep_for_jit(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0 / 0.13025 * x)

  def delete_conditioner(self) -> None:
    del self.conditioner

configs: Dict = {
  # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml
  "SDv1": {
    "default_weights_url": "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",
    "class": StableDiffusionV1,
    "args": {
      "model": {
        "adm_in_ch": None,
        "in_ch": 4,
        "out_ch": 4,
        "model_ch": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "n_heads": 8,
        "transformer_depth": [1, 1, 1, 1],
        "ctx_dim": 768,
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
      "num_timesteps": 1000,
    },
  },
  
  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
  "SDXL": {
    "default_weights_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "class": SDXL,
    "args": {
      "model": {
        "adm_in_ch": 2816,
        "in_ch": 4,
        "out_ch": 4,
        "model_ch": 320,
        "attention_resolutions": [4, 2],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4],
        "d_head": 64,
        "transformer_depth": [1, 2, 10],
        "ctx_dim": 2048,
      },
      "conditioner": {
        "embedders": [
          { "class": FrozenClosedClipEmbedder, "args": { "ret_layer_idx": 11 } },
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
      "num_timesteps": 1000,
    },
  },

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
  # "SDXL_Refiner": {
  #   "class": SDXL,
  #   "args": {
  #     "model": {
  #       "adm_in_ch": 2560,
  #       "in_ch": 4,
  #       "out_ch": 4,
  #       "model_ch": 384,
  #       "attention_resolutions": [4, 2],
  #       "num_res_blocks": 2,
  #       "channel_mult": [1, 2, 4, 4],
  #       "d_head": 64,
  #       "transformer_depth": [4, 4, 4, 4],
  #       "ctx_dim": [1280, 1280, 1280, 1280],
  #     },
  #     "conditioner": {
  #       "embedders": [
  #         { "class": FrozenClosedClipEmbedder, "args": { "ret_layer_idx": 11 } },
  #         { "class": FrozenOpenClipEmbedder,   "args": {} },
  #         { "class": ConcatTimestepEmbedderND, "args": { "input_key": "original_size_as_tuple" } },
  #         { "class": ConcatTimestepEmbedderND, "args": { "input_key": "crop_coords_top_left"   } },
  #         { "class": ConcatTimestepEmbedderND, "args": { "input_key": "target_size_as_tuple"   } },
  #       ],
  #     },
  #     "first_stage_model": {
  #       "ch": 128,
  #       "in_ch": 3,
  #       "out_ch": 3,
  #       "z_ch": 4,
  #       "ch_mult": [1, 2, 4, 4],
  #       "num_res_blocks": 2,
  #       "resolution": 256,
  #     },
  #     "denoiser": {
  #       "num_idx": 1000,
  #     },
  #     "num_timesteps": 1000,
  #   }
  # }
}

def compare_state_dicts(left_state_dict:Dict, right_state_dict:Dict, left_name:str="Model", right_name:str="Weights") -> None:
  left_keys, right_keys = set(left_state_dict.keys()), set(right_state_dict.keys())
  left_only, right_only = left_keys.difference(right_keys), right_keys.difference(left_keys)
  blocks = ["\n".join([f"\n{n} Only:"]+sorted(list(s))) for s,n in ((left_only,left_name), (right_only,right_name)) if len(s) > 0]
  print("Both state dicts contain the same keys" if len(blocks) == 0 else "\n".join(blocks)+"\n")

def from_pretrained(config_key:str, weights_fn:Optional[str]=None, weights_url:Optional[str]=None, fp16:bool=False) -> StableDiffusion:
  config = configs.get(config_key, None)
  assert config is not None, f"Invalid architecture key '{args.arch}', expected value in {list(configs.keys())}"
  model = config["class"](**config["args"])

  if weights_fn is not None:
    assert weights_url is None, "Got passed both a weights_fn and weights_url, options are mutually exclusive"
  else:
    weights_url = weights_url if weights_url is not None else config["default_weights_url"]
    weights_fn  = fetch(weights_url, os.path.basename(weights_url))

  loader_map = {
    "ckpt": lambda fn: torch_load(fn)["state_dict"],
    "safetensors": safe_load,
  }
  loader = loader_map.get(ext := str(weights_fn).split(".")[-1], None)
  assert loader is not None, f"Unsupported file extension '{ext}' for weights filename, expected value in {list(loader_map.keys())}"
  state_dict = loader(weights_fn)

  for k,v in state_dict.items():
    if fp16:
      state_dict[k] = v.cast(dtypes.float16)
    if re.match(r'model\.diffusion_model\..+_block.+proj_[a-z]+\.weight', k):
      state_dict[k] = v.squeeze()
  load_state_dict(model, state_dict, strict=False)

  return model

if __name__ == "__main__":
  arch_parser = argparse.ArgumentParser(description="Run SDXL", add_help=False)
  arch_parser.add_argument('--arch', type=str, default="SDv1", choices=list(configs.keys()))
  arch_args, _ = arch_parser.parse_known_args()
  defaults = {
    "SDv1": { "width": 512,  "height": 512,  "guidance": 7.5, "sampler": "basic" },
    "SDXL": { "width": 1024, "height": 1024, "guidance": 6.0, "sampler": "dpmpp2m" },
    # "SDXL_Refiner": { "width": 1024, "height": 1024 },
  }[arch_args.arch]
  sampler_options = list(configs[arch_args.arch]["class"].samplers.keys())

  parser = argparse.ArgumentParser(
    description="Run StableDiffusion. Note that changing the architecture with --arch will change some defaults and options, so set that option before running --help.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--arch',        type=str,   default="SDv1", choices=list(configs.keys()), help="Model architecture to use")
  parser.add_argument('--sampler',     type=str,   choices=sampler_options, default=sampler_options[0], help="Sampler to use for generation")
  parser.add_argument('--steps',       type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',      type=str,   default="a horse sized cat eating a bagel", help="Description of image to generate")
  parser.add_argument('--out',         type=str,   default=Path(tempfile.gettempdir())/"rendered.png", help="Output filename")
  parser.add_argument('--seed',        type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance',    type=float, default=defaults["guidance"], help="Prompt strength")
  parser.add_argument('--width',       type=int,   default=defaults["width"],  help="The output image width")
  parser.add_argument('--height',      type=int,   default=defaults["height"], help="The output image height")
  parser.add_argument('--aesthetic',   type=float, default=5.0, help="Aesthetic store for conditioning, only for SDXL_Refiner")
  parser.add_argument('--weights-fn',  type=str,   help="Filename of weights to load")
  parser.add_argument('--weights-url', type=str,   help="Url to download weights from")
  parser.add_argument('--fp16',        action='store_true', help="Loads the weights as float16")
  parser.add_argument('--noshow',      action='store_true', help="Don't show the image")
  parser.add_argument('--timing',      action='store_true', help="Print timing per step")
  args = parser.parse_args()

  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  model = from_pretrained(args.arch, args.weights_fn, args.weights_url, args.fp16)

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

  SamplerCls = model.samplers.get(args.sampler, None)
  assert SamplerCls is not None, f"Somehow failed to resolve sampler '{args.sampler}' from {model.__class__.__name__} class"

  sampler = SamplerCls(args.guidance, args.timing)
  with Context(BEAM=getenv("LATEBEAM")):
    z = sampler(model.denoise, randn, c, uc, args.steps)
  print("created samples")
  x = model.decode(z).realize()
  print("decoded samples")

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,args.height,args.width).permute(1,2,0).clip(0,1)*255
  print(x.shape)

  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()
