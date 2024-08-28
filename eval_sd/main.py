from tinygrad import Tensor, dtypes, Device, TinyJit # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict
from examples.sdxl import SDXL, DPMPP2MSampler, append_dims, configs # type: ignore
from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore

from typing import Dict, List, Tuple, Optional
from threading import Thread
import pandas as pd # type: ignore
import numpy as np
from PIL import Image
from functools import partial
import time


@TinyJit
def run_model(model, x, tms, ctx, y, c_out, add):
  return model(x, tms, ctx, y).mul(c_out).add(add).realize()

def denoise(self:SDXL, x:Tensor, sigma:Tensor, cond:Dict) -> Tuple[Tensor]:
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

  return run_model(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

def smart_print(sec:float) -> str:
  if sec < 60: return f"{sec:.0f} sec"
  mins = sec / 60.0
  if mins < 60: return f"{mins:.1f} mins"
  hours = mins / 60.0
  return f"{hours:.1f} hours"



def gen_images():

  Tensor.manual_seed(42)

  # Define constants

  GPUS = [f"{Device.DEFAULT}:{i}" for i in [4,5]]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  IMG_SIZE = 1024
  LATENT_SCALE = 8
  LATENT_SIZE = IMG_SIZE // LATENT_SCALE
  assert LATENT_SIZE * LATENT_SCALE == IMG_SIZE

  GUIDANCE_SCALE = 8.0
  NUM_STEPS = 2

  # Load model
  model = SDXL(configs["SDXL_Base"])
  load_state_dict(model, safe_load("/home/tiny/tinygrad/weights/sd_xl_base_1.0.safetensors"), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None))

  # Create sampler
  sampler = DPMPP2MSampler(GUIDANCE_SCALE)

  # Load dataset
  df = pd.read_csv("/home/tiny/tinygrad/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)
  captions = df["caption"].array
  with open("inputs/captions.txt", "w") as f:
    f.write("\n".join(captions))

  wall_time_start = time.time()

  def async_save(images:np.ndarray, start_i:int):
    for image_i in range(GLOBAL_BS):
      im = Image.fromarray(images[image_i])
      im.save(f"inputs/gen_{start_i+image_i:05d}.png")

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < len(captions):
    texts = captions[dataset_i:dataset_i+GLOBAL_BS].tolist()
    c, uc = model.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
    for t in  c.values(): t.shard_(GPUS, axis=0)
    for t in uc.values(): t.shard_(GPUS, axis=0)
    randn = Tensor.randn(GLOBAL_BS, 4, LATENT_SIZE, LATENT_SIZE).shard(GPUS, axis=0)
    z = sampler(partial(denoise, model), randn, c, uc, NUM_STEPS)
    x = model.decode(z).realize()
    x = (x + 1.0) / 2.0
    x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    x = x.numpy()

    Thread(target=async_save, args=(x,dataset_i)).start()

    dataset_i += GLOBAL_BS

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}")



def compute_clip():
  clip_enc = OpenClipEncoder(**clip_configs["ViT-H-14"]).load_from_pretrained()
  tokenizer = Tokenizer.ClipTokenizer()

  GPUS = [f"{Device.DEFAULT}:{i}" for i in [4,5]]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  with open("inputs/captions.txt", "r") as f:
    captions = f.read().split("\n")

  wall_time_start = time.time()
  all_scores = []

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < len(captions):
    texts = captions[dataset_i:dataset_i+GLOBAL_BS]
    tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64).reshape(1,-1) for text in texts]

    images = []
    for image_i in range(GLOBAL_BS):
      im = Image.open(f"inputs/gen_{dataset_i+image_i:05d}.png")
      images.append(clip_enc.prepare_image(im).unsqueeze(0))

    clip_score  = clip_enc.get_clip_score(Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
    scores      = clip_score.numpy()
    all_scores += scores.tolist()

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}, scores: {scores}")

    dataset_i += GLOBAL_BS

  print(f"average_clip_score: {sum(all_scores) / len(all_scores)}")

if __name__ == "__main__":
  gen_images()
