from tinygrad import Tensor, dtypes, Device, TinyJit, GlobalCounters # type: ignore
from tinygrad.multi import MultiLazyBuffer
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict, torch_load
from tinygrad.helpers import trange, fetch
from examples.sdxl import SDXL, DPMPP2MSampler, Guider, configs, append_dims, run # type: ignore
from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
from extra.models.inception import FidInceptionV3 # type: ignore

from typing import Tuple, List, Dict
import pandas as pd # type: ignore
import numpy as np
from scipy import linalg # type: ignore
from PIL import Image
from threading import Thread
import time, os



def smart_print(sec:float) -> str:
  if sec < 60: return f"{sec:.0f} sec"
  mins = sec / 60.0
  if mins < 60: return f"{mins:.1f} mins"
  hours = mins / 60.0
  return f"{hours:.1f} hours"



IMG_SIZE = 1024
LATENT_SCALE = 8
LATENT_SIZE = IMG_SIZE // LATENT_SCALE
assert LATENT_SIZE * LATENT_SCALE == IMG_SIZE

GUIDANCE_SCALE = 8.0
NUM_STEPS = 20

def gen_images():

  Tensor.manual_seed(42)

  # Define constants

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(2)]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  # Load model
  model = SDXL(configs["SDXL_Base"])
  load_state_dict(model, safe_load("/home/tiny/tinygrad/weights/sd_xl_base_1.0.safetensors"), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

  # Create sampler
  sampler = DPMPP2MSampler(GUIDANCE_SCALE, guider_cls=SplitVanillaCFG)

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
    z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
    x = model.decode(z).realize()
    x = (x + 1.0) / 2.0
    x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    x = x.numpy()

    # Thread(target=async_save, args=(x,dataset_i)).start()

    dataset_i += GLOBAL_BS

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}")


@TinyJit
def clip_step(model, tokens, images):
  return model(tokens, images).realize()

def compute_clip():
  GLOBAL_BS = 150

  clip_enc = OpenClipEncoder(**clip_configs["ViT-H-14"]).load_from_pretrained()
  tokenizer = Tokenizer.ClipTokenizer()

  with open("inputs/captions.txt", "r") as f:
    captions = f.read().split("\n")

  wall_time_start = time.time()
  all_scores = []

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < len(captions):
    texts  = captions[dataset_i:dataset_i+GLOBAL_BS]
    tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64).reshape(1,-1) for text in texts]

    images = []
    for image_i in range(GLOBAL_BS):
      im = Image.open(f"inputs/gen_{dataset_i+image_i:05d}.png")
      images.append(clip_enc.prepare_image(im).unsqueeze(0))

    clip_score  = clip_step(clip_enc.get_clip_score, Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
    scores      = clip_score.numpy()
    all_scores += scores.tolist()

    dataset_i += GLOBAL_BS

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}, batch_scores_mean: {scores.mean():.4f}")

  print(f"average_clip_score: {sum(all_scores) / len(all_scores)}")





def load_inception_model(gpus=None) -> FidInceptionV3:
  inception = FidInceptionV3().load_from_pretrained()
  for w in get_state_dict(inception).values():
    w2 = w.cast(dtypes.float16)
    if gpus is not None:
      w2 = w2.shard(gpus, axis=None)
    w.replace(w2).realize()
  return inception

def get_incp_act(model:FidInceptionV3, pil_ims:List[Image.Image], gpus=None):
  images = [Tensor(np.asarray(im), dtype=dtypes.float16).div(255.0).permute(2,0,1).unsqueeze(0) for im in pil_ims]
  x = Tensor.cat(*images, dim=0).realize()
  if gpus:
    x = x.shard(gpus, axis=0)
  print(f"input_shape: {x.shape}")
  incp_act = model(x.realize())
  return incp_act.to(Device.DEFAULT).reshape(incp_act.shape[:2]).realize()

def compute_fid():
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(1)]
  DEVICE_BS = 50
  GLOBAL_BS = DEVICE_BS * len(GPUS)
  TEST_SIZE = 300 # 30_000
  Tensor.no_grad = True

  inception = load_inception_model(GPUS)
  
  wall_time_start = time.time()
  all_incp_act = []

  dataset_i = 0
  assert TEST_SIZE % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide TEST_SIZE ({TEST_SIZE}) for now"
  while dataset_i < TEST_SIZE:
    pil_ims = [Image.open(f"output/rendered_2/gen_{dataset_i+image_i:05d}.png") for image_i in trange(GLOBAL_BS)]
    all_incp_act.append(get_incp_act(inception, pil_ims, GPUS))

    if len(all_incp_act) > 20:
      all_incp_act = [Tensor.cat(*all_incp_act, dim=0).realize()]

    dataset_i += GLOBAL_BS

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/TEST_SIZE)) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/TEST_SIZE:02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}")

  print("\n" + "="*80 + "\n")

  final_incp_acts = Tensor.cat(*all_incp_act, dim=0)
  fid_score = inception.compute_score(final_incp_acts)
  print(f"fid_score:  {fid_score}")

  print("")







# ##################################################################
# # TODO: upstream
# FidInceptionV3.m1 = None
# FidInceptionV3.s1 = None
# def compute_score(self:FidInceptionV3, inception_activations:Tensor) -> float:
#   if self.m1 is None or self.s1 is None:
#     with np.load("/home/tiny/tinygrad/datasets/coco2014/val2014_30k_stats.npz") as f:
#       self.m1, self.s1 = f['mu'][:], f['sigma'][:]
#     assert self.m1 is not None and self.s1 is not None
  
#   m2 = inception_activations.mean(axis=0).numpy()
#   s2 = np.cov(inception_activations.numpy(), rowvar=False) # FIXME: need to figure out how to do in pure tinygrad

#   return calculate_frechet_distance(self.m1, self.s1, m2, s2)
# FidInceptionV3.compute_score = compute_score
# #
# def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
#   mu1 = np.atleast_1d(mu1)
#   mu2 = np.atleast_1d(mu2)
#   sigma1 = np.atleast_2d(sigma1)
#   sigma2 = np.atleast_2d(sigma2)
#   assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape

#   diff = mu1 - mu2
#   covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#   if not np.isfinite(covmean).all():
#     offset = np.eye(sigma1.shape[0]) * eps
#     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#   if np.iscomplexobj(covmean):
#     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#       m = np.max(np.abs(covmean.imag))
#       raise ValueError(f"Imaginary component {m}")
#     covmean = covmean.real
  
#   tr_covmean = np.trace(covmean)

#   return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*tr_covmean
# #
# ##################################################################




##################################################################
# TODO: upstream
@TinyJit
def run2(model, x, tms, ctx, y):
  return (model(x, tms, ctx, y)).realize()
#
def denoise(self:SDXL, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

  def sigma_to_idx(s:Tensor) -> Tensor:
    dists = s - self.sigmas.unsqueeze(1)
    return dists.abs().argmin(axis=0).view(*s.shape)

  sigma = self.sigmas[sigma_to_idx(sigma)]
  sigma_shape = sigma.shape
  sigma = append_dims(sigma, x)

  c_out = -sigma
  c_in  = 1 / (sigma**2 + 1.0) ** 0.5
  tms   = sigma_to_idx(sigma.reshape(sigma_shape))

  def prep(*tensors:Tensor):
    return tuple(t.cast(dtypes.float16).realize() for t in tensors)

  args = prep(x*c_in, tms, cond["crossattn"], cond["vector"])
  return (run2(self.model.diffusion_model, *args)*c_out + x).realize()
SDXL.denoise = denoise
#
class SplitVanillaCFG(Guider):
  def __call__(self, denoiser, x:Tensor, s:Tensor, c:Dict, uc:Dict) -> Tensor:
    x_u = denoiser(x, s, uc)
    x_c = denoiser(x, s, c)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred
#
##################################################################



class Timing:
  def __init__(self, label:str, collection:List[str], print_fnx=(lambda l,d: f"{l}: {1e3*d:.1f} ms")):
    self.label = label
    self.collection = collection
    self.print_fnx = print_fnx
  def __enter__(self):
    self.start_time = time.time()
  def __exit__(self, *_):
    self.collection.append(self.print_fnx(self.label, (time.time() - self.start_time)))

def do_all():
  Tensor.manual_seed(42)
  Tensor.no_grad = True

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(1,6)]
  DEVICE_BS = 4
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  MAX_INCP_STORE_SIZE = 4
  SAVE_IMAGES = False
  SAVE_ROOT = "./output/rendered"
  if SAVE_IMAGES and not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)

  # Load generation model
  model = SDXL(configs["SDXL_Base"])
  weights_path = fetch("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "sd_xl_base_1.0.safetensors")
  load_state_dict(model, safe_load(weights_path), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

  # Load evaluation model
  clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
  weights_path = fetch("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin", "CLIP-ViT-H-14-laion2B-s32B-b79K.bin")
  load_state_dict(clip_enc, torch_load(weights_path), strict=False)
  tokenizer = Tokenizer.ClipTokenizer()
  inception = FidInceptionV3().load_from_pretrained()
  for w in get_state_dict(inception).values():
    w.replace(w.cast(dtypes.float16)).realize()

  # Create sampler
  sampler = DPMPP2MSampler(GUIDANCE_SCALE, guider_cls=SplitVanillaCFG)

  # Load dataset
  df = pd.read_csv("/raid/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)
  captions = df["caption"].array

  wall_time_start = time.time()
  all_clip_scores = []
  all_incp_act    = []

  def async_save(images:List[Image.Image], global_i:int):
    for image_i, im in enumerate(images):
      im.save(f"{SAVE_ROOT}/gen_{global_i+image_i:05d}.png")

  @TinyJit
  def decode_step(z:Tensor) -> Tensor:
    return model.decode(z).realize()

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < 300: #len(captions):
    timings = []

    # Generate Image
    with Timing("gen", timings):
      texts = captions[dataset_i:dataset_i+GLOBAL_BS].tolist()
      c, uc = model.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
      for t in  c.values(): t.shard_(GPUS, axis=0)
      for t in uc.values(): t.shard_(GPUS, axis=0)
      randn = Tensor.randn(GLOBAL_BS, 4, LATENT_SIZE, LATENT_SIZE).shard(GPUS, axis=0)
      z = sampler(model.denoise, randn, c, uc, NUM_STEPS).realize()

    # Decode Images
    with Timing("dec", timings):
      x = decode_step(z.realize())
      x = (x + 1.0) / 2.0
      x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).realize()
      ten_im = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8).numpy()
      pil_im = [Image.fromarray(ten_im[image_i]) for image_i in range(GLOBAL_BS)]

    # Save Images
    if SAVE_IMAGES:
      Thread(target=async_save, args=(pil_im,dataset_i)).start()

    # Evaluate CLIP Score and Inception Activations
    with Timing("eval", timings):
      images = [clip_enc.prepare_image(im).unsqueeze(0) for im in pil_im]
      tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64).reshape(1,-1) for text in texts]
      clip_scores = clip_step(clip_enc.get_clip_score, Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())

      clip_scores_np = (clip_scores * Tensor.eye(GLOBAL_BS)).sum(axis=-1).numpy()
      all_clip_scores += clip_scores_np.tolist()

      x_pil = Tensor.cat(*[Tensor(np.asarray(im), dtype=dtypes.float16).div(255.0).permute(2,0,1).unsqueeze(0) for im in pil_im], dim=0)
      incp_act = inception(x_pil.realize())

      all_incp_act.append(incp_act.reshape(incp_act.shape[:2]).realize())
      if len(all_incp_act) >= MAX_INCP_STORE_SIZE:
        all_incp_act = [Tensor.cat(*all_incp_act, dim=0)]

    # Print Progress
    dataset_i += GLOBAL_BS
    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}, clip: {clip_scores_np.mean():.4f}, " + ", ".join(timings))

  print("\n" + "="*80 + "\n")

  # Print Final CLIP Score
  print(f"clip_score: {sum(all_clip_scores) / len(all_clip_scores)}")

  # Compute Final FID Score
  final_incp_acts = Tensor.cat(*all_incp_act, dim=0)
  fid_score = inception.compute_score(final_incp_acts, "/raid/datasets/coco2014/val2014_30k_stats.npz")
  print(f"fid_score:  {fid_score}")

  print("")


if __name__ == "__main__":
  func_map = {
    "gen":  gen_images,
    "fid":  compute_fid,
    "clip": compute_clip,
    "all":  do_all,
  }

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('func', default="all", choices=list(func_map.keys()))
  args = parser.parse_args()

  func = func_map.get(args.func, None)
  assert func is not None
  func()
