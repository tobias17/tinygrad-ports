from tinygrad import Tensor, dtypes, Device, TinyJit, GlobalCounters # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict, torch_load
from tinygrad.helpers import trange
from examples.sdxl import SDXL, DPMPP2MSampler, SplitVanillaCFG, configs # type: ignore
from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
from extra.models.inception import FidInceptionV3 # type: ignore

from typing import Tuple, List
from threading import Thread
import pandas as pd # type: ignore
import numpy as np
from scipy import linalg # type: ignore
from PIL import Image
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
NUM_STEPS = 4

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



@TinyJit
def inception_step(model:FidInceptionV3, x:Tensor) -> Tensor:
  return model(x).realize()

def compute_fid():
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
  DEVICE_BS = 50
  GLOBAL_BS = DEVICE_BS * len(GPUS)
  TEST_SIZE = 30_000
  Tensor.no_grad = True

  inception = FidInceptionV3().load_from_pretrained()
  for w in get_state_dict(inception).values():
    w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()
  
  wall_time_start = time.time()
  inc_act = []

  dataset_i = 0
  assert TEST_SIZE % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide TEST_SIZE ({TEST_SIZE}) for now"
  while dataset_i < TEST_SIZE:
    images = []
    for image_i in trange(GLOBAL_BS):
      im = Image.open(f"inputs/gen_{dataset_i+image_i:05d}.png")
      images.append(Tensor(np.asarray(im), dtype=dtypes.float16).div(255.0).permute(2,0,1).unsqueeze(0))

    x = Tensor.cat(*images, dim=0).shard(GPUS, axis=0)

    inc_out = inception_step(inception, x.realize())
    inc_act.append(inc_out.squeeze(3).squeeze(2).to(Device.DEFAULT).realize())

    if len(inc_act) > 20:
      inc_act = [Tensor.cat(*inc_act, dim=0).realize()]

    dataset_i += GLOBAL_BS

    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/TEST_SIZE)) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/TEST_SIZE:02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}")

  inception_act = Tensor.cat(*inc_act, dim=0)
  fid_score = inception.compute_score(inception_act)
  print(f"fid_score:  {fid_score}")








##################################################################
# TODO: upstream
FidInceptionV3.m1 = None
FidInceptionV3.s1 = None
def compute_score(self:FidInceptionV3, inception_activations:Tensor) -> float:
  if self.m1 is None or self.s1 is None:
    with np.load("/home/tiny/tinygrad/datasets/coco2014/val2014_30k_stats.npz") as f:
      self.m1, self.s1 = f['mu'][:], f['sigma'][:]
    assert self.m1 is not None and self.s1 is not None
  
  m2 = inception_activations.mean(axis=0).numpy()
  s2 = np.cov(inception_activations.numpy(), rowvar=False) # FIXME: need to figure out how to do in pure tinygrad

  return calculate_frechet_distance(self.m1, self.s1, m2, s2)
FidInceptionV3.compute_score = compute_score
#
def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)
  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)
  assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape

  diff = mu1 - mu2
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError(f"Imaginary component {m}")
    covmean = covmean.real
  
  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*tr_covmean
#
##################################################################




class Timing:
  def __init__(self, label:str, collection:List[str], print_fnx=(lambda l,d: f"{l}: {1e3*d:.2f} ms")):
    self.label = label
    self.collection = collection
    self.print_fnx = print_fnx
  def __enter__(self):
    self.start_time = time.time()
  def __exit__(self, *_):
    self.collection.append(self.print_fnx(self.label, (time.time() - self.start_time)))

def do_all():
  Tensor.manual_seed(42)

  GPUS = [f"{Device.DEFAULT}:{i}" for i in [1]]
  DEVICE_BS = 1
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  MAX_INCP_STORE_SIZE = 20
  SAVE_IMAGES = True
  SAVE_ROOT = "./output"
  if SAVE_IMAGES and not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)

  # Load generation model
  model = SDXL(configs["SDXL_Base"])
  load_state_dict(model, safe_load("/home/tiny/tinygrad/weights/sd_xl_base_1.0.safetensors"), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

  # Load evaluation model
  clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
  load_state_dict(clip_enc, torch_load("/home/tiny/weights_cache/tinygrad/downloads/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin"), strict=False)
  tokenizer = Tokenizer.ClipTokenizer()
  inception = FidInceptionV3().load_from_pretrained()

  # Create sampler
  sampler = DPMPP2MSampler(GUIDANCE_SCALE, guider_cls=SplitVanillaCFG)

  # Load dataset
  df = pd.read_csv("/home/tiny/tinygrad/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)
  captions = df["caption"].array

  wall_time_start = time.time()
  all_clip_scores = []
  all_incp_actv   = []

  # @TinyJit
  def decode_step(z:Tensor) -> Tensor:
    print(f"Pre-decode: {GlobalCounters.global_mem/1e9:.3f}")
    return model.decode(z).realize()

  @TinyJit
  def evaluation_step(tokens:Tensor, images:Tensor, x:Tensor) -> Tuple[Tensor,Tensor]:
    return clip_enc.get_clip_score(tokens, images).realize(), inception(x).realize()

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < len(captions):
    timings = []

    # Generate Image
    with Timing("gen", timings):
      texts = captions[dataset_i:dataset_i+GLOBAL_BS].tolist()
      c, uc = model.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
      for t in  c.values(): t.shard_(GPUS, axis=0)
      for t in uc.values(): t.shard_(GPUS, axis=0)
      randn = Tensor.randn(GLOBAL_BS, 4, LATENT_SIZE, LATENT_SIZE).shard(GPUS, axis=0)
      z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
      x = decode_step(z.realize())
      x = (x + 1.0) / 2.0
      x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).realize()
      ten_im = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8).to(Device.DEFAULT).numpy()
      pil_im = [Image.fromarray(ten_im[image_i]) for image_i in range(GLOBAL_BS)]

    # Save Images
    if SAVE_IMAGES:
      with Timing("save", timings):
        for idx, im in enumerate(pil_im):
          im.save(os.path.join(SAVE_ROOT, f"gen_{dataset_i+idx:05d}.png"))

    # Evaluate CLIP Score and Inception Activations
    with Timing("eval", timings):
      images = [clip_enc.prepare_image(im).unsqueeze(0) for im in pil_im]
      tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64).reshape(1,-1) for text in texts]
      clip_scores, incp_act = evaluation_step(Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize(), x.to(Device.DEFAULT).realize())

      clip_scores_np = (clip_scores * Tensor.eye(GLOBAL_BS)).sum(axis=-1).numpy()
      all_clip_scores += clip_scores_np.tolist()

      all_incp_actv.append(incp_act.squeeze(3).squeeze(2))
      if len(all_incp_actv) >= MAX_INCP_STORE_SIZE:
        all_incp_actv = [Tensor.cat(*all_incp_actv)]

    # Print Progress
    dataset_i += GLOBAL_BS
    wall_time_delta = time.time() - wall_time_start
    eta_time = (wall_time_delta / (dataset_i/len(captions))) - wall_time_delta
    print(f"{dataset_i:05d}: {100.0*dataset_i/len(captions):02.2f}%, elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}, clip: {clip_scores_np.mean():.4f}, " + ", ".join(timings))

  print("\n" + "="*80 + "\n")

  # Print Final CLIP Score
  print(f"clip_score: {sum(all_clip_scores) / len(all_clip_scores)}")

  # Compute Final FID Score
  final_incp_acts = Tensor.cat(*all_incp_actv, dim=0)
  fid_score = inception.compute_score(final_incp_acts)
  print(f"fid_score:  {fid_score}")

  print("")


if __name__ == "__main__":
  do_all()
