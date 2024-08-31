from tinygrad import Tensor, dtypes, Device, TinyJit # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict
from tinygrad.helpers import trange
from examples.sdxl import SDXL, DPMPP2MSampler, configs # type: ignore
from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
from inception import FidInceptionV3 # type: ignore

from threading import Thread
import pandas as pd # type: ignore
import numpy as np
from PIL import Image
import time



def smart_print(sec:float) -> str:
  if sec < 60: return f"{sec:.0f} sec"
  mins = sec / 60.0
  if mins < 60: return f"{mins:.1f} mins"
  hours = mins / 60.0
  return f"{hours:.1f} hours"



def gen_images():

  Tensor.manual_seed(42)

  # Define constants

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
  DEVICE_BS = 4
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  IMG_SIZE = 1024
  LATENT_SCALE = 8
  LATENT_SIZE = IMG_SIZE // LATENT_SCALE
  assert LATENT_SIZE * LATENT_SCALE == IMG_SIZE

  GUIDANCE_SCALE = 8.0
  NUM_STEPS = 20

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
    z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
    x = model.decode(z).realize()
    x = (x + 1.0) / 2.0
    x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    x = x.numpy()

    Thread(target=async_save, args=(x,dataset_i)).start()

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


if __name__ == "__main__":
  gen_images()
