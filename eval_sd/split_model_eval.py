import time, os
start = time.perf_counter()
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, dtypes, GlobalCounters, TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, torch_load
from tinygrad.helpers import getenv, fetch, Context, BEAM, tqdm
def tlog(x): print(f"{x:25s}  @ {time.perf_counter()-start:5.2f}s")

out_folder = Path("./output/rendered")
if not out_folder.exists():
  os.makedirs(str(out_folder))

# Imports and Settings
import pandas as pd # type: ignore
from PIL import Image
from examples.sdxl import SDXL, DPMPP2MSampler, configs, SplitVanillaCFG # type: ignore
# from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
# from extra.models.inception import FidInceptionV3, compute_mu_and_sigma, calculate_frechet_distance # type: ignore
GPUS       = tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 6)))
CLIP_GPU   = GPUS[0]
INCP_GPUS  = GPUS if len(GPUS) == 1 else tuple(GPUS[1:])
CFG_SCALE  = getenv("CFG_SCALE",  8.0)
IMG_SIZE   = getenv("IMG_SIZE",   1024)
NUM_STEPS  = getenv("NUM_STEPS",  20)
DEV_GEN_BS = getenv("DEV_GEN_BS", 7)
DEV_EVL_BS = getenv("DEV_EVL_BS", 16)
GEN_BEAM   = getenv("GEN_BEAM",   getenv("BEAM", 1))
EVL_BEAM   = getenv("EVL_BEAM",   getenv("BEAM", 0))
DATASET    = "COCO/coco2014_5k"
GBL_GEN_BS = DEV_GEN_BS * len(GPUS)
GBL_EVL_BS = DEV_EVL_BS * len(INCP_GPUS)
BEAM.value = 0
LAT_SCALE  = 8
LAT_SIZE   = IMG_SIZE // LAT_SCALE
assert LAT_SIZE * LAT_SCALE == IMG_SIZE

def generate():
  # Configure Models and Load Weights
  mdl = SDXL(configs["SDXL_Base"])
  url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
  load_state_dict(mdl, safe_load(str(fetch(url, "sd_xl_base_1.0.safetensors"))), strict=False)
  for k,w in get_state_dict(mdl).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k == "sigmas":
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

  captions = pd.read_csv(f"{DATASET}/captions.tsv", sep='\t', header=0)
  sampler  = DPMPP2MSampler(CFG_SCALE, guider_cls=SplitVanillaCFG)

  @TinyJit
  def chunk_batches(z:Tensor):
    return [b.shard(GPUS, axis=0).realize() for b in z.to(GPUS[0]).chunk(DEV_GEN_BS)]
  @TinyJit 
  def decode_step(z:Tensor) -> Tensor:
    x = mdl.decode(z)
    x = (x + 1.0) / 2.0
    x = x.reshape(z.shape[0],3,IMG_SIZE,IMG_SIZE)
    x = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    return x.realize()

  def gen_batch(texts):
    c, uc = mdl.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
    for t in  c.values(): t.shard_(GPUS, axis=0)
    for t in uc.values(): t.shard_(GPUS, axis=0)
    randn = Tensor.randn(GBL_GEN_BS, 4, LAT_SIZE, LAT_SIZE).shard(GPUS, axis=0)
    pt = time.perf_counter()
    with Context(BEAM=GEN_BEAM):
      z = sampler(mdl.denoise, randn, c, uc, NUM_STEPS).realize()
      pil_im = []
      for b_in in chunk_batches(z.realize()):
        b_np = decode_step(b_in).numpy()
        pil_im += [Image.fromarray(b_np[image_i]) for image_i in range(len(GPUS))]
    return pil_im, pt

  print("\nFull Run")
  for dataset_i in range(0, len(captions), GBL_GEN_BS):
    padding = 0 if (dataset_i+GBL_GEN_BS <= len(captions)) else (dataset_i+GBL_GEN_BS) - len(captions)

    ds_slice = slice(dataset_i, dataset_i+GBL_GEN_BS)
    texts = captions["caption"].array[ds_slice].tolist()
    if padding > 0: texts += ["" for _ in range(padding)]
    pil_im, pt = gen_batch(texts)

    for frame_i, im in enumerate(pil_im):
      im.save(str(out_folder/f"frame_{dataset_i+frame_i}.png"))
    gt = time.perf_counter()

    curr_i = min(dataset_i+GBL_GEN_BS, len(captions))
    print(f"{curr_i:04d}: {100.0*curr_i/len(captions):02.2f}%, {(gt-st)*1000:.0f} ms step ({(pt-st)*1000:.0f} prep, {(gt-pt)*1000:.0f} gen)")
    st = gt


# def evaluate():
#   print("\nEvaluating")

#   # Load Evaluation Models
#   tokenizer = Tokenizer.ClipTokenizer()
#   clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
#   url = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin"
#   load_state_dict(clip_enc, torch_load(str(fetch(url, "CLIP-ViT-H-14-laion2B-s32B-b79K.bin"))), strict=False)
#   for w in get_parameters(clip_enc): w.replace(w.cast(dtypes.float16).to(CLIP_GPU)).realize()
#   inception = FidInceptionV3().load_from_pretrained()
#   for w in get_parameters(inception): w.replace(w.cast(dtypes.float16).shard(INCP_GPUS)).realize()

#   @TinyJit
#   def clip_step(tokens:Tensor, images:Tensor):
#     return clip_enc.get_clip_score(tokens, images).realize()
#   def load_incp_img(im:Image.Image) -> Tensor:
#     x = Tensor(np.array(im)).cast(dtypes.float16).div(255.0)
#     if x.ndim == 2: x = x.unsqueeze(-1).expand(*x.shape, 3)
#     return x.permute(2,0,1).interpolate((299,299), mode='linear')

#   all_clip_scores = []
#   all_incp_acts_1 = []
#   all_incp_acts_2 = []

#   tracker = tqdm(total=len(captions))
#   while len(gens.imgs) > 0:
#     imgs, texts, fns, padding = gens.slice_batch(GBL_EVL_BS)

#     # Evaluate Images
#     tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64, device=CLIP_GPU) for text in texts]
#     images = [clip_enc.prepare_image(im) for im in imgs]
#     incp_imgs = [Image.open(f"{DATASET}/calibration/{fn}") for fn in fns] + imgs
#     incp_xs   = [load_incp_img(im) for im in incp_imgs]
#     with Context(BEAM=EVL_BEAM):
#       clip_scores = clip_step(Tensor.stack(*tokens, dim=0).realize(), Tensor.stack(*images, dim=0).realize())
#       incp_act = inception(Tensor.stack(*incp_xs, dim=0).shard(INCP_GPUS, axis=0).realize()).to(INCP_GPUS[0])

#     pad_slice = slice(None, None if padding == 0 else -padding)
#     all_clip_scores += (clip_scores * Tensor.eye(GBL_EVL_BS, device=CLIP_GPU)).sum(axis=-1)[pad_slice].tolist()
#     incp_act_1, incp_act_2 = incp_act.chunk(2)
#     all_incp_acts_1.append(incp_act_1.squeeze()[pad_slice].realize())
#     all_incp_acts_2.append(incp_act_2.squeeze()[pad_slice].realize())

#     tracker.update(GBL_EVL_BS - padding)

#   # Final Score Computation
#   m1, s1 = compute_mu_and_sigma(Tensor.cat(*all_incp_acts_1, dim=0).realize())
#   m2, s2 = compute_mu_and_sigma(Tensor.cat(*all_incp_acts_2, dim=0).realize())
#   fid_score = calculate_frechet_distance(m1, s1, m2, s2)

#   timings.append(("Evaluate", time.perf_counter() - eval_start))

#   print(f"\n clip_score: {sum(all_clip_scores) / len(all_clip_scores):.5f}")
#   print(f" fid_score:  {fid_score:.4f}")


if __name__ == "__main__":
  # inference only
  Tensor.training = False
  Tensor.no_grad = True

  generate()
