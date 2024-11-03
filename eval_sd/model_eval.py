import time
start = time.perf_counter()
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, dtypes, GlobalCounters, TinyJit
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, get_state_dict, torch_load
from tinygrad.helpers import getenv
def tlog(x): print(f"{x:25s}  @ {time.perf_counter()-start:5.2f}s")

def eval_sd():
  # Imports and Settings
  Tensor.no_grad = True
  import pandas as pd # type: ignore
  from PIL import Image
  from tinygrad.helpers import fetch, Context, BEAM
  from examples.sdxl import SDXL, DPMPP2MSampler, configs, SplitVanillaCFG # type: ignore
  from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
  from extra.models.inception import FidInceptionV3 # type: ignore
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 6))]
  INCP_GPU   = GPUS[1 % len(GPUS)]
  CLIP_GPU   = GPUS[2 % len(GPUS)]
  CFG_SCALE  = getenv("CFG_SCALE", 8.0)
  IMG_SIZE   = getenv("IMG_SIZE",  1024)
  NUM_STEPS  = getenv("NUM_STEPS", 20)
  DEVICE_BS  = getenv("DEVICE_BS", 7)
  BEAM_VALUE = getenv("BEAM",      1)
  BEAM.value = 0
  GLOBAL_BS  = DEVICE_BS * len(GPUS)
  LAT_SCALE  = 8
  LAT_SIZE   = IMG_SIZE // LAT_SCALE
  assert LAT_SIZE * LAT_SCALE == IMG_SIZE
  MAX_INCP_STORE_SIZE = 32

  # Configure Models and Load Weights
  mdl = SDXL(configs["SDXL_Base"])
  load_state_dict(mdl, safe_load(fetch("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "sd_xl_base_1.0.safetensors")), strict=False)
  for k,w in get_state_dict(mdl).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k == "sigmas":
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()
  tokenizer = Tokenizer.ClipTokenizer()
  clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
  weights_path = fetch("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin", "CLIP-ViT-H-14-laion2B-s32B-b79K.bin")
  load_state_dict(clip_enc, torch_load(weights_path), strict=False)
  for w in get_parameters(clip_enc): w.replace(w.cast(dtypes.float16).to(CLIP_GPU)).realize()
  inception = FidInceptionV3().load_from_pretrained()
  for w in get_parameters(inception): w.replace(w.cast(dtypes.float16).to(INCP_GPU)).realize()

  sampler  = DPMPP2MSampler(CFG_SCALE, guider_cls=SplitVanillaCFG)
  captions = pd.read_csv("/raid/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)["caption"].array
  all_clip_scores = []
  all_incp_act    = []

  @TinyJit
  def chunk_batches(z:Tensor):
    return [b.shard(GPUS, axis=0).realize() for b in z.to(INCP_GPU).chunk(DEVICE_BS)]
  @TinyJit 
  def decode_step(z:Tensor):
    x = mdl.decode(z)
    x = (x + 1.0) / 2.0
    x = x.reshape(z.shape[0],3,IMG_SIZE,IMG_SIZE)
    inc_x = x.to(INCP_GPU)
    x = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    return x.realize(), inc_x.realize()
  @TinyJit
  def clip_step(tokens:Tensor, images:Tensor):
    return clip_enc.get_clip_score(tokens, images).realize()

  st = time.perf_counter()
  for dataset_i in range(0, len(captions), GLOBAL_BS):
    padding = 0 if (dataset_i+GLOBAL_BS <= len(captions)) else (dataset_i+GLOBAL_BS) - len(captions)

    # Prepare Inputs
    texts = captions[dataset_i:dataset_i+GLOBAL_BS].tolist()
    if padding > 0: texts += ["" for _ in range(padding)]
    c, uc = mdl.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
    for t in  c.values(): t.shard_(GPUS, axis=0)
    for t in uc.values(): t.shard_(GPUS, axis=0)
    randn = Tensor.randn(GLOBAL_BS, 4, LAT_SIZE, LAT_SIZE).shard(GPUS, axis=0)
    pt = time.perf_counter()

    # Generate Images
    with Context(BEAM=BEAM_VALUE):
      z = sampler(mdl.denoise, randn, c, uc, NUM_STEPS).realize()
      pil_im, xs = [], []
      for b_in in chunk_batches(z.realize()):
        b_im, b_x = decode_step(b_in)
        xs.append(b_x)
        b_np = b_im.numpy()
        pil_im += [Image.fromarray(b_np[image_i]) for image_i in range(len(GPUS))]
    gt = time.perf_counter()

    # Evaluate Images
    tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64, device=CLIP_GPU).reshape(1,-1) for text in texts]
    images = [clip_enc.prepare_image(im).unsqueeze(0).to(CLIP_GPU) for im in pil_im]
    with Context(BEAM=BEAM_VALUE):
      x_incp = Tensor.cat(*xs, dim=0)
      clip_scores = clip_step(Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
      incp_act = inception(x_incp.realize())

    clip_scores_np = (clip_scores * Tensor.eye(GLOBAL_BS, device=CLIP_GPU)).sum(axis=-1).numpy()
    if padding > 0: clip_scores_np = clip_scores_np[:-padding]
    all_clip_scores += clip_scores_np.tolist()

    if padding > 0: incp_act = incp_act[:-padding]
    all_incp_act.append(incp_act.reshape(incp_act.shape[:2]).realize())
    if len(all_incp_act) >= MAX_INCP_STORE_SIZE:
      all_incp_act = [Tensor.cat(*all_incp_act, dim=0).realize()]
    et = time.perf_counter()

    curr_i = min(dataset_i+GLOBAL_BS, len(captions))
    print(f"{curr_i:05d}: {100.0*curr_i/len(captions):02.2f}%, {(et-st)*1000:.0f} ms step ({(pt-st)*1000:.0f} prep, {(gt-pt)*1000:.0f} gen, {(et-gt)*1000:.0f} eval), {clip_scores_np.mean():.4f} clip score")
    st = et

  # Final Score Computation
  print("\n" + "="*80 + "\n")
  print(f"clip_score: {sum(all_clip_scores) / len(all_clip_scores):.5f}")
  final_incp_acts = Tensor.cat(*all_incp_act, dim=0)
  fid_score = inception.compute_score(final_incp_acts, "/raid/datasets/coco2014/val2014_30k_stats.npz")
  print(f"fid_score:  {fid_score:.4f}")
  print(f"exec_time:  {(time.perf_counter()-start)/3600:.3f} hours")
  print("")

if __name__ == "__main__":
  eval_sd()
