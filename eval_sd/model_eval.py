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
  from examples.sdxl import SDXL, DPMPP2MSampler, Guider, configs, append_dims, SplitVanillaCFG # type: ignore
  from extra.models.clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
  from extra.models.inception import FidInceptionV3 # type: ignore
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 6))]
  GEN_GPUS   = GPUS[:-1] if len(GPUS) > 1 else GPUS
  EVAL_GPU   = GPUS[-1]
  CFG_SCALE  = getenv("CFG_SCALE", 8.0)
  IMG_SIZE   = getenv("IMG_SIZE",  1024)
  NUM_STEPS  = getenv("NUM_STEPS", 20)
  DEVICE_BS  = getenv("DEVICE_BS", 8)
  BEAM_VALUE = getenv("BEAM",      1)
  BEAM.value = 0
  GLOBAL_BS  = DEVICE_BS * GEN_GPUS
  LAT_SCALE  = 8
  LAT_SIZE   = IMG_SIZE // LAT_SCALE
  assert LAT_SIZE * LAT_SCALE == IMG_SIZE
  MAX_INCP_STORE_SIZE = 32

  # Configure Models and Load Weights
  mdl = SDXL(configs["SDXL_Base"])
  load_state_dict(mdl, safe_load(fetch("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "sd_xl_base_1.0.safetensors")), strict=False)
  for k,w in get_state_dict(mdl).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16).shard(GEN_GPUS, axis=None)).realize()
  tokenizer = Tokenizer.ClipTokenizer()
  clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
  weights_path = fetch("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin", "CLIP-ViT-H-14-laion2B-s32B-b79K.bin")
  load_state_dict(clip_enc, torch_load(weights_path), strict=False)
  for w in get_parameters(clip_enc): w.replace(w.cast(dtypes.float16).to(EVAL_GPU)).realize()
  inception = FidInceptionV3().load_from_pretrained()
  for w in get_parameters(inception): w.replace(w.cast(dtypes.float16).to(EVAL_GPU)).realize()

  # Loop Objects
  sampler = DPMPP2MSampler(CFG_SCALE, guider_cls=SplitVanillaCFG)
  captions = pd.read_csv("/raid/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)["caption"].array

  @TinyJit
  def chunk_batches(z:Tensor):
    return [b.shard(GEN_GPUS, axis=0).realize() for b in z.to(EVAL_GPU).chunk(GEN_GPUS)]
  @TinyJit
  def decode_step(z:Tensor):
    x = mdl.decode(z)
    x = (x + 1.0) / 2.0
    x = x.reshape(z.shape[0],3,IMG_SIZE,IMG_SIZE)
    inc_x = x.to(EVAL_GPU)
    x = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    return x.realize(), inc_x.realize()
  @TinyJit
  def clip_step(tokens, images):
    return clip_enc(tokens, images).realize()

  all_clip_scores = []
  all_incp_act    = []

  for dataset_i in range(0, len(captions), GLOBAL_BS):
    padding = 0 if (dataset_i+GLOBAL_BS <= len(captions)) else (dataset_i+GLOBAL_BS) - len(captions)

    # Generate Images
    texts = captions[dataset_i:dataset_i+GLOBAL_BS].tolist()
    if padding > 0: texts += ["" for _ in range(padding)]
    c, uc = mdl.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
    for t in  c.values(): t.shard_(GEN_GPUS, axis=0)
    for t in uc.values(): t.shard_(GEN_GPUS, axis=0)
    randn = Tensor.randn(GLOBAL_BS, 4, LAT_SIZE, LAT_SIZE).shard(GPUS, axis=0)
    with Context(BEAM=BEAM_VALUE):
      z = sampler(mdl.denoise, randn, c, uc, NUM_STEPS).realize()
      pil_im, xs = [], []
      for b_in in chunk_batches(z.realize()):
        b_im, b_x = decode_step(b_in)
        xs.append(b_x)
        b_np = b_im.numpy()
        pil_im += [Image.fromarray(b_np[image_i]) for image_i in range(len(GPUS))]
    
    # Evaluate Images
    tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64, device=EVAL_GPU).reshape(1,-1) for text in texts]
    images = [clip_enc.prepare_image(im).unsqueeze(0).to(EVAL_GPU) for im in pil_im]
    with Context(BEAM=BEAM_VALUE):
      x_incp = Tensor.cat(*xs, dim=0)
      clip_scores = clip_step(clip_enc.get_clip_score, Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
      incp_act = inception(x_incp.realize())
    
    clip_scores_np = (clip_scores * Tensor.eye(GLOBAL_BS, device=EVAL_GPU)).sum(axis=-1).numpy()
    if padding > 0: clip_scores_np = clip_scores_np[:-padding]
    all_clip_scores += clip_scores_np.tolist()

    if padding > 0: incp_act = incp_act[:-padding]
    all_incp_act.append(incp_act.reshape(incp_act.shape[:2]).realize())
    if len(all_incp_act) >= MAX_INCP_STORE_SIZE:
      all_incp_act = [Tensor.cat(*all_incp_act, dim=0).realize()]

  print("\n" + "="*80 + "\n")
  print(f"clip_score: {sum(all_clip_scores) / len(all_clip_scores)}")
  final_incp_acts = Tensor.cat(*all_incp_act, dim=0)
  fid_score = inception.compute_score(final_incp_acts, "/raid/datasets/coco2014/val2014_30k_stats.npz")
  print(f"fid_score:  {fid_score}")
  print("")
