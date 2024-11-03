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
  from extra.models.inception import FidInceptionV3, calculate_frechet_distance # type: ignore
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 2))]
  INCP_GPU   = GPUS[1 % len(GPUS)]
  CLIP_GPU   = GPUS[2 % len(GPUS)]
  CFG_SCALE  = getenv("CFG_SCALE",  8.0)
  IMG_SIZE   = getenv("IMG_SIZE",   1024)
  NUM_STEPS  = getenv("NUM_STEPS",  20)
  DEV_GEN_BS = getenv("DEV_GEN_BS", 3)
  DEV_EVL_BS = getenv("DEV_EVL_BS", 2)
  GEN_BEAM   = getenv("GEN_BEAM",   getenv("BEAM", 1))
  EVL_BEAM   = getenv("EVL_BEAM",   getenv("BEAM", 0))
  WARMUP     = getenv("WARMUP",     3)
  EVALUATE   = getenv("EVALUATE",   1)
  BEAM.value = 0
  GBL_GEN_BS = DEV_GEN_BS * len(GPUS)
  GBL_EVL_BS = DEV_EVL_BS
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

  sampler  = DPMPP2MSampler(CFG_SCALE, guider_cls=SplitVanillaCFG)
  captions = pd.read_csv("COCO/coco2014/captions.tsv", sep='\t', header=0)[:7]
  gens     = []

  @TinyJit 
  def decode_step(z:Tensor) -> Tensor:
    x = mdl.decode(z)
    x = (x + 1.0) / 2.0
    x = x.reshape(z.shape[0],3,IMG_SIZE,IMG_SIZE)
    x = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    return x.realize()
  @TinyJit
  def chunk_batches(z:Tensor):
    return [b.shard(GPUS, axis=0).realize() for b in z.to(INCP_GPU).chunk(DEV_GEN_BS)]

  def gen_batch(texts):
    # Prepare Inputs
    c, uc = mdl.create_conditioning(texts, IMG_SIZE, IMG_SIZE)
    for t in  c.values(): t.shard_(GPUS, axis=0)
    for t in uc.values(): t.shard_(GPUS, axis=0)
    randn = Tensor.randn(GBL_GEN_BS, 4, LAT_SIZE, LAT_SIZE).shard(GPUS, axis=0)
    pt = time.perf_counter()

    # Generate Images
    with Context(BEAM=GEN_BEAM):
      z = sampler(mdl.denoise, randn, c, uc, NUM_STEPS).realize()
      pil_im = []
      for b_in in chunk_batches(z.realize()):
        b_np = decode_step(b_in).numpy()
        pil_im += [Image.fromarray(b_np[image_i]) for image_i in range(len(GPUS))]
    return pil_im, pt

  print("\nWarming Up")
  for _ in range(WARMUP):
    gen_batch([""]*GBL_GEN_BS)

  print("\nFull Run")
  st = time.perf_counter()
  for dataset_i in range(0, len(captions), GBL_GEN_BS):
    padding = 0 if (dataset_i+GBL_GEN_BS <= len(captions)) else (dataset_i+GBL_GEN_BS) - len(captions)

    ds_slice = slice(dataset_i, dataset_i+GBL_GEN_BS)
    texts = captions["caption"].array[ds_slice].tolist()
    if padding > 0: texts += ["" for _ in range(padding)]
    pil_im, pt = gen_batch(texts)

    pad_slice = slice(None, None if padding == 0 else -padding)
    values = [pil_im[pad_slice], texts[pad_slice], captions["file_name"].array[ds_slice].tolist()]
    assert len(values[0]) == len(values[1]) and len(values[0]) == len(values[2])
    gens.append(zip(*values))
    gt = time.perf_counter()

    curr_i = min(dataset_i+GBL_GEN_BS, len(captions))
    print(f"{curr_i:05d}: {100.0*curr_i/len(captions):02.2f}%, {(gt-st)*1000:.0f} ms step ({(pt-st)*1000:.0f} prep, {(gt-pt)*1000:.0f} gen)")
    st = gt

  decode_step.reset()
  chunk_batches.reset()

  # Evaluation
  if EVALUATE > 0:
    # Load Evaluation Data
    tokenizer = Tokenizer.ClipTokenizer()
    clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"])
    weights_path = fetch("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/open_clip_pytorch_model.bin", "CLIP-ViT-H-14-laion2B-s32B-b79K.bin")
    load_state_dict(clip_enc, torch_load(weights_path), strict=False)
    for w in get_parameters(clip_enc): w.replace(w.cast(dtypes.float16).to(CLIP_GPU)).realize()
    inception = FidInceptionV3().load_from_pretrained()
    for w in get_parameters(inception): w.replace(w.cast(dtypes.float16).to(INCP_GPU)).realize()

    @TinyJit
    def clip_step(tokens:Tensor, images:Tensor):
      return clip_enc.get_clip_score(tokens, images).realize()
    def create_batches():
      imgs, texts, fns = [], [], []
      for gen_batch in gens:
        for img, text, fn in gen_batch:
          imgs.append(img)
          texts.append(text)
          fns.append(fn)
          if len(imgs) == GBL_EVL_BS:
            yield imgs, texts, fns, 0
            imgs, texts, fns = [], [], []
      if len(imgs) > 0:
        yield imgs, texts, fns, GBL_EVL_BS - len(imgs)
      yield None
    batch_iter = create_batches()
    all_clip_scores = []
    all_incp_acts   = [[], []]

    while True:
      # Prepare Inputs
      batch = next(batch_iter)
      if batch is None:
        break
      imgs, texts, fns, padding = batch
      if padding > 0:
        imgs  += [imgs [-1]]*padding
        texts += [texts[-1]]*padding
        fns   += [fns  [-1]]*padding
      print(f"{len(imgs)=}")
      print(f"{len(texts)=}")
      print(f"{len(fns)=}")
      print(f"{padding=}")

      # Evaluate Images
      tokens = [Tensor(tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64, device=CLIP_GPU) for text in texts]
      images = [clip_enc.prepare_image(im).to(CLIP_GPU) for im in imgs]
      incp_imgs = [Image.open(f"COCO/coco2014/calibration/{fn}") for fn in fns] + imgs
      incp_xs   = [Tensor(np.array(im), device=INCP_GPU).cast(dtypes.float16).div(255.0).permute(2,0,1).interpolate((299,299), mode='linear') for im in incp_imgs]
      with Context(BEAM=EVL_BEAM):
        clip_scores = clip_step(Tensor.stack(*tokens, dim=0).realize(), Tensor.stack(*images, dim=0).realize())
        incp_act = inception(Tensor.stack(*incp_xs, dim=0).realize())

      clip_scores_np = (clip_scores * Tensor.eye(GBL_EVL_BS, device=CLIP_GPU)).sum(axis=-1).numpy()
      if padding > 0: clip_scores_np = clip_scores_np[:-padding]
      all_clip_scores += clip_scores_np.tolist()

      for incp_act_z, all_act_z in zip(incp_act.chunk(2), all_incp_acts):
        if padding > 0: incp_act_z = incp_act_z[:-padding]
        all_act_z.append(incp_act_z.reshape(incp_act_z.shape[:2]).realize())
        if len(all_act_z) >= MAX_INCP_STORE_SIZE:
          all_act_z.clear()
          all_act_z += [Tensor.cat(*all_act_z, dim=0).realize()]

    # Final Score Computation
    all_incp_acts_1 = Tensor.cat(*all_incp_acts[0], dim=0).realize()
    m1 = all_incp_acts_1.mean(axis=0).numpy()
    s1 = np.cov(all_incp_acts_1.numpy(), rowvar=False)
    all_incp_acts_2 = Tensor.cat(*all_incp_acts[1], dim=0).realize()
    m2 = all_incp_acts_2.mean(axis=0).numpy()
    s2 = np.cov(all_incp_acts_2.numpy(), rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    print("\n" + "="*80 + "\n")
    print(f"clip_score: {sum(all_clip_scores) / len(all_clip_scores):.5f}")
    print(f"fid_score:  {fid_score:.4f}")
    print(f"exec_time:  {(time.perf_counter()-start)/3600:.3f} hours")
    print("")

if __name__ == "__main__":
  eval_sd()
