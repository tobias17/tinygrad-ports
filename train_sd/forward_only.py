from tinygrad import Tensor, dtypes, TinyJit, Device # type: ignore
from tinygrad.helpers import BEAM # type: ignore
from tqdm import tqdm # type: ignore

import time, os, datetime, math
from typing import Dict, Tuple
import numpy as np

from tinygrad.nn.state import load_state_dict, torch_load, get_parameters, get_state_dict, safe_save # type: ignore

from extra.models.unet import UNetModel, timestep_embedding # type: ignore
from examples.sdv2 import params, FrozenOpenClipEmbedder, get_alphas_cumprod # type: ignore


if __name__ == "__main__":
  seed = 42
  Tensor.manual_seed(seed)

  __OUTPUT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
  def get_output_root(*folders:str) -> str:
    dirpath = os.path.join(__OUTPUT_ROOT, *folders)
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    return dirpath

  TRAIN_DTYPE = dtypes.float16

  # GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))]

  GPUS = [f'{Device.DEFAULT}:{i}' for i in [1,2,3,4,5]]
  DEVICE_BS = 24

  # GPUS = [f'{Device.DEFAULT}:{i}' for i in [4,5]]
  # DEVICE_BS = 1

  GLOBAL_BS = DEVICE_BS * len(GPUS)
  EVAL_EVERY = math.ceil(512000.0 / GLOBAL_BS)
  print(f"Configured to Eval every {EVAL_EVERY} steps")

  # Model, Conditioner, Other Variables

  class TokenEmbedModel:
    def __init__(self, cond_stage_config:Dict, **kwargs):
      self.cond_stage_model = FrozenOpenClipEmbedder(**cond_stage_config)
  token_model = TokenEmbedModel(**params)
  # for w in get_state_dict(token_model).values():
  #   w.to_(TOKENIZE_DEVICE)
  load_state_dict(token_model, torch_load("/home/tiny/tinygrad/weights/512-base-ema.ckpt")["state_dict"], strict=False)

  model = UNetModel(**params["unet_config"])
  for w in get_state_dict(model).values():
    w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

  alphas_cumprod      = get_alphas_cumprod()
  alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
  sqrt_alphas_cumprod = alphas_cumprod.sqrt()
  sqrt_on_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()


  # Dataset

  def collate_fnx(batch):
    arr = np.zeros((len(batch),8,64,64), dtype=np.float32)
    for i, e in enumerate(batch):
      arr[i,:,:,:] = e["npy"]
    return { "moments": Tensor(arr), "txt": [e["txt"] for e in batch] }

  def sample_moments(moments:Tensor) -> Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clip(-30.0, 20.0)
    std = Tensor.exp(logvar * 0.5)
    return mean + std * Tensor.rand(*mean.shape)

  from webdataset import WebDataset, WebLoader # type: ignore
  urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
  keep_only_keys = { "npy", "txt" }
  def filter_fnx(sample): return {k:v for k,v in sample.items() if k in keep_only_keys}
  dataset = WebDataset(urls=urls, resampled=True, cache_size=-1, cache_dir=None)
  dataset = dataset.shuffle(size=1000).decode().map(filter_fnx).batched(GLOBAL_BS, partial=False, collation_fn=collate_fnx)
  dataloader = WebLoader(dataset, batch_size=None, shuffle=False, num_workers=1, persistent_workers=True)


  # Train Funcs and Utils

  @TinyJit
  def train_step(x:Tensor, x_noisy:Tensor, t_emb:Tensor, c:Tensor) -> Tensor:
    Tensor.training = True

    output = model(x_noisy, t_emb, c)

    loss = (x - output).square().mean()
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    return loss.realize()

  @TinyJit
  def tokenize_step(tokens:Tensor) -> Tensor:
    return token_model.cond_stage_model.embed_tokens(tokens).realize()

  def prep_for_jit(*inputs:Tensor) -> Tuple[Tensor,...]:
    return tuple(i.cast(TRAIN_DTYPE).shard(GPUS, axis=0).realize() for i in inputs)

  MAX_QUEUE_SIZE = 10

  MAX_ITERS = 50000

  BEAM_VAL = BEAM.value
  BEAM.value = 0

  # Main Train Loop

  for i, entry in enumerate(dataloader):
    if i >= MAX_ITERS:
      break

    st = time.perf_counter()

    c = tokenize_step(Tensor.cat(*[token_model.cond_stage_model.tokenize(t) for t in entry["txt"]]))
    x = (sample_moments(entry["moments"]) * 0.18215)
    t = Tensor.randint(x.shape[0], low=0, high=1000)
    noise = Tensor.randn(x.shape)
    x_noisy =   sqrt_alphas_cumprod         [t].reshape(GLOBAL_BS, 1, 1, 1) * x \
              + sqrt_on_minus_alphas_cumprod[t].reshape(GLOBAL_BS, 1, 1, 1) * noise
    t_emb = timestep_embedding(t, 320).cast(TRAIN_DTYPE)
    inputs = prep_for_jit(x, x_noisy, t_emb, c)

    pt = time.perf_counter()

    BEAM.value = BEAM_VAL
    loss = train_step(*inputs).numpy().item()
    BEAM.value = 0

    et = time.perf_counter()
    tqdm.write(f"{i:05d}: {(et-st)*1000.0:6.0f} ms run, {(pt-st)*1000.0:6.0f} ms prep, {(et-pt)*1000.0:6.0f} ms step, {loss:>2.5f} train loss")
