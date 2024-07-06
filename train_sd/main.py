from tinygrad import Tensor, dtypes, TinyJit, Device # type: ignore
from tinygrad.nn.optim import AdamW, SGD # type: ignore
from tinygrad.helpers import getenv # type: ignore
# from tinygrad.helpers import tqdm
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt

from typing import Dict, Tuple
import numpy as np
import time, os, datetime
import torch

from tinygrad.nn.state import load_state_dict, torch_load, get_parameters, get_state_dict # type: ignore

from extra.models.unet import UNetModel, timestep_embedding # type: ignore
from examples.sdv2 import params, FrozenOpenClipEmbedder, get_alphas_cumprod # type: ignore


# TODO:
# - Figure out AdamW
# - Investigate the cond_stage_model
#   - Use a GPU for pre-processing?
#   - Use the CPU for this?


if __name__ == "__main__":
  seed = 42
  Tensor.manual_seed(seed)
  torch.manual_seed(seed)

  __OUTPUT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
  def get_output_root() -> str:
    if not os.path.exists(__OUTPUT_ROOT):
      os.mkdir(__OUTPUT_ROOT)
    return __OUTPUT_ROOT

  TRAIN_DTYPE = dtypes.float16

  # GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))]
  GPUS = [f'{Device.DEFAULT}:{i}' for i in [1,2,3,4,5]]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)


  # Model, Conditioner, Other Variables

  class WrapperModel:
    def __init__(self, cond_stage_config:Dict, **kwargs):
      self.cond_stage_model = FrozenOpenClipEmbedder(**cond_stage_config)
  wrapper = WrapperModel(**params)
  load_state_dict(wrapper, torch_load("/home/tiny/tinygrad/weights/512-base-ema.ckpt")["state_dict"])
  # for w in get_state_dict(wrapper).values():
  #   w.to(WRAPPER_DEVICE)

  model = UNetModel(**params["unet_config"])
  for w in get_state_dict(model).values():
    w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, b1=0.9, b2=0.999, weight_decay=0.01)
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, eps=1.0)
  optimizer = SGD(get_parameters(model), lr=1.25e-7 *50)

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

  from webdataset import WebDataset, WebLoader # type: ignore
  urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
  keep_only_keys = { "npy", "txt" }
  def filter_fnx(sample): return {k:v for k,v in sample.items() if k in keep_only_keys}
  dataset = WebDataset(urls=urls, resampled=True, cache_size=-1, cache_dir=None)
  dataset = dataset.shuffle(size=1000).decode().map(filter_fnx).batched(GLOBAL_BS, partial=False, collation_fn=collate_fnx)
  dataloader = WebLoader(dataset, batch_size=None, shuffle=False, num_workers=4, persistent_workers=True)

  def sample_moments(moments:Tensor) -> Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clip(-30.0, 20.0)
    std = Tensor.exp(logvar * 0.5)
    return mean + std * Tensor.rand(*mean.shape)

  # Train Funcs and Utils

  @TinyJit
  def train_step(x:Tensor, c:Tensor, x_noisy:Tensor, t_emb:Tensor) -> Tensor:
    Tensor.training = True
    
    output = model(x_noisy, t_emb, c)

    loss = (x.cast(dtypes.float32) - output.cast(dtypes.float32)).square().mean()
    optimizer.zero_grad()
    loss.backward()
    lt = time.perf_counter()

    optimizer.step()
    ot = time.perf_counter()

    return loss.realize(), lt, ot

  def prep_for_jit(*inputs:Tensor) -> Tuple[Tensor,...]:
    return tuple(i.cast(TRAIN_DTYPE).shard(GPUS, axis=0).realize() for i in inputs)

  MAX_ITERATIONS = 50000
  MEAN_EVERY = 10
  GRAPH_EVERY = 100
  losses = []
  saved_losses = []


  # Main Train Loop

  for i, entry in enumerate(dataloader):
    if i >= MAX_ITERATIONS:
      break

    st = time.perf_counter()

    x = (sample_moments(entry["moments"]) * 0.18215)
    c = Tensor.cat(*[wrapper.cond_stage_model(t) for t in entry["txt"]], dim=0)
    t = Tensor.randint(x.shape[0], low=0, high=1000)
    noise = Tensor.randn(x.shape)
    x_noisy =   sqrt_alphas_cumprod         [t].reshape(GLOBAL_BS, 1, 1, 1) * x \
              + sqrt_on_minus_alphas_cumprod[t].reshape(GLOBAL_BS, 1, 1, 1) * noise
    t_emb = timestep_embedding(t, 320).cast(TRAIN_DTYPE)

    inputs = prep_for_jit(x, c, x_noisy, t_emb)
    pt = time.perf_counter()

    loss, lt, ot = train_step(*inputs)
    loss = loss.numpy().item()
    losses.append(loss)

    et = time.perf_counter()
    tqdm.write(f"{i:05d}: {(et-st)*1000.0:6.0f} ms run, {(pt-st)*1000.0:6.0f} ms prep, {(lt-pt)*1000.0:6.0f} ms loss, {(ot-lt)*1000.0:6.0f} ms step, {loss:>2.5f} train loss")

    if i > 0 and i % MEAN_EVERY == 0:
      saved_losses.append(sum(losses) / len(losses))
      losses = []
    if i > 0 and i % GRAPH_EVERY == 0:
      plt.clf()
      plt.plot(np.arange(len(saved_losses))*MEAN_EVERY, saved_losses)
      plt.savefig(os.path.join(get_output_root(), "loss"))
