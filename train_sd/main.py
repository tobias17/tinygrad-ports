from tinygrad import Tensor, dtypes, TinyJit, Device # type: ignore
from tinygrad.nn.optim import AdamW, SGD # type: ignore

from typing import Dict, Tuple
import numpy as np

from tinygrad.nn.state import load_state_dict, torch_load, get_parameters, get_state_dict # type: ignore

from extra.models.unet import UNetModel # type: ignore
from examples.sdv2 import params, FrozenOpenClipEmbedder, get_alphas_cumprod # type: ignore

if __name__ == "__main__":
  BS = 1
  TRAIN_DTYPE = dtypes.float32

  class WrapperModel:
    def __init__(self, cond_stage_config:Dict, **kwargs):
      self.cond_stage_model = FrozenOpenClipEmbedder(**cond_stage_config)
  wrapper = WrapperModel(**params)
  load_state_dict(wrapper, torch_load("/home/tiny/tinygrad/weights/512-base-ema.ckpt")["state_dict"])
  # for w in get_state_dict(wrapper).values():
  #   w.to(WRAPPER_DEVICE)

  model = UNetModel(**params["unet_config"])
  for w in get_state_dict(model).values():
    w.replace(w.cast(dtypes.float16)).realize()
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, b1=0.9, b2=0.999, weight_decay=0.01)
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, eps=1.0)
  optimizer = SGD(get_parameters(model), lr=1.25e-7)



  alphas_cumprod      = get_alphas_cumprod()
  alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
  sqrt_alphas_cumprod = alphas_cumprod.sqrt()
  sqrt_on_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()




  def collate_fnx(batch):
    arr = np.zeros((len(batch),8,64,64), dtype=np.float32)
    for i, e in enumerate(batch):
      arr[i,:,:,:] = e["npy"]
    return { "moments": Tensor(arr), "txt": [e["txt"] for e in batch] }

  from webdataset import WebDataset, WebLoader # type: ignore
  # urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
  urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00010}.tar"
  keep_only_keys = { "npy", "txt" }
  def filter_fnx(sample): return {k:v for k,v in sample.items() if k in keep_only_keys}
  dataset = WebDataset(urls=urls, resampled=True, cache_size=-1, cache_dir=None)
  dataset = dataset.shuffle(size=1000).decode().map(filter_fnx).batched(BS, partial=False, collation_fn=collate_fnx)
  dataloader = WebLoader(dataset, batch_size=None, shuffle=False, num_workers=4, persistent_workers=True)

  def sample_moments(moments:Tensor) -> Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clip(-30.0, 20.0)
    std = Tensor.exp(logvar * 0.5)
    return mean + std * Tensor.rand(*mean.shape)

  # @TinyJit
  def train_step(x:Tensor, c:Tensor, t:Tensor) -> None:
    Tensor.training = True
    
    noise   = Tensor.randn(x.shape)
    x_noisy =   sqrt_alphas_cumprod         .gather(-1, t).reshape(x.shape[0], 1, 1, 1) * x \
              + sqrt_on_minus_alphas_cumprod.gather(-1, t).reshape(x.shape[0], 1, 1, 1) * noise
    output  = model(x_noisy, t, c)

    loss = (x.cast(dtypes.float32) - output.cast(dtypes.float32)).square().mean([1, 2, 3]).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lv = loss.numpy()
    print(lv)

  def prep_for_jit(*inputs:Tensor) -> Tuple[Tensor,...]:
    return tuple(i.cast(TRAIN_DTYPE).realize() for i in inputs)

  for i, entry in enumerate(dataloader):
    x = (sample_moments(entry["moments"]) * 0.18215)
    c = Tensor.cat(*[wrapper.cond_stage_model(t) for t in entry["txt"]], dim=0)
    t = Tensor.randint(x.shape[0], low=0, high=1000)
    print(f"Running Step {i}")
    train_step(*prep_for_jit(x, c, t))
    if i >= 25:
      break

