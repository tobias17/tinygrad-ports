from tinygrad import Tensor, dtypes # type: ignore
from tinygrad.nn.optim import AdamW # type: ignore

from PIL import Image
from typing import Dict, Tuple

from tinygrad.nn.state import load_state_dict, torch_load # type: ignore

from extra.models.unet import UNetModel # type: ignore
from examples.sdv2 import params, FrozenOpenClipEmbedder, get_alphas_cumprod # type: ignore

if __name__ == "__main__":
  BS = 2
  TRAIN_DTYPE = dtypes.float16




  class WrapperModel:
    def __init__(self, cond_stage_config:Dict, **kwargs):
      self.cond_stage_model = FrozenOpenClipEmbedder(**cond_stage_config)
  wrapper = WrapperModel(**params)
  load_state_dict(wrapper, torch_load("/home/tiny/tinygrad/weights/512-base-ema.ckpt")["state_dict"])

  model = UNetModel(**params["unet_config"])




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

  import numpy as np
  for i, entry in enumerate(dataloader):
    break

  print(entry.keys())
  print(type(entry["moments"]))
  print(entry["moments"].shape)

  def sample_moments(moments:Tensor) -> Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clip(-30.0, 20.0)
    std = Tensor.exp(logvar * 0.5)
    return mean + std * Tensor.rand(*mean.shape)

  x = (sample_moments(entry["moments"]) * 0.18215).cast(TRAIN_DTYPE)
  c = Tensor.cat(*[wrapper.cond_stage_model(t) for t in entry["txt"]], dim=0)
  t = Tensor.randint(x.shape[0], low=0, high=1000)

  def extract_into_tensor(a:Tensor, t:Tensor, x_shape:Tuple[int,...]) -> Tensor:
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

  def train_step(x:Tensor, c:Tensor):
    noise   = Tensor.randn(x.shape)
    x_noisy = sqrt_alphas_cumprod         .gather(-1, t).reshape(x.shape[0], 1, 1, 1) * x \
            + sqrt_on_minus_alphas_cumprod.gather(-1, t).reshape(x.shape[0], 1, 1, 1) * noise
    output  = model(x_noisy, t, c)

    loss = (x - output).square().mean([1, 2, 3]).mean()




