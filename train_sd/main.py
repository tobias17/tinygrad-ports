from tinygrad import Tensor # type: ignore
from PIL import Image

from tinygrad.nn.state import load_state_dict, torch_load # type: ignore

from extra.models.unet import UNetModel # type: ignore
from examples.sdv2 import params, StableDiffusionV2 # type: ignore

if __name__ == "__main__":
  BS = 10

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
    # print(entry["txt"])
    # latent = np.frombuffer(entry["npy"], dtype=np.float16)
    # # side = int((latent.shape[0] // 4) ** 0.5)
    # # assert side * side * 4 == latent.shape[0]
    # # latent = latent.reshape((side,side,4))
    # print(latent.shape, latent.shape[0]//4, ((latent.shape[0]//4)**0.5)*8)
    # print(latent.dtype)
    # if i > 10:
    #   break

  print(entry.keys())
  print(type(entry["moments"]))
  print(entry["moments"].shape)
  # print(type(entry["npy"][0]))
  # print(len(entry["npy"][0]))

  model = StableDiffusionV2(**params)
  state_dict = torch_load("/raid/weights/512-base-ema.ckpt")["state_dict"]
  load_state_dict(model, state_dict, strict=False)
  print("model loaded")

  x = entry["moments"][0:1, 0:4] * 0.18215
  print(x.shape)
  im = Image.fromarray(model.decode(x, 512, 512).numpy())
  im.save("/tmp/decoded.png")

  print(entry["txt"])

  # model = UNetModel(**params["unet_config"])
