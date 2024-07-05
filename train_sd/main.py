from tinygrad import Tensor

from extra.models.unet import UNetModel
from examples.sdv2 import params

from torch.utils.data import default_collate

if __name__ == "__main__":
  BS = 10

  # def collate_fnx(batch):
  #   arr = np.array((len(batch),*batch[0]["npy"].shape))
  #   for i, e in enumerate(batch):
  #     a = np.frombuffer(e["npy"], dtype=np.float16)
  #     if i == 0:
  #     arr[i] = np.frombuffer()
  #   output = { "latent": Tensor(arr) "txt": [e["txt"] for e in batch] }
  #   print(type(batch))
  #   print(len(batch))
  #   print(type(batch[0]))
  #   print(batch[0].keys())
  #   print()

  from webdataset import WebDataset, WebLoader
  # urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
  urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00010}.tar"
  keep_only_keys = { "npy", "txt" }
  def filter_fnx(sample): return {k:v for k,v in sample.items() if k in keep_only_keys}
  dataset = WebDataset(urls=urls, resampled=True, cache_size=-1, cache_dir=None)
  dataset = dataset.shuffle(size=1000).map(filter_fnx).batched(BS, partial=False, collation_fn=default_collate)
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

  # model = UNetModel(**params["unet_config"])
