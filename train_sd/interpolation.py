from tinygrad import Tensor, dtypes # type: ignore
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F

def bilinear_interp(x:Tensor, out_size=(299,299), align_corners=False) -> Tensor:
  inp_size = x.shape[-2:]
  for i in range(-len(out_size),0):
    scale = (inp_size[i] - (1 if align_corners else 0)) / (out_size[i] - (1 if align_corners else 0))
    index = Tensor.arange(out_size[i]).cast(dtypes.float32)
    index = scale * index if align_corners else (scale * (index + 0.5)) - 0.5
    _, high, perc = (low := index.floor()), index.ceil(), index - low
    # np.set_printoptions(suppress=True,precision=3)
    # print(low.numpy())
    # print(high.numpy())
    # print(perc.numpy())
    resh_shape, exp_shape = [1,]*len(x.shape), list(x.shape)
    resh_shape[i] = out_size[i]
    exp_shape[i]  = out_size[i]
    low, high, perc = [y.reshape(resh_shape).expand(exp_shape) for y in (low,high,perc)]
    x = x.gather(i, low)*(1.0 - perc) + x.gather(i, high)*perc
  return x

if __name__ == "__main__":
  im = Image.open("/home/tiny/tinygrad/examples/stable_diffusion_seed0.png")
  x = Tensor(np.array(im)).cast(dtypes.float32)
  x = x.permute(2, 0, 1)
  print(x.shape)


  x_pytorch = torch.tensor(x.unsqueeze(0).numpy())
  x_pytorch = F.interpolate(x_pytorch, size=(299,299), mode="bilinear", align_corners=False)
  x = bilinear_interp(x)

  a,b = x.numpy(),x_pytorch.cpu().numpy()
  print(f"| {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")

  diff = (x.permute(1, 2, 0).numpy() - x_pytorch.squeeze(0).permute(1, 2, 0).cpu().numpy()) + 127.0
  imd = Image.fromarray(diff.astype(np.uint8))
  imd.save(f"/tmp/resized_diff.png")

  print(x.shape)
  x = x.permute(1, 2, 0).cast(dtypes.uint8)
  print(x.shape)
  im = Image.fromarray(x.numpy())
  im.save(f"/tmp/resized.png")

  im2 = Image.fromarray((x_pytorch).squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy())
  im2.save(f"/tmp/resized_pytorch.png")
