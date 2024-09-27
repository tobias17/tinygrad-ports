from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import trange
from tinygrad.nn.state import get_state_dict
from extra.models.inception import FidInceptionV3 # type: ignore
from typing import List
from PIL import Image
import numpy as np

def call(self:FidInceptionV3, x:Tensor):
   values = [x.numpy()]

   x = x.interpolate((299,299), mode="linear")
   x = (x * 2) - 1
   values.append(x.numpy())

   x = x.sequential([
      self.Conv2d_1a_3x3,
      self.Conv2d_2a_3x3,
      self.Conv2d_2b_3x3,
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, dilation=1),
   ])
   values.append(x.numpy())

   x = x.sequential([
      self.Conv2d_3b_1x1,
      self.Conv2d_4a_3x3,
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, dilation=1),
   ])
   values.append(x.numpy())

   x = x.sequential([
      self.Mixed_5b,
      self.Mixed_5c,
      self.Mixed_5d,
      self.Mixed_6a,
      self.Mixed_6b,
      self.Mixed_6c,
      self.Mixed_6d,
      self.Mixed_6e,
   ])
   values.append(x.numpy())

   x = x.sequential([
      self.Mixed_7a,
      self.Mixed_7b,
      self.Mixed_7c,
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8)),
   ])
   values.append(x.numpy())

   return x, values
FidInceptionV3.__call__ = call



def get_incp_values(model:FidInceptionV3, pil_ims:List[Image.Image], gpus=None):
  images = [Tensor(np.asarray(im), dtype=dtypes.float16).div(255.0).permute(2,0,1).unsqueeze(0) for im in pil_ims]
  x = Tensor.cat(*images, dim=0).realize()
  if gpus:
    x = x.shard(gpus, axis=0)
  incp_act, values = model(x.realize())
  return incp_act.to(Device.DEFAULT).reshape(incp_act.shape[:2]).realize(), values

def main():
   GPUS = [f"{Device.DEFAULT}:{i}" for i in [1,2,3,4]]
   GLOBAL_BS = 40

   model_sgl = FidInceptionV3().load_from_pretrained()
   model_mlt = FidInceptionV3().load_from_pretrained()

   for w in get_state_dict(model_sgl).values():
      w.replace(w.cast(dtypes.float16)).realize()
   for w in get_state_dict(model_mlt).values():
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()

   pil_ims = [Image.open(f"output/rendered_2/gen_{image_i:05d}.png") for image_i in trange(GLOBAL_BS)]

   incp_sgl, values_sgl = get_incp_values(model_sgl, pil_ims, None)
   incp_mlt, values_mlt = get_incp_values(model_mlt, pil_ims, GPUS)

   np.testing.assert_allclose(incp_sgl.numpy(), incp_mlt.numpy(), atol=1e-6, rtol=1e-6)

   for value_sgl, value_mlt in zip(values_sgl, values_mlt):
      np.testing.assert_allclose(value_sgl, value_mlt, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
   main()
