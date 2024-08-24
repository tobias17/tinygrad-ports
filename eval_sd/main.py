from tinygrad import Tensor, dtypes, Device # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict
from examples.sdxl import SDXL, DPMPP2MSampler, configs # type: ignore

from typing import Dict, List, Tuple
import pandas as pd # type: ignore
from PIL import Image

# temporary overwrite to fix function for batching and external
def create_conditioning(self:SDXL, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float=5.0) -> Tuple[Dict,Dict]:
  N = 1
  batch_c : Dict = {
    "txt": pos_prompt,
    "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
    "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
    "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
    "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
  }
  batch_uc: Dict = {
    "txt": "",
    "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
    "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
    "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
    "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
  }
  return self.conditioner(batch_c), self.conditioner(batch_uc, force_zero_embeddings=["txt"])
SDXL.create_conditioning = create_conditioning


def main():

  # Define constants

  GPUS = [f"{Device.DEFAULT}:{i}" for i in [4,5]]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  IMG_SIZE = 1024
  LATENT_SCALE = 8
  LATENT_SIZE = IMG_SIZE // LATENT_SCALE
  assert LATENT_SIZE * LATENT_SCALE == IMG_SIZE

  GUIDANCE_SCALE = 8.0
  NUM_STEPS = 20

  # Load model
  model = SDXL(configs["SDXL_Base"])
  load_state_dict(model, safe_load("/home/tiny/tinygrad/weights/sd_xl_base_1.0.safetensors"), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model."):
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None))
  
  # Create sampler
  sampler = DPMPP2MSampler(GUIDANCE_SCALE)

  # Load dataset
  df = pd.read_csv("/home/tiny/tinygrad/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)
  captions = df["caption"].array

  dataset_i = 0
  assert len(captions) % GLOBAL_BS == 0, f"GLOBAL_BS ({GLOBAL_BS}) needs to evenly divide len(captions) ({len(captions)}) for now"
  while dataset_i < len(captions):
    batch_c, batch_uc = [], []
    for text in captions[dataset_i:dataset_i+GLOBAL_BS]:
      c, uc = model.create_conditioning(text, IMG_SIZE, IMG_SIZE)
      batch_c.append(c)
      batch_uc.append(uc)
    
    c, uc = {}, {}
    for key in batch_c [0]: c [key] = Tensor.cat(*[bc[key] for bc in batch_c ]).shard(GPUS, axis=0).realize()
    for key in batch_uc[0]: uc[key] = Tensor.cat(*[bu[key] for bu in batch_uc]).shard(GPUS, axis=0).realize()

    randn = Tensor.randn(GLOBAL_BS, 4, LATENT_SIZE, LATENT_SIZE)
    z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
    x = model.decode(z).realize()
    x = (x + 1.0) / 2.0
    x = x.reshape(GLOBAL_BS,3,IMG_SIZE,IMG_SIZE).permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8)
    x = x.to(Device.DEFAULT).realize()
    print(x.shape)

    for i in range(GLOBAL_BS):
      im = Image.fromarray(x.numpy())
      im.save(f"/tmp/eval_gen_{i}.png")
    input("next? ")

    dataset_i += GLOBAL_BS

if __name__ == "__main__":
  main()
