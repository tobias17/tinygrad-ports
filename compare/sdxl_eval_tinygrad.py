from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, safe_load, get_parameters
from examples.sdxl import SDXL, DPMPP2MSampler, configs
from PIL import Image
import numpy as np

GUIDANCE = 8.0
STEPS = 20

pos_prompt = "a horse sized cat eating a bagel"
neg_prompt = ""

def main():
  mdl = SDXL(configs["SDXL_Base"])
  url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
  load_state_dict(mdl, safe_load(str(fetch(url, "sd_xl_base_1.0.safetensors"))), strict=False)
  for w in get_parameters(mdl):
    w.replace(w.cast(dtypes.float16).realize())

  sampler = DPMPP2MSampler(GUIDANCE)
  noise = Tensor(np.load("out_noise.npy"), dtype=dtypes.float16)

  c, uc = mdl.create_conditioning([pos_prompt], 1024, 1024)
  z = sampler(mdl.denoise, noise, c, uc, STEPS)
  x = mdl.decode(z).realize()

  x = x.add(1.0).div(2.0).reshape(3,1024,1024).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)

  im = Image.fromarray(x.numpy())
  im.save("out_img_tinygrad.png")

if __name__ == "__main__":
  main()
