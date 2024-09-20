from tinygrad import Tensor, dtypes, fetch
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_load
from examples.sdxl import SDXL, configs, DPMPP2MSampler, Guider, VanillaCFG, SplitVanillaCFG, run # type: ignore
from PIL import Image

IMG_SIZE = 1024
LATENT_SIZE = IMG_SIZE // 8
GUIDANCE_SCALE = 8.0
NUM_STEPS = 12

class TestCFG(Guider):
  def __call__(self, denoiser, x:Tensor, s:Tensor, c, uc) -> Tensor:
    x_u = denoiser(x, s, uc)
    x_c = denoiser(x, s, c)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred

def main():
  Tensor.manual_seed(42)

  # Load generation model
  model = SDXL(configs["SDXL_Base"])
  weights_path = fetch('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors', 'sd_xl_base_1.0.safetensors')
  load_state_dict(model, safe_load(weights_path), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16)).realize()

  prompt = "A horse size cat eating a bagel."
  c, uc = model.create_conditioning([prompt], IMG_SIZE, IMG_SIZE)
  randn = Tensor.randn(1, 4, LATENT_SIZE, LATENT_SIZE)

  # to_test = [VanillaCFG, SplitVanillaCFG]
  to_test = [TestCFG]

  for guider_cls in to_test:
    Tensor.manual_seed(42)
    sampler = DPMPP2MSampler(GUIDANCE_SCALE, guider_cls=guider_cls)
    z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
    x = model.decode(z).realize()
    x = (x + 1.0) / 2.0
    x = x.reshape(1, 3, IMG_SIZE, IMG_SIZE).realize()
    ten_im = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8).numpy()
    pil_im = Image.fromarray(ten_im[0])
    pil_im.save(f"./output/test_{guider_cls.__name__}.png")
    # run.reset()

if __name__ == "__main__":
  main()
