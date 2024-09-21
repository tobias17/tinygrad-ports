from tinygrad import Tensor, dtypes, fetch, TinyJit
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_load
from examples.sdxl import SDXL, configs, DPMPP2MSampler, Guider, VanillaCFG, SplitVanillaCFG, append_dims, timestep_embedding # type: ignore
from PIL import Image

from examples import sdxl # type: ignore

IMG_SIZE = 1024
LATENT_SIZE = IMG_SIZE // 8
GUIDANCE_SCALE = 8.0
NUM_STEPS = 12

class TestCFG(Guider):
  def __call__(self, denoiser, x:Tensor, s:Tensor, c, uc) -> Tensor:
    # sdxl.run = run1
    x_u = denoiser(x, s, uc)
    # sdxl.run = run2
    x_c = denoiser(x, s, c)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred

# @TinyJit
def run1(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()
@TinyJit
def run2(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()



def denoise(self:SDXL, x:Tensor, sigma:Tensor, cond) -> Tensor:

  def sigma_to_idx(s:Tensor) -> Tensor:
    dists = s - self.sigmas.unsqueeze(1)
    return dists.abs().argmin(axis=0).view(*s.shape)

  sigma = self.sigmas[sigma_to_idx(sigma)]
  sigma_shape = sigma.shape
  sigma = append_dims(sigma, x)

  c_out = -sigma
  c_in  = 1 / (sigma**2 + 1.0) ** 0.5
  tms   = sigma_to_idx(sigma.reshape(sigma_shape))

  def prep(*tensors:Tensor):
    return tuple(t.cast(dtypes.float16).realize() for t in tensors)

  args = prep(x*c_in, tms, cond["crossattn"], cond["vector"], c_out, x)
  if not self.warmed_up:
    for i in range(3):
      print(f"Warmup {i+1}")
      run1(self.model.diffusion_model, *[Tensor.rand(a.shape, dtype=a.dtype, device=a.device).realize() for a in args])
    self.warmed_up = True

  return run1(self.model.diffusion_model, *args)
SDXL.denoise = denoise
SDXL.warmed_up = False


def main():
  Tensor.manual_seed(42)

  # Load generation model
  model = SDXL(configs["SDXL_Base"])
  weights_path = fetch('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors', 'sd_xl_base_1.0.safetensors')
  load_state_dict(model, safe_load(weights_path), strict=False)
  for k,w in get_state_dict(model).items():
    if k.startswith("model.") or k.startswith("first_stage_model.") or k.startswith("sigmas"):
      w.replace(w.cast(dtypes.float16)).realize()

  prompts = ["A horse size cat eating a bagel.", "Photograph of a white and blue bathroom."][:]
  # to_test = [VanillaCFG, TestCFG]
  to_test = [TestCFG]

  for p_i, prompt in enumerate(prompts):
    for guider_cls in to_test:
      Tensor.manual_seed(42)
      randn = Tensor.randn(1, 4, LATENT_SIZE, LATENT_SIZE)
      c, uc = model.create_conditioning([prompt], IMG_SIZE, IMG_SIZE)
      sampler = DPMPP2MSampler(GUIDANCE_SCALE, guider_cls=guider_cls)
      z = sampler(model.denoise, randn, c, uc, NUM_STEPS)
      x = model.decode(z).realize()
      x = (x + 1.0) / 2.0
      x = x.reshape(1, 3, IMG_SIZE, IMG_SIZE).realize()
      ten_im = x.permute(0,2,3,1).clip(0,1).mul(255).cast(dtypes.uint8).numpy()
      pil_im = Image.fromarray(ten_im[0])
      pil_im.save(f"./output/test_{guider_cls.__name__}_{p_i}.png")
      # run.reset()

if __name__ == "__main__":
  main()
