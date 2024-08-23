from tinygrad import Tensor # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load, get_state_dict
from examples.sdxl import SDXL, configs # type: ignore

def main():
  MODEL_WEIGHTS_ROOT = "/home/tiny/tinygrad/weights/stable_diffusion_fp16/checkpoint_pipe"
  model = SDXL(configs["SDXL_Base"])

  sd1 = get_state_dict(model.model.diffusion_model)
  sd2 = safe_load(f"{MODEL_WEIGHTS_ROOT}/unet/diffusion_pytorch_model.safetensors")

  k1 = sorted(list(sd1.keys()))
  k2 = sorted(list(sd2.keys()))

  load_state_dict(model.model.diffusion_model, safe_load(f"{MODEL_WEIGHTS_ROOT}/unet/diffusion_pytorch_model.safetensors"))
  load_state_dict(model.first_stage_model, safe_load(f"{MODEL_WEIGHTS_ROOT}/vae/diffusion_pytorch_model.safetensors"))

if __name__ == "__main__":
  main()
