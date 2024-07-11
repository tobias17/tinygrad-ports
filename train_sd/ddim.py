from tinygrad import Tensor, TinyJit, dtypes # type: ignore
from tinygrad.helpers import tqdm # type: ignore
from examples.sdv2 import get_alphas_cumprod # type: ignore
from extra.models.unet import timestep_embedding # type: ignore

import numpy as np

TOTAL_STEPS = 1000

class DdimSampler:
  def __init__(self):
    pass

  def sample(self, model, batch_size:int, c:Tensor, uc:Tensor, num_steps:int=50, cfg_scale:float=8.0) -> Tensor:
    ddim_timesteps = np.arange(1, TOTAL_STEPS+1, TOTAL_STEPS//num_steps)
    
    alphas_cumprod = get_alphas_cumprod()
    alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    
    # ddim_sigmas, ddim_alphas, ddim_alphas_prev

    tms_tensor  = Tensor(ddim_timesteps)
    alphas      = alphas_cumprod[tms_tensor]
    alphas_prev = Tensor.cat(alphas_cumprod[:1], alphas_cumprod[tms_tensor[:-1]])
    sigmas      = Tensor.zeros_like(alphas)

    x_t = Tensor.randn(batch_size, 4, 64, 64)
    time_range = np.flip(ddim_timesteps)

    @TinyJit
    def run(model, x, t, c):
      return model(x,t,c).realize()

    for i, step in enumerate(tqdm(time_range)):
      x_t = Tensor(np.load(f"/home/tiny/weights_cache/ddim/img_step_{i}.npy"))
      print(f"ITER {i}")

      index = num_steps - i - 1
      tms = Tensor.full((batch_size,), int(step))
      t_emb = timestep_embedding(tms, 320)

      def fp16r(z):
        return z.cast(dtypes.float16).realize()

      latent_uc, latent_c = run(model, fp16r(Tensor.cat(x_t,x_t)), fp16r(Tensor.cat(t_emb,t_emb)), fp16r(Tensor.cat(uc,c))).chunk(2)
      output = latent_uc + cfg_scale * (latent_c - latent_uc)

      a,b = output.numpy(), np.load(f"/home/tiny/weights_cache/ddim/model_output_step_{index}.npy")
      print(f"| out | {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")
      output = Tensor(b)

      shape = (batch_size, 1, 1, 1)
      e_t =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * output \
            + sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * x_t

      a,b = e_t.numpy(),np.load(f"/home/tiny/weights_cache/ddim/e_t_step{index}.npy")
      print(f"| e_t | {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")

      a_t               = alphas_cumprod[tms]               .reshape(batch_size,1,1,1)
      a_prev            = alphas_prev[index]                .reshape(batch_size,1,1,1)
      sqrt_one_minus_at = sqrt_one_minus_alphas_cumprod[tms].reshape(batch_size,1,1,1)

      print(f"A_PREV: {a_prev.numpy()}")

      pred_x0 =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * x_t \
                - sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * output

      a,b = pred_x0.numpy(),np.load(f"/home/tiny/weights_cache/ddim/pred_x0_step{index}.npy")
      print(f"| px0 | {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")

      dir_xt = (1.0 - a_prev).sqrt() * e_t

      a,b = dir_xt.numpy(),np.load(f"/home/tiny/weights_cache/ddim/dir_xt_step{index}.npy")
      print(f"| dir | {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")

      x_t = a_prev.sqrt() * pred_x0 + dir_xt

      a,b = x_t.numpy(),np.load(f"/home/tiny/weights_cache/ddim/x_prev_step{index}.npy")
      print(f"| x_t | {np.mean(np.abs(a-b)):.4f} | {np.mean(np.abs(a)):.4f} | {np.mean(np.abs(b)):.4f} |")

    x_t = Tensor(np.load("/home/tiny/weights_cache/ddim/out.npy"))

    return x_t
