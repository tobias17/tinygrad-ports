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
    alphas_prev = alphas_cumprod_prev[tms_tensor]
    sigmas      = Tensor.zeros_like(alphas)

    x_t = Tensor.randn(batch_size, 4, 64, 64)
    time_range = np.flip(ddim_timesteps)

    @TinyJit
    def run(model, x, t, c):
      return model(x,t,c).realize()

    for i, step in enumerate(tqdm(time_range)):
      index = num_steps - i - 1
      tms = Tensor.full((batch_size,), int(step))
      t_emb = timestep_embedding(tms, 320)

      def fp16r(z):
        return z.cast(dtypes.float16).realize()

      latent_uc, latent_c = run(model, fp16r(Tensor.cat(x_t,x_t)), fp16r(Tensor.cat(t_emb,t_emb)), fp16r(Tensor.cat(uc,c))).chunk(2)
      output = latent_uc + cfg_scale * (latent_c - latent_uc)

      shape = (batch_size, 1, 1, 1)
      e_t =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * output \
            + sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * x_t

      a_t               = alphas_cumprod[tms]               .reshape(batch_size,1,1,1)
      a_prev            = alphas_cumprod_prev[tms]          .reshape(batch_size,1,1,1)
      sqrt_one_minus_at = sqrt_one_minus_alphas_cumprod[tms].reshape(batch_size,1,1,1)

      pred_x0 =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * x_t \
                - sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * output

      dir_xt = (1.0 - a_prev).sqrt() * e_t
      x_t = a_prev.sqrt() * pred_x0 + dir_xt

    return x_t
