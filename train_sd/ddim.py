from tinygrad import Tensor # type: ignore
from tinygrad.helpers import tqdm # type: ignore
from examples.sdv2 import get_alphas_cumprod

import numpy as np

TOTAL_STEPS = 1000

class DdimSampler:
  def __init__(self):
    pass

  def sample(self, model, batch_size:int, c:Tensor, uc:Tensor, num_steps:int=50, cfg_scale:float=8.0) -> Tensor:
    ddim_timesteps = np.arange(1, TOTAL_STEPS+1, num_steps//TOTAL_STEPS)
    
    alphas_cumprod = get_alphas_cumprod()
    alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    
    # ddim_sigmas, ddim_alphas, ddim_alphas_prev

    alphas      = alphas_cumprod[ddim_timesteps]
    alphas_prev = alphas_cumprod_prev[ddim_timesteps]
    sigmas      = Tensor.zeros_like(alphas)

    x_t     = Tensor.randn(*batch_size, 4, 64, 64)
    time_range = np.flip(ddim_timesteps)

    for i, step in enumerate(tqdm(time_range)):
      index = num_steps - i - 1
      tms = Tensor.full((batch_size,), step)

      latent_uc, latent_c = model(Tensor.cat(x_t,x_t), Tensor.cat(tms,tms), Tensor.cat(uc,c))
      output = latent_uc + cfg_scale * (latent_c - latent_uc)

      shape = (batch_size, 1, 1, 1)
      e_t =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * output \
            + sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * x_t

      a_t               = Tensor.full(shape, alphas_cumprod[tms])
      a_prev            = Tensor.full(shape, alphas_cumprod_prev[tms])
      sqrt_one_minus_at = Tensor.full(shape, sqrt_one_minus_alphas_cumprod[tms])

      pred_x0 =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * x_t \
                - sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * output

      dir_xt = (1.0 - a_prev).sqrt() * e_t
      x_t = a_prev.sqrt() * pred_x0 + dir_xt

    return x_t
