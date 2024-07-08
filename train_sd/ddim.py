from tinygrad import Tensor # type: ignore
from examples.sdv2 import get_alphas_cumprod

import numpy as np

TOTAL_STEPS = 1000

class DdimSampler:
  def __init__(self):
    pass

  def sample(self, batch_size:int, conditioning:Tensor, num_steps:int=50) -> Tensor:
    ddim_timesteps = np.arange(1, TOTAL_STEPS+1, num_steps//TOTAL_STEPS)
    
    alphas_cumprod = get_alphas_cumprod()
    alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
    
    # ddim_sigmas, ddim_alphas, ddim_alphas_prev

    alphas      = alphas_cumprod[ddim_timesteps]
    alphas_prev = alphas_cumprod_prev[ddim_timesteps]
    sigmas      = Tensor.zeros_like(alphas)
    size        = (batch_size, 4, 64, 64)

    
