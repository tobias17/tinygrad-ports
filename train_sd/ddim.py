from tinygrad import Tensor, TinyJit, dtypes # type: ignore
from tinygrad.helpers import tqdm # type: ignore
from examples.sdv2 import get_alphas_cumprod # type: ignore
from extra.models.unet import timestep_embedding # type: ignore

import numpy as np

TOTAL_STEPS = 1000

class DdimSampler:
  def __init__(self):
    pass

  @TinyJit
  def run(self, model, x, t, c):
    return model.pre_embedded(x,t,c).realize()

  def sample(self, model, batch_size:int, c:Tensor, uc:Tensor, num_steps:int=50, cfg_scale:float=8.0, shard_fnx=(lambda x: x), all_fnx_=(lambda x: x), dtype=dtypes.float16) -> Tensor:
    ddim_timesteps = np.arange(1, TOTAL_STEPS+1, TOTAL_STEPS//num_steps)
    
    alphas_cumprod                = get_alphas_cumprod()
    sqrt_alphas_cumprod           = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    alphas_prev                   = Tensor.cat(alphas_cumprod[:1], alphas_cumprod[Tensor(ddim_timesteps[:-1])])

    all_fnx_(alphas_cumprod), all_fnx_(sqrt_alphas_cumprod), all_fnx_(sqrt_one_minus_alphas_cumprod), all_fnx_(alphas_prev)

    x_t = shard_fnx(Tensor.randn(batch_size, 4, 96, 96))
    time_range = np.flip(ddim_timesteps)

    for i, step in enumerate(tqdm(time_range)):

      index = num_steps - i - 1
      tms = Tensor.full((batch_size,), int(step))
      t_emb = timestep_embedding(tms, 320).realize()

      def fp16r(z): return Tensor.cast(z, dtype).realize()
      x_t, t_emb = fp16r(x_t), fp16r(shard_fnx(t_emb))

      # TODO: this should be doable with the cat batch and chunk approach, just need to be clever with sharding
      latent_uc = self.run(model, x_t, t_emb, fp16r(shard_fnx(uc)))
      latent_c  = self.run(model, x_t, t_emb, fp16r(shard_fnx(c )))
      output = latent_uc + cfg_scale * (latent_c - latent_uc)

      shape = (batch_size, 1, 1, 1)
      e_t     =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * output \
                + sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * x_t
      pred_x0 =   sqrt_alphas_cumprod          .gather(-1, tms).reshape(shape) * x_t \
                - sqrt_one_minus_alphas_cumprod.gather(-1, tms).reshape(shape) * output

      a_prev = alphas_prev[index].reshape(1,1,1,1)
      dir_xt = (1.0 - a_prev).sqrt() * e_t
      x_t = (a_prev.sqrt() * pred_x0 + dir_xt).realize()

    return x_t
