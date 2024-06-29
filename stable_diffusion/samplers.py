from tinygrad import Tensor # type: ignore
from tinygrad.helpers import trange # type: ignore
from stable_diffusion import append_dims, LegacyDDPMDiscretization # type: ignore

from typing import Dict, Tuple, Optional

class VanillaCFG:
  def __init__(self, scale:float):
    self.scale = scale

  def prepare_inputs(self, x:Tensor, s:float, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor,Tensor]:
    c_out = {}
    for k in c:
      assert k in ["vector", "crossattn", "concat"]
      c_out[k] = Tensor.cat(uc[k], c[k], dim=0)
    return Tensor.cat(x, x), Tensor.cat(s, s), c_out

  def __call__(self, x:Tensor, sigma:float) -> Tensor:
    x_u, x_c = x.chunk(2)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L21
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
class DPMPP2MSampler:
  def __init__(self, cfg_scale:float):
    self.discretization = LegacyDDPMDiscretization()
    self.guider = VanillaCFG(cfg_scale)

  def sampler_step(self, old_denoised:Optional[Tensor], prev_sigma:Optional[Tensor], sigma:Tensor, next_sigma:Tensor, denoiser, x:Tensor, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor]:
    denoised = denoiser(*self.guider.prepare_inputs(x, sigma, c, uc))
    denoised = self.guider(denoised, sigma)

    t, t_next = sigma.log().neg(), next_sigma.log().neg()
    h = t_next - t
    r = None if prev_sigma is None else (t - prev_sigma.log().neg()) / h

    mults = [t_next.neg().exp()/t.neg().exp(), (-h).exp().sub(1)]
    if r is not None:
      mults.extend([1 + 1/(2*r), 1/(2*r)])
    mults = [append_dims(m, x) for m in mults]

    x_standard = mults[0]*x - mults[1]*denoised
    if (old_denoised is None) or (next_sigma.sum().numpy().item() < 1e-14):
      return x_standard, denoised

    denoised_d = mults[2]*denoised - mults[3]*old_denoised
    x_advanced = mults[0]*x        - mults[1]*denoised_d
    x = Tensor.where(append_dims(next_sigma, x) > 0.0, x_advanced, x_standard)
    return x, denoised

  def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
    sigmas = self.discretization(num_steps)
    x *= Tensor.sqrt(1.0 + sigmas[0] ** 2.0)
    num_sigmas = len(sigmas)

    old_denoised = None
    for i in trange(num_sigmas - 1):
      x, old_denoised = self.sampler_step(
        old_denoised=old_denoised,
        prev_sigma=(None if i==0 else sigmas[i-1].reshape(x.shape[0])),
        sigma=sigmas[i].reshape(x.shape[0]),
        next_sigma=sigmas[i+1].reshape(x.shape[0]),
        denoiser=denoiser,
        x=x,
        c=c,
        uc=uc,
      )
      x.realize()
      old_denoised.realize()

    return x