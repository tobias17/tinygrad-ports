from tinygrad import Tensor # type: ignore
from tinygrad.nn import Conv2d, GroupNorm # type: ignore
from stable_diffusion import tensor_identity # type: ignore

from typing import List, Callable, Any

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L136
class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = GroupNorm(32, in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L74
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L102
class Downsample:
  def __init__(self, dims:int):
    self.conv = Conv2d(dims, dims, kernel_size=3, stride=2, padding=(0,1,0,1))

  def __call__(self, x:Tensor) -> Tensor:
    return self.conv(x)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L58
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L83
class Upsample:
  def __init__(self, dims:int):
    self.conv = Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    B,C,Y,X = x.shape
    x = x.reshape(B, C, Y, 1, X, 1).expand(B, C, Y, 2, X, 2).reshape(B, C, Y*2, X*2)
    return self.conv(x)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L204
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L17
class AttnBlock:
  def __init__(self, dim:int):
    self.norm = GroupNorm(32, dim)
    self.q = Conv2d(dim, dim, 1)
    self.k = Conv2d(dim, dim, 1)
    self.v = Conv2d(dim, dim, 1)
    self.proj_out = Conv2d(dim, dim, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x:Tensor) -> Tensor:
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
    h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
    return x + self.proj_out(h_)

class MidEntry:
  def __init__(self, block_in:int):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1  = AttnBlock  (block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L487
class Encoder:
  def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
    self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
    in_ch_mult = (1,) + tuple(ch_mult)

    class BlockEntry:
      def __init__(self, block:List[ResnetBlock], downsample):
        self.block = block
        self.downsample = downsample
    self.down: List[BlockEntry] = []
    for i_level in range(len(ch_mult)):
      block = []
      block_in  = ch * in_ch_mult[i_level]
      block_out = ch * ch_mult   [i_level]
      for _ in range(num_res_blocks):
        block.append(ResnetBlock(block_in, block_out))
        block_in = block_out

      downsample = tensor_identity if (i_level == len(ch_mult)-1) else Downsample(block_in)
      self.down.append(BlockEntry(block, downsample))

    self.mid = MidEntry(block_in)

    self.norm_out = GroupNorm(32, block_in)
    self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    h = self.conv_in(x)
    for down in self.down:
      for block in down.block:
        h = block(h)
      h = down.downsample(h)

    h = h.sequential([self.mid.block_1, self.mid.attn_1, self.mid.block_2])
    h = h.sequential([self.norm_out,    Tensor.swish,    self.conv_out   ])
    return h

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L604
class Decoder:
  def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
    block_in = ch * ch_mult[-1]
    curr_res = resolution // 2 ** (len(ch_mult) - 1)
    self.z_shape = (1, z_ch, curr_res, curr_res)

    self.conv_in = Conv2d(z_ch, block_in, kernel_size=3, stride=1, padding=1)

    self.mid = MidEntry(block_in)

    class BlockEntry:
      def __init__(self, block:List[ResnetBlock], upsample):
        self.block = block
        self.upsample = upsample
    self.up: List[BlockEntry] = []
    for i_level in reversed(range(len(ch_mult))):
      block = []
      block_out = ch * ch_mult[i_level]
      for _ in range(num_res_blocks + 1):
        block.append(ResnetBlock(block_in, block_out))
        block_in = block_out

      upsample = tensor_identity if i_level == 0 else Upsample(block_in)
      self.up.insert(0, BlockEntry(block, upsample))

    self.norm_out = GroupNorm(32, block_in)
    self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

  def __call__(self, z:Tensor) -> Tensor:
    h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])

    for up in self.up[::-1]:
      for block in up.block:
        h = block(h)
      h = up.upsample(h)

    h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
    return h

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L508
class FirstStageModel:
  def __init__(self, embed_dim:int=4, **kwargs):
    self.encoder = Encoder(**kwargs)
    self.decoder = Decoder(**kwargs)
    self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
    self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

  def decode(self, z:Tensor) -> Tensor:
    return z.sequential([self.post_quant_conv, self.decoder])
