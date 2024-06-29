from tinygrad import Tensor, dtypes # type: ignore
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm # type: ignore
from stable_diffusion import tensor_identity, timestep_embedding

from typing import Optional, Union, List, Any
import math

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L136
class ResBlock:
  def __init__(self, channels:int, emb_channels:int, out_channels:int):
    self.in_layers = [
      GroupNorm(32, channels),
      Tensor.silu,
      Conv2d(channels, out_channels, 3, padding=1),
    ]
    self.emb_layers = [
      Tensor.silu,
      Linear(emb_channels, out_channels),
    ]
    self.out_layers = [
      GroupNorm(32, out_channels),
      Tensor.silu,
      lambda x: x,  # needed for weights loading code to work
      Conv2d(out_channels, out_channels, 3, padding=1),
    ]
    self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else tensor_identity

  def __call__(self, x:Tensor, emb:Tensor) -> Tensor:
    h = x.sequential(self.in_layers)
    emb_out = emb.sequential(self.emb_layers)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = h.sequential(self.out_layers)
    return self.skip_connection(x) + h

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L163
class CrossAttention:
  def __init__(self, query_dim:int, ctx_dim:int, n_heads:int, d_head:int):
    self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = Linear(ctx_dim,   n_heads*d_head, bias=False)
    self.to_v = Linear(ctx_dim,   n_heads*d_head, bias=False)
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    ctx = x if ctx is None else ctx
    q,k,v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
    q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
    attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
    h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
    return h_.sequential(self.to_out)

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L180
class GEGLU:
  def __init__(self, dim_in:int, dim_out:int):
    self.proj = Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def __call__(self, x:Tensor) -> Tensor:
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * gate.gelu()

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L189
class FeedForward:
  def __init__(self, dim:int, mult:int=4):
    self.net = [
      GEGLU(dim, dim*mult),
      lambda x: x,  # needed for weights loading code to work
      Linear(dim*mult, dim)
    ]

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.net)

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L200
class BasicTransformerBlock:
  def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff    = FeedForward(dim)
    self.attn2 = CrossAttention(dim, ctx_dim, n_heads, d_head)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.norm3 = LayerNorm(dim)

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    x = x + self.attn1(self.norm1(x))
    x = x + self.attn2(self.norm2(x), ctx=ctx)
    x = x + self.ff(self.norm3(x))
    return x

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L215
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/attention.py#L619
class SpatialTransformer:
  def __init__(self, channels:int, n_heads:int, d_head:int, ctx_dim:Union[int,List[int]], depth:int=1):
    if isinstance(ctx_dim, int):
      ctx_dim = [ctx_dim]*depth
    else:
      assert isinstance(ctx_dim, list) and depth == len(ctx_dim)
    self.norm = GroupNorm(32, channels)
    assert channels == n_heads * d_head
    self.proj_in = Linear(channels, n_heads * d_head)
    self.transformer_blocks = [BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head) for d in range(depth)]
    self.proj_out = Linear(n_heads * d_head, channels)

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    B, C, H, W = x.shape
    x_in = x
    x = self.norm(x)
    x = x.reshape(B, C, H*W).permute(0,2,1) # b c h w -> b c (h w) -> b (h w) c
    x = self.proj_in(x)
    for block in self.transformer_blocks:
      x = block(x, ctx=ctx)
    x = self.proj_out(x)
    x = x.permute(0,2,1).reshape(B, C, H, W) # b (h w) c -> b c (h w) -> b c h w
    return x + x_in

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L235
class Downsample:
  def __init__(self, channels:int):
    self.op = Conv2d(channels, channels, 3, stride=2, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    return self.op(x)

# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L242
class Upsample:
  def __init__(self, channels:int):
    self.conv = Conv2d(channels, channels, 3, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    bs,c,py,px = x.shape
    z = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    return self.conv(z)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/openaimodel.py#L472
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L257
class UNetModel:
  def __init__(self, adm_in_channels:int, in_channels:int, out_channels:int, model_channels:int, attention_resolutions:List[int], num_res_blocks:int, channel_mult:List[int], d_head:int, transformer_depth:List[int], ctx_dim:Union[int,List[int]]):
    self.in_channels = in_channels
    self.model_channels = model_channels
    self.out_channels = out_channels
    self.num_res_blocks = [num_res_blocks] * len(channel_mult)

    self.attention_resolutions = attention_resolutions
    self.dropout = 0.0
    self.channel_mult = channel_mult
    self.conv_resample = True
    self.num_classes = "sequential"
    self.use_checkpoint = False
    self.d_head = d_head

    time_embed_dim = model_channels * 4
    self.time_embed = [
      Linear(model_channels, time_embed_dim),
      Tensor.silu,
      Linear(time_embed_dim, time_embed_dim),
    ]

    self.label_emb = [
      [
        Linear(adm_in_channels, time_embed_dim),
        Tensor.silu,
        Linear(time_embed_dim, time_embed_dim),
      ]
    ]

    self.input_blocks = [
      [Conv2d(in_channels, model_channels, 3, padding=1)]
    ]
    input_block_channels = [model_channels]
    ch = model_channels
    ds = 1
    for idx, mult in enumerate(channel_mult):
      for _ in range(self.num_res_blocks[idx]):
        layers: List[Any] = [
          ResBlock(ch, time_embed_dim, model_channels*mult),
        ]
        ch = mult * model_channels
        if ds in attention_resolutions:
          n_heads = ch // d_head
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))

        self.input_blocks.append(layers)
        input_block_channels.append(ch)

      if idx != len(channel_mult) - 1:
        self.input_blocks.append([
          Downsample(ch),
        ])
        input_block_channels.append(ch)
        ds *= 2

    n_heads = ch // d_head
    self.middle_block: List = [
      ResBlock(ch, time_embed_dim, ch),
      SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[-1]),
      ResBlock(ch, time_embed_dim, ch),
    ]

    self.output_blocks = []
    for idx, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(self.num_res_blocks[idx] + 1):
        ich = input_block_channels.pop()
        layers = [
          ResBlock(ch + ich, time_embed_dim, model_channels*mult),
        ]
        ch = model_channels * mult

        if ds in attention_resolutions:
          n_heads = ch // d_head
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))

        if idx > 0 and i == self.num_res_blocks[idx]:
          layers.append(Upsample(ch))
          ds //= 2
        self.output_blocks.append(layers)

    self.out = [
      GroupNorm(32, ch),
      Tensor.silu,
      Conv2d(model_channels, out_channels, 3, padding=1),
    ]

  def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Tensor) -> Tensor:
    t_emb = timestep_embedding(tms, self.model_channels).cast(dtypes.float16)
    emb   = t_emb.sequential(self.time_embed)

    assert y.shape[0] == x.shape[0]
    emb = emb + y.sequential(self.label_emb[0])

    emb = emb.cast(dtypes.float16)
    ctx = ctx.cast(dtypes.float16)
    x   = x  .cast(dtypes.float16)

    def run(x:Tensor, bb) -> Tensor:
      if isinstance(bb, ResBlock): x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer): x = bb(x, ctx)
      else: x = bb(x)
      return x

    saved_inputs = []
    for b in self.input_blocks:
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)
    for bb in self.middle_block:
      x = run(x, bb)
    for b in self.output_blocks:
      x = x.cat(saved_inputs.pop(), dim=1)
      for bb in b:
        x = run(x, bb)

    return x.sequential(self.out)


class DiffusionModel:
  def __init__(self, *args, **kwargs):
    self.diffusion_model = UNetModel(*args, **kwargs)
