from tinygrad import Tensor, TinyJit, dtypes
from extra.models.unet import ResBlock, SpatialTransformer, Downsample, Upsample, timestep_embedding # type: ignore

from typing import Optional, List, Union, Tuple, Any
from tinygrad.nn import Linear, Conv2d, GroupNorm

class UNetModel:
  def __init__(self, adm_in_ch:Optional[int], in_ch:int, out_ch:int, model_ch:int, attention_resolutions:List[int], num_res_blocks:int, channel_mult:List[int], transformer_depth:List[int], ctx_dim:Union[int,List[int]], use_linear:bool=False, d_head:Optional[int]=None, n_heads:Optional[int]=None):
    self.model_ch = model_ch
    self.num_res_blocks = [num_res_blocks] * len(channel_mult)

    self.attention_resolutions = attention_resolutions
    self.d_head  = d_head
    self.n_heads = n_heads
    def get_d_and_n_heads(dims:int) -> Tuple[int,int]:
      if self.d_head is None:
        assert self.n_heads is not None, f"d_head and n_heads cannot both be None"
        return dims // self.n_heads, self.n_heads
      else:
        assert self.n_heads is None, f"d_head and n_heads cannot both be non-None"
        return self.d_head, dims // self.d_head

    time_embed_dim = model_ch * 4
    self.time_embed = [
      Linear(model_ch, time_embed_dim),
      Tensor.silu,
      Linear(time_embed_dim, time_embed_dim),
    ]

    if adm_in_ch is not None:
      self.label_emb = [
        [
          Linear(adm_in_ch, time_embed_dim),
          Tensor.silu,
          Linear(time_embed_dim, time_embed_dim),
        ]
      ]

    self.input_blocks: List[Any] = [
      [Conv2d(in_ch, model_ch, 3, padding=1)]
    ]
    input_block_channels = [model_ch]
    ch = model_ch
    ds = 1
    for idx, mult in enumerate(channel_mult):
      for _ in range(self.num_res_blocks[idx]):
        layers: List[Any] = [
          ResBlock(ch, time_embed_dim, model_ch*mult),
        ]
        ch = mult * model_ch
        if ds in attention_resolutions:
          d_head, n_heads = get_d_and_n_heads(ch)
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[idx]))

        self.input_blocks.append(layers)
        input_block_channels.append(ch)

      if idx != len(channel_mult) - 1:
        self.input_blocks.append([
          Downsample(ch),
        ])
        input_block_channels.append(ch)
        ds *= 2

    d_head, n_heads = get_d_and_n_heads(ch)
    self.middle_block: List = [
      ResBlock(ch, time_embed_dim, ch),
      SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[-1]),
      ResBlock(ch, time_embed_dim, ch),
    ]

    self.output_blocks = []
    for idx, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(self.num_res_blocks[idx] + 1):
        ich = input_block_channels.pop()
        layers = [
          ResBlock(ch + ich, time_embed_dim, model_ch*mult),
        ]
        ch = model_ch * mult

        if ds in attention_resolutions:
          d_head, n_heads = get_d_and_n_heads(ch)
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[idx]))

        if idx > 0 and i == self.num_res_blocks[idx]:
          layers.append(Upsample(ch))
          ds //= 2
        self.output_blocks.append(layers)

    self.out = [
      GroupNorm(32, ch),
      Tensor.silu,
      Conv2d(model_ch, out_ch, 3, padding=1),
    ]

  def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Optional[Tensor]=None) -> Tensor:
    t_emb = timestep_embedding(tms, self.model_ch).cast(dtypes.float16)
    emb   = t_emb.sequential(self.time_embed)

    if y is not None:
      assert y.shape[0] == x.shape[0]
      emb = emb + y.sequential(self.label_emb[0])

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


@TinyJit
def run_yes(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()

def run_no(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.manual_seed(0)

  model = UNetModel(adm_in_ch=36, in_ch=4, out_ch=4, model_ch=48, attention_resolutions=[4, 2], num_res_blocks=2, channel_mult=[1, 2, 4], d_head=8, transformer_depth=[1, 2, 10], ctx_dim=64, use_linear=True)

  shps = (
    (1, 4, 32, 32),
    (1,),
    (1, 22, 64),
    (1, 36),
    (1, 1, 1, 1),
    (1, 4, 32, 32),
  )

  for i in range(5):
    x_pred = {}
    args1 = [Tensor.randn(shp, dtype=dtypes.float32).realize() for shp in shps]
    args2 = [Tensor.randn(shp, dtype=dtypes.float32).realize() for shp in shps]
    for use_jit in [True, False]:
      def run(*args): return run_yes(*args) if use_jit else run_no(*args)
      x_u = run(model, *args1)
      x_c = run(model, *args2)
      x_pred[use_jit] = x_u + 8.0*(x_c - x_u)

    print(f"ITER {i:02d}: delta={Tensor.abs(x_pred[True] - x_pred[False]).mean().numpy():.4f}")
