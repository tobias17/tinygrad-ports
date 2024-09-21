from tinygrad import Tensor, dtypes, nn
from tinygrad.nn.state import get_parameters
from extra.models.unet import CrossAttention, BasicTransformerBlock, SpatialTransformer, UNetModel # type: ignore
from typing import List, Callable
import numpy as np
import unittest

def make_input(*shape:int, count:int=1, dtype=dtypes.float16) -> List[Tensor]:
  return [Tensor.rand(*shape, dtype=dtype) for _ in range(count)]

def randomize_weights(model):
  for w in get_parameters(model):
    w.replace(Tensor.rand(*w.shape)).realize()
  return model

def make_calls(func:Callable[[Tensor,Tensor],Tensor], b1_i1:Tensor, b2_i1:Tensor, b1_i2:Tensor, b2_i2:Tensor):
  x1, x2 = func(Tensor.cat(b1_i1, b2_i1), Tensor.cat(b1_i2, b2_i2)).chunk(2)
  y1, y2 = func(b1_i1, b1_i2), func(b2_i1, b2_i2)

  np.testing.assert_allclose(x1.numpy(), y1.numpy(), atol=1e-6, rtol=1e-3)
  np.testing.assert_allclose(x2.numpy(), y2.numpy(), atol=1e-6, rtol=1e-3)

class Dual_Call_Tests(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(42)

  def test_add(self):
    make_calls(Tensor.add, *make_input(1, 32, count=4))

  def test_linear(self):
    l = randomize_weights(nn.Linear(32, 8))
    make_calls((lambda x,y: l(x)+l(y)), *make_input(1, 32, count=4))

  def test_cross_attention(self):
    ca = randomize_weights(CrossAttention(32, 64, 4, 8))
    make_calls(ca, *make_input(1, 32, count=2), *make_input(1, 64, count=2))

  def test_self_attention(self):
    ca = randomize_weights(CrossAttention(32, 32, 4, 8))
    make_calls((lambda x,y: ca(x)+ca(y)), *make_input(1, 32, count=4))

  def test_basic_transformer_block(self):
    btb = randomize_weights(BasicTransformerBlock(32, 64, 4, 8))
    make_calls(btb, *make_input(1, 4, 32, count=2), *make_input(1, 4, 64, count=2))

  def test_spatial_transformer(self):
    st = randomize_weights(SpatialTransformer(16, 8, 2, 32, True))
    make_calls(st, *make_input(1, 16, 8, 8, count=2), *make_input(1, 4, 32, count=2))

  def test_unet(self):
    params = {"adm_in_ch": 32, "in_ch": 4, "out_ch": 4, "model_ch": 128, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 16, "transformer_depth": [1, 2, 10], "ctx_dim": 88, "use_linear": True}
    unet = randomize_weights(UNetModel(**params))
    b1_x, b2_x = make_input(1, 4, 32, 32, count=2)
    b1_y, b2_y = make_input(1, 32, count=2)
    b1_c, b2_c = make_input(1, 17, 88, count=2)

    tms = Tensor([50.0])
    b1_t, b2_t = unet(Tensor.cat(b1_x, b2_x), Tensor.cat(tms, tms), Tensor.cat(b1_c, b2_c), Tensor.cat(b1_y, b2_y)).chunk(2)
    b1_s, b2_s = unet(b1_x, tms, b1_c, b1_y), unet(b2_x, tms, b2_c, b2_y)

    np.testing.assert_allclose(b1_t.numpy(), b1_s.numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(b2_t.numpy(), b2_s.numpy(), atol=1e-6, rtol=1e-3)

if __name__ == "__main__":
  unittest.main()
