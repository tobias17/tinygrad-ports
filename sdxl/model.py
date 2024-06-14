from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Conv2d
from tinygrad.nn.state import safe_load
from typing import Dict, List, Union, Callable
import os

# configs:
# https://github.com/Stability-AI/generative-models/blob/main/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/main/configs/inference/sd_xl_refiner.yaml

configs: Dict = {
   "SDXL_Base": {
      "model": {"adm_in_channels": 2816, "in_channels": 4, "out_channels": 4, "model_channels": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "num_head_channels": 64, "transformer_depth": [1, 2, 10], "context_dim": 2048},
      "conditioner": {},
      "first_stage_model": {},
   },
   "SDXL_Refiner": {
      "model": {"adm_in_channels": 2560, "in_channels": 4, "out_channels": 4, "model_channels": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "num_head_channels": 64, "transformer_depth": [4, 4, 4, 4], "context_dim": [1280, 1280, 1280, 1280]},
      "conditioner": {},
      "first_stage_model": {},
   }
}

# https://github.com/Stability-AI/generative-models/blob/059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9/sgm/modules/diffusionmodules/openaimodel.py#L472
class UNetModel:
   def __init__(self, adm_in_channels:int, in_channels:int, out_channels:int, model_channels:int, attention_resolutions:List[int], num_res_blocks:int, channel_mult:List[int], num_head_channels:int, transformer_depth:List[int], context_dim:Union[int,List[int]]):
      self.in_channels = in_channels
      self.model_channels = model_channels
      self.out_channels = out_channels
      self.num_res_blocks = [num_res_blocks] * len(channel_mult)

      self.attention_resolutions = attention_resolutions
      self.dropout = 0.0
      self.channel_mult = channel_mult
      self.conv_resample = True
      self.num_classes = None
      self.use_checkpoint = False
      self.num_heads = -1
      self.num_head_channels = num_head_channels
      self.num_heads_upsample = -1

      time_embed_dim = model_channels * 4
      self.time_embed = [
         Linear(model_channels, time_embed_dim),
         Tensor.silu,
         Linear(time_embed_dim, time_embed_dim),
      ]

      self.label_emb = [
         Linear(adm_in_channels, time_embed_dim),
         Tensor.silu,
         Linear(time_embed_dim, time_embed_dim),
      ]

      self.input_blocks = [
         Conv2d(in_channels, model_channels, 3, padding=1)
      ]
      input_block_channels = [model_channels]
      ch = model_channels
      ds = 1
      for idx, mult in enumerate(channel_mult):
         for nr in range(self.num_res_blocks[idx]):
            layers = [
               
            ]

   def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Tensor) -> Tensor:
      time_emb = timestep_embedding(tms, self.model_channels)
      return x

class Conditioner:
   pass

class FirstStageModel:
   pass

class SDXL:
   def __init__(self):
      self.model = UNetModel()
      self.conditioner = None
      self.first_stage_model = None

if __name__ == "__main__":
   weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
   d = safe_load(weight_path)
