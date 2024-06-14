from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load
from typing import Dict
import os

# configs:
# https://github.com/Stability-AI/generative-models/blob/main/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/main/configs/inference/sd_xl_refiner.yaml

configs: Dict = {
   "SDXL_Base": {
      "model": {},
      "conditioner": {},
      "first_stage_model": {},
   },
   "SDXL_Refiner": {
      "model": {},
      "conditioner": {},
      "first_stage_model": {},
   }
}

class UNetModel:
   def __call__(x:Tensor) -> Tensor:
      return x

class Conditioner:
   pass

class FirstStageModel:
   pass

class SDXL:
   model: UNetModel

   def __init__(self):
      model = UNetModel()
      conditioner = None
      first_stage_model = None

if __name__ == "__main__":
   weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
   d = safe_load(weight_path)
