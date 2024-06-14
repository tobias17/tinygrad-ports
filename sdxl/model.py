from tinygrad.nn.state import safe_load
import os

weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
d = safe_load(weight_path)
print(d)
