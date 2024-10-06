from tinygrad import Tensor, TinyJit, dtypes
from extra.models.unet import UNetModel # type: ignore

@TinyJit
def run_yes(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()

def run_no(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.manual_seed(0)

  model = UNetModel(adm_in_ch=2816, in_ch=4, out_ch=4, model_ch=320, attention_resolutions=[4, 2], num_res_blocks=2, channel_mult=[1, 2, 4], d_head=64, transformer_depth=[1, 2, 10], ctx_dim=2048, use_linear=True)

  shps = (
    (1, 4, 32, 32),
    (1,),
    (1, 77, 2048),
    (1, 2816),
    (1, 1, 1, 1),
    (1, 4, 32, 32),
  )

  for i in range(10):
    x_pred = {}
    args1 = [Tensor.randn(shp, dtype=dtypes.float16).realize() for shp in shps]
    args2 = [Tensor.randn(shp, dtype=dtypes.float16).realize() for shp in shps]
    for use_jit in [True, False]:
      def run(*args): return run_yes(*args) if use_jit else run_no(*args)
      x_u = run(model, *args1)
      x_c = run(model, *args2)
      x_pred[use_jit] = x_u + 8.0*(x_c - x_u)

    print(f"ITER {i:02d}: delta={Tensor.abs(x_pred[True] - x_pred[False]).mean().numpy():.4f}")
