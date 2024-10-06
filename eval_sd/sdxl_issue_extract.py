from tinygrad import Tensor, TinyJit, dtypes

@TinyJit
def run_yes(x, c_out, add):
  return (x*c_out + add).realize()

def run_no(x, c_out, add):
  return (x*c_out + add).realize()

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.manual_seed(0)

  shps = (
    (1, 4, 32, 32),
    (1, 1, 1, 1),
    (1, 4, 32, 32),
  )

  for i in range(5):
    x_pred = {}
    args1 = [Tensor.randn(shp, dtype=dtypes.float32).realize() for shp in shps]
    args2 = [Tensor.randn(shp, dtype=dtypes.float32).realize() for shp in shps]
    for use_jit in [True, False]:
      def run(*args): return run_yes(*args) if use_jit else run_no(*args)
      x_u = run(*args1)
      x_c = run(*args2)
      x_pred[use_jit] = x_u + 8.0*(x_c - x_u)

    print(f"ITER {i:02d}: delta={Tensor.abs(x_pred[True] - x_pred[False]).mean().numpy():.4f}")
