from tinygrad import Tensor, TinyJit, dtypes

@TinyJit
def run_yes(x, add):
  return (x + add).realize()

def run_not(x, add):
  return (x + add).realize()

if __name__ == "__main__":
  Tensor.manual_seed(0)

  shps = (
    (1, 4, 32, 32),
    (1, 4, 32, 32),
  )

  for i in range(5):
    args1 = [Tensor.randn(shp).realize() for shp in shps]
    args2 = [Tensor.randn(shp).realize() for shp in shps]

    y_out_1, y_out_2 = run_yes(*args1), run_yes(*args2)
    n_out_1, n_out_2 = run_not(*args1), run_not(*args2)

    def comp(o1, o2) -> Tensor:
      return o1 + 8.0*(o2 - o1)

    delta = comp(y_out_1, y_out_2) - comp(n_out_1, n_out_2)
    print(f"ITER {i:02d}: delta={delta.abs().mean().numpy():.4f}")
