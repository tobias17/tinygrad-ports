from tinygrad import Tensor, TinyJit

@TinyJit
def run_yes(x, add) -> Tensor:
  return (x + add).realize()

def run_not(x, add) -> Tensor:
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

    y_comp = (y_out_1 - y_out_2).realize()
    n_comp = (n_out_1 - n_out_2).realize()

    print(f"ITER {i:02d}: delta={Tensor.abs(y_comp - n_comp).mean().numpy():.4f}")
