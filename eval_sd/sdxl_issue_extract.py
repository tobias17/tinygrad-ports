from tinygrad import Tensor, TinyJit

@TinyJit
def run_yes(x, add) -> Tensor:
  return (x + add).realize()

def run_not(x, add) -> Tensor:
  return (x + add).realize()

if __name__ == "__main__":
  Tensor.manual_seed(0)

  for i in range(5):
    a1, a2, a3, a4 = [Tensor.randn(32, 32) for _ in range(4)]

    y_comp = run_yes(a1, a2) - run_yes(a3, a4)
    n_comp = run_not(a1, a2) - run_not(a3, a4)

    print(f"ITER {i:02d}: delta={Tensor.abs(y_comp - n_comp).mean().numpy():.4f}")
