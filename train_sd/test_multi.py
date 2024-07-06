from tinygrad import Tensor, Device, TinyJit # type: ignore

if __name__ == "__main__":
  GPUS = [f'{Device.DEFAULT}:{i}' for i in range(4)]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  def initial_test():
    print("\nINITIAL TEST")
    a = Tensor.randn(GLOBAL_BS, 4).shard(GPUS, axis=0)

    b = Tensor.arange(1000).sqrt().realize()
    t = Tensor.randint(GLOBAL_BS, low=0, high=1000)
    b_t = b[t].reshape(GLOBAL_BS, 1)

    c = b_t.shard(GPUS, axis=0) * a
    print(c.numpy())
  # initial_test()

  def gather_test():
    print("\nGATHER TEST")
    b = Tensor.arange(1000).sqrt().realize().shard(GPUS, axis=None)
    t = Tensor.randint(GLOBAL_BS, low=0, high=1000).shard(GPUS, axis=0)
    print(f"b.device: {b.device}")
    print(f"t.device: {t.device}")
    b_t = b.gather(-1, t)
    print(b_t.numpy())
    print(b_t.device)
  # gather_test()

  def jit_shard_test():
    @TinyJit
    def run(a, b) -> Tensor:
      return (a.shard(GPUS, axis=0) + b).realize()
    for _ in range(8):
      aa = Tensor.randn(GLOBAL_BS, 4)
      bb = Tensor.randn(GLOBAL_BS, 4).shard(GPUS, axis=0)
      print(run(aa.realize(), bb.realize()).numpy())
  jit_shard_test()
