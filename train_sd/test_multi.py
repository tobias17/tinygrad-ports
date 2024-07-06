from tinygrad import Tensor, Device # type: ignore

if __name__ == "__main__":
  GPUS = [f'{Device.DEFAULT}:{i}' for i in range(4)]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  a = Tensor.randn(GLOBAL_BS, 4).shard(GPUS, axis=0)

  b = Tensor.arange(1000).sqrt().realize()
  t = Tensor.randint(GLOBAL_BS, low=0, high=1000)
  b_t = b[t].reshape(GLOBAL_BS, 1)

  c = b_t.shard(GPUS, axis=0) * a
  print(c.numpy())
