from tinygrad import Tensor, Device
from typing import Tuple

GPUS = tuple([f"{Device.DEFAULT}:{i}" for i in [3,4,5]])
DEVICE_BS = 1
GLOBAL_BS = DEVICE_BS * len(GPUS)

def ver1():

  def pack(a:Tensor, b:Tensor) -> Tensor:
    return Tensor.cat(a, b)
    # return Tensor.cat(a.unsqueeze(1), b.unsqueeze(1)).reshape(a.shape[0]+b.shape[0],*a.shape[1:])

  def unpack(x:Tensor) -> Tuple[Tensor,Tensor]:
    a, b = x.chunk(2)
    return (a, b)

  a_in = Tensor.randn(GLOBAL_BS,64,64).realize().shard(GPUS, axis=0)
  b_in = Tensor.randn(GLOBAL_BS,64,64).realize().shard(GPUS, axis=0)
  print(f"a_in:  {a_in.shape}")
  print(f"b_in:  {b_in.shape}")

  x_out = pack(a_in, b_in)
  print(f"x_out: {x_out.shape}")

  a_out, b_out = unpack(x_out)
  print(f"a_out: {a_out.shape}")
  print(f"b_out: {b_out.shape}")

def ver2():
  a, b = [Tensor.randn(GLOBAL_BS,64,64).shard(GPUS, axis=0) for _ in range(2)]
  c = Tensor.cat(a.unsqueeze(0), b.unsqueeze(0), dim=0)
  c = c.reshape(a.shape[0]+b.shape[0],*a.shape[1:])

def ver3():
  a = Tensor.randn(GLOBAL_BS,2,2).pad(((0,GLOBAL_BS),None,None)).shard(GPUS+GPUS).shrink(((0,GLOBAL_BS),None,None))
  b = Tensor.randn(GLOBAL_BS,2,2).pad(((GLOBAL_BS,0),None,None)).shard(GPUS+GPUS).shrink(((GLOBAL_BS,2*GLOBAL_BS),None,None))
  c = Tensor.cat(a, b)
  print(f"full thing:\n{c.numpy()}\n")
  for i,v in enumerate(c.chunk(2)):
    print(f"chunk {i}:\n{v.numpy()}\n")

if __name__ == "__main__":
  ver3()
