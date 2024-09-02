from tinygrad import Tensor, Device # type: ignore
from typing import Tuple
import itertools

GPUS = tuple([f"{Device.DEFAULT}" for _ in range(3)])
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

def ver4():
  from tinygrad.multi import MultiLazyBuffer, sint # type: ignore

  def cat_multi(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    if len(args) == 0: return self
    catargs = [self, *args]
    assert all(isinstance(y.lazydata, MultiLazyBuffer) and all(y.lazydata.real) and (y.lazydata.axis == dim) for y in catargs)
    assert all(len(self.shape) == len(y.shape) and self.requires_grad == y.requires_grad and self.dtype == y.dtype and all(y.shape[i] == s for i,s in enumerate(self.shape)) for y in args)
    return Tensor(MultiLazyBuffer(tuple(lb for y in catargs for lb in y.lazydata.lbs), dim), tuple(dev for y in catargs for dev in y.device), self.dtype)
  
  def shrink(self:MultiLazyBuffer, arg:Tuple[Tuple[sint, sint], ...]):
    if self.axis is None or arg[self.axis] == (0, self.shape[self.axis]):
      bounds_subset = None
    else:
      bounds_subset = []
      for bound in self.bounds:
        if bound[0] >= arg[self.axis][0] and bound[1] <= arg[self.axis][1]:
          bounds_subset.append(bound)
      if any(b[0] == arg[self.axis][0] for b in bounds_subset) and any(b[1] == arg[self.axis][1] for b in bounds_subset):
        assert all(arg[i] == (0, s) or i == self.axis for i,s in enumerate(self.shape)), "cannot shrink sharded and non-sharded axis at the same time"
        idxs = [self.bounds.index(b) for b in bounds_subset]
        return MultiLazyBuffer([lb if i in idxs else lb.const(0) for i,lb in enumerate(self.lbs)], self.axis, [i in idxs for i in range(len(self.lbs))])

    assert self.axis is None or arg[self.axis] == (0, self.shape[self.axis]) or arg[self.axis] in self.bounds, f"shrinking not supported for {arg=}"
    if self.axis is not None and arg[self.axis] in self.bounds and arg[self.axis] != (0, self.shape[self.axis]):
      assert all(arg[i] == (0, s) or i == self.axis for i,s in enumerate(self.shape)), "cannot shrink sharded and non-sharded axis at the same time"
      idx = self.bounds.index(arg[self.axis])
      # zero out other lbs to not create lb reference
      return MultiLazyBuffer([lb if i==idx else lb.const(0) for i,lb in enumerate(self.lbs)], self.axis, [i==idx for i in range(len(self.lbs))])
    return MultiLazyBuffer([x.shrink(tuple((0, x.shape[self.axis]) if a == self.axis else s for a,s in enumerate(arg))) for x in self.lbs],
                           self.axis, self.real)
  MultiLazyBuffer.shrink = shrink

  a, b = [Tensor.randn(GLOBAL_BS,2,2).shard(GPUS, axis=0) for _ in range(2)]
  c = cat_multi(a, b)
  print(f"full thing:\n{c.numpy()}\n")
  for i,v in enumerate(c.chunk(2)):
    print(f"chunk {i}:\n{v.numpy()}\n")

def test_multi_shrink():
  a = Tensor.randn(3,2,2).shard(GPUS, axis=0)
  b = a.shrink(((0,1),None,None))
  print(b.numpy())

if __name__ == "__main__":
  test_multi_shrink()
