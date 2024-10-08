from tinygrad import Tensor, Device # type: ignore
from typing import Tuple

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
  a = Tensor.randn(GLOBAL_BS,2,2).pad(((0,GLOBAL_BS),None,None)).shard(GPUS+GPUS, axis=0).shrink(((0,GLOBAL_BS),None,None))
  b = Tensor.randn(GLOBAL_BS,2,2).pad(((GLOBAL_BS,0),None,None)).shard(GPUS+GPUS, axis=0).shrink(((GLOBAL_BS,2*GLOBAL_BS),None,None))
  c = Tensor.cat(a, b)
  print(f"full thing:\n{c.numpy()}\n")
  for i,v in enumerate(c.chunk(2)):
    print(f"chunk {i}:\n{v.numpy()}\n")

def ver4():
  from tinygrad.multi import MultiLazyBuffer, sint # type: ignore

  def cat_multi(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    if len(args) == 0: return self
    catargs = [self, *args]
    if all(isinstance(y.lazydata, MultiLazyBuffer) and all(y.lazydata.real) and (y.lazydata.axis == dim) for y in catargs) \
        and all(len(self.shape) == len(y.shape) and self.requires_grad == y.requires_grad and self.dtype == y.dtype and \
                all(y.shape[i] == s for i,s in enumerate(self.shape)) for y in args):
      return Tensor(MultiLazyBuffer(tuple(lb for y in catargs for lb in y.lazydata.lbs), dim), \
                    tuple(dev for y in catargs for dev in y.device), self.dtype)
  
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

  from tinygrad.lazy import LazyBuffer # type: ignore
  from tinygrad.ops import BinaryOps # type: ignore
  from typing import List
  import functools, itertools
  def copy_to_device(self:MultiLazyBuffer, device:str) -> LazyBuffer:
    if self.axis is None:
      # if we already have a copy on the device, return that
      for lb in self.real_lbs:
        if lb.device == device: return lb
      return self.real_lbs[0].copy_to_device(device)
    # copy lbs to device, pad to final shape, and sum
    llbs:List[LazyBuffer] = []
    deltas = [end-start if r else 0 for r,(start,end) in zip(self.real, self.bounds)]
    real_bounds = [(end-delta, end) for delta, end in zip(deltas, itertools.accumulate(deltas))]
    for lb,real,(start,end) in zip(self.lbs, self.real, real_bounds):
      if not real: continue
      pad_arg = tuple((0,0) if a != self.axis else (start, real_bounds[-1][1]-end) for a in range(len(lb.shape)))
      llbs.append(lb.copy_to_device(device).pad(pad_arg))
    return functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), llbs)
  MultiLazyBuffer.copy_to_device = copy_to_device

  a, b = [Tensor.randn(3,2,2).shard(GPUS, axis=0) for _ in range(2)]
  c = cat_multi(a, b)
  # print(f"full thing:\n{c.numpy()}\n")
  for i,v in enumerate(c.chunk(2)):
    print(v.shape)
    x = v.contiguous()
    y = x.to("CLANG")
    z = y.realize()
    print(f"chunk {i}:\n{z.numpy()}\n")

def test_multi_shrink():
  a = Tensor.randn(6,2,2).shard(GPUS, axis=0)
  b = a.shrink(((0,2),None,None))
  x = b.contiguous()
  y = x.to("CLANG")
  z = y.realize()
  print(z.numpy())

def ver5():
  if False:
    from tinygrad.multi import MultiLazyBuffer, sint # type: ignore
    from tinygrad.lazy import LazyBuffer # type: ignore
    from tinygrad.ops import BinaryOps # type: ignore
    from typing import List
    import functools, itertools
    def copy_to_device(self:MultiLazyBuffer, device:str) -> LazyBuffer:
      if self.axis is None:
        # if we already have a copy on the device, return that
        for lb in self.real_lbs:
          if lb.device == device: return lb
        return self.real_lbs[0].copy_to_device(device)
      # copy lbs to device, pad to final shape, and sum
      llbs:List[LazyBuffer] = []
      deltas = [end-start if r else 0 for r,(start,end) in zip(self.real, self.bounds)]
      real_bounds = [(end-delta, end) for delta, end in zip(deltas, itertools.accumulate(deltas))]
      for lb,real,(start,end) in zip(self.lbs, self.real, real_bounds):
        if not real: continue
        pad_arg = tuple((0,0) if a != self.axis else (start, real_bounds[-1][1]-end) for a in range(len(lb.shape)))
        llbs.append(lb.copy_to_device(device).pad(pad_arg))
      return functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), llbs)
    MultiLazyBuffer.copy_to_device = copy_to_device

  a = Tensor.rand(4,16).shard([Device.DEFAULT]*2, axis=0).shrink(((0,2),None)).realize()
  print(f'1: {a.shape}')
  print(f'2: {a.to("CLANG").shape}')
  print(f'3: {a.numpy().shape}')
  print(f'4: {a.to("CLANG").numpy().shape}')

def ver6():
  GPUS = tuple([Device.DEFAULT]*2)
  a = Tensor.rand(8,4).shard(GPUS, axis=0)
  b, c = a.chunk(2)
  b = b + Tensor.rand(4,4).shard((GPUS[0],))
  c = c - Tensor.rand(4,4).shard((GPUS[1],))
  d = Tensor.cat(b, c)
  print(d.numpy())

def ver7():
  a = Tensor.rand(4,4).shard(('NV:1','NV:2','NV:1','NV:2'), axis=0)
  b = Tensor.rand(4,4).shard(('NV:1','NV:2'), axis=None)
  c = a * b

def ver8():
  from tinygrad.nn import Linear
  from tinygrad.nn.state import get_state_dict

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(4)]
  DEVICE_BS = 2
  GLOBAL_BS = DEVICE_BS * len(GPUS)

  class Model:
    def __init__(self, size:int):
      self.in_1 = Linear(size, size*2)
      self.in_2 = Linear(size, size*2)
      self.out  = Linear(size*2, size)
    def __call__(self, x, cond):
      return self.out(self.in_1(x) + self.in_2(cond))

  model = Model(10)
  for w in get_state_dict(model).values():
    w.shard_(GPUS, axis=None)

  ITERATIONS = 10
  x      = Tensor.randn(GLOBAL_BS,10).shard(GPUS, axis=0)
  cond_u = Tensor.zeros(GLOBAL_BS,10).shard(GPUS, axis=0)
  cond_c = Tensor.randn(GLOBAL_BS,10).shard(GPUS, axis=0)
  for i in range(ITERATIONS):
    # latent_u, latent_c = model(Tensor.cat(x, x), Tensor.cat(cond_u, cond_c)).chunk(2)
    latent_u = model(x, cond_u)
    latent_c = model(x, cond_c)
    scaled = latent_u + 8.0 * (latent_c - latent_u)
    x = x + ((1.0 / ITERATIONS) * (x - scaled))
  print(x.numpy())

if __name__ == "__main__":
  # ver3()
  # ver4()
  # test_multi_shrink()
  # ver5()
  # ver6()
  # ver7()
  ver8()
