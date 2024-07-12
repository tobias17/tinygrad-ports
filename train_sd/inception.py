from tinygrad import Tensor # type: ignore
from tinygrad.nn import Conv2d, BatchNorm2d # type: ignore

class BasicConv2d:
  def __init__(self, in_ch:int, out_ch:int, **kwargs):
    self.conv = Conv2d(in_ch, out_ch, bias=False, **kwargs)
    self.bn   = BatchNorm2d(out_ch, eps=0.001)
  
  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential([self.conv, self.bn, Tensor.relu])

class InceptionA:
  def __init__(self, in_ch:int, pool_feat:int):
    self.branch1x1 = BasicConv2d(in_ch, 64, kernel_size=1)

    self.branch5x5_1 = BasicConv2d(in_ch, 48, kernel_size=1)
    self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

    self.branch_pool = BasicConv2d(in_ch, pool_feat, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch5x5_1, self.branch5x5_2]),
      x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2, self.branch3x3dbl_3]),
      self.branch_pool(x.avg_pool2d(kernel_size=3, stride=1, dilation=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionB:
  def __init__(self, in_ch:int):
    self.branch3x3 = BasicConv2d(in_ch, 384, kernel_size=3, stride=2)

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
  
  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch3x3(x),
      x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2, self.branch3x3dbl_3]),
      x.max_pool2d(kernel_size=3, stride=2)
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionC:
  def __init__(self, in_ch, ch_7x7):
    self.branch1x1 = BasicConv2d(in_ch, 192, kernel_size=1)

    self.branch7x7_1 = BasicConv2d(in_ch, ch_7x7, kernel_size=1)
    self.branch7x7_2 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7_3 = BasicConv2d(ch_7x7, 192, kernel_size=(7, 1), padding=(3, 0))

    self.branch7x7dbl_1 = BasicConv2d(in_ch, ch_7x7, kernel_size=1)
    self.branch7x7dbl_2 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7dbl_3 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7dbl_4 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7dbl_5 = BasicConv2d(ch_7x7, 192, kernel_size=(1, 7), padding=(0, 3))

    self.branch_pool = BasicConv2d(in_ch, 192, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch7x7_1, self.branch7x7_2, self.branch7x7_3]),
      x.sequential([self.branch7x7dbl_1, self.branch7x7dbl_2, self.branch7x7dbl_3, self.branch7x7dbl_4, self.branch7x7dbl_5]),
      self.branch_pool(x.avg_pool2d(x, kernel_size=3, stride=1, dilation=1))
    ]
    return Tensor.cat(*outputs, dim=1)


class Inception3:
  def __init__(self, num_classes:int=1000, aux_logits:bool=False):
    self.aux_logits = aux_logits
    self.transform_input = False
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.Mixed_5b = inception_a(192, pool_features=32)
    self.Mixed_5c = inception_a(256, pool_features=64)
    self.Mixed_5d = inception_a(288, pool_features=64)
    self.Mixed_6a = inception_b(288)
    self.Mixed_6b = inception_c(768, channels_7x7=128)
    self.Mixed_6c = inception_c(768, channels_7x7=160)
    self.Mixed_6d = inception_c(768, channels_7x7=160)
    self.Mixed_6e = inception_c(768, channels_7x7=192)


def fid_inception_v3():
  pass

class InceptionV3:
  def __init__(self):
    self.output_blocks = [2048]
    self.blocks = [
      []
    ]

