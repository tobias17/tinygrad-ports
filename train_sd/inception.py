from tinygrad import Tensor # type: ignore
from tinygrad.nn import Conv2d, BatchNorm2d, Linear # type: ignore

from typing import Optional, Dict

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
      x.max_pool2d(kernel_size=3, stride=2),
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
      self.branch_pool(x.avg_pool2d(x, kernel_size=3, stride=1, dilation=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionD:
  def __init__(self, in_ch:int):
    self.branch3x3_1 = BasicConv2d(in_ch, 192, kernel_size=1)
    self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

    self.branch7x7x3_1 = BasicConv2d(in_ch, 192, kernel_size=1)
    self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      x.sequential([self.branch3x3_1, self.branch3x3_2]),
      x.sequential([self.branch7x7x3_1, self.branch7x7x3_2, self.branch7x7x3_3, self.branch7x7x3_4]),
      x.max_pool2d(kernel_size=3, stride=2),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionE:
  def __init__(self, in_ch:int):
    self.branch1x1 = BasicConv2d(in_ch, 320, kernel_size=1)

    self.branch3x3_1  = BasicConv2d(in_ch, 384, kernel_size=1)
    self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 448, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
    self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch_pool = BasicConv2d(in_ch, 192, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    branch3x3 = self.branch3x3_1(x)
    branch3x3dbl = x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2])
    outputs = [
      self.branch1x1(x),
      Tensor.cat(self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3), dim=1),
      Tensor.cat(self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl), dim=1),
      self.branch_pool(x.avg_pool2d(kernel_size=3, stride=1, dilation=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionAux:
  def __init__(self, in_ch:int, num_classes:int):
    self.conv0 = BasicConv2d(in_ch, 128, kernel_size=1)
    self.conv1 = BasicConv2d(128, 768, kernel_size=5)
    # self.conv1.stddev = 0.01
    self.fc = Linear(768, num_classes)
    # self.fc.stddev = 0.001

  def __call__(self, x:Tensor) -> Tensor:
    x = x.avg_pool2d(kernel_size=5, stride=3).sequential([self.conv0, self.conv1])
    x = x.adaptive_avg_pool2d((1,1)).flatten()
    return self.fc(x)

class Inception3:
  def __init__(self, num_classes:int=1000, aux_logits:bool=False, cls_map:Optional[Dict]=None):
    if cls_map is None: cls_map = {}

    self.aux_logits = aux_logits
    self.transform_input = False
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
    self.maxpool1 = lambda x: Tensor.max_pool2d(x, kernel_size=3, stride=2)
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
    self.maxpool2 = lambda x: Tensor.max_pool2d(x, kernel_size=3, stride=2)
    self.Mixed_5b  = cls_map.get("A",InceptionA)(192, pool_features=32)
    self.Mixed_5c  = cls_map.get("A",InceptionA)(256, pool_features=64)
    self.Mixed_5d  = cls_map.get("A",InceptionA)(288, pool_features=64)
    self.Mixed_6a  = cls_map.get("B",InceptionB)(288)
    self.Mixed_6b  = cls_map.get("C",InceptionC)(768, channels_7x7=128)
    self.Mixed_6c  = cls_map.get("C",InceptionC)(768, channels_7x7=160)
    self.Mixed_6d  = cls_map.get("C",InceptionC)(768, channels_7x7=160)
    self.Mixed_6e  = cls_map.get("C",InceptionC)(768, channels_7x7=192)
    self.AuxLogits = cls_map.get("Aux",InceptionAux)(768, num_classes)
    self.Mixed_7a  = cls_map.get("D",InceptionD)(768)
    self.Mixed_7b  = cls_map.get("E1",InceptionE)(1280)
    self.Mixed_7c  = cls_map.get("E2",InceptionE)(2048)
    self.avgpool = AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(p=dropout)
    self.fc = Linear(2048, num_classes)


def fid_inception_v3():
  pass

class InceptionV3:
  def __init__(self):
    self.output_blocks = [2048]
    self.blocks = [
      []
    ]

