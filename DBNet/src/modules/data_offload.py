# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Supported dataset operations applied on devices"""
import mindspore as ms
from mindspore import ops, nn
from mindspore.dataset.engine.offload import RandomColorAdjust


class Normalize(nn.Cell):
    """
    Normalize the input image with respect to mean and standard deviation.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__(auto_prefix=False)
        self.mean = ms.Tensor(mean, ms.float32)
        self.std = ms.Tensor(std, ms.float32)

    def construct(self, img):
        img = (img - self.mean.reshape((1, 1, 1, -1))) / self.std.reshape((1, 1, 1, -1))
        return img


class HWC2CHW(nn.Cell):
    """
    Transpose the input image from shape <H, W, C> to <C, H, W>.
    """
    def construct(self, img):
        img = ops.transpose(img, (0, 3, 1, 2))
        return img


class DataProcess(nn.Cell):
    """
    dataset pre process, include RandomColorAdjust, Normalize, HWC2CHW.
    """

    def __init__(self, mean, std, use_color_adjust=True):
        super(DataProcess, self).__init__()
        self.use_color_adjust = use_color_adjust
        self.color_adjust = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5, contrast=(1, 1), hue=(0, 0))
        self.norm = Normalize(mean=mean, std=std)
        self.hwc2chw = HWC2CHW()

    def construct(self, img):
        if self.use_color_adjust and self.training:
            img = self.color_adjust(img)
        img = self.norm(img)
        img = self.hwc2chw(img)
        img = ops.stop_gradient(img)
        return img
