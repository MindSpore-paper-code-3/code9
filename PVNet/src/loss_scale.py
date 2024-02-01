# Copyright 2021 Huawei Technologies Co., Ltd
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
"""dynamic loss scale """
import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore.nn import Cell
from mindspore import Tensor, RowTensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """tensor_grad_scale"""
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    """tensor_grad_scale_row_tensor"""
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    """_tensor_grad_overflow"""
    return grad_overflow(grad)


@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    """_tensor_grad_overflow_row_tensor"""
    return grad_overflow(grad.values)


class DynamicLossScaleUpdateCell(Cell):
    r"""
    Dynamic Loss scale update cell.

    For loss scaling training, the initial loss scaling value will be set to be `loss_scale_value`.
    In each training step, the loss scaling value  will be updated by loss scaling value/`scale_factor`
    when there is an overflow. And it will be increased by loss scaling value * `scale_factor` if there is no
    overflow for a continuous `scale_window` steps. This cell is used for Graph mode training in which all
    logic will be executed on device side(Another training mode is normal(non-sink) mode in which some logic will be
    executed on host).

    Args:
        loss_scale_value (float): Initializes loss scale.
        scale_factor (int): Coefficient of increase and decrease.
        scale_window (int): Maximum continuous training steps that do not have overflow.

    Inputs:
        - **inputs** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Raises:
        TypeError: If dtype of `inputs` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 loss_scale_value,
                 scale_factor,
                 scale_window):
        """__init__"""
        super(DynamicLossScaleUpdateCell, self).__init__()

        self.scale_window = Tensor(scale_window, dtype=mstype.int32)
        self.scale_factor = Tensor(scale_factor, dtype=mstype.float32)
        self.loss_scale_value = loss_scale_value

        self.cur_iter = Parameter(Tensor(1, dtype=mstype.int32), name="current_iterator_step")
        self.last_overflow_iter = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        self.select = P.Select()
        self.max = P.Maximum()
        self.minimum_loss_scale = Tensor(1.0, dtype=mstype.float32)
        self.reciprocal = P.Reciprocal()
        self.less_equal = P.LessEqual()
        self.logic_and = P.LogicalAnd()
        self.logic_not = P.LogicalNot()
        self.logic_or = P.LogicalOr()
        self.const_true = Tensor(True, dtype=mstype.bool_)

    def get_loss_scale(self):
        """get_loss_scale"""
        return self.loss_scale_value

    def construct(self, loss_scale, overflow):
        """construct"""
        overflow_cond = overflow
        loss_scale_on_overflow = self.select(overflow_cond, self.max(loss_scale * self.reciprocal(self.scale_factor),
                                                                     self.minimum_loss_scale), loss_scale)
        should_inc = self.less_equal(self.scale_window, self.cur_iter - self.last_overflow_iter)
        last_iter_cond = self.logic_or(overflow_cond, should_inc)
        last_overflow_iter = self.select(last_iter_cond, self.cur_iter, self.last_overflow_iter)
        last_iter = F.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        F.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        inc_cur_iter = F.depend(inc_cur_iter, last_iter)
        F.assign(self.cur_iter, inc_cur_iter)
        return overflow


class FixedLossScaleUpdateCell(Cell):
    """
    Static scale update cell, the loss scaling value will not be updated.

    For usage, refer to `DynamicLossScaleUpdateCell`.

    Args:
        loss_scale_value (float): Initializes loss scale.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, loss_scale_value):
        """__init__"""
        super(FixedLossScaleUpdateCell, self).__init__()
        self.loss_scale_value = loss_scale_value

    def get_loss_scale(self):
        """get_loss_scale"""
        return self.loss_scale_value

    def construct(self, _, overflow):
        """construct"""
        return overflow


GRADIENT_CLIP_TYPE = 0
GRADIENT_CLIP_VALUE = 1.0


class ClipGradients(Cell):
    """
    Clip gradients.
    Inputs:
        grads (tuple[Tensor]): Gradients.
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.

    Outputs:
        tuple[Tensor], clipped gradients.
    """

    def __init__(self):
        """__init__"""
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """construct"""
        if clip_type not in (0, 1):
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads


class TrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    r"""
    Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    TrainOneStepWithLossScaleCell will be compiled to be graph which takes `*inputs` as input data.
    The Tensor type of `scale_sense` is acting as loss scaling value. If you want to update it on host side,
    the value must be provided. If  the Tensor type of `scale_sense` is not given, the loss scale update logic
    must be provied by Cell type of `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        scale_sense (Union[Tensor, Cell]): If this value is Cell type, the loss scaling update logic cell.If this value
                                          is Tensor type, Tensor with shape :math:`()` or :math:`(1,)`.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scaling value.

        - **loss** (Tensor) -  Tensor with shape :math:`()`.
        - **overflow** (Tensor) -  Tensor with shape :math:`()`, type is bool.
        - **loss scaling value** (Tensor) -  Tensor with shape :math:`()`

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, network, optimizer, scale_sense):
        """__init__"""
        super(TrainOneStepWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.hyper_map = C.HyperMap()
        self.base = Tensor(1, mstype.float32)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.allreduce = P.AllReduce()
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.gpu_target = (context.get_context("device_target") == "GPU")
        self.loss_scaling_manager = None
        self.clip_gradients = ClipGradients()
        if isinstance(scale_sense, Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of scale_sense must be (1,) or (), but got {}".format(scale_sense.shape))
        else:
            raise TypeError("The scale_sense must be Cell or Tensor, but got {}".format(type(scale_sense)))

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        # grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens
