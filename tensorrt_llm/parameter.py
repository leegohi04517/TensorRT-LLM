# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import math
from typing import Optional, Sequence, Union

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on

from ._utils import str_dtype_to_trt, torch_to_numpy, trt_dtype_to_torch
from .functional import Tensor, constant
from .logger import logger


class Parameter:
    _DEFAULT_DTYPE = trt.DataType.FLOAT

    def __init__(self,
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 shape: Sequence[int] = None,
                 dtype: Union[str, trt.DataType] = None):
        if dtype is None:
            logger.warning(
                f'Parameter dtype is None, using default dtype: {self._DEFAULT_DTYPE}, it is recommended to always specify dtype explicitly'
            )
        dtype = self._DEFAULT_DTYPE if dtype is None else dtype
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        if value is None:
            assert isinstance(shape, (list, tuple))
            if len(shape) == 2:
                # Xavier initialization see https://paperswithcode.com/method/xavier-initialization
                v_range = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
            else:
                v_range = 0.1

            if dtype == trt.DataType.INT8:
                upper = math.ceil(128 * v_range)
                value = torch.randint(-upper,
                                      upper, (shape),
                                      dtype=trt_dtype_to_torch(dtype),
                                      device='cuda')
                # value ~ U[int(-128 * v_range), int(128 * v_range)]
            else:
                value = torch.randn(
                    (shape), dtype=trt_dtype_to_torch(dtype),
                    device='cuda') * 2 - 1
                # value ~ N[-v_range, v_range]
                value = value * v_range
        self._value = self._regularize_value(value)

    @property
    def value(self) -> Tensor:
        if isinstance(self._value, np.ndarray):
            self._value = constant(self._value)

        return self._value

    @property
    def raw_value(self) -> np.ndarray:
        assert isinstance(
            self._value, np.ndarray
        ), "Must be np.ndarray. Proper usage: get parameter.raw_value before getting parameter.value"
        return self._value

    @value.setter
    def value(self, v: Union[np.ndarray, torch.Tensor]):
        v = self._regularize_value(v)
        assert v.shape == self._value.shape, \
            f'The value updated is not the same shape as the original. ' \
            f'Updated: {v.shape}, original: {self._value.shape}'
        self._value = v

    def _get_weights(self) -> trt.Weights:
        return self._value.producer.weights if isinstance(self._value,
                                                          Tensor) else None

    def _regularize_value(self, value):
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, torch.Tensor):
            return torch_to_numpy(value)
        raise TypeError(
            f'Expected numpy.ndarray or torch.Tensor, got {type(value)}')
