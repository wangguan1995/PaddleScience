# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Union

import paddle
import paddle.nn as nn

from .activation import get_activation
from .base import NetBase


class MLP(NetBase):
    """Multi Layer Perceptron Network
    """
    def __init__(self,
        input_keys: List[str],
        output_keys: List[str],
        num_layers: Optional[int],
        hidden_size: Union[int, List[int]],
        activation: str="tanh"
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        if isinstance(hidden_size, (tuple, list)):
            if num_layers is not None:
                raise ValueError(
                    f"num_layers must be None when hidden_size is specified"
                )
        elif isinstance(hidden_size, int):
            if not isinstance(num_layers, int):
                raise ValueError(
                    f"num_layers must be an int when hidden_size is an int"
                )
            hidden_size = [hidden_size] * num_layers
        else:
            raise ValueError(
                f"hidden_size must be list of int or int"
                f"but got {type(hidden_size)}"
            )

        # initialize FC layer(s)
        cur_size = len(self.input_keys)
        for _size in hidden_size:
            self.linears.append(nn.Linear(cur_size, _size))
            cur_size = _size
        self.linears.append(nn.Linear(cur_size, len(self.output_keys)))
        self.linears = nn.LayerList(self.linears)

        # initialize activation function
        self.act = get_activation(activation)

    def forward_tensor(self, x: paddle.Tensor):
        y = x
        for i, linear in enumerate(self.linears):
            y = linear(y)
            if i < len(self.linears) - 1:
                y = self.act(y)
        return y

    def forward(self, x: Dict[str, paddle.Tensor]):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_data(x, self.input_keys)
        y = self.forward_tensor(y)
        y = {
            key: y[:, i:i+1]
            for i, key in enumerate(self.output_keys)
        }

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y
