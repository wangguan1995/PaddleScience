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

from copy import deepcopy

from .integral import IntegralLoss
from .l1 import L1Loss
from .l2 import L2Loss
from .mse import MSELoss

__all__ = [
    "IntegralLoss",
    "L1Loss",
    "L2Loss",
    "MSELoss",
    "build_loss",
]


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (AttrDict): Loss config.
    Returns:
        Loss: Callable loss object.
    """
    cfg = deepcopy(cfg)

    loss_cls = cfg.pop("name")
    loss = eval(loss_cls)(**cfg)
    return loss
