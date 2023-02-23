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

from typing import Dict, List

import paddle.nn as nn

from .lr_scheduler import *
from .optimizer import *

__all__ = ["build_optimizer"]

def build_optimizer(
    opt_name: str,
    lr_sch_name: str,
    opt_cfg: Dict,
    lr_sch_cfg: Dict,
    model_list: List[nn.Layer],
):
    lr_sch = eval(lr_sch_name)(**lr_sch_cfg)

    optimizer = eval(opt_name)(
        learning_rate=lr_sch,
        **opt_cfg
    )(model_list)

    return optimizer
