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

from collections import OrderedDict
from copy import deepcopy

from .mae import MAE
from .mse import MSE
from .rl2 import RelativeL2
from .rmse import RMSE

__all__ = [
    "MAE",
    "MSE",
    "RelativeL2",
    "RMSE",
    "build_metric",
]


def build_metric(cfg):
    """Build metric.

    Args:
        cfg (List[AttrDict]): List of metric config.

    Returns:
        Dict[str, Metric]: Dict of callable metric object.
    """
    cfg = deepcopy(cfg)

    metric_dict = OrderedDict()
    for _item in cfg:
        metric_cls = next(iter(_item.keys()))
        metric_cfg = _item.pop(metric_cls)
        metric = eval(metric_cls)(**metric_cfg)
        metric_dict[metric_cls] = metric
    return metric_dict
