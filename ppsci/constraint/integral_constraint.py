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

from typing import Callable

import numpy as np
import sympy

from ..data.dataset import NamedArrayDataset
from ..geometry import Geometry
from ..utils.config import AttrDict
from ..utils.misc import convert_to_dict
from .base import Constraint


class IntegralrConstraint(Constraint):
    def __init__(
        self,
        label_expr,
        label_dict,
        geom: Geometry,
        criteria: Callable,
        data_cfg: AttrDict,
        loss,
        weight_dict=None,
        name="IgC"
    ):
        self.label_expr = label_expr
        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())
        input = geom.sample_interior(
            data_cfg.batch_size * data_cfg.iters_per_epoch,
            criteria=criteria
        )
        input = convert_to_dict(input, self.input_keys)

        label = {}
        for key, value in label_dict.items():
            if isinstance(value, (int, float)):
                label[key] = np.full_like(
                    next(iter(input.values())),
                    float(value)
                )
            elif isinstance(value, sympy.Basic):
                func = sympy.lambdify(
                    sympy.Symbol(geom.dim_keys),
                    value,
                    "numpy"
                )
                label[key] = func(**input)
            else:
                raise NotImplementedError(
                    f"type of {type(value)} is invalid yet."
                )

        weight = {
            key: np.ones_like(next(iter(label.values())))
            for key in label
        }
        if weight_dict is not None:
            for key in weight_dict:
                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(
                        next(iter(label.values())),
                        float(value)
                    )
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        sympy.Symbol(geom.dim_keys),
                        value,
                        "numpy"
                    )
                    weight[key] = func(**input)
                else:
                    raise NotImplementedError(
                        f"type of {type(value)} is invalid yet."
                    )
        dataset = NamedArrayDataset(input, label, weight)
        super().__init__(
            dataset,
            data_cfg,
            loss,
            name
        )

