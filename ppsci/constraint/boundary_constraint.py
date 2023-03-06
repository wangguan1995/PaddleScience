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

import types
from typing import Callable

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr

from ..data.dataset import NamedArrayDataset
from ..geometry import Geometry
from ..utils.config import AttrDict
from .base import Constraint


class BoundaryConstraint(Constraint):
    """Class for boundary constraint.

    Args:
        label_expr (Dict[str, sympy.Basic]): Expression of how to compute label.
        label_dict (Dict[str, Union[float, sympy.Basic]]): Value of label.
        geom (Geometry): Geometry which constraint applied on.
        dataloader_cfg (AttrDict): Config of building a dataloader.
        loss (LossBase): Loss object.
        random (str, optional): Random method for sampling points in geometry.
            Defaults to "pseudo".
        criteria (Callable, optional): Criteria for finely define subdomain in geometry.
            Defaults to None.
        evenly (bool, optional): Whether to use envely distribution in sampling.
            Defaults to False.
        weight_dict (Dict[str, Union[float, sympy.Basic]], optional): Weight for label
            if specified. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "BC".
    """
    def __init__(
        self,
        label_expr,
        label_dict,
        geom,
        dataloader_cfg,
        loss,
        random="pseudo",
        criteria=None,
        evenly=False,
        weight_dict=None,
        name="BC"
    ):
        self.label_expr = label_expr
        for label_name, label_expr in self.label_expr.items():
            if isinstance(label_expr, str):
                self.label_expr[label_name] = parse_expr(label_expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())
        if isinstance(criteria, str):
            criteria = eval(criteria)

        input = geom.sample_boundary(
            dataloader_cfg["batch_size"] * dataloader_cfg["iters_per_epoch"],
            random,
            criteria,
            evenly
        )

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
            elif isinstance(value, types.FunctionType):
                func = value
                label[key] = func(input)
            else:
                raise NotImplementedError(
                    f"type of {type(value)} is invalid yet."
                )

        weight = {
            key: np.ones_like(next(iter(label.values())))
            for key in label
        }
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(
                        next(iter(label.values())),
                        float(value)
                    )
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in geom.dim_keys],
                        value,
                        "numpy"
                    )
                    weight[key] = func(**{k: input[k] for k in geom.dim_keys})
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                else:
                    raise NotImplementedError(
                        f"type of {type(value)} is invalid yet."
                    )
        dataset = NamedArrayDataset(input, label, weight)
        super().__init__(
            dataset,
            dataloader_cfg,
            loss,
            name
        )
