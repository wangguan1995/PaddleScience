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

import sympy
from .base import PDE


class NavierStokes(PDE):
    """NavierStokes

    Args:
        nu (float): _description_
        rho (float): _description_
        dim (int): _description_
        time (bool): _description_
    """
    def __init__(self, nu: float, rho: float, dim: int, time: bool):
        nu = sympy.Number(nu)
        rho = sympy.Number(rho)

        # independent variable
        t = sympy.Symbol("t")
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")
        invars = [x, y, z][: dim]
        if time:
            invars = [t] + invars

        # dependent variable
        u = sympy.Function("u")(*invars)
        v = sympy.Function("v")(*invars)
        w = sympy.Function("w")(*invars) if dim == 3 else sympy.Number(0)
        p = sympy.Function("p")(*invars)

        # continuity equation
        continuity = u.diff(x) + v.diff(y) + w.diff(z)

        # momentum equation
        momentum_x = u.diff(t) + u * u.diff(x) + v * u.diff(
            y) + w * u.diff(z) - nu / rho * u.diff(x).diff(
                x) - nu / rho * u.diff(y).diff(y) - nu / rho * u.diff(
                    z).diff(z) + 1.0 / rho * p.diff(x)
        momentum_y = v.diff(t) + u * v.diff(x) + v * v.diff(
            y) + w * v.diff(z) - nu / rho * v.diff(x).diff(
                x) - nu / rho * v.diff(y).diff(y) - nu / rho * v.diff(
                    z).diff(z) + 1.0 / rho * p.diff(y)
        momentum_z = w.diff(t) + u * w.diff(x) + v * w.diff(
            y) + w * w.diff(z) - nu / rho * w.diff(x).diff(
                x) - nu / rho * w.diff(y).diff(y) - nu / rho * w.diff(
                    z).diff(z) + 1.0 / rho * p.diff(z)

        super().__init__()
        self.equations["continuity"] = continuity
        self.equations["momentum_x"] = momentum_x
        self.equations["momentum_y"] = momentum_y
        if dim == 3:
            self.equations["momentum_z"] = momentum_z
