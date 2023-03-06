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
        super().__init__()
        nu = sympy.Number(nu)
        rho = sympy.Number(rho)

        # independent variable
        t, x, y, z = self.create_symbols("t x y z")
        invars = [x, y, z][: dim]
        if time:
            invars = [t] + invars

        # dependent variable
        u = self.create_function("u", invars)
        v = self.create_function("v", invars)
        w = self.create_function("w", invars) if dim == 3 else sympy.Number(0)
        p = self.create_function("p", invars)

        # continuity equation
        continuity = u.diff(x) + v.diff(y) + w.diff(z)

        # momentum equation
        momentum_x = u.diff(t) + u * u.diff(x) + v * u.diff(y) + w * u.diff(z) - \
            nu / rho * u.diff(x, 2) - nu / rho * u.diff(y, 2) - \
            nu / rho * u.diff(z, 2) + 1.0 / rho * p.diff(x)

        momentum_y = v.diff(t) + u * v.diff(x) + v * v.diff(y) + w * v.diff(z) - \
            nu / rho * v.diff(x, 2) - nu / rho * v.diff(y, 2) - \
            nu / rho * v.diff(z, 2) + 1.0 / rho * p.diff(y)

        momentum_z = w.diff(t) + u * w.diff(x) + v * w.diff(y) + w * w.diff(z) - \
            nu / rho * w.diff(x, 2) - nu / rho * w.diff(y, 2) - \
            nu / rho * w.diff(z, 2) + 1.0 / rho * p.diff(z)

        self.equations["continuity"] = continuity
        self.equations["momentum_x"] = momentum_x
        self.equations["momentum_y"] = momentum_y
        if dim == 3:
            self.equations["momentum_z"] = momentum_z
