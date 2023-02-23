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

from __future__ import annotations

import types
from typing import Dict

import paddle
import sympy
from ppsci.arch.mlp import MLP
from ppsci.gradient import clear, hessian, jacobian


class ExpressionSolver(paddle.nn.Layer):
    """Expression Solver

    Args:
        input_keys (Dict[str]): List of string for input keys.
        output_keys (Dict[str]): List of string for output keys.
        model (nn.Layer): Model to get output variables from input variables.
    """
    def __init__(self, input_keys, output_keys, model):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.model = model
        self.expr_dict = {}
        self.output_dict = {}

    def solve_expr(self, f):
        if getattr(f, "name", None) in self.output_dict:
            return self.output_dict[f.name]
        if isinstance(f, sympy.Symbol):
            f_name = f.name
            if f_name in self.model.output_keys:
                out_dict = self.model(self.output_dict)
                self.output_dict.update(out_dict)
                return self.output_dict[f_name]
            else:
                raise ValueError(
                    f"varname {f_name} not exist!"
                )
        elif isinstance(f, sympy.Function):
            f_name = f.name
            out_dict = self.model(self.output_dict)
            self.output_dict.update(out_dict)
            return self.output_dict[f_name]
        elif isinstance(f, sympy.Derivative):
            ys = self.solve_expr(f.args[0])
            ys_name = f.args[0].name
            if ys_name not in self.output_dict:
                self.output_dict[ys_name] = ys
            xs = self.solve_expr(f.args[1][0])
            xs_name = f.args[1][0].name
            if xs_name not in self.output_dict:
                self.output_dict[xs_name] = xs
            order = f.args[1][1]
            if order == 1:
                der = jacobian(self.output_dict[ys_name], self.output_dict[xs_name])
                out_name = f"{ys_name}__{xs_name}"
                if out_name not in self.output_dict:
                    self.output_dict[out_name] = der
                return der
            elif order == 2:
                der = hessian(self.output_dict[ys_name], self.output_dict[xs_name])
                out_name = f"{ys_name}__{xs_name}__{xs_name}"
                if out_name not in self.output_dict:
                    self.output_dict[out_name] = der
                return der
        elif isinstance(f, sympy.Number):
            return float(f)
        elif isinstance(f, sympy.Add):
            results = [self.solve_expr(arg) for arg in f.args]
            out = results[0]
            for i in range(1, len(results)):
                out = out + results[i]
            return out
        elif isinstance(f, sympy.Mul):
            results = [self.solve_expr(arg) for arg in f.args]
            return results[0] * results[1]
        elif isinstance(f, sympy.Pow):
            results = [self.solve_expr(arg) for arg in f.args]
            return results[0] ** results[1]
        else:
            raise ValueError(
                f"Node type {type(f)} is unknown"
            )

    def forward(self, input_dict):
        self.output_dict = input_dict
        for name, expr in self.expr_dict.items():
            if isinstance(expr, sympy.Basic):
                self.output_dict[name] = self.solve_expr(expr)
            elif isinstance(expr, types.FunctionType):
                output_dict = expr(input_dict)
                self.output_dict.update(output_dict)
            else:
                raise TypeError(
                    f"expr type({type(expr)}) is invalid"
                )

        # clear gradient cache
        clear()

        return {
            k: self.output_dict[k]
            for k in self.output_keys
        }

    def add_target_expr(self, expr, expr_name):
        self.expr_dict[expr_name] = expr

    def __str__(self):
        ret = f"input: {self.input_keys}, output: {self.output_keys}\n" + \
            "\n".join(
                [f"{name} = {expr}" for name, expr in self.expr_dict.items()]
            )
        return ret


if __name__ == "__main__":
    nu = 0.01
    rho = 3.0
    nu = sympy.Number(nu)
    rho = sympy.Number(rho)
    t = sympy.Symbol("t")
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    d = {
        "t": t,
        "x": x,
        "y": y
    }
    # dependent variable
    u = sympy.Function("u")(*d)
    v = sympy.Function("v")(*d)
    p = sympy.Function("p")(*d)
    # continuity equation
    continuity = u.diff(x) + v.diff(y)
    momentum_x = u.diff(t) + u * u.diff(x) + v * u.diff(y) - nu / rho * u.diff(x).diff(x) - nu / rho * u.diff(y).diff(y) + sympy.Number(1.0) / rho * p.diff(x)
    momentum_y = v.diff(t) + u * v.diff(x) + v * v.diff(y) - nu / rho * v.diff(x).diff(x) - nu / rho * v.diff(y).diff(y) + sympy.Number(1.0) / rho * p.diff(y)

    N = 64
    input_keys = ["t", "x", "y"]
    model_output_keys = ["u", "v", "p"]
    in_dim = len(input_keys)
    out_dim = len(model_output_keys)
    input_data = paddle.randn([N, in_dim])
    input_data.stop_gradient = False
    input_dict = {
        key: input_data[:, i:i+1]
        for i, key in enumerate(input_keys)
    }

    model = MLP(input_keys, model_output_keys, num_layers=None, hidden_size=[16, 64, 16])
    output_keys = model_output_keys + ["continuity", "momentum_x", "momentum_y"]
    evaluator = ExpressionSolver(input_keys, output_keys, model)
    evaluator.add_target_expr(continuity, "continuity")
    evaluator.add_target_expr(momentum_x, "momentum_x")
    evaluator.add_target_expr(momentum_y, "momentum_y")

    # 动态图结果
    output_dict = evaluator(input_dict)
    momentum_x_val = output_dict["momentum_x"]
    continuity_val = output_dict["continuity"]
    momentum_y_val = output_dict["momentum_y"]
    clear()

    # 动转静转换
    # from paddle.static import InputSpec
    # static_evaluator = paddle.jit.to_static(
    #     evaluator,
    #     input_spec=[{"t": InputSpec(shape=[None, 1], name="t"), "x":InputSpec(shape=[None, 1], name="x"), "y":InputSpec(shape=[None, 1], name="y")}]
    # )
    # # 动转静保存
    # paddle.jit.save(evaluator, "./expr_solver")

    # # 动转静加载
    # static_evaluator = paddle.jit.load("./expr_solver")
    # static_evaluator.eval()
    # # 动转静测试
    # out = static_evaluator(*list(input_dict.values()))
    # for v in out:
    #     print(f"{v.shape}")
    # exit()

    txy = paddle.concat([input_dict[k] for k in input_keys], axis=-1)
    _t, _x, _y = txy[:, 0:1], txy[:, 1:2], txy[:, 2:3]
    _t.stop_gradient = False
    _x.stop_gradient = False
    _y.stop_gradient = False
    uvp = model.forward_tensor(paddle.concat([_t, _x, _y], axis=-1))
    _u, _v, _p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    momentum_x_gt = jacobian(_u, _t) + _u * jacobian(_u, _x) + _v * jacobian(_u, _y) - \
        float(nu) / float(rho) * hessian(_u, _x) - \
            float(nu) / float(rho) * hessian(_u, _y) + \
                1.0 / float(rho) * jacobian(_p, _x)
    momentum_y_gt = jacobian(_v, _t) + _u * jacobian(_v, _x) + _v * jacobian(_v, _y) - \
        float(nu) / float(rho) * hessian(_v, _x) - \
            float(nu) / float(rho) * hessian(_v, _y) + \
                1.0 / float(rho) * jacobian(_p, _y)
    continuity_gt = jacobian(_u, _x) + jacobian(_v, _y)
    clear()

    print(paddle.allclose(momentum_x_val, momentum_x_gt))
    print(paddle.allclose(momentum_y_val, momentum_y_gt))
    print(paddle.allclose(continuity_val, continuity_gt))
