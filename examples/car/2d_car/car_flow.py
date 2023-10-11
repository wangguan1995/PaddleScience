# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.s


import numpy as np
import paddle

import ppsci
from ppsci.loss import base
from ppsci.utils import config
from ppsci.utils import logger


class MSELoss(base.Loss):
    def __init__(
        self,
        input_str,
    ):
        super().__init__("False")

    def forward(self, output_dict, label_dict, weight_dict=None):
        return 0


class Mass_imbalance_metric(ppsci.metric.base.Metric):
    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}

        metric_dict["mass imbalance"] = paddle.sum(
            output_dict["area"] * paddle.abs(output_dict["continuity"])
        )
        return metric_dict


class Momentum_imbalance_metric(ppsci.metric.base.Metric):
    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        metric_dict["momentum imbalance"] = output_dict["area"].T.matmul(
            (
                paddle.abs(output_dict["momentum_x"])
                + paddle.abs(output_dict["momentum_y"])
            )
        )
        return metric_dict


class Integral_translate:
    def __init__(self):
        self.trans = 0
        self.trans_array = 5 * np.random.random((128 * 100,)) - channel_length[1]
        self.trans_array[0] = -1.03928

    def __call__(self, x, y):
        n = x.shape[0]
        x = x + self.trans
        if self.trans > -1.6 and self.trans < 0.6:  # x for car head  # x for car tail
            # print("Sample number on integral Line ", n)
            x = x.reshape(n, 1)
            y = y.reshape(n, 1)
            z = np.full_like(x, geo.bounds[2][0])
            points = np.concatenate([x, y, z], 1)
            logic_sdf = geo_extrusion.sdf_func(points) <= 0
            # print(logic_sdf)
        else:
            logic_sdf = np.full(n, True)
        return logic_sdf

    def set_trans(self, index):
        self.trans = self.trans_array[index]
        return self.trans


if __name__ == "__main__":
    # initialization
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    OUTPUT_DIR = "./output" if not args.output_dir else args.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # params for domain
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    gap = 0.15 + 0.1
    inlet_vel = 1.5
    nu = 0.01
    max_distance = (channel_width[1] - channel_width[0]) / 2

    # define geometry
    channel = ppsci.geometry.Channel(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )

    geo_extrusion = ppsci.geometry.Mesh("../data/extrusion.stl")
    geo_area = 4.11548
    geo = ppsci.geometry.Mesh("../data/1010_car_fluid.stl", geo_extrusion, geo_area)

    geo.compute_properties(True, True, True)
    geo_inlet = ppsci.geometry.Line(
        (channel_length[0], channel_width[0]),
        (channel_length[0], channel_width[1]),
        (-1, 0),
    )
    geo_outlet = ppsci.geometry.Line(
        (channel_length[1], channel_width[0]),
        (channel_length[1], channel_width[1]),
        (1, 0),
    )

    integral_line = ppsci.geometry.Line(
        (0, channel_width[0]), (0, channel_width[1]), (1, 0)
    )

    equation = ppsci.utils.misc.Prettydefaultdict()
    equation["ZeroEquation"] = ppsci.equation.ZeroEquation(0.01, max_distance)
    equation["NavierStokes"] = ppsci.equation.NavierStokes(
        equation["ZeroEquation"].expr, 1.0, 2, False, True
    )

    equation["GradNormal"] = ppsci.equation.GradNormal(grad_var="c", dim=2, time=False)
    equation["NormalDotVec"] = ppsci.equation.NormalDotVec(("u", "v"))

    ITERS_PER_EPOCH = 1000

    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": False,
        },
        "num_workers": 1,
    }

    def parabola(
        input, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
    ):
        x = input["y"]
        factor = (4 * height) / (-(inter_1**2) - inter_2**2 + 2 * inter_1 * inter_2)
        return factor * (x - inter_1) * (x - inter_2)

    # integral continuity
    integral_criteria = Integral_translate()
    integral_continuity = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vel": 1},
        integral_line,
        {**train_dataloader_cfg, "batch_size": 4, "integral_batch_size": 128},
        ppsci.loss.IntegralLoss("sum"),
        criteria=integral_criteria,
        weight_dict={"normal_dot_vel": 0.1},
        name="integral_continuity",
    )

    constraint_inlet = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "c": lambda d: d["c"]},
        {"u": parabola, "v": 0, "c": 0},
        geo_inlet,
        {**train_dataloader_cfg, "batch_size": 64},
        ppsci.loss.MSELoss("sum"),
        name="inlet",
    )

    constraint_outlet = ppsci.constraint.BoundaryConstraint(
        {"p": lambda d: d["p"]},
        {"p": 0},
        geo_outlet,
        {**train_dataloader_cfg, "batch_size": 64},
        ppsci.loss.MSELoss("sum"),
        name="outlet",
    )

    constraint_channel_wall = ppsci.constraint.BoundaryConstraint(
        {
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "normal_gradient_c": lambda d: equation["GradNormal"].equations[
                "normal_gradient_c"
            ](d),
        },
        {"u": 0, "v": 0, "normal_gradient_c": 0},
        channel,
        {**train_dataloader_cfg, "batch_size": 2500},
        ppsci.loss.MSELoss("sum"),
        weight_dict={"u": 1, "v": 1, "normal_gradient_c": 1},
        name="channel_wall",
    )

    constraint_car = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"]},
        {"u": 0, "v": 0},
        geo,
        {**train_dataloader_cfg, "batch_size": 500},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (x > -2) & (x < 1) & (y < 0.2),
        name="hs_wall",
    )

    constraint_flow_interior = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geo,
        {**train_dataloader_cfg, "batch_size": 4800},
        ppsci.loss.MSELoss("sum"),
        weight_dict={"continuity": "sdf", "momentum_x": "sdf", "momentum_y": "sdf"},
        name="interior_flow",
    )

    constraint = {
        constraint_inlet.name: constraint_inlet,
        constraint_outlet.name: constraint_outlet,
        constraint_channel_wall.name: constraint_channel_wall,
        constraint_car.name: constraint_car,
        constraint_flow_interior.name: constraint_flow_interior,
        integral_continuity.name: integral_continuity,
    }

    model_flow = ppsci.arch.MLP(
        ("x", "y"), ("u", "v", "p"), 6, 512, "silu", weight_norm=True
    )
    model_heat = ppsci.arch.MLP(("x", "y"), ("c"), 6, 512, "silu", weight_norm=True)

    model = ppsci.arch.ModelList((model_flow, model_heat))

    # set training hyper-parameters
    EPOCHS = 500 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        0.95,
        5000,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=1,
        log_freq=100,
        equation=equation,
        checkpoint_path="./output/checkpoints/epoch_270",
    )

    # solver.train()

    import meshio

    mesh = meshio.read("../data/ChannelMesh.msh")
    input_dict = {
        "x": mesh.points[:, 0].reshape(-1, 1),
        "y": mesh.points[:, 1].reshape(-1, 1),
    }
    output_dict = solver.predict(input_dict, batch_size=1000)
    output_dict = {k: v.numpy() for k, v in output_dict.items()}
    ppsci.visualize.save_vtu_to_mesh(
        "./predict.vtu",
        {**input_dict, **output_dict},
        ("x", "y"),
        ("u", "v", "p"),
        mesh=mesh,
    )
