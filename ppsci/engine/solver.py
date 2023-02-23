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

import copy
import os

import numpy as np
import paddle
import paddle.amp as amp
import paddle.optimizer as optim
import ppsci
import sympy
from ppsci.utils import config, logger
from ppsci.utils.misc import AverageMeter

from PaddleScience_refactor.ppsci import utils


class Solver(object):
    def __init__(self, cfg, mode="train"):
        """Initialization

        Args:
            cfg (AttrDict): Configuration parsed from yaml.
            mode (str, optional): Running mode. Defaults to "train".
        """
        self.config = cfg
        self.mode = mode

        logger.init_logger(log_file="./refactor_test.log")
        self.device = paddle.set_device("gpu")
        logger.info("train with paddle {} and device {}".format(paddle.__version__, self.device))

        # gradient accumulation
        self.update_freq = 1
        geo = ppsci.geometry.Rectangle([-0.05, -0.05], [0.05, 0.05])

        pde = ppsci.equation.pde.NavierStokes(nu=0.01, rho=1.0)

        self.model = ppsci.arch.MLP(["x", "y"], ["u", "v", "p"], 9, 50)
        # self.model.load_dict(paddle.load("output/ldc2d/epoch_200.pdparams"))


        interior_constraint = ppsci.constraint.InteriorConstraint(
            pde.equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geo,
            None,
            config.AttrDict({
                "batch_size":9801,
                "iters_per_epoch":100,
                "shuffle":True,
                "drop_last":True,
                "num_workers":2,
                "seed":42,
                "device":self.device,
                "use_shared_memory":False
            }),
            ppsci.loss.MSELoss("sum"),
            {"continuity": 1e-4, "momentum_x": 1e-4, "momentum_y": 1e-4}
        )

        x = sympy.Symbol("x")
        boundary_constraint_top = ppsci.constraint.BoundaryConstraint(
            {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
            {"u": 1.0, "v": 0.0},
            geo,
            lambda x, y: np.isclose(y, 0.05),
            config.AttrDict({
                "batch_size": 101,
                "iters_per_epoch": 100,
                "shuffle": True,
                "drop_last": True,
                "num_workers": 2,
                "seed": 42,
                "device": self.device,
                "use_shared_memory": True
            }),
            ppsci.loss.MSELoss("sum"),
            {"u": 1.0 - 20.0 * sympy.Abs(x)},
            name="BC_top"
        )
        boundary_constraint_down = ppsci.constraint.BoundaryConstraint(
            {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
            {"u": 0.0, "v": 0.0},
            geo,
            lambda x, y: np.isclose(y, -0.05),
            config.AttrDict({
                "batch_size": 101,
                "iters_per_epoch": 100,
                "shuffle": True,
                "drop_last": True,
                "num_workers": 2,
                "seed": 42,
                "device": self.device,
                "use_shared_memory": True
            }),
            ppsci.loss.MSELoss("sum"),
            name="BC_down"
        )
        boundary_constraint_left = ppsci.constraint.BoundaryConstraint(
            {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
            {"u": 0.0, "v": 0.0},
            geo,
            lambda x, y: np.isclose(x, -0.05),
            config.AttrDict({
                "batch_size": 99,
                "iters_per_epoch": 100,
                "shuffle": True,
                "drop_last": True,
                "num_workers": 2,
                "seed": 42,
                "device": self.device,
                "use_shared_memory": True
            }),
            ppsci.loss.MSELoss("sum"),
            name="BC_left"
        )
        boundary_constraint_right = ppsci.constraint.BoundaryConstraint(
            {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
            {"u": 0.0, "v": 0.0},
            geo,
            lambda x, y: np.isclose(x, 0.05),
            config.AttrDict({
                "batch_size": 99,
                "iters_per_epoch": 100,
                "shuffle": True,
                "drop_last": True,
                "num_workers": 2,
                "seed": 42,
                "device": self.device,
                "use_shared_memory": True
            }),
            ppsci.loss.MSELoss("sum"),
            name="BC_right"
        )
        self.constraints = [
            interior_constraint,
            boundary_constraint_top,
            boundary_constraint_down,
            boundary_constraint_left,
            boundary_constraint_right
        ]

        validator = ppsci.validate.NumpyValidator(
            {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
            {"u": 0.0, "v": 0.0},
            geo,
            None,
            config.AttrDict({
                "total_size":12001,
                "batch_size":12001,
                "iters_per_epoch":1,
                "shuffle":False,
                "drop_last":False,
                "num_workers":2,
                "seed":42,
                "device":self.device,
                "use_shared_memory":False
            }),
            ppsci.loss.MSELoss("sum"),
            ppsci.metric.RMSE()
        )
        self.validator = [
            validator
        ]


    def train(self):
        """Training
        """
        self.epoch = 200
        self.iters_per_epoch = 100
        self.lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
            self.epoch,
            self.iters_per_epoch,
            0.001,
            warmup_epoch=int(0.05 * self.epoch),
            by_epoch=False
        )()

        self.optimizer = optim.Adam(
            self.lr_scheduler,
            parameters=self.model.parameters()
        )

        self.log_freq = self.cfg["Global"].get("log_freq", 20)
        self.save_freq = self.cfg["Global"].get("save_freq", 1)
        self.eval_freq = self.cfg["Global"].get("eval_freq", 1)
        self.update_freq = self.cfg["Global"].get("update_freq", 1)

        self.use_amp = "AMP" in self.cfg
        if self.use_amp:
            self.amp_level = self.cfg["AMP"].pop("level", "O1").upper()
            self.scaler = amp.GradScaler(
                True,
                **self.cfg["AMP"]
            )

        self.output_info = {}
        self.time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        self.global_step = 0
        self.train_epoch_func = ppsci.engine.train.train_epoch_func
        for epoch_id in range(1, self.epoch + 1):
            self.train_epoch_func(self, epoch_id, self.print_batch_step)
            if epoch_id % self.eval_freq == 0:
                eval_result = self.eval(epoch_id)
                print(eval_result)

            if epoch_id % self.save_freq == 0:
                utils.save_checkpoint(
                    self.model,
                    self.optimizer,
                    eval_result,
                    "output",
                    "ldc2d",
                    f"epoch_{epoch_id}"
                )

    def eval(self, epoch_id=0):
        """Evaluation
        """
        self.model.eval()

        self.eval_func = ppsci.engine.eval.eval_func
        result = self.eval_func(epoch_id)

        self.model.train()
        return result


    def predict(self):
        """Prediction
        """


    def export(self):
        """Export to inference model
        """
        if self.config["Global"]["pretrained_model"] is not None:
            utils.load_pretrain(
                self.model,
                self.config["Global"]["pretrained_model"]
            )
        self.model.eval()

        input_spec = copy.deepcopy(self.config["Export"]["input_shape"])
        config.replace_shape_with_inputspec_(input_spec)
        static_model = paddle.jit.to_static(
            self.model,
            input_spec=input_spec
        )

        export_dir = self.config["Global"]["save_inference_dir"]
        save_path = os.path.join(
            export_dir,
            "inference"
        )
        paddle.jit.save(static_model, save_path)
        logger.info(
            f"The inference model has been exported to {export_dir}."
        )


if __name__ == "__main__":
    cfg = None
    engine = Solver(cfg)
    engine.train()
