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
import random

import numpy as np
import paddle
import paddle.amp as amp
import paddle.distributed as dist
import ppsci
from packaging.version import Version
from ppsci.utils import config, logger
from ppsci.utils.misc import AverageMeter
from ppsci.utils.save_load import (load_checkpoint, load_pretrain,
                                   save_checkpoint)
from visualdl import LogWriter


class Solver(object):
    """

    Args:
        cfg (AttrDict): Configuration parsed from yaml.
        mode (str, optional): Running mode. Defaults to "train".
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        rank = dist.get_rank()

        # set random seed
        seed = self.cfg["Global"].get("seed", 42)
        if seed is not None and seed is not False:
            assert isinstance(seed, int), \
                f"Global.seed({seed}) must be a integer"
            paddle.seed(seed + rank)
            np.random.seed(seed + rank)
            random.seed(seed + rank)

        # init logger
        self.output_dir = self.cfg["Global"]["output_dir"]
        log_file = os.path.join(
            self.output_dir,
            self.cfg["Arch"]["name"],
            f"{mode}.log"
        )
        self.log_freq = self.cfg["Global"].get("log_freq", 20)
        logger.init_logger(log_file=log_file)
        config.print_config(cfg)

        # init VisualDL
        self.vdl_writer = None
        if rank == 0 and self.cfg["Global"]["use_visualdl"]:
            vdl_writer_path = os.path.join(self.output_dir, "visualdl")
            os.makedirs(vdl_writer_path, exist_ok=True)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.cfg["Global"]["device"] in \
            ["cpu", "gpu", "xpu", "npu", "mlu", "ascend"]
        self.device = paddle.set_device(self.cfg["Global"]["device"])
        version = paddle.__version__ \
            if Version(paddle.__version__) != Version("0.0.0") else \
                f"develop({paddle.version.commit[:7]})"
        logger.info(f"Using paddlepaddle {version} on device {self.device}")

        # build geometry(ies)
        self.geom = ppsci.geometry.build_geometry(
            self.cfg["Geometry"]
        )

        # build model
        self.model = ppsci.arch.build_model(self.cfg["Arch"])

        # build equations
        self.equation = ppsci.equation.build_equation(self.cfg["Equation"])

        # init AMP
        self.use_amp = "AMP" in self.cfg
        if self.use_amp:
            self.amp_level = self.cfg["AMP"].pop("level", "O1").upper()
            self.scaler = amp.GradScaler(True, **self.cfg["AMP"])
        else:
            self.amp_level = "O0"


        # # build constraint(s)
        # pde = ppsci.equation.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)

        # self.model.load_dict(paddle.load("output/ldc2d/epoch_200.pdparams"))


        # interior_constraint = ppsci.constraint.InteriorConstraint(
        #     pde.equations,
        #     {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        #     geo,
        #     None,
        #     config.AttrDict({
        #         "batch_size":9801,
        #         "iters_per_epoch":100,
        #         "shuffle":True,
        #         "drop_last":True,
        #         "num_workers":2,
        #         "seed":42,
        #         "device":self.device,
        #         "use_shared_memory":False
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     {"continuity": 1e-4, "momentum_x": 1e-4, "momentum_y": 1e-4}
        # )

        # x = sympy.Symbol("x")
        # boundary_constraint_top = ppsci.constraint.BoundaryConstraint(
        #     {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
        #     {"u": 1.0, "v": 0.0},
        #     geo,
        #     lambda x, y: np.isclose(y, 0.05),
        #     config.AttrDict({
        #         "batch_size": 101,
        #         "iters_per_epoch": 100,
        #         "shuffle": True,
        #         "drop_last": True,
        #         "num_workers": 2,
        #         "seed": 42,
        #         "device": self.device,
        #         "use_shared_memory": True
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     {"u": 1.0 - 20.0 * sympy.Abs(x)},
        #     name="BC_top"
        # )
        # boundary_constraint_down = ppsci.constraint.BoundaryConstraint(
        #     {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
        #     {"u": 0.0, "v": 0.0},
        #     geo,
        #     lambda x, y: np.isclose(y, -0.05),
        #     config.AttrDict({
        #         "batch_size": 101,
        #         "iters_per_epoch": 100,
        #         "shuffle": True,
        #         "drop_last": True,
        #         "num_workers": 2,
        #         "seed": 42,
        #         "device": self.device,
        #         "use_shared_memory": True
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     name="BC_down"
        # )
        # boundary_constraint_left = ppsci.constraint.BoundaryConstraint(
        #     {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
        #     {"u": 0.0, "v": 0.0},
        #     geo,
        #     lambda x, y: np.isclose(x, -0.05),
        #     config.AttrDict({
        #         "batch_size": 99,
        #         "iters_per_epoch": 100,
        #         "shuffle": True,
        #         "drop_last": True,
        #         "num_workers": 2,
        #         "seed": 42,
        #         "device": self.device,
        #         "use_shared_memory": True
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     name="BC_left"
        # )
        # boundary_constraint_right = ppsci.constraint.BoundaryConstraint(
        #     {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
        #     {"u": 0.0, "v": 0.0},
        #     geo,
        #     lambda x, y: np.isclose(x, 0.05),
        #     config.AttrDict({
        #         "batch_size": 99,
        #         "iters_per_epoch": 100,
        #         "shuffle": True,
        #         "drop_last": True,
        #         "num_workers": 2,
        #         "seed": 42,
        #         "device": self.device,
        #         "use_shared_memory": True
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     name="BC_right"
        # )

        # validator = ppsci.validate.NumpyValidator(
        #     {"u": sympy.Symbol("u"), "v": sympy.Symbol("v")},
        #     {"u": 0.0, "v": 0.0},
        #     geo,
        #     None,
        #     config.AttrDict({
        #         "total_size":12001,
        #         "batch_size":12001,
        #         "iters_per_epoch":1,
        #         "shuffle":False,
        #         "drop_last":False,
        #         "num_workers":2,
        #         "seed":42,
        #         "device":self.device,
        #         "use_shared_memory":False
        #     }),
        #     ppsci.loss.MSELoss("sum"),
        #     ppsci.metric.RMSE()
        # )
        # self.validator = [
        #     validator
        # ]


    def train(self):
        """Training
        """
        epochs = self.cfg["Global"]["epochs"]
        self.iters_per_epoch = self.cfg["Global"]["iters_per_epoch"]

        save_freq = self.cfg["Global"].get("save_freq", 1)

        eval_during_train = self.cfg["Global"].get("eval_during_train", True)
        eval_freq = self.cfg["Global"].get("eval_freq", 1)
        start_eval_epoch = self.cfg["Global"].get("start_eval_epoch", 1)

        # gradient accumulation
        self.update_freq = self.cfg["Global"].get("update_freq", 1)
        self.global_step = 0

        best_metric = {
            "metric": float("inf"),
            "epoch": 0,
        }

        self.optimizer, self.lr_scheduler = ppsci.optimizer.build_optimizer(
            self.cfg["Optimizer"],
            [self.model],
            epochs,
            self.iters_per_epoch
        )
        if self.cfg["Global"]["checkpoints"] is not None:
            loaded_metric = load_checkpoint(
                self.cfg["Global"]["checkpoints"],
                self.model,
                self.optimizer
            )
            if isinstance(loaded_metric, dict):
                best_metric.update(loaded_metric)

        self.constraints = ppsci.constraint.build_constraint(
            self.cfg["Constraint"],
            self.equation,
            self.geom
        )

        self.train_output_info = {}
        self.train_time_info = {
            "batch_cost": AverageMeter("batch_cost", ".5f", postfix=" s,"),
            "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,")
        }

        # init train func
        self.train_mode = self.cfg["Global"].get("train_mode", None)
        if self.train_mode is None:
            self.train_epoch_func = ppsci.solver.train.train_epoch_func
        else:
            self.train_epoch_func = ppsci.solver.train.train_LBFGS_epoch_func

        # train epochs
        for epoch_id in range(1, epochs + 1):
            self.train_epoch_func(self, epoch_id, self.log_freq)

            metric_msg = ", ".join(
                [self.train_output_info[key].avg_info for key in self.train_output_info]
            )
            logger.info(f"[Train][Epoch {epoch_id}/{epochs}][Avg] {metric_msg}")
            self.train_output_info.clear()

            # evaluate during training
            if eval_during_train and \
                    epoch_id % eval_freq == 0 and \
                    epoch_id >= start_eval_epoch:
                cur_metric = self.eval(epoch_id)
                if cur_metric < best_metric["metric"]:
                    best_metric["metric"] = cur_metric
                    best_metric["epoch"] = epoch_id
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        self.cfg["Arch"]["name"],
                        "best_model"
                    )
            logger.info(
                f"[Eval][Epoch {epoch_id}][best metric: {best_metric['metric']}]"
            )
            logger.scaler(
                "eval_metric",
                cur_metric,
                epoch_id,
                self.vdl_writer
            )

            if self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model by save_freq
            if save_freq > 0 and epoch_id % save_freq == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.output_dir,
                    self.cfg["Arch"]["name"],
                    f"epoch_{epoch_id}"
                )

            # always save the latest model
            save_checkpoint(
                self.model,
                self.optimizer,
                {"metric": cur_metric, "epoch": epoch_id},
                self.output_dir,
                self.cfg["Arch"]["name"],
                "latest",
            )

        # close VisualDL
        if self.vdl_writer is not None:
            self.vdl_writer.close()

    def eval(self, epoch_id=0):
        """Evaluation
        """
        # load pretrained model if specified
        if self.cfg["Global"]["pretrained_model"] is not None:
            load_pretrain(self.model, self.cfg["Global"]["pretrained_model"])

        # init train func
        self.model.eval()

        self.eval_func = ppsci.solver.eval.eval_func
        # for directly evaluation
        if not hasattr(self, "validator"):
            self.validator = ppsci.validate.build_validator(
                self.cfg["Validator"],
                self.geom
            )

        self.eval_output_info = {}
        self.eval_time_info = {
            "batch_cost": AverageMeter("batch_cost", ".5f", postfix=" s,"),
            "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,")
        }

        result = self.eval_func(self, epoch_id, self.log_freq)
        logger.info(f"[Eval][Epoch {epoch_id}] {self.eval_output_info}")
        self.eval_output_info.clear()

        self.model.train()
        return result


    def predict(self):
        """Prediction
        """


    def export(self):
        """Export to inference model
        """
        pretrained_path = self.cfg["Global"]["pretrained_model"]
        if pretrained_path is not None:
            load_pretrain(self.model, pretrained_path)

        self.model.eval()

        input_spec = copy.deepcopy(self.cfg["Export"]["input_shape"])
        config.replace_shape_with_inputspec_(input_spec)
        static_model = paddle.jit.to_static(self.model, input_spec=input_spec)

        export_dir = self.cfg["Global"]["save_inference_dir"]
        save_path = os.path.join(export_dir, "inference")
        paddle.jit.save(static_model, save_path)
        logger.info(f"The inference model has been exported to {export_dir}.")


if __name__ == "__main__":
    cfg = None
    engine = Solver(cfg)
    engine.train()
