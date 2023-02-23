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

import time
from collections import defaultdict

import paddle.amp as amp

from ..utils import profiler
from ..utils.expression import ExpressionSolver
from .printer import log_info, update_loss


def train_LBFGS_epoch_func(solver, epoch_id, log_freq):
    """Train function for one epoch

    Args:
        solver (Solver): Main solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log training information every `log_freq` steps.
    """
    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        reader_cost = 0
        batch_cost = 0
        loss_dict = defaultdict(float)
        def closure():
            """Closure function for LBFGS optimizer.

            Returns:
                Tensor: Computed loss.
            """
            total_loss = 0
            total_batch_size = []
            reader_tic = time.perf_counter()
            for _constraint in solver.constraints:
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)

                # profile code below
                profiler.add_profiler_step(solver.cfg["profiler_options"])
                if iter_id == 5:
                    # 5 step for warmup
                    for key in solver.time_info:
                        solver.time_info[key].reset()
                reader_cost += time.perf_counter() - reader_tic
                total_batch_size.append(
                    sum([v.shape[0] for v in input_dict.values()])
                )

                for v in input_dict.values():
                    v.stop_gradient = False
                evaluator = ExpressionSolver(
                    _constraint.input_keys,
                    _constraint.output_keys,
                    solver.model
                )
                for label_name, label_formula in _constraint.label_expr.items():
                    evaluator.add_target_expr(label_formula, label_name)

                # forward for every constraint
                with amp.auto_cast(solver.use_amp, level=solver.amp_level):
                    output_dict = evaluator(input_dict)
                    constraint_loss = _constraint.loss(output_dict, label_dict, weight_dict)
                    total_loss += constraint_loss

                loss_dict[_constraint.name] += float(constraint_loss)

                reader_tic = time.perf_counter()

            total_loss = total_loss
            solver.optimizer.clear_grad()
            total_loss.backward()
            return total_loss

        solver.optimizer.step(closure)
        solver.lr_scheduler.step()
        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        total_batch_size = sum(total_batch_size)
        solver.time_info["reader_cost"].update(reader_cost)
        solver.time_info["batch_cost"].update(batch_cost)
        update_loss(solver, loss_dict, total_batch_size)
        if iter_id % log_freq == 0:
            log_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()
