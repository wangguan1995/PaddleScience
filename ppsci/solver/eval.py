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

import paddle
import paddle.amp as amp

from ..utils import profiler
from ..utils.expression import ExpressionSolver
from ..visualize import save_vtk
from .printer import log_eval_info, update_eval_loss, update_eval_metric


def eval_func(solver, epoch_id, log_freq):
    """Evaluation program

    Args:
        solver (Solver): Main Solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log evaluation information every `log_freq` steps.

    Returns:
        Dict[str, Any]: Metric collected during evaluation.
    """
    target_metric = None
    for _, _validator in solver.validator.items():
        # all_input = defaultdict(list)
        all_output = defaultdict(list)
        all_label = defaultdict(list)
        num_samples = len(_validator.dataset)

        loss_dict = defaultdict(float)
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        for iter_id, batch in enumerate(_validator.data_loader, start=1):
            input_dict, label_dict, _ = batch

            # profile code below
            profiler.add_profiler_step(solver.cfg["profiler_options"])
            if iter_id == 5:
                # 5 step for warmup
                for key in solver.eval_time_info:
                    solver.eval_time_info[key].reset()
            reader_cost = time.perf_counter() - reader_tic
            total_batch_size = sum(
                [v.shape[0] for v in input_dict.values()]
            )

            for v in input_dict.values():
                v.stop_gradient = False
            evaluator = ExpressionSolver(
                _validator.input_keys,
                _validator.output_keys,
                solver.model
            )
            for label_name, label_formula in _validator.label_expr.items():
                evaluator.add_target_expr(label_formula, label_name)

            # forward for every validator
            with amp.auto_cast(solver.use_amp, level=solver.amp_level):
                output_dict = evaluator(input_dict)
                validator_loss = _validator.loss(output_dict, label_dict)
                loss_dict[_validator.name] = float(validator_loss)
                # for key, input in input_dict.items():
                #     all_input[key].append(input)
                for key, output in output_dict.items():
                    all_output[key].append(output)
                for key, label in label_dict.items():
                    all_label[key].append(label)

            batch_cost = time.perf_counter() - batch_tic
            solver.eval_time_info["reader_cost"].update(reader_cost)
            solver.eval_time_info["batch_cost"].update(batch_cost)
            update_eval_loss(solver, loss_dict, total_batch_size)
            if iter_id == 1 or iter_id % log_freq == 0:
                log_eval_info(
                    solver,
                    total_batch_size,
                    epoch_id,
                    len(_validator.data_loader),
                    iter_id
                )

            reader_tic = time.perf_counter()
            batch_tic = time.perf_counter()

        for key in all_output:
            all_output[key] = paddle.concat(all_output[key], 0)
            if len(all_output[key]) > num_samples:
                all_output[key] = all_output[key][:num_samples]
        for key in all_label:
            all_label[key] = paddle.concat(all_label[key], 0)
            if len(all_label[key]) > num_samples:
                all_label[key] = all_label[key][:num_samples]

        metric = {}
        for metric_name, metric_func in _validator.metric.items():
            metric_value = metric_func(all_output, all_label)
            metric[metric_name] = metric_value

        solver.eval_output_info[_validator.name] = metric
        if target_metric is None:
            tmp = metric
            while isinstance(tmp, dict):
                tmp = next(iter(tmp.values()))
            assert isinstance(tmp, (int, float)), \
                f"target metric({type(tmp)}) must be a number"
            target_metric = tmp

    return target_metric
