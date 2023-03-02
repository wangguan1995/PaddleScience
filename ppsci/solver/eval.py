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

import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict

import paddle
import paddle.amp as amp

from ..utils import misc, profiler
from ..utils.expression import ExpressionSolver
from ..visualize import save_vtu_from_dict
from .printer import log_eval_info, update_eval_loss


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
        all_input = defaultdict(list)
        all_output = defaultdict(list)
        all_label = defaultdict(list)
        num_samples = len(_validator.dataset)

        loss_dict = defaultdict(float)
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        for iter_id, batch in enumerate(_validator.data_loader, start=1):
            input_dict, label_dict, _ = batch

            # profile code
            profiler.add_profiler_step(solver.cfg["profiler_options"])
            if iter_id == 5:
                # 5 step for warmup
                for key in solver.eval_time_info:
                    solver.eval_time_info[key].reset()
            reader_cost = time.perf_counter() - reader_tic

            for v in input_dict.values():
                v.stop_gradient = False
            evaluator = ExpressionSolver(
                _validator.input_keys,
                _validator.output_keys,
                solver.model
            )
            for label_name, label_formula in _validator.label_expr.items():
                evaluator.add_target_expr(label_formula, label_name)

            # forward
            with amp.auto_cast(solver.use_amp, level=solver.amp_level):
                output_dict = evaluator(input_dict)
                validator_loss = _validator.loss(output_dict, label_dict)
                loss_dict[_validator.name] = float(validator_loss)

            # collect data from all trainer
            for key, input in input_dict.items():
                all_input[key].append(
                    input if solver.world_size == 1
                    else misc.all_gather(input)
                )
            for key, output in output_dict.items():
                all_output[key].append(
                    output if solver.world_size == 1
                    else misc.all_gather(output)
                )
            for key, label in label_dict.items():
                all_label[key].append(
                    label if solver.world_size == 1
                    else misc.all_gather(label)
                )

            batch_cost = time.perf_counter() - batch_tic
            solver.eval_time_info["reader_cost"].update(reader_cost)
            solver.eval_time_info["batch_cost"].update(batch_cost)
            total_batch_size = sum([v.shape[0] for v in input_dict.values()])
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

        # gather
        for key in all_input:
            all_input[key] = paddle.concat(all_input[key])
            if len(all_input[key]) > num_samples:
                all_input[key] = all_input[key][:num_samples]
        for key in all_output:
            all_output[key] = paddle.concat(all_output[key])
            if len(all_output[key]) > num_samples:
                all_output[key] = all_output[key][:num_samples]
        for key in all_label:
            all_label[key] = paddle.concat(all_label[key])
            if len(all_label[key]) > num_samples:
                all_label[key] = all_label[key][:num_samples]

        metric = OrderedDict()
        for metric_name, metric_func in _validator.metric.items():
            metric_value = metric_func(all_output, all_label)
            metric[metric_name] = metric_value
        solver.eval_output_info[_validator.name] = metric

        if target_metric is None:
            tmp = metric
            while isinstance(tmp, dict):
                tmp = next(iter(tmp.values()))
            assert isinstance(tmp, (int, float)), \
                f"target metric({type(tmp)}) should be a number"
            target_metric = tmp

        visual_dir = osp.join(
            solver.output_dir,
            solver.cfg["Arch"]["name"],
            "visual",
            f"epoch_{epoch_id}"
        )
        if solver.rank == 0:
            os.makedirs(visual_dir, exist_ok=True)
            save_vtu_from_dict(
                osp.join(visual_dir, _validator.name),
                {**all_output, **all_input},
                _validator.input_keys,
                _validator.output_keys,
                _validator.num_timestamp
            )

    return target_metric
