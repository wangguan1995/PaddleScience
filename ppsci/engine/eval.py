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

from collections import defaultdict

import paddle

from ..utils.expression import ExpressionSolver
from ..visualize import save_vtk



def eval_func(engine, epoch_id, print_batch_step):
    for _validator in engine.validator:
        total_input = defaultdict(list)
        total_output = defaultdict(list)
        total_label = defaultdict(list)
        total_len = len(_validator.dataset)

        for iter_id, batch in enumerate(_validator.data_loader):
            input_dict, label_dict, _ = batch
            for v in input_dict.values():
                v.stop_gradient = False
            evaluator = ExpressionSolver(
                _validator.input_keys,
                _validator.output_keys,
                engine.model
            )
            for label_name, label_formula in _validator.label_expr.items():
                evaluator.add_target_expr(label_formula, label_name)
            output_dict = evaluator(input_dict)
            for key, input in input_dict.items():
                total_input[key].append(input)
            for key, output in output_dict.items():
                total_output[key].append(output)
            for key, label in label_dict.items():
                total_label[key].append(label)

            if iter_id % print_batch_step == 0:
                print(f"eval - epoch [{epoch_id}] iter [{iter_id}/{len(_validator.data_loader)}]")

        for key in total_input:
            total_input[key] = paddle.concat(total_input[key], 0)
            if len(total_input[key]) > total_len:
                total_input[key] = total_input[key][:total_len]
        for key in total_output:
            total_output[key] = paddle.concat(total_output[key], 0)
            if len(total_output[key]) > total_len:
                total_output[key] = total_output[key][:total_len]
        for key in total_label:
            total_label[key] = paddle.concat(total_label[key], 0)
            if len(total_label[key]) > total_len:
                total_label[key] = total_label[key][:total_len]

        coord = paddle.concat([total_input[k] for k in _validator.input_keys], -1).numpy()
        data =  paddle.concat([total_output[k] for k in _validator.output_keys], -1).numpy()
        save_vtk(f"output/ldc2d/validate_{epoch_id}", coord, data)
        metric = _validator.metric(total_output, total_label)

    return metric
