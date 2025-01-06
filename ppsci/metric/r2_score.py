# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Dict

import paddle

from ppsci.metric import base


class R2Score(base.Metric):
    r"""Coefficient of Determination (R^2 Score).

    $$
    R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
    $$

    $$
    \mathbf{y}, \mathbf{\hat{y}} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import paddle
        >>> from ppsci.metric import R2Score
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> metric = R2Score()
        >>> result = metric(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               -3.75000000), 'v': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               0.00000000)}
        >>> metric = R2Score(keep_batch=True)
        >>> result = metric(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [-0.64000008, -6.88000011]), 'v': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [0.00000000, 0.00000000])}
    """

    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict) -> Dict[str, "paddle.Tensor"]:
        r2score_dict = {}

        for key in label_dict:
            output = output_dict[key]
            label = label_dict[key]

            # Calculate mean of label
            label_mean = label.mean(axis=-1, keepdim=True)

            # Calculate total sum of squares (TSS)
            tss = ((label - label_mean) ** 2).sum(axis=-1)

            # Calculate residual sum of squares (RSS)
            rss = ((output - label) ** 2).sum(axis=-1)

            # Calculate R^2 score
            r2score = 1 - (
                rss / (tss + 1e-8)
            )  # Add small epsilon to avoid division by zero

            if self.keep_batch:
                r2score_dict[key] = r2score
            else:
                r2score_dict[key] = r2score.mean()

        return r2score_dict
