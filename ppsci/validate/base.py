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

import random
from functools import partial

import numpy as np
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

from ..data.dataloader import InfiniteDataLoader


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """callback function on each worker subprocess after seeding and before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1]
        num_workers (int): Number of subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in non-distributed environment, it is a constant number `0`.
        seed (int): Random seed
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Validator(object):
    """Base class for validators"""

    def __init__(
        self,
        dataset,
        dataloader_cfg,
        loss,
        metric,
        name
    ):
        cfg = dataloader_cfg
        self.dataset = dataset
        if dist.get_world_size() > 1:
            batch_sampler = BatchSampler(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                drop_last=False,
            )
        else:
            batch_sampler = DistributedBatchSampler(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                drop_last=False,
            )

        self.data_loader = DataLoader(
            dataset=dataset,
            num_workers=cfg["num_workers"],
            return_list=True,
            use_shared_memory=cfg["use_shared_memory"],
            batch_sampler=batch_sampler,
        )
        self.data_iter = iter(self.data_loader)
        self.loss = loss
        self.metric = metric
        self.name = name
