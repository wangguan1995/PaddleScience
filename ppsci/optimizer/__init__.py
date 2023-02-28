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

from copy import deepcopy

from .lr_scheduler import (ConstLR, Cosine, Linear, MultiStepDecay, Piecewise,
                           Step)
from .optimizer import SGD, Adam, AdamW, Momentum, RMSProp

__all__ = [
    "ConstLR",
    "Cosine",
    "Linear",
    "MultiStepDecay",
    "Piecewise",
    "Step",
    "SGD",
    "Adam",
    "AdamW",
    "Momentum",
    "RMSProp",
    "build_optimizer",
    "build_lr_scheduler"
]


def build_lr_scheduler(cfg, epochs, iters_per_epoch):
    """Build learning rate scheduler.

    Args:
        cfg (AttrDict): Learing rate scheduler config.
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        LRScheduler: Learing rate scheduler.
    """
    cfg = deepcopy(cfg)
    cfg.update({"epochs": epochs, "iters_per_epoch": iters_per_epoch})
    lr_scheduler_cls = cfg.pop("name")
    lr_scheduler = eval(lr_scheduler_cls)(**cfg)
    return lr_scheduler()


def build_optimizer(cfg, model_list, epochs, iters_per_epoch):
    """Build optimizer and learing rate scheduler

    Args:
        cfg (AttrDict): Learing rate scheduler config.
        model_list (List[nn.Layer]): List of model(s).
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        Optimizer, LRScheduler: Optimizer and learing rate scheduler.
    """
    # build lr_scheduler
    cfg = deepcopy(cfg)
    lr_cfg = cfg.pop("lr")
    lr_scheduler = build_lr_scheduler(
        lr_cfg,
        epochs,
        iters_per_epoch
    )

    # build optimizer
    opt_cls = cfg.pop("name")
    optimizer = eval(opt_cls)(
        learning_rate=lr_scheduler,
        **cfg
    )(model_list)

    return optimizer, lr_scheduler
