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

from typing import List, Dict

import numpy as np
from paddle.io import Dataset


class NamedArrayDataset(Dataset):
    def __init__(self, input: Dict[str, np.ndarray], label: Dict[str, np.ndarray], weight: Dict[str, np.ndarray]=None):
        super().__init__()
        self.input = input # [str, ndarray]
        self.label = label # [str, ndarray]
        self.weight = weight # [str, ndarray]
        self._len = len(next(iter(input.values())))

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        input_item = {}
        label_item = {}
        weight_item = {}
        for key, value in self.input.items():
            input_item[key] = value[idx]
        for key, value in self.label.items():
            label_item[key] = value[idx]
        for key, value in self.weight.items():
            weight_item[key] = value[idx]
        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len
