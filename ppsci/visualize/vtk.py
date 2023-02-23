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

import numpy as np
from pyevtk.hl import pointsToVTK
import paddle

def save_vtk(filename="output", cordinate=None, data=None):

    if isinstance(cordinate, paddle.Tensor):
        cordinate = cordinate.numpy()
    if isinstance(cordinate, paddle.Tensor):
        data = data.numpy()

    npoints = len(cordinate)
    ndims = len(cordinate[0])

    if data is None:
        data = np.ones((npoints, 1), dtype=type(cordinate[0, 0]))

    data_vtk = dict()

    for i in range(len(data[0, :])):
        data_vtk[str(i + 1)] = np.ascontiguousarray(data[:, i])

    if ndims == 3:
        axis_x = np.ascontiguousarray(cordinate[:, 0])
        axis_y = np.ascontiguousarray(cordinate[:, 1])
        axis_z = np.ascontiguousarray(cordinate[:, 2])
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = np.ascontiguousarray(cordinate[:, 0])
        axis_y = np.ascontiguousarray(cordinate[:, 1])
        axis_z = np.zeros(npoints, dtype='float32')
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
