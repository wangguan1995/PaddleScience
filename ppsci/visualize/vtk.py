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
import paddle
from pyevtk.hl import pointsToVTK


def save_vtk_from_array(filename="output", coord=None, data=None):
    if isinstance(coord, paddle.Tensor):
        coord = coord.numpy()
    if isinstance(coord, paddle.Tensor):
        data = data.numpy()

    npoints = len(coord)
    ndims = len(coord[0])

    if data is None:
        data = np.ones((npoints, 1), dtype=type(coord[0, 0]))

    data_vtk = dict()

    for i in range(len(data[0, :])):
        data_vtk[str(i + 1)] = np.ascontiguousarray(data[:, i])

    if ndims == 3:
        axis_x = np.ascontiguousarray(coord[:, 0])
        axis_y = np.ascontiguousarray(coord[:, 1])
        axis_z = np.ascontiguousarray(coord[:, 2])
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = np.ascontiguousarray(coord[:, 0])
        axis_y = np.ascontiguousarray(coord[:, 1])
        axis_z = np.zeros(npoints, dtype="float32")
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)

def save_vtk_from_dict(filename, data_dict, coord_keys, value_keys, num_timestamp=1):
    """Save vtu file from dict.

    Args:
        filename (str): output filename.
        data_dict (Dict[str, np.ndarray]): Data to be saved in dict.
        coord_keys (List[str]): List of coord key. such as ["x", "y"].
        value_keys (List[str]): List of value key. such as ["u", "v"].
        ndim (int): Number of coord dimension in data_dict.
        num_timestamp (int, optional): Number of timestamp in data_dict. Defaults to 1.
    """
    ndim = len(coord_keys)
    assert ndim in [2, 3, 4], \
        f"ndim({ndim}) must be 2, 3 or 4"

    ntotal = len(next(iter(data_dict.values())))
    nx = ntotal // num_timestamp
    assert nx * num_timestamp == ntotal, \
        f"number of space data({nx}) * number of timestamps({num_timestamp}) " \
        f"must be equal to ntotal({ntotal})"

    for t in range(num_timestamp):
        coord = [
            data_dict[key][t*nx: (t+1)*nx] for key in coord_keys if key != "t"
        ]
        for i in range(len(coord)):
            if isinstance(coord[i], paddle.Tensor):
                coord[i] = coord[i].numpy()
            coord[i] = np.ascontiguousarray(coord[i])
        if len(coord) == 2:
            # use 0 matrix for z-axis coord
            coord.append(np.full_like(coord[-1], 0))
        value = {
            key: data_dict[key][t*nx: (t+1)*nx] for key in value_keys
        }
        for key in value:
            if isinstance(value[key], paddle.Tensor):
                value[key] = value[key].numpy()
            value[key] = np.ascontiguousarray(value[key].flatten())
        if num_timestamp > 1:
            filename_t = f"{filename}_t-{t}"
        else:
            filename_t = filename
        pointsToVTK(filename_t, *coord, data=value)
