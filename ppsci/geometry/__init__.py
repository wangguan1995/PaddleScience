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

__all__ = [
    "Cuboid",
    "Disk",
    "Geometry",
    "Hypercube",
    "Hypersphere",
    "Interval",
    "Mesh",
    "Polygon",
    "Rectangle",
    "Sphere",
    "TimeDomain",
    "TimeXGeometry",
    "Triangle",
    "build_geometry"
]
from copy import deepcopy

from .geometry import Geometry
from .geometry_1d import Interval
from .geometry_2d import Disk, Polygon, Rectangle, Triangle
from .geometry_3d import Cuboid, Sphere
from .geometry_nd import Hypercube, Hypersphere
from .mesh import Mesh
from .timedomain import TimeDomain, TimeXGeometry


def build_geometry(cfg):
    """Build geometry(ies)

    Args:
        cfg (List[AttrDict]): Geometry config list.

    Returns:
        Dict[str, Geometry]: Geometry(ies) in dict.
    """
    cfg = deepcopy(cfg)

    geom_dict = {}
    for _item in cfg:
        geom_cls = next(iter(_item.keys()))
        geom_cfg = _item[geom_cls]
        geom_name = geom_cfg.pop("name", geom_cls)
        geom_dict[geom_name] = eval(geom_cls)(**geom_cfg)
    return geom_dict
