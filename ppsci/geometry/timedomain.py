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

"""
Code below is heavily based on https://github.com/lululxvi/deepxde
"""

import itertools

import numpy as np

from ..utils import misc
from .geometry import Geometry
from .geometry_1d import Interval
from .geometry_2d import Rectangle
from .geometry_3d import Cuboid
from .geometry_nd import Hypercube


class TimeDomain(Interval):
    def __init__(self, t0, t1, time_step=None):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
        self.time_step = time_step
        self.num_timestamp = None if time_step is None \
            else (int(np.ceil(self.diam / time_step)) + 1)

    def on_initial(self, t):
        return np.isclose(t, self.t0).flatten()


class TimeXGeometry(Geometry):
    def __init__(self, timedomain, geometry):
        self.timedomain = timedomain
        self.geometry = geometry
        self.ndim = geometry.ndim + timedomain.ndim

    @property
    def dim_keys(self):
        return ["t"] + self.geometry.dim_keys

    def on_boundary(self, x):
        # [N, txyz]
        return self.geometry.on_boundary(x[:, 1:])

    def on_initial(self, x):
        # [N, 1]
        return self.timedomain.on_initial(x[:, :1])

    def boundary_normal(self, x):
        # [N, xyz]
        normal = self.geometry.boundary_normal(x[:, 1:])
        return np.hstack((np.zeros([len(normal), 1]), normal))

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            nx = int(np.ceil(n / nt))
        else:
            nx = int(
                np.ceil((n * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                        / self.timedomain.diam)**0.5)
            )
            nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)
        if boundary and (self.timedomain.time_step is None):
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=boundary,
                dtype="float32"
            )[:, None][::-1]
        xt = []
        for ti in t:
            xt.append(np.hstack((np.full([nx, 1], ti[0]), x)))
        xt = np.vstack(xt)
        if len(xt) > n:
            xt = xt[:n]
        return xt

    def random_points(self, n, random="pseudo"):
        # time evenly and geometry random, if time_step if specified
        if self.timedomain.time_step is not None:
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype="float32"
            )[:, None][::-1] # [nt, 1]
            nx = int(np.ceil(n / nt))
            x = self.geometry.random_points(nx, random)
            xt = []
            for ti in t:
                xt.append(np.hstack((np.full([nx, 1], ti[0]), x)))
            xt = np.vstack(xt)
            if len(xt) > n:
                xt = xt[:n]
            return xt

        if isinstance(self.geometry, Interval):
            geom = Rectangle(
                [self.timedomain.t0, self.geometry.l],
                [self.timedomain.t1, self.geometry.r]
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, Rectangle):
            geom = Cuboid(
                [
                    self.timedomain.t0,
                    self.geometry.xmin[0], self.geometry.xmin[1]
                ],
                [
                    self.timedomain.t1,
                    self.geometry.xmax[0], self.geometry.xmax[1]
                ]
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, (Cuboid, Hypercube)):
            geom = Hypercube(
                np.append(self.timedomain.t0, self.geometry.xmin),
                np.append(self.timedomain.t1, self.geometry.xmax)
            )
            return geom.random_points(n, random=random)

        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.permutation(t)
        return np.hstack((t, x))

    def uniform_boundary_points(self, n):
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        if self.geometry.ndim == 1:
            nx = 2
        else:
            s = 2 * sum(
                map(lambda l: l[0] * l[1],
                    itertools.combinations(
                        self.geometry.bbox[1] - self.geometry.bbox[0], 2
                    )
                )
            )
            nx = int((n * s / self.timedomain.diam)**0.5)
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        nx = len(x)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype="float32"
        )
        xt = []
        for ti in t:
            xt.append(np.hstack((np.full([nx, 1], ti), x)))
        xt = np.vstack(xt)
        if len(xt) > n:
            xt = xt[:n]
        return xt

    def random_boundary_points(self, n, random="pseudo"):
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype="float32"
            )
            nx = int(np.ceil(n / nt))
            x = self.geometry.random_boundary_points(nx, random=random)
            xt = []
            for ti in t:
                xt.append(np.hstack((np.full([nx, 1], ti), x)))
            xt = np.vstack(xt)
            if len(xt) > n:
                xt = xt[:n]
            return xt
        else:
            t = self.timedomain.random_points(n, random=random)
            t = np.random.permutation(t)
            return np.hstack((t, x))

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if len(x) > n:
            x = x[:n]
        return np.hstack((np.full([len(x), 1], t, dtype="float32"), x))

    def random_initial_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((np.full([n, 1], t, dtype="float32"), x))

    def periodic_point(self, x, component):
        xp = self.geometry.periodic_point(x[:, 1:], component)
        return np.hstack((x[:, :1], xp))


    def sample_initial_interior(self, n: int, random: str="pseudo", criteria=None, evenly=False):
        """Sample random points in the time-geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.ndim), dtype="float32")
        _size, _ntry, _nsuc = 0, 0, 0
        while _size < n:
            if evenly:
                points = self.uniform_initial_points(n)
            else:
                points = self.random_initial_points(n, random)

            if criteria is not None:
                criteria_mask = criteria(*np.split(points, self.ndim, axis=1)).flatten()
                points = points[criteria_mask]

            if len(points) > n - _size:
                points = points[:n - _size]
            x[_size:_size + len(points)] = points

            _size += len(points)
            _ntry += 1
            if len(points) > 0:
                _nsuc += 1

            if _ntry >= 1000 and _nsuc == 0:
                raise RuntimeError(
                    f"sample initial interior failed"
                )
        return misc.convert_to_dict(x, self.dim_keys)
