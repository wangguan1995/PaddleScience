# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddlescience as psci
import numpy as np
import random
import paddle
import paddle.distributed as dist

cfg = psci.utils.parse_args()

if cfg is not None:
    # Geometry
    npoints = cfg['Geometry']['npoints']
    seed_num = cfg['Geometry']['seed']
    sampler_method = cfg['Geometry']['sampler_method']
    # Network
    epochs = cfg['Global']['epochs']
    num_layers = cfg['Model']['num_layers']
    hidden_size = cfg['Model']['hidden_size']
    activation = cfg['Model']['activation']
    # Optimizer
    learning_rate = cfg['Optimizer']['lr']['learning_rate']
    # Post-processing
    solution_filename = cfg['Post-processing']['solution_filename']
    vtk_filename = cfg['Post-processing']['vtk_filename']
    checkpoint_path = cfg['Post-processing']['checkpoint_path']
else:
    # Geometry
    # 101**2
    npoints = 10201
    seed_num = 1
    sampler_method = 'uniform'
    # Network
    epochs = 2
    num_layers = 5
    hidden_size = 20
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_laplace2d'
    vtk_filename = 'output_laplace2d'
    checkpoint_path = 'checkpoints'

# initialize parallel environment and set random seed
if dist.get_world_size() > 1:
    dist.init_parallel_env()
    paddle.seed(seed_num + dist.get_rank())
    np.random.seed(seed_num + dist.get_rank())
    random.seed(seed_num + dist.get_rank())
else:
    paddle.seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

# analytical solution
# 定义标准解
ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

# set geometry and boundary
# 定义求解的几何区域，包括描述区域的端点、空间维数，并初始化边界判定条件集合、法线判定条件集合
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))

# 为这个几何区域添加边界判定条件
geo.add_boundary(
    name="around",
    criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

# discretize geometry
# 针对几何区域，获得划分好的点集，包括内部点、边界点等
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# Laplace
# 用sympy描述方程，包括定义自变量，因变量，以及自变量与因变量之间的关系式（包括偏微分关系式）
pde = psci.pde.Laplace(dim=2)

# set bounday condition
# 给u(x,y)定义边值条件
bc_around = psci.bc.Dirichlet('u', rhs=ref_sol)

# add bounday and boundary condition
# 添加刚才定义的边值条件bc_around
pde.add_bc("around", bc_around)

# discretization pde
# 针对pde，对rhs在内部点、边界点、初始点进行离散化；
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(
    learning_rate=learning_rate, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=epochs)

psci.visu.save_vtk(
    filename=vtk_filename, geo_disc=pde_disc.geometry, data=solution)

psci.visu.save_npy(
    filename=solution_filename, geo_disc=pde_disc.geometry, data=solution)

# MSE
# TODO: solution array to dict: interior, bc
cord = pde_disc.geometry.interior
ref = ref_sol(cord[:, 0], cord[:, 1])
# 计算内部点的mse^2
for i in range(len(solution)):
    solution[i] = psci.utils.gather_nprocs(paddle.to_tensor(solution[i])).numpy()
    # print(solution[i].shape)
# exit(0)
mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2)**2

n = 1
for cord in pde_disc.geometry.boundary.values():
    ref = ref_sol(cord[:, 0], cord[:, 1])
    mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2)**2
    n += 1

mse = mse2 / npoints

print("MSE is: ", mse)
