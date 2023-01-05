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

import vtk
import copy
import numpy as np
import paddle
import paddlescience as psci
import paddle.distributed as dist
from pyevtk.hl import pointsToVTK
import sample_boundary_training_data as sample_data
from load_lbm_data import load_ic_data, load_supervised_data

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()

# time arraep 
t_start = 2000.1
t_end = 2001.1
t_step = 0.1
time_num = int((t_end - t_start)/t_step)
time_tmp = np.linspace(t_start-2000, t_end-2000, time_num+1, endpoint=True)
time_array = time_tmp

# num points to sample per GPU
num_points = 100000

# discretize node by geo
inlet_txyz, outlet_txyz, cylinder_txyz, interior_txyz = sample_data.sample_data(t_step=len(time_array), nr_points=num_points)

i_t = interior_txyz[:, 0]
i_x = interior_txyz[:, 1]
i_y = interior_txyz[:, 2]
i_z = interior_txyz[:, 3]

# bc inlet nodes discre
b_inlet_t = inlet_txyz[:, 0]
b_inlet_x = inlet_txyz[:, 1]
b_inlet_y = inlet_txyz[:, 2]
b_inlet_z = inlet_txyz[:, 3]

# bc outlet nodes discre
b_outlet_t = outlet_txyz[:, 0]
b_outlet_x = outlet_txyz[:, 1]
b_outlet_y = outlet_txyz[:, 2]
b_outlet_z = outlet_txyz[:, 3]

# bc cylinder nodes discre
b_cylinder_t = cylinder_txyz[:, 0]
b_cylinder_x = cylinder_txyz[:, 1]
b_cylinder_y = cylinder_txyz[:, 2]
b_cylinder_z = cylinder_txyz[:, 3]

# bc & interior nodes for nn
inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
inputbc2 = np.stack((b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
inputbc3 = np.stack((b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)

# N-S, Re=3900, D=80, u=1, nu=8/390
pde = psci.pde.NavierStokes(nu=0.0205, rho=1.0, dim=3, time_dependent=True)

# set bounday condition
bc_inlet_u = psci.bc.Dirichlet('u', rhs=1)
bc_inlet_v = psci.bc.Dirichlet('v', rhs=0)
bc_inlet_w = psci.bc.Dirichlet('w', rhs=0)

bc_cylinder_u = psci.bc.Dirichlet('u', rhs=0)
bc_cylinder_v = psci.bc.Dirichlet('v', rhs=0)
bc_cylinder_w = psci.bc.Dirichlet('w', rhs=0)

bc_outlet_p = psci.bc.Dirichlet('p', rhs=0)

# add bounday and boundary condition
pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
pde.set_bc("outlet", bc_outlet_p)

# Network
net = psci.network.FCNet(
    num_ins=4, num_outs=4, num_layers=6, hidden_size=50, activation='tanh')

net.initialize('checkpoint/static_model_params_16000.pdparams')

outeq = net(inputeq)
outbc1 = net(inputbc1)
outbc2 = net(inputbc2)
outbc3 = net(inputbc3)

# eq loss
losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

# bc loss
lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
lossbc3 = psci.loss.BcLoss("cylinder", netout=outbc3)

# total loss
loss = losseq1 + losseq2 + losseq3 + losseq4 + lossbc1 + lossbc2 + lossbc3
# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo)
solution = solver.predict()

n = int(i_x.shape[0] / len(time_array))
i_x = i_x.astype("float32")
i_y = i_y.astype("float32")
i_z = i_z.astype("float32")

cord = np.stack((i_x[0:n], i_y[0:n], i_z[0:n]), axis=1)
psci.visu.save_vtk_cord(filename="./vtk/output", time_array=time_array, cord=cord, data=solution)

