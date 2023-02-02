import numpy as np

import paddlescience as psci
from paddlescience.neo_geometry import Disk, Mesh, Rectangle

x_mesh = Mesh("./x.stl")
y_mesh = Mesh("./y.stl")
z_mesh = Mesh("./z.stl")
right_mesh = x_mesh | y_mesh | z_mesh

ball_mesh = Mesh("./ball.stl")
box_mesh = Mesh("./box.stl")
left_mesh = ball_mesh & box_mesh

up_mesh = left_mesh - right_mesh

up_mesh.add_sample_config("interior1", batch_size=1000)
up_mesh.add_sample_config("boundary1", batch_size=1000)
up_mesh.add_sample_config("top", batch_size=1000, criteria=lambda x, y, z: (0.99 <= z) & (z <= 1.0))
up_mesh.add_sample_config("bottom", batch_size=1000, criteria=lambda x, y, z: (-1.0 <= z) & (z <= -0.99))

points_dict = up_mesh.fetch_batch_data()

for key, points in points_dict.items():
    print(f"{key} {points.shape}")
    psci.visu.__save_vtk_raw(key, points, np.full((len(points), 1), 1.0))


disk = Disk([1.0, 1.0], 0.5)
rect = Rectangle([-1.0, -1.0], [1.0, 1.0])
merge = disk | rect

merge.add_sample_config("interior", batch_size=1000)
merge.add_sample_config("boundary", batch_size=1000)
merge.add_sample_config("left", batch_size=1000, criteria=lambda x, y: x <= 0)
merge.add_sample_config("right", batch_size=1000, criteria=lambda x, y: x > 0)

points_dict = merge.fetch_batch_data()

for key, points in points_dict.items():
    print(f"{key} {points.shape}")
    psci.visu.__save_vtk_raw(key+"2", points, np.full((len(points), 1), 1.0))
