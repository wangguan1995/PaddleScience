import numpy as np
import paddlescience as psci
from paddlescience.modulus.geometry.primitives_3d import Box, Sphere, Cylinder
from paddlescience.modulus.utils.io.vtk import var_to_polyvtk

domain_coordinate_interval_dict = {1:[0,800], 2:[0,400], 3:[0,300]}
def normalize(max_domain, min_domain, array, index):
    #array_min = min(array[:,index])
    #array_max = max(array[:,index])
    diff = max_domain - min_domain
    if abs(diff) < 0.00001:
        array[:,index] = 0.0
    else:
        array[:,index] = (array[:, index] - min_domain)/diff

def sample_data(t_step=50, nr_points = 4000):
    # make standard constructive solid geometry example
    # make primitives
    # box = Box(point_1=(-20, -20, -15), point_2=(60, 20, 15))
    # box2 = Box(point_1=(-10, -10, -15), point_2=(40, 10, 15))

    box = Box(point_1=(0, 0, 0), point_2=(800, 400, 300))
    box2 = Box(point_1=(120, 120, 0), point_2=(400, 280, 300))
    #sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(200, 200, 150), radius=40, height=300)
    # cylinder_1 = Cylinder(center=(0, 0, 0), radius=4, height=30)
    #cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 1.0), axis="x")
    #cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 1.0), axis="y")

    # combine with boolean operations
    #all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    #box_minus_sphere = box & sphere
    #geo = box_minus_sphere - all_cylinders
    #all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    # geo = box + box2 - cylinder_1
    geo = box - cylinder_1
    geo1 = box2 - cylinder_1


    print("Sampling Boundary data ......")
    # sample geometry for plotting in Paraview
    # 5: Inlet, 4:Outlet, 12:cylinder
    boundaries, s  = geo.sample_boundary(nr_points=nr_points, curve_index_filters=[4, 5, 6])
    var_to_polyvtk(s, "boundary")
    print("Surface Area: {:.3f}".format(np.sum(s["area"])))

    # inlet = boundaries[0]   
    inlet = boundaries[1]   # inlet不是boundaries[0],应该是boundaries[1]
    inlet = convert_float64_to_float32(inlet)
    inlet_xyz = np.concatenate((inlet['x'], inlet['y'], inlet['z']), axis=1)
    inlet_txyz = replicate_t(t_step, inlet_xyz)

    # outlet = boundaries[1]
    outlet = boundaries[0]  # outlet不是boundaries[1],应该是boundaries[0]
    outlet = convert_float64_to_float32(outlet)
    outlet_xyz = np.concatenate((outlet['x'], outlet['y'], outlet['z']), axis=1)
    outlet_txyz = replicate_t(t_step, outlet_xyz)

    cylinder = boundaries[2]
    cylinder = convert_float64_to_float32(cylinder)
    cylinder_xyz = np.concatenate((cylinder['x'], cylinder['y'], cylinder['z']), axis=1)
    cylinder_txyz = replicate_t(t_step, cylinder_xyz)

    print("Sampling Domain data ......")
    interior = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    interior = convert_float64_to_float32(interior)
    interior_xyz = np.concatenate((interior['x'], interior['y'], interior['z']), axis=1)
    interior1 = geo1.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    interior1 = convert_float64_to_float32(interior1)
    interior1_xyz = np.concatenate((interior1['x'], interior1['y'], interior1['z']), axis=1)
    interior2_xyz = np.concatenate((interior_xyz, interior1_xyz), axis=0)
    # interior_txyz = replicate_t(t_step, interior_xyz)
    interior2_txyz = replicate_t(t_step, interior2_xyz)
    var_to_polyvtk(interior, "interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))

    for item in [inlet_txyz, outlet_txyz, cylinder_txyz, interior2_txyz]:
    # Normalize x,y,z to [0,1]
        for coordinate, interval in domain_coordinate_interval_dict.items():
            min_domain = interval[0]
            max_domain = interval[1]
            normalize(min_domain, max_domain, item, coordinate)
 
    return inlet_txyz, outlet_txyz, cylinder_txyz, interior2_txyz

def convert_float64_to_float32(dict_a):
    for k in dict_a:
        dict_a[k]=dict_a[k].astype(np.float32)
    return dict_a

def replicate_t(t_step, data):
    full_data = None 
    for time in range(1, (t_step+1)):
        t_len = data.shape[0]
        t_extended = np.array([time] * t_len, dtype=np.float32).reshape((-1, 1))
        t_data = np.concatenate((t_extended, data), axis=1)
        if full_data is None:
            full_data = t_data
        else:
            full_data = np.concatenate((full_data, t_data))

    return full_data
