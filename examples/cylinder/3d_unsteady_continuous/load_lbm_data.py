import glob
import os
import time

import numpy as np
import pandas as pd

dir_name = '/workspace/hesensen/PaddleScience_dev_3d/examples/cylinder/3d_unsteady_continuous/data/all'
#file_pattern = 'point_70.000000_9.000000_27.000000.plt'


scale = 10.0
pressure_base = 101325.0
# time_step = 1 # need change to 0.1
time_start_per_file = 2000.0
domain_coordinate_interval_dict = {1:[0,800], 2:[0,400], 3:[0,300]}

def normalize(max_domain, min_domain, array, index):
    #array_min = min(array[:,index])
    #array_max = max(array[:,index])
    diff = max_domain - min_domain
    if abs(diff) < 0.00001:
        array[:,index] = 0.0
    else:
        array[:,index] = (array[:, index] - min_domain)/diff

def scale_value(array, scale, index):
    array[:, index] = array[:, index] * scale

#'2002.000000,10.000000,14.000000,8.000000,0.937419,-0.129196,-0.008992,101325.328523,4.933044'
def load_ic_data(t):
    print("Loading IC data ......")
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter( os.path.isfile, glob.glob(dir_name + '*')))

    # Iterate over sorted list of files and print the file paths one by one.
    # (63916, 9)
    flow_array = None
    for file_path in list_of_files:
        #print(file_path)
        filename = file_path.strip('.plt').split('_')
        #data = pd.read_table(file_path, sep=',', header=None)
        with open(file_path) as fp:
            # line = fp.readline()
            line = fp.readlines(0)[1]   #2000.1开始
            data_list = line.strip('\n').split(',')
        txyz_uvwpe = np.array([data_list],dtype=float)
        if flow_array is None:
            flow_array = txyz_uvwpe
        else:
            flow_array = np.concatenate((flow_array, txyz_uvwpe))

    # Set ic t=0
    flow_array[:,0] = 0
    # Scale xyz by 10.0
    for i in [1,2,3]:
        scale_value(flow_array, scale, i)
    # Normalize x,y,z to [0,1]
    for coordinate, interval in domain_coordinate_interval_dict.items():
        min_domain = interval[0]
        max_domain = interval[1]
        normalize(min_domain, max_domain, flow_array, coordinate)
    # Cast pressure baseline
    flow_array[:,7] = flow_array[:,7] - pressure_base

    # txyzuvwpe
    print("IC data shape: {}".format(flow_array.shape))
    return flow_array.astype(np.float32)

def load_supervised_data(t_start, t_end, t_step, t_ic, num_points):
    print("Loading Supervised data ......")
    # row_index = int((t_start - time_start_per_file) / time_step)
    row_index = round((t_start - time_start_per_file) / t_step)  # time_step=0.1,row_index从第2行开始
    # row_length = int((t_end - t_start) / time_step)
    row_length = int((t_end - t_start) / t_step)

    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter( os.path.isfile, glob.glob(dir_name + '*') ) )
    random_files = np.random.permutation(list_of_files)


    # Iterate over sorted list of files and print the file paths one by one.
    # (63916, 9)
    flow_array = None
    for file_path in random_files[:num_points]:
        #print(file_path)
        filename = file_path.strip('.plt').split('_')
        #data = pd.read_table(file_path, sep=',', header=None)
        with open(file_path) as fp:
            for index, line in enumerate(fp):
                if index in range(row_index, row_index+row_length):
                    data_list = line.strip('\n').split(',')
                    txyz_uvwpe = np.array([data_list], dtype=float)
                    if flow_array is None:
                        flow_array = txyz_uvwpe
                    else:
                        flow_array = np.concatenate((flow_array, txyz_uvwpe))

    # Normalize t to [0, ..]
    flow_array[:,0] = (flow_array[:, 0] - t_ic)/t_step

    # Scale xyz by 10.0
    for i in [1,2,3]:
        scale_value(flow_array, scale, i)
    # Normalize x,y,z to [0,1]
    for  coordinate, interval in domain_coordinate_interval_dict.items():
        min_domain = interval[0]
        max_domain = interval[1]
        normalize(min_domain, max_domain, flow_array, coordinate)
    # Cast pressure baseline
    flow_array[:,7] = flow_array[:,7] - pressure_base

    # txyzuvwpe
    print("Supervised data shape: {}".format(flow_array.shape))
    return flow_array.astype(np.float32)

if __name__ == '__main__':
    start = time.perf_counter()
    ic_t = 2000.0
    #flow_array = load_ic_data(ic_t)
    ic_array = load_ic_data(ic_t)
    flow_array = load_supervised_data(2000.0, 2000.4, 100)
    end = time.perf_counter()
    print("spent:{}s".format(end-start))
