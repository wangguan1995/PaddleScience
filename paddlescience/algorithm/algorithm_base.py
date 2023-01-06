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


class AlgorithmBase(object):
    def __init__(self):
        pass

    # print label
    def __print_label(self, label):
        print(" ** labels ** ")
        for i in label:
            print(i.shape)
        print("")

    # print label_attr
    def __print_label_attr(self, attr):

        print("** interior-equations ** ")
        for i in attr["interior"]["equations"]:
            print(i)

        print("** bc **")
        for k in attr["bc"].keys():
            print("- key: ", k)
            for i in attr["bc"][k]:
                print(i)

        print("** ic **")
        for i in attr["ic"]:
            print(i)

        print("** user-equations ** ")
        for i in attr["user"]["equations"]:
            print(i)

        print("** user-data ** ")
        for i in attr["user"]["data_next"]:
            print(i)

        print("")

    def __timespace(self, time, space):

        nt = len(time)
        ns = len(space)
        ndims = len(space[0])
        time_r = np.repeat(time, ns).reshape((nt * ns, 1))
        space_r = np.tile(space, (nt, 1)).reshape((nt * ns, ndims))
        timespace = np.concatenate([time_r, space_r], axis=1)
        return timespace

    def __repeatspace(self, time, space):

        nt = len(time)
        ns = len(space)
        space_r = np.tile(space, (nt, 1)).reshape((nt * ns))
        return space_r

    def __sqrt(self, x):
        if np.isscalar(x):
            return np.sqrt(x)
        else:
            return paddle.sqrt(x)

    def __padding_array(self, nprocs, array):
        npad = (nprocs - len(array) % nprocs) % nprocs  # pad npad elements
        if array.ndim == 2:
            datapad = array[-1, :].reshape((-1, array[-1, :].shape[0]))
            for i in range(npad):
                array = np.append(array, datapad, axis=0)
        elif array.ndim == 1:
            datapad = array[-1]
            for i in range(npad):
                array = np.append(array, datapad)
        return array
