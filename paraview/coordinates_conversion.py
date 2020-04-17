import math
import operator
import os
from functools import reduce
from os.path import join as jp

import numpy as np

# %% Set directories
cwd = os.getcwd()
data_dir = jp(cwd, 'paraview', 'data')
coord_file = jp(data_dir, 'coordinates_block_topo.txt')
ep_file = jp(data_dir, 'end_points.txt')


# %% Read data


def read_file(file=None, header=0):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        op = np.array([list(map(float, i.split())) for i in fr.readlines()[header:]])
    return op


blocks = read_file(coord_file)  # Raw mesh info
blocks2d_flat = blocks[:, 1:-3]  # Flat list of polygon vertices
rho = blocks[:, -1]  # Resistivity
blocks2d = blocks2d_flat.reshape(-1, 4, 2)  # Reshape in (n, 4, 2)


# %% Order vertices in each block to correspond to VTK requirements


def order_vertices(vertices):
    # Compute center of vertices
    center = \
        tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), vertices), [len(vertices)] * 2))

    # Sort vertices according to angle
    so = \
        sorted(vertices,
               key=lambda coord: (math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))))

    return np.array(so)


blocks2d_vo = np.array([order_vertices(vs) for vs in blocks2d])  # Blocks' vertices are now correctly ordered

# columns = ['n', 'x0', 'z0', 'x1', 'z1', 'x2', 'z2', 'x3', 'z3', 'xc', 'zc', 'rho']
# df = pd.DataFrame(data=blocks, columns=columns)

# # Load hk array
# hk = np.load(jp(results_dir, 'hk.npy')).reshape(-1)
# cells = [("quad", np.array([list(np.arange(i*4, i*4+4))])) for i in range(len(blocks))]
#
# meshio.write_points_cells(
#     "foo.vtk",
#     blocks3d,
#     cells,
#     # Optionally provide extra data on points, cells, etc.
#     # point_data=point_data,
#     cell_data={'hk': hk},
#     # field_data=field_data
#     )
#
# diavatly.model_map(blocks2d, hk, log=1)
# plt.show()
