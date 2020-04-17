import math
import operator
import os
from functools import reduce
from os.path import join as jp

import meshio
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

# %% Add a new axis to make coordinates 3-D
# We have now the axis along the profile line and the depth.

shp = blocks2d_vo.shape
# Create empty array
blocks3d = np.zeros((shp[0], shp[1], 3))

# Insert 0 value for each vertices
for i in range(len(blocks2d_vo)):
    for j in range(shp[1]):
        blocks3d[i, j] = np.insert(blocks2d_vo[i, j], 1, 0)

# Flatten
blocks3d = blocks3d.reshape(-1, 3)
# Index of vertices
cells = [("quad", np.array([list(np.arange(i*4, i*4+4))])) for i in range(shp[0])]

# Write file
meshio.write_points_cells(
    "foo.vtk",
    blocks3d,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data={'res': rho},
    # field_data=field_data
    )
