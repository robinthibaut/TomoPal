import math
import operator
import os
from functools import reduce
from os.path import join as jp

import numpy as np

cwd = os.getcwd()
data_dir = jp(cwd, 'paraview', 'data')
coord_file = jp(data_dir, 'coordinates_block_topo.txt')
ep_file = jp(data_dir, 'end_points.txt')


def read_file(file=None, header=0):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        op = np.array([list(map(float, i.split())) for i in fr.readlines()[header:]])
    return op


blocks = read_file(coord_file)
blocks2d = blocks[:, 1:-3]
rho = blocks[:, -1]

blocks2d_flat = blocks2d.reshape(-1, 2)

test = blocks2d_flat[:4]

coords = test
center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))

so = sorted(test,
            key=lambda coord: (math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))))
print(so)

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
