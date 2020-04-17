import math
import operator
import os
from functools import reduce
from os.path import join as jp

import meshio
import numpy as np
import rasterio
from geographiclib.geodesic import Geodesic

# %% Set directories

cwd = os.getcwd()
data_dir = jp(cwd, 'paraview', 'data')
coord_file = jp(data_dir, 'coordinates_block_topo.txt')
ep_file = jp(data_dir, 'end_points.txt')
tif_file = jp(data_dir, "n11_e108_1arc_v3.tif")

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

# Set the maximum elevation at 0
blocks3d[:, 2] -= np.min((np.abs(blocks3d[:, 2].min()), np.abs(blocks3d[:, 2].max())))


# %% Coordinates conversion

geod = Geodesic.WGS84  # define the WGS84 ellipsoid
# Load profile bounds, must be in the correct format:
# [[ lat1, lon1], [lat2, lon2]]
bounds = np.flip(read_file(ep_file), axis=1)
profile = geod.InverseLine(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1])


def lat_lon(distance):
    """
    Returns the WGS coordinates given a distance between two coordinates.
    :param distance: Distance in meters
    :return:
    """

    g = profile.Position(distance, Geodesic.STANDARD | Geodesic.LONG_UNROLL)

    return g['lat2'], g['lon2']


blocks_wgs = np.copy(blocks3d)


# %% Insert elevation

# Load tif file
dataset = rasterio.open(tif_file)

# Elevation data:
r = dataset.read(1)


def elevation(lat_, lon_):
    idx = dataset.index(lon_, lat_)
    return r[idx]


# Convert distance along axis to lat/lon and add elevation
for i in range(len(blocks_wgs)):
    lat, lon = lat_lon(blocks_wgs[i, 0])
    blocks_wgs[i, 0] = lat
    blocks_wgs[i, 1] = lon
    altitude = elevation(lat, lon) - blocks_wgs[i, 2]
    blocks_wgs[i, 2] = altitude


# %% VTK file creation

# Index of vertices

cells = [("quad", np.array([list(np.arange(i*4, i*4+4))])) for i in range(shp[0])]

# Write file
meshio.write_points_cells(
    "foo.vtk",
    blocks_wgs,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    cell_data={'res': rho},
    # field_data=field_data
    )
