import os
from os.path import join as jp

import rasterio

# %% Set directories

cwd = os.getcwd()
data_dir = jp(cwd, 'paraview', 'data')
coord_file = jp(data_dir, 'coordinates_block_topo.txt')
ep_file = jp(data_dir, 'end_points.txt')
tif_file = jp(data_dir, "n11_e108_1arc_v3.tif")

# Load tif file
dataset = rasterio.open(tif_file)
# Elevation data:
r = dataset.read(1)


def elevation(lat_, lon_):
    idx = dataset.index(lon_, lat_)
    return r[idx]

