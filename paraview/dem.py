import itertools
import os
from os.path import join as jp

import numpy as np
import rasterio

# %% Set directories
cwd = os.getcwd()
data_dir = jp(cwd, 'paraview', 'data')
tif_file = jp(data_dir, "n11_e108_1arc_v3.tif")

# Load tif file
dataset = rasterio.open(tif_file)
# Elevation data:
r = dataset.read(1)


def elevation(lat_, lon_):
    idx = dataset.index(lon_, lat_)
    return r[idx]


# %% Define bounds of polygon in which to build DEM

bbox = [[11.147984, 108.422479], [11.284767, 108.602519]]
# list(itertools.combinations([1,2, 3,4], 2))
lats = np.linspace(bbox[0][0], bbox[1][0], 1000)
longs = np.linspace(bbox[0][1], bbox[1][1], 1000)
cs = list(itertools.product(lats, longs))

dem = [[c[0], c[1], elevation(c[0], c[1])] for c in cs]

