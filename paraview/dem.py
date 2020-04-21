import itertools
import math
import os
from os.path import join as jp

import meshio
import numpy as np
import rasterio
from geographiclib.geodesic import Geodesic
from scipy.spatial import Delaunay

from paraview.coordinates_conversion import order_vertices

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


geod = Geodesic.WGS84  # define the WGS84 ellipsoid
# %% Define bounds of polygon in which to build DEM

bbox = [[11.147984, 108.422479], [11.284767, 108.602519]]

lats = np.linspace(bbox[0][0], bbox[1][0], 100)
longs = np.linspace(bbox[0][1], bbox[1][1], 100)
cs = list(itertools.product(lats, longs))

dem_wgs = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in cs])

lat_origin, long_origin = 11.207775, 108.529248


def dem_local_system(arg):
    """
    Given an origin, converts the WGS84 coordinates into meters around that point.
    :param lat_p:
    :param lon_p:
    :return:
    """
    line = geod.InverseLine(lat_origin, long_origin, arg[0], arg[1])
    azi = line.azi1
    dis = line.s13
    return dis * math.sin(math.radians(azi)), dis * math.cos(math.radians(azi)), arg[2]


dem_local = list(map(dem_local_system, dem_wgs))
tri = Delaunay(dem_local)
simp = tri.simplices
points = tri.points

blocks = points[simp]
shp = blocks.shape

blocks_ordered = np.array([order_vertices(vs) for vs in blocks]).reshape(shp[0]*shp[1], 3)
# Blocks' vertices are now correctly ordered
cells = [("quad", np.array([list(np.arange(i * 4, i * 4 + 4))])) for i in range(len(blocks))]

# cells = [("quad", s) for s in simp]
# # Write file
meshio.write_points_cells(
    filename="points.vtk",
    points=blocks_ordered,
    cells=cells
)

