import itertools
import math
import os
from os.path import join as jp

import meshio
import numpy as np
import rasterio
from geographiclib.geodesic import Geodesic
from scipy.spatial import Delaunay

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

lats = np.linspace(bbox[0][0], bbox[1][0], 200)
longs = np.linspace(bbox[0][1], bbox[1][1], 200)
cs = list(itertools.product(lats, longs))

tri = Delaunay(cs)
simp = tri.simplices
shp = simp.shape
points = tri.points
tri_del = points[simp].reshape(-1, 2)

dem_wgs = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in tri_del])

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


dem_local = np.array(list(map(dem_local_system, dem_wgs)))
cells_struct = np.array([list(np.arange(i * 3, i * 3 + 3)) for i in range(shp[0])])
elev = np.mean(dem_local[:, 2][cells_struct], axis=1)
cells = [("triangle", np.array([list(np.arange(i * 3, i * 3 + 3))])) for i in range(shp[0])]

# # Write file
meshio.write_points_cells(
    filename="points.vtk",
    points=dem_local,
    cells=cells,
    cell_data={'elevation': elev}
)

