#  Copyright (c) 2020. Robin Thibaut, Ghent University

import itertools
import math
import os
from os.path import join as jp

import meshio
import numpy as np
import rasterio
from geographiclib.geodesic import Geodesic
from scipy.spatial import Delaunay


def dem():
    # %% Set directories
    cwd = os.getcwd()
    data_dir = jp(cwd, 'data')
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

    lats = np.linspace(bbox[0][0], bbox[1][0], 250)
    longs = np.linspace(bbox[0][1], bbox[1][1], 250)
    cs = list(itertools.product(lats, longs))

    tri = Delaunay(cs)  # Perform Delaunay triangulation on wgs coordinates
    simp = tri.simplices  # Extract simplices (=vertices)
    shp = simp.shape
    points = tri.points  # Corresponding points
    tri_del = points[simp].reshape(-1, 2)  # 2D array of points

    dem_wgs = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in tri_del])  # For each vertices, extract elevation

    lat_origin, long_origin = 11.207775, 108.529248

    def dem_local_system(arg):
        """
        Given an origin, converts the WGS84 coordinates into meters around that point.
        :param arg: [lat (decimal degree wgs84), lon (decimal degree wgs84), elev (m)]
        :return:
        """
        line = geod.InverseLine(lat_origin, long_origin, arg[0], arg[1])
        azi = line.azi1
        dis = line.s13
        return dis * math.sin(math.radians(azi)), dis * math.cos(math.radians(azi)), arg[2]

    dem_local = np.array(list(map(dem_local_system, dem_wgs)))  # Convert WGS to local coordinates in meters
    cells_struct = np.array([list(np.arange(i * 3, i * 3 + 3)) for i in range(shp[0])])  # Indexes of triangles corners
    elev = np.mean(dem_local[:, 2][cells_struct], axis=1)  # Elevation of the middle of each triangle
    cells = [("triangle", np.array([list(np.arange(i * 3, i * 3 + 3))])) for i in range(shp[0])]  # vtk default

    # # Write file
    meshio.write_points_cells(
        filename=".\\vtk\\dem.vtk",
        points=dem_local,
        cells=cells,
        cell_data={'elevation': elev}
    )


if __name__ == '__main__':
    dem()
