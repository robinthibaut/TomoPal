#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import operator
import os
from functools import reduce
from os.path import join as jp

import numpy as np
import rasterio
import vtk
from geographiclib.geodesic import Geodesic


def read_file(file=None, header=0):
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        op = np.array([list(map(float, i.split())) for i in fr.readlines()[header:]])
    return op


def order_vertices(vertices):
    """
    Paraview expects vertices in a particular order for Quad object, with the origin at the bottom left corner.
    :param vertices: (x, y) coordinates of the quad vertices
    :return: Sorted vertices
    """
    # Compute center of vertices
    center = \
        tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), vertices), [len(vertices)] * 2))

    # Sort vertices according to angle
    so = \
        sorted(vertices,
               key=lambda coord: (math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))))

    return np.array(so)


def conversion(name):

    # Set directories
    cwd = os.getcwd()
    data_dir = jp(cwd, 'data')
    coord_file = jp(data_dir, name, 'p{}.dat'.format(name))
    ep_file = jp(data_dir, name, 'end_points.txt')
    tif_file = jp(data_dir, "n11_e108_1arc_v3.tif")

    blocks = read_file(coord_file)  # Raw mesh info

    blocks2d_flat = blocks[:, 1:9]  # Flat list of polygon vertices

    rho = blocks[:, -1]  # Resistivity

    # Load profile bounds, must be in the correct format:
    # [[ lat1, lon1], [lat2, lon2]]
    bounds = np.flip(read_file(ep_file), axis=1)  # Flip only in this case as the format is incorrect.

    # Elevation data
    # Load tif file
    dataset = rasterio.open(tif_file)
    # Elevation data:
    r = dataset.read(1)

    blocks2d = blocks2d_flat.reshape(-1, 4, 2)  # Reshape in (n, 4, 2)

    # %% Order vertices in each block to correspond to VTK requirements
    # This might actually not be necessary for meshio.
    blocks2d_vo = np.array([order_vertices(vs) for vs in blocks2d])  # Blocks' vertices are now correctly ordered
    # %% Add a new axis to make coordinates 3-D
    # We have now the axis along the profile line and the depth.
    shp = blocks2d_vo.shape
    # Create 3D empty array
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

    # Create an 'InverseLine' bounded by the profile endpoints.
    profile = geod.InverseLine(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1])

    def lat_lon(distance):
        """
        Returns the WGS coordinates given a distance along the axis of the profile.
        :param distance: Distance along the profile from its origin (m)
        :return: latitude WGS84 in decimal degrees, longitude WGS84 in decimal degrees
        """

        g = profile.Position(distance, Geodesic.STANDARD | Geodesic.LONG_UNROLL)

        return g['lat2'], g['lon2']

    blocks_wgs = np.copy(blocks3d)

    # %% Insert elevation

    def elevation(lat_, lon_):
        """
        Gets the elevation from a raster file given a pair of latitude/longitude coordinates
        :param lat_: latitude WGS84 in decimal degrees
        :param lon_: longitude WGS84 in decimal degrees
        :return: Elevation (m)
        """
        idx = dataset.index(lon_, lat_)
        return r[idx]

    # Convert distance along axis to lat/lon and add elevation
    for i in range(len(blocks_wgs)):
        lat, lon = lat_lon(blocks_wgs[i, 0])
        blocks_wgs[i, 0] = lat
        blocks_wgs[i, 1] = lon
        altitude = elevation(lat, lon) - np.abs(blocks_wgs[i, 2])
        blocks_wgs[i, 2] = altitude

    # %% Set in local coordinate system

    lat_origin, long_origin = 11.207775, 108.529248  # Arbitrary origin

    def local_system(lat_p, lon_p):
        """
        Given an origin, converts the WGS84 coordinates into meters around that point.
        :param lat_p: latitude (decimal degree wgs84)
        :param lon_p: longitude (decimal degree wgs84)
        :return:
        """
        line = geod.InverseLine(lat_origin, long_origin, lat_p, lon_p)
        azi = line.azi1
        dis = line.s13
        return dis * math.sin(math.radians(azi)), dis * math.cos(math.radians(azi))

    blocks_local = np.copy(blocks_wgs)
    # Update coordinates
    for i in range(len(blocks_wgs)):
        x, y = local_system(blocks_local[i, 0], blocks_local[i, 1])
        blocks_local[i, 0], blocks_local[i, 1] = x, y

    # %% VTK file creation

    # Initiate points and ugrid
    points = vtk.vtkPoints()
    ugrid = vtk.vtkUnstructuredGrid()

    for e, b in enumerate(blocks_local):
        points.InsertNextPoint(b)

    # Insert cells in UGrid
    [ugrid.InsertNextCell(vtk.VTK_QUAD, 4, list(range(e*4, e*4+4))) for e in range(len(blocks))]
    ugrid.SetPoints(points)  # Set points

    # Initiate array and give it a name
    resArray = vtk.vtkDoubleArray()
    resArray.SetName("res")

    [resArray.InsertNextValue(r) for r in rho]
    ugrid.GetCellData().AddArray(resArray)  # Add array to unstructured grid

    # Save grid
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(ugrid)
    writer.SetFileName(".\\vtk\\{}.vtu".format(name))
    writer.Write()

    return 0


if __name__ == '__main__':
    conversion('28')
