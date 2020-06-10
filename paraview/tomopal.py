#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import operator
import os
from functools import reduce

import numpy as np
import rasterio
import vtk
from geographiclib.geodesic import Geodesic
from scipy.interpolate import interp1d


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


class Transformation:

    def __init__(self, file, bounds, dem=None, origin=None, name=None):
        """
        :param file: Results file containing block coordinates and associated values
        :param bounds: tuple: ((lat1, lon1), (lat2, lon2))
        :param dem: Digital Elevation Model file
        :param origin: Coordinates of the origin of the map (lat, lon)
        :param name: str: Name of the output file
        """
        self.block_file = file
        self.bounds = bounds
        self.elevation_file = dem
        self.origin = origin

        if name is None:
            self.name = os.path.splitext(os.path.basename(self.block_file))[0]
        else:
            self.name = name

        self.output_dir = os.path.join(os.getcwd(), 'vtk')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def conversion(self):

        blocks = read_file(self.block_file)  # Raw mesh info
        # TODO: Optimize parse - implement for several output (res, ip..)
        blocks2d_flat = blocks[:, 1:9]  # Flat list of polygon vertices
        rho = blocks[:, -1]  # Resistivity

        # Load profile bounds, must be in the correct format:
        # [[ lat1, lon1], [lat2, lon2]]
        bounds = self.bounds  # Flip only in this case as the format is incorrect.

        # Elevation data
        if self.elevation_file is not None:
            tif = 0
            if '.tif' in self.elevation_file.lower():  # If tif file
                tif = 1
                # Load tif file
                z = rasterio.open(self.elevation_file)
                # Elevation data:
                r = z.read(1)

                def elevation(lat_, lon_):
                    """
                    Gets the elevation from a raster file given a pair of latitude/longitude coordinates
                    :param lat_: latitude WGS84 in decimal degrees
                    :param lon_: longitude WGS84 in decimal degrees
                    :return: Elevation (m)
                    """
                    idx = z.index(lon_, lat_)
                    return r[idx]

            else:  # If 2D x - z data
                z = read_file(self.elevation_file)
                # Interpolating function to fill missing values
                fi = interp1d(z[:, 0], z[:, 1], fill_value='extrapolate')

                def elevation(x_):
                    """
                    :return: Elevation (m)
                    """
                    return fi(x_)
        else:
            def elevation(*args):
                return 0

        blocks2d = blocks2d_flat.reshape(-1, 4, 2)  # Reshape in (n, 4, 2)

        # %% Order vertices in each block to correspond to VTK requirements
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

        # Convert distance along axis to lat/lon and add elevation
        if tif:
            for i in range(len(blocks_wgs)):
                lat, lon = lat_lon(blocks_wgs[i, 0])
                blocks_wgs[i, 0] = lat
                blocks_wgs[i, 1] = lon
                altitude = elevation(lat, lon) - np.abs(blocks_wgs[i, 2])
                blocks_wgs[i, 2] = altitude
        else:
            for i in range(len(blocks_wgs)):
                altitude = elevation(blocks_wgs[i, 0]) - np.abs(blocks_wgs[i, 2])
                lat, lon = lat_lon(blocks_wgs[i, 0])
                blocks_wgs[i, 0] = lat
                blocks_wgs[i, 1] = lon
                blocks_wgs[i, 2] = altitude

        # %% Set in local coordinate system

        lat_origin, long_origin = self.origin  # Arbitrary origin

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
        [ugrid.InsertNextCell(vtk.VTK_QUAD, 4, list(range(e * 4, e * 4 + 4))) for e in range(len(blocks))]
        ugrid.SetPoints(points)  # Set points

        # Initiate array and give it a name
        resArray = vtk.vtkDoubleArray()
        resArray.SetName("res")

        [resArray.InsertNextValue(r) for r in rho]
        ugrid.GetCellData().AddArray(resArray)  # Add array to unstructured grid

        # Save grid

        vtu_file = os.path.join(self.output_dir, f'{self.name}.vtu')

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(ugrid)
        writer.SetFileName(vtu_file)
        writer.Write()

        return 0

    def dem(self, dem_file, bounding_box, n_x=100, n_y=100):
        """
        :param dem_file: Path to dem file
        :param bounding_box: tuple: Bounding box (rectangle) of the DEM ((lat1, long1), (lat2, long2))
        :param n_x: int: Number of cells in x direction (longitude)
        :param n_y: int: Number of cells in y direction (latitude)
        :return:
        """
        # TODO: add more DEM files input options

        if '.tif' in dem_file.lower():
            # Load tif file
            dataset = rasterio.open(dem_file)
            # Elevation data:
            r = dataset.read(1)

            def elevation(lat_, lon_):
                idx = dataset.index(lon_, lat_)
                return r[idx]
        else:  # Expects (lat, lon, elev) data file
            dataset = read_file(dem_file)

            points = dataset[:, :2]
            values = dataset[:, -1]

            def elevation(lat_, lon_):
                """Inverse Square Distance"""
                d = np.sqrt((lon_ - points[:, 1]) ** 2 + (lat_ - points[:, 0]) ** 2)
                if d.min() > 0:
                    v = np.sum(values * (1 / d ** 2) / np.sum(1 / d ** 2))
                    return v
                else:
                    return values[d.argmin()]

        geod = Geodesic.WGS84  # define the WGS84 ellipsoid
        # %% Define bounds of polygon in which to build DEM
        bbox = bounding_box

        lats = np.linspace(bbox[0][0], bbox[1][0], n_y)
        longs = np.linspace(bbox[0][1], bbox[1][1], n_x)

        # Define meshgrid whose vertices will be used to triangulate the area
        xv, yv = np.meshgrid(lats, longs, sparse=False, indexing='xy')
        cs = np.stack((xv, yv), axis=2).reshape((-1, 2))  # Stack coordinates
        dem_raw = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in cs])  # Extract elevation

        lat_origin, long_origin = self.origin

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

        dem_local = np.array(list(map(dem_local_system, dem_raw)))  # Convert WGS to local coordinates in meters

        # %%
        points = vtk.vtkPoints()
        [points.InsertNextPoint(c) for c in dem_local]

        aPolyData = vtk.vtkPolyData()
        aPolyData.SetPoints(points)

        aCellArray = vtk.vtkCellArray()

        # Start triangulation - define boundary
        boundary = vtk.vtkPolyData()
        boundary.SetPoints(aPolyData.GetPoints())
        boundary.SetPolys(aCellArray)
        # Perform Delaunay 2D
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(aPolyData)
        delaunay.SetSourceData(boundary)

        delaunay.Update()

        # Extract the polydata object from the triangulation = all the triangles
        trianglePolyData = delaunay.GetOutput()

        # Clean the polydata so that the edges are shared !
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(trianglePolyData)

        # Use a filter to smooth the data (will add triangles and smooth) - default parameters
        smooth_loop = vtk.vtkLoopSubdivisionFilter()
        smooth_loop.SetNumberOfSubdivisions(3)
        smooth_loop.SetInputConnection(cleanPolyData.GetOutputPort())

        # Save Polydata to XML format. Use smooth_loop.GetOutput() to obtain filtered polydata
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(smooth_loop.GetOutput())
        writer.SetFileTypeToBinary()
        writer.SetFileName(os.path.join(self.output_dir, 'dem.vtk'))
        writer.Update()