#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import os
from os.path import join as jp

import numpy as np
import rasterio
import vtk
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

lats = np.linspace(bbox[0][0], bbox[1][0], 50)
longs = np.linspace(bbox[0][1], bbox[1][1], 50)
# cs = list(itertools.product(lats, longs))
xv, yv = np.meshgrid(lats, longs, sparse=False, indexing='xy')
cs = np.stack((xv, yv), axis=2).reshape((-1, 2))

tri = Delaunay(cs)  # Perform Delaunay triangulation on wgs coordinates
simp = tri.simplices  # Extract simplices (=vertices)
shp = simp.shape
points = tri.points  # Corresponding points
tri_del = points[simp].reshape(-1, 2)  # 2D array of points

dem_wgs = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in tri_del])  # For each vertices, extract elevation
dem_wgs_pt = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in cs])
# dem_wgs_pt = np.array([[xx, yy, elevation(xx, yy)] for xx in xv for yy in yv]).reshape(-1, 3)

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

cells_triangles = [list(np.arange(i * 3, i * 3 + 3)) for i in range(shp[0])]

points = vtk.vtkPoints()
triangles = vtk.vtkCellArray()

for i, p in enumerate(dem_local):
    points.InsertNextPoint(p)

for k, c in enumerate(cells_triangles):
    triangle = vtk.vtkTriangle()
    [triangle.GetPointIds().SetId(j, c[j]) for j in range(3)]
    triangles.InsertNextCell(triangle)

# Create a polydata object
trianglePolyData = vtk.vtkPolyData()

# Add the geometry and topology to the polydata
trianglePolyData.SetPoints(points)

trianglePolyData.SetPolys(triangles)

# Clean the polydata so that the edges are shared !
cleanPolyData = vtk.vtkCleanPolyData()
cleanPolyData.SetInputData(trianglePolyData)

# Use a filter to smooth the data (will add triangles and smooth)
smooth_loop = vtk.vtkLoopSubdivisionFilter()
smooth_loop.SetNumberOfSubdivisions(3)
smooth_loop.SetInputConnection(cleanPolyData.GetOutputPort())

# Create a mapper and actor for smoothed dataset
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(smooth_loop.GetOutputPort())
actor_loop = vtk.vtkActor()
actor_loop.SetMapper(mapper)
actor_loop.GetProperty().SetInterpolationToFlat()

# Update the pipeline so that vtkCellLocator finds cells !
# smooth_loop.Update()

# Visualize
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add actors and render
renderer.AddActor(actor_loop)

renderer.SetBackground(1, 1, 1)  # Background color white
renderWindow.SetSize(800, 800)
renderWindow.Render()
renderWindowInteractor.Start()

writer = vtk.vtkPolyDataWriter()
writer.SetInputData(smooth_loop.GetOutput())
writer.SetFileName('mysmoothloop.vtk')
writer.Update()