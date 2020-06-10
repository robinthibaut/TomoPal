#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import os
from os.path import join as jp

import numpy as np
import rasterio
import vtk
from geographiclib.geodesic import Geodesic

# %% Set directories
cwd = os.getcwd()
data_dir = jp(cwd, 'tomopal', 'data')
vtk_dir = jp(cwd, 'tomopal', 'vtk')
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

# Define meshgrid whose vertices will be used to triangulate the area
xv, yv = np.meshgrid(lats, longs, sparse=False, indexing='xy')
cs = np.stack((xv, yv), axis=2).reshape((-1, 2))  # Stack coordinates
dem_raw = np.array([[c[0], c[1], elevation(c[0], c[1])] for c in cs])  # Extract elevation


def dem_local_system(arg):
    """
    Given an origin, converts the WGS84 coordinates into meters around that point.
    :param arg: [lat (decimal degree wgs84), lon (decimal degree wgs84), elev (m)]
    :return:
    """
    lat_origin, long_origin = 11.207775, 108.529248
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

# The triangulation has texture coordinates generated so we can map
# a texture onto it.
tmapper = vtk.vtkTextureMapToPlane()
tmapper.SetInputConnection(smooth_loop.GetOutputPort())

# We scale the texture coordinate to get some repeat patterns.
xform = vtk.vtkTransformTextureCoords()
xform.SetInputConnection(tmapper.GetOutputPort())
# xform.SetScale(4, 4, 1)

bmpReader = vtk.vtkJPEGReader()
bmpReader.SetFileName(jp(data_dir, 'bb2c.jpg'))
atext = vtk.vtkTexture()
atext.SetInputConnection(bmpReader.GetOutputPort())
atext.InterpolateOn()

# Create a mapper and actor for smoothed dataset
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(xform.GetOutputPort())
actor_loop = vtk.vtkActor()
actor_loop.SetMapper(mapper)
actor_loop.SetTexture(atext)
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

renderer.SetBackground(0, 0, 0)  # Background color white
renderWindow.SetSize(800, 800)
renderWindow.Render()
renderWindowInteractor.Start()

# Save Polydata to XML format. Use smooth_loop.GetOutput() to obtain filtered polydata
writer = vtk.vtkPolyDataWriter()
writer.SetInputData(smooth_loop.GetOutput())
writer.SetFileTypeToBinary()
writer.SetFileName(jp(vtk_dir, 'texture.vtk'))
writer.Update()
