#  Copyright (c) 2020. Robin Thibaut, Ghent University

import math
import operator
import os
import time
from functools import reduce

import numpy as np
import vtk


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


class TomoVTK:

    def __init__(self, output_dir, name=None):

        self.output_dir = output_dir

        if name is None:
            self.name = str(round(time.time()))
        else:
            self.name = name

    def grid_to_vtk(self, blocks, values, values_names=None):
        """
        # TODO: define values shape
        :param blocks:
        :param values:
        :param values_names:
        :return:
        """

        if values_names is None:
            values_names = [f'val{i}' for i in range(values.shape[1])]

        # Initiate points and ugrid
        points = vtk.vtkPoints()
        ugrid = vtk.vtkUnstructuredGrid()

        for e, b in enumerate(blocks):
            points.InsertNextPoint(b)

        ncells = len(blocks) // 4
        # Insert cells in UGrid
        [ugrid.InsertNextCell(vtk.VTK_QUAD, 4, list(range(e * 4, e * 4 + 4))) for e in range(ncells)]
        ugrid.SetPoints(points)  # Set points

        # Initiate array and give it a name
        for e, val in enumerate(values_names):
            resArray = vtk.vtkDoubleArray()
            resArray.SetName(f'{val}')
            [resArray.InsertNextValue(r) for r in values[:, e]]
            ugrid.GetCellData().AddArray(resArray)  # Add array to unstructured grid

        # Save grid

        vtu_file = os.path.join(self.output_dir, f'{self.name}.vtu')

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(ugrid)
        writer.SetFileName(vtu_file)
        writer.Write()

    def dem_to_vtk(self, dem):
        # %%
        points = vtk.vtkPoints()
        [points.InsertNextPoint(c) for c in dem]

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

        smooth_loop.Update()

        # Save Polydata to XML format. Use smooth_loop.GetOutput() to obtain filtered polydata
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(smooth_loop.GetOutput())
        writer.SetFileName(os.path.join(self.output_dir, 'dem.vtp'))
        writer.Write()

        return 0
