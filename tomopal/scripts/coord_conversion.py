#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

import numpy as np

from tomopal.geoview.iotomo import Transformation

if __name__ == '__main__':
    cwd = os.path.dirname(os.getcwd())
    files = ['28']
    dem_ = os.path.join(cwd, 'data', 'n11_e108_1arc_v3.tif')
    # Arbitrary geographic origin (WGS84)
    origin = (11.207775, 108.529248)

    # Generate DEM
    demio = Transformation(origin=origin)
    # Geographic bounds for DEM
    dem_bbox = ((11.147984, 108.422479), (11.284767, 108.602519))
    demio.dem(dem_file=dem_, bounding_box=dem_bbox, n_x=100, n_y=100)

    for f in files:
        data_ = os.path.join(cwd, 'data', f, f'{f}.dat')

        # Geographic tomography bounds (start point, end point)
        with open(os.path.join(cwd, 'data', 'mycoordinates.txt'), 'r') as cf:
            pc = np.array([line.split() for line in cf.readlines()])
        # Load tomography bounds
        p_bounds = np.flip(np.array(pc[np.where(f'P{f}' in pc[:, 0])][0][1:].astype(np.float)).reshape(-1, 2), axis=1)

        # Initiate transformation object
        tio = Transformation(data_, bounds=p_bounds, origin=origin, dem=dem_)

        # Perform conversion
        tio.conversion()




