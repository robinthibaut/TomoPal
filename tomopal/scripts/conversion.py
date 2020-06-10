#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

from tomopal.geoview.iotomo import Transformation

if __name__ == '__main__':
    cwd = os.path.dirname(os.getcwd())
    file = '28'
    data_ = os.path.join(cwd, 'data', file, f'p{file}.dat')
    dem_ = os.path.join(cwd, 'data', 'n11_e108_1arc_v3.tif')

    # Arbitrary geographic origin (WGS84)
    origin = (11.207775, 108.529248)

    # Geographic bounds for DEM
    dem_bbox = ((11.147984, 108.422479), (11.284767, 108.602519))

    # Geographic tomography bounds (start point, end point)
    tomo_bounds = ((11.21201674, 108.5593435), (11.21594119, 108.5563131))

    # Initiate transformation object
    tio = Transformation(data_, bounds=tomo_bounds, origin=origin, dem=dem_)

    # Perform conversion
    tio.conversion()

    # Generate DEM
    tio.dem(dem_file=dem_, bounding_box=dem_bbox, n_x=100, n_y=100)


