#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os

from tomopal.geoview.iotomo import Transformation

cwd = os.getcwd()
file = '28'
data_ = os.path.join(cwd, 'data', file)
dem_ = os.path.join(cwd, 'data', 'n11_e108_1arc_v3.tif')

bounds = ((11.147984, 108.422479), (11.284767, 108.602519))
origin = (11.207775, 108.529248)

tio = Transformation(data_, bounds=bounds, origin=origin, dem=dem_)


