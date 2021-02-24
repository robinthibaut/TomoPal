#  Copyright (c) 2020. Robin Thibaut, Ghent University

from os.path import join as jp

import numpy as np

from tomopal.crtomopy.crtomo.crc import Crtomo, datread, mesh_geometry, import_res, mtophase
from ..parent import inventory
from ...geoview.diavatly import model_map  # To plot results

# %% Directories

# Input here the folders to structure your project. It is not necessary to previously create them
# (except the data folder)
# they will be automatically generated once you initialize a crtomo object.
# Note: the function 'jp' simply joins the arguments to build a path.
main_dir = inventory.hello()  # Current working directory of the project
data_dir = jp(main_dir, 'data', 'demo')  # Data files directory
mesh_dir = jp(main_dir, 'mesh', 'demo')  # Mesh files directory
iso_dir = jp(main_dir, 'iso', 'demo')  # ISO file dir
ref_dir = jp(main_dir, 'ref', 'demo')  # Reference model files dir
start_dir = jp(main_dir, 'start', 'demo')  # Start model files dir
results_dir = jp(main_dir, 'results', 'demo')  # Results files directory

# %% Exe names

# Input here the path to your exe files.

mesh_exe_name = jp(main_dir, 'mesh.exe')
crtomo_exe_name = jp(main_dir, 'crtomo.exe')

# %%  Create crtomo object

# Folders will be generated here if they don't exist already.

myinv = Crtomo(working_dir=main_dir,
               data_dir=data_dir,
               mesh_dir=mesh_dir,
               iso_dir=iso_dir,
               ref_dir=ref_dir,
               start_dir=start_dir,
               crtomo_exe=crtomo_exe_name,
               mesh_exe=mesh_exe_name)

# %%  Generating the mesh

# Data file name A B M N in meters

df = jp(data_dir, 'demo_elecs.dat')  # Path to electrode configuration file
dat = datread(df)  # Use built-in function to extract data (optional)

# Electrode spacing in meters
es = 5

#  Electrodes elevation
ef = jp(data_dir, 'demo_elevation.dat')  # Data elevation file name X Z in meters
elev = datread(ef)  # Use built-in function to extract data (optional)

# %% Build the mesh

# The following command generates the mesh in the folder indicated previously.

# It requires 3 arguments:
# the numpy array of electrodes position of shape (n, 4) (required)
# the electrode spacing (required)
# the elevation data (optional)

myinv.meshmaker(abmn=dat[:, [0, 1, 2, 3]],
                electrode_spacing=es,
                elevation_data=elev)
# If you already have generated a mesh, comment the line above and instead
# load the previously generated Mesh.dat file as described below.

# %% Read the mesh data (number of cells, blocks coordinates, x-y coordinates of the center of the blocks) from Mesh.dat

mshf = jp(mesh_dir, 'Mesh.dat')  # Path to the generated 'Mesh.dat' file.
ncol, nlin, nelem, blocks, centerxy = mesh_geometry(mshf)  # Extract mesh properties

# %% Build configuration file

# 0 Mesh.dat file
mesh_file = mshf

# 1 elec.dat file
elec_file = jp(mesh_dir, 'elec.dat')

# 2 Data file
data_file = jp(data_dir, 'demo_data.dat')

# 3 Results folder file

# Specify the path where the results will be loaded

frname = ''  # If you want to save the results in a sub-folder in the main results folder

result_folder = jp(results_dir, frname)

# 8 Flag for reference model constraint (0/1)
reference_model = 0
#
reference_model_file = None

# %% 12 File for reference model (model weights)

reference_weights_file = None

# You can use the tool ModelMaker from mohinh to interactively create prior models, and automatically save the results
# in a dat file if you provide a file name.
# Otherwise you can access the final results with (ModelMaker object).final_results and export it yourself.

# Example with a background resistivity of 100 ohm.m :

# rfwm = ModelMaker(blocks=blocks, values_log=1, bck=100)
# my_model = rfwm.final_results

# Alternatively, use a simpler approach to produce a reference model file:

# with open(reference_weights_file, 'w') as rw:
#     rw.write(str(nelem)+'\n')
#     [rw.write('0.1'+'\n') for i in range(nelem)]
#     rw.close()

# %% 22 Maximum numbers of iterations
iterations = 20

# 23 Min data RMS
rms = 1.0000

# 24 Flag for DC inversion (0 = with IP / 1 = only DC)
dc = 1

# 25 Flag for robust inversion (0/1)
robust = 1

# 26 Flag for checking polarity (0/1)
check_polarity = 1

# 27 Flag for final phase improvement (0/1)
final_phase_improvement = 1

# 29 Relative magnitude error level (%)
error_level = 2.5

# 30 Minimum absolute magnitude error (ohm)
min_abs_error = 0.00015

# 31 Error in phase (mrad)
phase_error = 0.5

# 36 Flag for MGS inversion (0/1)
mgs = 0

# 37 Beta value
beta = 0.002

# 38 Flag for starting model (0/1)
starting_model = 0

# 39 Starting model file
starting_model_file = None

# %% 19 ISO file 1
iso_file1 = jp(iso_dir, 'iso.dat')

# dm = datread(starting_model_file, start=1)[:, 0]
# isom = ModelMaker(blocks=blocks, values=dm, values_log=1, bck=1)
# #
# with open(iso_file1, 'w') as rw:
#     rw.write(str(nelem)+'\n')
#     [rw.write('{} 1'.format(str(i))+'\n') for i in isom.final_results]
#     rw.close()


# %% Generate configuration file

# If erase = 1, every item in the result folder will be deleted. If you don't want that, pick 0 instead.

# Use help(Crtomo.write_config) to see which parameters you can implement.

myinv.write_config(erase=1,
                   mesh_file=mesh_file,
                   elec_file=elec_file,
                   data_file=data_file,
                   result_folder=result_folder,
                   reference_model=reference_model,
                   reference_model_file=reference_model_file,
                   reference_weights_file=reference_weights_file,
                   iso_file1=iso_file1,
                   iterations=iterations,
                   rms=rms,
                   dc=dc,
                   robust=robust,
                   check_polarity=check_polarity,
                   final_phase_improvement=final_phase_improvement,
                   error_level=error_level,
                   min_abs_error=min_abs_error,
                   phase_error=phase_error,
                   mgs=mgs,
                   beta=beta,
                   starting_model=starting_model,
                   starting_model_file=starting_model_file)

# Forward modeling example :

# # Results folder file
# fwname = 'fwd'  # If you want to save the results in a sub-folder in the main results folder
#
# result_folder_fwd = jp(results_dir, fwname)
#
# myfwd = Crtomo(working_dir=cwd,
#                data_dir=data_dir,
#                mesh_dir=mesh_dir,
#                crtomo_exe=crtomo_exe_name)
#
# # # res2mod(jp(result_folder, 'rho1.txt'))
# myfwd.write_config(mesh_file=mesh_file,
#                    elec_file=elec_file,
#                    fwd_only=1,
#                    result_folder=result_folder_fwd,
#                    starting_model_file=jp(cwd, 'rho1.dat'))
# myfwd.run()

# %% Run CRTOMO

# This will make your Crtomo object run the inversion. The configuration files are
# automatically saved in the results folder

myinv.run()

# %% Import results

if dc == 0:  # If you have IP results to load
    res, ip = import_res(result_folder=result_folder)
    m2p = mtophase(ncycles=1, pulse_l=3.5, tmin=0.02, tmax=2.83)
    ipt = ip[:] * m2p
else:  # if you only have resistivity data to load
    res, files = import_res(result_folder=result_folder, return_file=1)
    rest = np.copy(res[0])

# If you want to convert a crtomo result file in a prior model for future inversions for example:
# modf = res2mod(files[0])

# Let's plot the results:
# Remove outliers (arbitrary)
cut = np.log10(4500)
rest[rest > cut] = cut
res_levels = 10 ** np.linspace(min(rest), cut, 10)  # Define a linear space for the color map
rtp = 10 ** np.copy(rest)

# Use the model_map function to display the computed resistivity:
# log=1 because we want a logarithmic scale.
# cbpos is for the position of the color bar.
model_map(polygons=blocks,
          vals=rtp,
          log=1,
          cbpos=0.4,
          levels=res_levels,
          folder=result_folder,
          figname='demo_res_levels')

# %% if IP
if dc == 0:
    ip = np.copy(res[1])
    # crtomo works in phase so we perform the conversion to go back to "mv/v".
    m2p = mtophase(ncycles=1, pulse_l=3.5, tmin=0.02, tmax=2.83)
    ipt = np.copy(np.abs(ip / m2p))

    # Arbitrarily cut outliers
    hist = np.histogram(ipt, bins='auto')
    cut = 260
    ipt[ipt > cut] = cut

    # Define levels to be plotted
    ip_levels = [0, 10, 20, 30, 40, 50, 60, 70, 260]

    model_map(polygons=blocks,
              vals=ipt,
              log=0,
              levels=ip_levels,
              folder=result_folder,
              figname='demo_ip_level')
