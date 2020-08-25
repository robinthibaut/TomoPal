#  Copyright (c) 2020. Robin Thibaut, Ghent University

import ntpath
import os
import shutil
import subprocess as sp
import warnings
from os import listdir
from os.path import isfile, join
from os.path import join as jp
from shutil import copyfile

import numpy as np
from scipy.interpolate import interp1d as f1d

from tomopal.crtomopy.parent import inventory


#  Functions


def path_leaf(path):
    """Extracts file name from a path name"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def datread(file=None, start=0, end=None):
    # end must be set to None and NOT -1
    """Reads space separated dat file"""
    with open(file, 'r') as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array([list(map(float, line.split())) for line in lines])
        except ValueError:
            op = [line.split() for line in lines]
    return op


def res2mod(file, processing_function=None):
    """Converts a crtomo result txt file to a starting/reference model dat file"""

    # ref or start model:
    # nelem
    # resistivity (ohm*m) -- 0 -- log10(resistivity) -- IP (mrad)

    fext = file.split('.')[-1]

    f = datread(file)
    sm = [f[i][2] for i in range(1, len(f))]

    if processing_function:
        sm = processing_function(np.array(sm))

    fname = file.replace(fext, 'dat')

    with open(fname, 'w') as smf:
        smf.write(str(int(f[0][0])) + '\n')
        [smf.write(str(10 ** (sm[i])) + '\t' + '0' + '\t' + str(sm[i]) + '\t' + '0' + '\n')
         for i in range(0, len(sm))]
        smf.close()

    return fname


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flag(n):
    """
    Used to build configuration file
    :param n: 0 or 1
    :return: F or T
    """

    if n == 0:
        return 'F'
    else:
        return 'T'


def write_data(nelem=0,
               electrode_spacing=1,
               data=None,
               data_op_file=None,
               m2p=1):
    """

    Given the arguments below, writes crtomo-readable file.

    :param nelem: number of cells
    :param electrode_spacing: electrode spacing
    :param data: np.array containing A B M N locations in METERS, RESISTANCE, and IP (mrad)
    :param data_op_file: name of the output data file
    :param m2p Factor to convert IP data to mrad
    :return: nothing

    """

    crdata = data
    es = electrode_spacing

    crdata[:, [0, 1, 2, 3]] = (crdata[:,
                               [0, 1, 2, 3]] + es) / es  # Convert the electrode x-position to electrodes number.

    if m2p != 1:
        crdata[:, -1] *= m2p

    crdataf = str(nelem) + '\n' + '\n'.join([' '.join(list(map(str, l))) for l in crdata])

    with open(data_op_file, 'w') as ndf:
        ndf.write(crdataf)
        ndf.close()


def mtophase(ncycles=0,
             pulse_l=0,
             tmin=0,
             tmax=0,
             mpath=None):
    """
    Run mtophase.exe and loads return the conversion factor.
    :param ncycles: number of cycles of injected square wave (with 50% duty cycle)
    :param pulse_l: pulse length (in sec)
    :param tmin: t_min of chargeability time window (in sec)
    :param tmax: t_max of chargeability time window (in sec)
    :param mpath: str: Path to the folder containing IP exe and config files
    :return: m2p factor
    """

    if mpath is None:
        main_dir = inventory.hello()
        mpath = jp(main_dir, 'ip')

    if not os.path.exists(mpath):
        warnings.warn(mpath + ' folder not found')

    params = list(map(str, [ncycles, pulse_l, tmin, tmax]))  # Transforms the params input to string

    minf = jp(mpath, 'MtoPhase.cfg')  # Writing config file

    with open(minf, 'w') as ms:
        ms.write('\n'.join(params))
        ms.close()

    #  Running m2p exe file
    if not os.path.exists(jp(mpath, 'mtophase.exe')):
        warnings.warn('mtophase.exe not found')

    os.chdir(mpath)
    try:
        sp.call([jp(mpath, 'mtophase.exe')])  # Run
    except Exception as e:
        print(e)
    os.chdir(os.path.dirname(mpath))

    mm = open(jp(mpath, 'MtoPhase.dat'), 'r').readlines()
    ms = mm[0].split()
    f = float(ms[0])

    return f


def crtomo_file_shortener(f1, f2):
    """CRTOMO cannot deal with long folder path, especially not with spaces within it..."""
    fs = f2

    if f2:  # If file is different than None

        exefolder = f1.replace(jp('\\', path_leaf(f1)), '')  # Remove file name from path

        if exefolder in f2:
            fs = '.\\' + f2.replace(exefolder, '')  # CRtomo can not deal with long paths
        else:
            fs = f2
            if len(f2) == len(path_leaf(f2)):
                fs = '.\\' + f2
    return fs


def import_res(result_folder,
               iteration=0,
               return_file=0):
    """
    :param return_file: bool: if True, returns the path of the created file
    :param result_folder: str: FOLDER containing results files .rho .pha
    :param iteration: int: Iteration number, by default the last one is selected
    :return: r_array, p_array if the case
    """

    iteration = iteration - 1

    onlyfiles = [f for f in listdir(result_folder) if isfile(join(result_folder, f))]  # Lists all the file in
    # requested folder

    rho_files = [jp(result_folder, x) for x in onlyfiles if 'rho' in x and '.txt' in x]
    pha_files = [jp(result_folder, x) for x in onlyfiles if '.pha' in x]
    volt_files = [jp(result_folder, x) for x in onlyfiles if 'volt' in x and '.dat' in x]
    sens = jp(result_folder, 'sens.dat')

    r_array = np.array([])
    p_array = np.array([])
    v_array = np.array([])

    if len(rho_files) > 0:

        # Load resistance
        its_res = [int(os.path.basename(r).split('.')[0].strip('rho0').strip('rho')) for r in rho_files]
        nits = max(its_res)

        if iteration == -1:
            iter = its_res.index(nits)
        else:
            iter = its_res.index(iteration)

        rlast = rho_files[iter]
        rholast = np.array(datread(rlast, start=1))
        r_array = np.array([rholast[r][2] for r in range(len(rholast))])

        # Load volt
        its_volt = [int(os.path.basename(v).split('.')[0].strip('volt0').strip('volt')) for v in volt_files]

        if iteration == -1:
            iter_v = its_volt.index(nits)
        else:
            iter_v = its_volt.index(iteration)

        vlast = volt_files[iter_v]
        v_array = np.array(datread(vlast, start=1))

        #  Load phase
        if len(pha_files) > 0:
            its_pha = [int(os.path.basename(r).split('.')[0].strip('rho0').strip('rho')) for r in pha_files]
            if iteration == -1:
                iterip = its_pha.index(nits)
            else:
                iterip = its_pha.index(iteration)

            iplast = pha_files[iterip]
            phalast = np.array(datread(iplast, start=1))
            p_array = np.array([phalast[r][2] for r in range(len(phalast))])
        else:
            iplast = ''

        # Load sensitivity
        try:
            s_array = np.array(datread(sens, start=1))
        except FileNotFoundError:
            s_array = []

    else:
        warnings.warn('no results found')

    if return_file:
        return [r_array, p_array, v_array, s_array], [rlast, iplast, vlast, sens]

    else:
        return [r_array, p_array, v_array, s_array]


def dirmaker(dirp):
    """
    Given a folder path, check if it exists, and if not, creates it
    :param dirp: path
    :return:
    """
    try:
        if not os.path.exists(dirp):
            os.makedirs(dirp)
    except Exception as e:
        print(e)


def deldir(dirp):
    """
    Used to delete the contents of the folder
    :param dirp: path to the folder to be emptied
    :return:
    """
    for the_file in os.listdir(dirp):
        file_path = os.path.join(dirp, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def mesh_geometry(mesh_file):
    """
    Extracts the mesh properties from the mesh.dat file.

    :param mesh_file: mesh.dat file
    :return: ncol, nlin, nelem, blocks, centerxy
    """

    # Reading mesh results
    # msh = datread(mesh_file) # Deprecated

    with open(mesh_file, 'r') as fr:
        msh = np.array([list(map(float, l.replace('T', '').split())) for l in fr.readlines()])

    nnodes = int(msh[0][0])
    nelem = int(msh[1][1])
    ncol = int(msh[0][3])
    nlin = int(msh[0][4])

    nodes = np.array(msh[4:(nnodes + 4)])
    xn = np.array(list(chunks([nodes[i][1] for i in range(nnodes)], ncol + 1))).flatten()
    yn = np.array(list(chunks([nodes[i][2] for i in range(nnodes)], nlin + 1))).flatten()

    xy = np.array([[xn[i], yn[i]] for i in range(len(xn))])

    layers = np.array(list(chunks(xy, ncol + 1)))

    # Computing the 4-corners coordinates of each blocks based on the node position

    blocks = []

    s = layers.shape

    for i in range(s[0] - 1):
        for j in range(s[1] - 1):
            blocks.append([
                [layers[i, j, 0], layers[i, j, 1]],
                [layers[i + 1, j, 0], layers[i + 1, j, 1]],
                [layers[i + 1, j + 1, 0], layers[i + 1, j + 1, 1]],
                [layers[i + 1, j + 1, 0], layers[i, j + 1, 1]]
            ])

    blocks = np.array(blocks)

    centerxy = np.array([[np.mean(blocks[i, :, 0]), np.mean(blocks[i, :, 1])] for i in range(nelem)])

    return ncol, nlin, nelem, blocks, centerxy, nodes


def neighbor(abcd, h):
    # TODO: Write this in c++
    """

    Function to fill the final mesh file.

    :param abcd: adj [4 cols]
    :param h: int:
    :return: [el1, el2, el3, el4]
    """

    Nelem = len(abcd)

    a = abcd[h][0]
    b = abcd[h][1]
    c = abcd[h][2]
    d = abcd[h][3]

    el1, el2, el3, el4 = 0, 0, 0, 0

    N = 0

    for j in range(0, Nelem - 1):

        if N == 4:
            break

        if a in abcd[j, :] and b in abcd[j, :] and j != h:
            N += 1
            el1 = j + 1

        if b in abcd[j, :] and c in abcd[j, :] and j != h:
            N += 1
            el2 = j + 1

        if c in abcd[j, :] and d in abcd[j, :] and j != h:
            N += 1
            el3 = j + 1

        if d in abcd[j, :] and a in abcd[j, :] and j != h:
            N += 1
            el4 = j + 1

    return [el1, el2, el3, el4]


class Crtomo:

    def __init__(self,
                 working_dir='\\',  # Some defaults folders
                 data_dir='\\',
                 mesh_dir='\\',
                 iso_dir='\\',
                 ref_dir='\\',
                 start_dir='\\',
                 results_dir='\\',
                 crtomo_exe='crtomo.exe',
                 mesh_exe='mesh.exe'):

        self.working_dir = working_dir
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.iso_dir = iso_dir
        self.ref_dir = ref_dir
        self.start_dir = start_dir

        dirmaker(mesh_dir)
        dirmaker(iso_dir)
        dirmaker(ref_dir)
        dirmaker(start_dir)

        # self.results_dir = results_dir
        #
        # try:
        #     if not os.path.exists(results_dir):
        #         os.mkdir(results_dir)
        # except:
        #     pass

        self.crtomo_exe = crtomo_exe

        if not os.path.exists(crtomo_exe):  # Check if the crtomo exe files can be found.
            warnings.warn('Can not find crtomo executable')

        self.mesh_exe = mesh_exe

        if not os.path.exists(mesh_exe):  # Check if the crtomo exe files can be found.
            warnings.warn('Can not find mesh executable')

        self.rf = ''  # Results folder
        self.mf = ''  # Mesh data file
        self.ef = ''  # Elec data file
        self.df = ''  # Data file
        self.rwf = ''  # Reference model weights file
        self.iso_f1 = ''  # ISO file 1
        self.iso_f2 = ''  # ISO file 1
        self.smf = ''  # Starting model file
        self.f1 = ''  # F1
        self.rmf = ''  # Reference model file
        self.f3 = ''  # F3

    def meshmaker(self,
                  abmn=None,
                  electrode_spacing=5,
                  elevation_data=None
                  ):

        """
        Generates crtomo-readable mesh files.
        :param abmn: np.array containing x-coordinates of electrodes in METERS
        :param electrode_spacing:
        :param elevation_data: np.array containing [[x_i, elev_i]...[x_f, elev_f]]
        :return:
        """

        dat = abmn
        mesh_dir = self.mesh_dir
        mesh_exe_name = self.mesh_exe
        cwd = self.working_dir
        elev = elevation_data
        es = electrode_spacing  # Electrode spacing

        #  Electrodes x-position

        extent = int(max(dat.flatten()))  # Maximum electrode x-position

        epx = list(range(0, extent + es, es))

        epn = 'elec1.dat'
        epfn = jp(mesh_dir, epn)  # Electrode position file

        with open(epfn, 'w') as elec:
            [elec.write(str(i + 1) + ' ' + str(i * es) + '\n') for i in range(0, len(epx))]
            elec.close()

        #  Electrodes elevation

        fnel = f1d(elev[:, 0], elev[:, 1], kind='cubic',
                   fill_value="extrapolate")  # Interpolation function for the elevation to fill the gaps.
        elint = list(map(float, list(map(fnel, epx))))

        evn = 'topo1.dat'
        evf = jp(mesh_dir, evn)  # Electrode elevation file

        with open(evf, 'w') as elev:
            [elev.write(str(i * es) + ' ' + str(elint[i]) + '\n') for i in range(0, len(epx))]
            elev.close()

        ms_exe_f = jp(mesh_dir, path_leaf(mesh_exe_name))

        # if not os.path.exists(self.mesh_exe):  # Check if the mesh exe files can be found.
        #     if os.path.exists(jp(self.working_dir, path_leaf(mesh_exe_name))):
        #         copyfile(jp(self.working_dir, path_leaf(mesh_exe_name)), ms_exe_f)
        #     else:
        #         print('Can not find mesh executable')

        if os.path.exists(self.mesh_exe):  # Check if the mesh exe files can be found.
            if not os.path.exists(jp(self.mesh_dir, path_leaf(mesh_exe_name))):
                copyfile(self.mesh_exe, ms_exe_f)
        else:
            print('Can not find mesh executable')
        self.mesh_exe = ms_exe_f

        mesh_short = '.\\'
        epfn = crtomo_file_shortener(ms_exe_f, epfn)
        evf = crtomo_file_shortener(ms_exe_f, evf)

        #  Writing mesh.in file
        meshparams = ["{}".format(mesh_short),
                      "{}".format(epfn),
                      "Mesh", "2", "{}".format(evf),
                      "0 0",
                      "0.1 20 0.01 0.05"]

        meshinf = jp(os.path.dirname(self.mesh_exe), 'mesh.in')

        with open(meshinf, 'w') as ms:
            ms.write('\n'.join(meshparams))
            ms.close()

        #  Running mesh exe file

        mmdir = jp(mesh_dir, 'Model')  # CRTOMO automatically loads the results in a folder called 'Model'

        try:
            if not os.path.exists(mmdir):
                os.makedirs(mmdir)
        except Exception as e:
            print(e)

        os.chdir(self.mesh_dir)
        sp.call([self.mesh_exe])  # Run
        os.chdir(self.working_dir)

        msh = datread(jp(mmdir, 'Mesh.msh'))
        nelem = int(msh[1][1])

        #  Builing final mesh

        # 1 - Where there are 4 columns, move them to the right and add a column beginning by 1 incrementing by 1
        # each line.

        nc = [len(e) for e in msh]

        cidx1 = nc.index(4)  # Beginning of 4 columns
        b1 = nc.index(2)  # End of 4 columns when 2 columns appear
        [msh[i].insert(0, i + 1 - cidx1) for i in range(cidx1, b1)]

        # 2 - Where's there's one column, take it and place it on the right of the two previous one. Move the three
        # of them to the right and add a new column like step 1.

        l1 = [i for i, x in enumerate(nc) if x == 1]  # Index of rows of length 1
        cidx2 = l1[0]
        b2 = l1[-1]

        l2 = [i for i, x in enumerate(nc) if x == 2]  # Index of rows of length 2
        cidx3 = l2[0]
        b3 = l2[-1]

        for i in range(len(l2)):
            msh[l2[i]] += msh[l1[i]]  # Inserting column

        [msh[l2[i]].insert(0, int(i + 1)) for i in range(len(l2))]  # Inserting column 1 2 3 ...

        msh = np.delete(msh, l1)  # Deleting column

        nc2 = [len(e) for e in msh]  # New len array

        l5 = [i for i, x in enumerate(nc2) if x == 5]  # Index of rows of length 5
        l5.pop(0)  # First element = header, not necessary
        for j in range(len(l5)):  # Adding columns as required
            msh[l5[j]] += ['T', 'T', 'T', 'T', int(10 + j), 'T', int(2 + j), 'T']

        adj = [msh[l5[i]][1:5] for i in range(len(l5))]  # Preparing 'adj' file
        adj = np.array([list(map(int, a)) for a in adj])

        print('neighbourg process begins, there are {} elements'.format(nelem))

        adji = [neighbor(adj, i) for i in range(nelem)]

        print('neighbourg process over')

        for j in range(len(l5)):  # Adding columns as required
            msh[l5[j]] += list(map(str, adji[j]))

        # Export final mesh

        mesh_file_name = jp(mesh_dir, 'Mesh.dat')  # Export Mesh.msh as Mesh.dat and Mesh.elc as elec.dat

        meshdat = '\n'.join(['\t'.join(list(map(str, l))) for l in msh])

        with open(mesh_file_name, 'w') as md:
            md.write(meshdat)
            md.close()

        elec_file_name = jp(mesh_dir, 'elec.dat')  #

        copyfile(jp(mmdir, 'Mesh.elc'), elec_file_name)  # Mesh.elc -> elec.dat

        print('mesh generated')

    def write_config(self,
                     erase=0,
                     mesh_file='Mesh.dat',
                     elec_file='elec.dat',
                     data_file='dataset.dat',
                     result_folder='results',
                     difference_inversion=0,
                     fdi1=None,
                     reference_model_file=None,
                     fdi3=None,
                     reference_model=0,
                     data_difference=0,
                     stochastic_inversion=0,
                     prior_si=0,
                     reference_weights_file=None,
                     fincr=1,
                     grid_type=1,
                     arbitrary=5,
                     vario=1,
                     smoothing_x=1,
                     smoothing_y=1,
                     iso_file1='iso.dat',
                     iso_file2='iso.dat',
                     variogram_regularization=0,
                     iterations=20,
                     rms=1.0000,
                     dc=1,
                     robust=1,
                     check_polarity=1,
                     final_phase_improvement=0,
                     individual_error=0,
                     error_level=1,
                     min_abs_error=0.00015,
                     phase_error=0.15,
                     hom_bkg_res=0,
                     bkg_mag=160,
                     bkg_pha=0,
                     resolution_mtx=0,
                     mgs=0,
                     beta=0.0001,
                     starting_model=0,
                     starting_model_file='startmodel.dat',
                     fwd_only=0,
                     sink_node=0,
                     node_bumber=3348,
                     next_dset=0):

        """
        Writes the crtomo configuration file crtritime.cfg.

        :param erase: flag for erasing the content of a result folder (0/1)
        :param result_folder:
        :param mesh_file: Mesh data file
        :param elec_file: Elec data file
        :param data_file: Data file
        :param difference_inversion:
        :param fdi1:
        :param reference_model_file: Reference model file
        :param fdi3:
        :param reference_model:
        :param data_difference:
        :param stochastic_inversion:
        :param prior_si:
        :param reference_weights_file:
        :param fincr:
        :param grid_type:
        :param arbitrary:
        :param vario:
        :param smoothing_x:
        :param smoothing_y:
        :param iso_file1:
        :param iso_file2:
        :param variogram_regularization:
        :param iterations:
        :param rms:
        :param dc:
        :param robust:
        :param check_polarity:
        :param final_phase_improvement:
        :param individual_error:
        :param error_level:
        :param min_abs_error:
        :param phase_error:
        :param hom_bkg_res:
        :param bkg_mag:
        :param bkg_pha:
        :param resolution_mtx:
        :param mgs:
        :param beta:
        :param starting_model: If doing forward modeling only, select 1
        :param starting_model_file: If doing forward modeling only, input here the model
        :param fwd_only:
        :param sink_node:
        :param node_bumber:
        :param next_dset:
        :return:
        """

        if not os.path.exists(result_folder):  # If the result folder does not exists:
            os.makedirs(result_folder)  # Creates it
        else:
            if erase:  # If it exists and erase option enabled, empties its content first!
                deldir(result_folder)
            else:
                pass

        self.rf = result_folder
        rf_crtomo = crtomo_file_shortener(self.crtomo_exe, result_folder)

        self.mf = mesh_file  # Mesh data file
        mesh_file = crtomo_file_shortener(self.crtomo_exe, mesh_file)

        self.ef = elec_file  # Elec data file
        elec_file = crtomo_file_shortener(self.crtomo_exe, elec_file)

        self.df = data_file  # Data file
        data_file = crtomo_file_shortener(self.crtomo_exe, data_file)

        self.rwf = reference_weights_file  # Reference model weights file
        reference_weights_file = crtomo_file_shortener(self.crtomo_exe, reference_weights_file)

        self.iso_f1 = iso_file1  # ISO file 1
        iso_file1 = crtomo_file_shortener(self.crtomo_exe, iso_file1)

        iso_file2 = iso_file1

        self.iso_f2 = iso_file2  # ISO file 2
        iso_file2 = crtomo_file_shortener(self.crtomo_exe, iso_file2)

        self.smf = starting_model_file  # Starting model file
        starting_model_file = crtomo_file_shortener(self.crtomo_exe, starting_model_file)

        self.f1 = fdi1  # F1
        fdi1 = crtomo_file_shortener(self.crtomo_exe, fdi1)

        self.rmf = reference_model_file  # F2
        reference_model_file = crtomo_file_shortener(self.crtomo_exe, reference_model_file)

        self.f3 = fdi3  # F3
        fdi3 = crtomo_file_shortener(self.crtomo_exe, fdi3)

        template = """***Files****
{0}
{1}
{2}
{3}
{4} ! difference inversion?
{5}
{6}
{7}
{8} ! Reference model constraint
{9} ! Data difference only
{10} ! stochastic regularization?
{11} ! Prior model Cm^-1 (m-m0)
{12}
{13} ! fincr
{14} ! Grid type?1 triangular and 0 rectangular
***PARAMETERS***
{15} ! arbitrary
{16} ! 1 spheric 2 gaussian 3 power 4 exponential
{17} ! smoothing parameter in x-direction or correlation length (range X)
{18} ! smoothing parameter in z-direction or correlation length (range Y)
{19}
{20}
{21} ! Variogram regularization (TH)
{22} ! max.# inversion iterations
{23} ! min.data RMS
{24} ! DC inversion?
{25} ! robust inversion?
{26} ! check polarity
{27} ! final phase improvement
{28} ! individual error?
{29} ! rel.magnitude error level (%)
{30} ! min.abs.magnitude error (ohm)
{31} ! error in phase (mrad)
{32} ! homogeneous background resistivity?
{33} ! background magnitude (ohm*m)
{34} ! background magnitude (mrad)
{35} ! Compute resolution matrix
{36} ! MGS?
{37} !beta
{38} 'starting model?
{39}
{40} ! Forward modelling only
{41} ! Sink node activated
{42} ! Node number (no boarder and away from electrode)
{43} ! another dataset?""".format(mesh_file,
                                  elec_file,
                                  data_file,
                                  rf_crtomo,
                                  flag(difference_inversion),
                                  fdi1,
                                  reference_model_file,
                                  fdi3,
                                  flag(reference_model),
                                  flag(data_difference),
                                  flag(stochastic_inversion),
                                  flag(prior_si),
                                  reference_weights_file,
                                  fincr,
                                  grid_type,
                                  arbitrary,
                                  vario,
                                  smoothing_x,
                                  smoothing_y,
                                  iso_file1,
                                  iso_file2,
                                  flag(variogram_regularization),
                                  iterations,
                                  rms,
                                  flag(dc),
                                  flag(robust),
                                  flag(check_polarity),
                                  flag(final_phase_improvement),
                                  flag(individual_error),
                                  error_level,
                                  min_abs_error,
                                  phase_error,
                                  flag(hom_bkg_res),
                                  bkg_mag,
                                  bkg_pha,
                                  flag(resolution_mtx),
                                  flag(mgs),
                                  beta,
                                  flag(starting_model),
                                  starting_model_file,
                                  flag(fwd_only),
                                  flag(sink_node),
                                  node_bumber,
                                  flag(next_dset))

        with open(jp(os.path.dirname(self.crtomo_exe), 'crtritime.cfg'), 'w') as cf:
            cf.write(template)
            cf.close()

    def run(self):

        """
        Copies all the input files to a 'param' folder within the inversion results folder and run the exe.

        :return:
        """

        print(self.rf)

        try:
            if not os.path.exists(self.iso_f1):  # If no iso file exists, it automatically creates one
                ratio = 1
                ncol, nlin, nelem, blocks, centerxy = mesh_geometry(self.mf)
                isod = [[1, 1 * ratio] for i in range(nelem)]
                isodat = str(nelem) + '\n' + '\n'.join([' '.join(list(map(str, l))) for l in isod])
                with open(self.iso_f1, 'w') as md:
                    md.write(isodat)
                    md.close()
        except Exception as e:
            print(e)

        print('starting Crtomo')

        file_list = [self.mf, self.ef, self.df, self.rwf,
                     self.iso_f1, self.iso_f2, self.smf, self.f1, self.rmf, self.f3]

        pfn = jp(self.rf, 'config')
        if not os.path.exists(pfn):
            os.mkdir(pfn)

        copyfile(jp(os.path.dirname(self.crtomo_exe), 'crtritime.cfg'), jp(pfn, 'crtritime.cfg'))  # Copies cfg file

        for f in file_list:  # Copies each important file in the 'config' folder
            try:
                fname = path_leaf(f)
                copyfile(f, jp(pfn, fname))
            except Exception as e:
                print(e)

        sp.call([self.crtomo_exe])  # Runs the exe

        print('process over')
