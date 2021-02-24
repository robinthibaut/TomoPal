#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_res(file):
    """Reads ABEM type output text files. Lowers the columns and removes special characters."""
    data = pd.read_csv(file, delimiter='\t')
    data.columns = [re.sub('[^A-Za-z0-9]+', '', col.lower()) for col in data.columns]

    return data


def export(file, normal_reciprocal):
    """Export (n, 2) normal, reciprocal measurement to text file"""
    np.savetxt(file, normal_reciprocal)


def display(nor_rec):
    # Plot
    plt.plot(nor_rec[:, 0], nor_rec[:, 1], 'ko')
    plt.show()


def hist(nor_rec, bins, quantile=None):
    """Plot histogram
    :param nor_rec: np.array: Array (n, 2) containing n normal and reciprocal measurements
    :param quantile: float: Quantile threshold
    :param bins: int: Number of bins
    """
    if quantile is None:
        quantile = 1
    # Create DF and compute relative (%) reciprocal error
    diff = pd.DataFrame(data=np.abs(np.subtract(nor_rec[:, 0], nor_rec[:, 1]) / nor_rec[:, 0]), columns=['diff'])
    # Display some statistics
    print(diff.describe())
    # Extracts value corresponding to desired quantile
    vt = diff.quantile(quantile).values[0]
    # Cut
    diffT = diff[diff['diff'] <= vt]
    # Plot
    diffT.hist(bins=bins)
    plt.xlabel('Reciprocal error (%)', weight='bold', size=12)
    plt.ylabel('Count', weight='bold', size=12)
    plt.title('Histogram of reciprocal error', weight='bold', size=12)
    plt.show()


class Reciprocal:

    def __init__(self, normal_file, reciprocal_file, stack_tres):
        """

        :param normal_file: str: path to the normal measurements file
        :param reciprocal_file: str: path to the reciprocal measurements file
        :param stack_tres: float: Measurements repeatability (var %) threshold
        """

        self.fN = normal_file
        self.fR = reciprocal_file
        self.ts = stack_tres

    def parse(self):
        """
        Reads the results text files and parses them.
        It will cut data above repeatability threshold.
        :return: resNR, varNR - two np arrays of pairs of resistance and repeatability error
        """
        # Read normal and reciprocal data
        pN = read_res(self.fN)
        pR = read_res(self.fR)

        # Filter stack error
        pN = pN[pN['var'] < self.ts]
        pR = pR[pR['var'] < self.ts]

        # Extract normal and reciprocal subsets
        abmnN = pN[['ax', 'bx', 'mx', 'nx', 'rohm', 'var']]
        abmnR = pR[['ax', 'bx', 'mx', 'nx', 'rohm', 'var']]

        # Concatenate them
        conc = pd.concat([abmnN, abmnR])

        # To use a dict as a key you need to turn it into something that may be hashed first. If the dict you wish to
        # use as key consists of only immutable values, you can create a hashable representation of it with frozenset
        conc['id'] = conc.apply(lambda row: frozenset(Counter(row[['ax', 'bx', 'mx', 'nx']]).keys()), axis=1)

        # Group by same identifiers = same electrode pairs
        df1 = conc.groupby('id')['rohm'].apply(np.array).reset_index(name='rhos')

        # Extract list containing res values [N, R]
        rhos = [d for d in df1.rhos.values if len(d) == 2]
        # Flatten and reshape
        resNR = np.array([item for sublist in rhos for item in sublist]).reshape((-1, 2))

        # Extract repeatability error as well:
        df2 = conc.groupby('id')['var'].apply(np.array).reset_index(name='vars')

        # Extract list containing var values [N, R]
        var = [d for d in df2.vars.values if len(d) == 2]
        # Flatten and reshape
        varNR = np.array([item for sublist in var for item in sublist]).reshape((-1, 2))

        return resNR, varNR


if __name__ == '__main__':
    # Directories
    cwd = os.path.dirname(os.getcwd())
    data_dir = os.path.join(cwd, 'misc')
    # Files
    fN = os.path.join(data_dir, 'Project27_Gradient8_1.txt')
    fR = os.path.join(data_dir, 'Project27_Grad_8_R_1.txt')
    # Initiate and parse
    ro = Reciprocal(fN, fR, stack_tres=.5)
    res_nr, var_nr = ro.parse()
    # Plot histogram
    hist(res_nr, quantile=.99, bins=20)
    # Linear plot
    display(res_nr)
