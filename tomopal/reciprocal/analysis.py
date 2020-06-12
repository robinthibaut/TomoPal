#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import re
from collections import Counter

import pandas as pd

cwd = os.path.dirname(os.getcwd())

data_dir = os.path.join(cwd, 'misc')

fN = os.path.join(data_dir, 'Project27_Gradient8_1.txt')
fR = os.path.join(data_dir, 'Project27_Grad_8_R_1.txt')


def read_res(file):
    data = pd.read_csv(file, delimiter='\t')
    data.columns = [re.sub('[^A-Za-z0-9]+', '', col.lower()) for col in data.columns]

    return data


pN = read_res(fN)
pN.rohm *= -1
abmnN = pN[['ax', 'bx', 'mx', 'nx', 'rohm']]

pR = read_res(fR)
abmnR = pR[['ax', 'bx', 'mx', 'nx', 'rohm']]

nr = pd.concat([abmnN, abmnR])

# To use a dict as a key you need to turn it into something that may be hashed first. If the dict you wish to use as
# key consists of only immutable values, you can create a hashable representation of it with frozenset
nr['id'] = nr.apply(lambda row: frozenset(Counter(row[['ax', 'bx', 'mx', 'nx']]).keys()), axis=1)

df1 = nr.groupby('id')['rohm'].apply(list).reset_index(name='rhos')


# https://stackoverflow.com/questions/29464234/compare-python-pandas-dataframes-for-matching-rows
