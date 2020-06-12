#  Copyright (c) 2020. Robin Thibaut, Ghent University

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
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
pR = read_res(fR)

# Filter stack error
ts = .5
pN = pN[pN['var'] < ts]
pR = pR[pR['var'] < ts]

# Extract normal and reciprocal subsets
abmnN = pN[['ax', 'bx', 'mx', 'nx', 'rohm']]
abmnR = pR[['ax', 'bx', 'mx', 'nx', 'rohm']]

# Concatenate them
nr = pd.concat([abmnN, abmnR])

# To use a dict as a key you need to turn it into something that may be hashed first. If the dict you wish to use as
# key consists of only immutable values, you can create a hashable representation of it with frozenset
nr['id'] = nr.apply(lambda row: frozenset(Counter(row[['ax', 'bx', 'mx', 'nx']]).keys()), axis=1)

# Group by same identifiers = same electrode pairs
df1 = nr.groupby('id')['rohm'].apply(np.array).reset_index(name='rhos')

# Extract list containing res values [N, R]
rhos = [d for d in df1.rhos.values if len(d) == 2]
# Flatten and reshape
flat_list = np.array([item for sublist in rhos for item in sublist]).reshape((-1, 2))
# Plot
plt.plot(flat_list[:, 0], flat_list[:, 1], 'ko')
plt.show()
