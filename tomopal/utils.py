import itertools
import os
import shutil

import numpy as np
import pandas as pd
from loguru import logger


def flatten_array(arr: np.array) -> np.array:
    arr_flat = np.array([item for sublist in arr for item in sublist])
    return arr_flat.reshape(1, -1)


def i_am_framed(array: np.array, ids: dict = None, flat: bool = True) -> pd.DataFrame:
    """Build a panda's dataframe that contains the flattened samples, their id's and the original shape"""
    # Remember original shape
    physical_shape = array.shape

    # Flatten for dimension reduction
    if flat:
        array = flatten_array(array)

    # Save DataFrames
    # Initiate array
    training_df_target = pd.DataFrame(data=array)
    # Set id's
    if ids is not None:
        for key in ids:
            training_df_target[key] = ids[key]
        training_df_target.set_index([k for k in ids], inplace=True)
    # Save original shape as a dataframe attribute
    training_df_target.attrs["physical_shape"] = physical_shape

    return training_df_target


def data_read(file: str = None, start: int = 0, end: int = None):
    # end must be set to None and NOT -1
    """Reads space-separated dat file"""
    with open(file, "r") as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array(
                [list(map(float, line.split())) for line in lines], dtype=object
            )
        except ValueError:
            op = [line.split() for line in lines]
    return op


def folder_reset(folder: str, exceptions: list = None):
    """Deletes files in folder"""
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]
    try:
        for filename in os.listdir(folder):
            if filename not in exceptions:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning("Failed to delete %s. Reason: %s" % (file_path, e))
    except FileNotFoundError:
        pass


def dirmaker(dird: str, erase: bool = False):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :param erase: bool: Whether to delete existing folder or not.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            if erase:
                shutil.rmtree(dird)
                os.makedirs(dird)
            return 1
    except Exception as e:
        logger.warning(e)
        return 0


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [
        list(itertools.combinations(combi, i)) for i in range(1, combi[-1] + 1)
    ]  # Get all possible wel combinations
    # Flatten and reverse to get all combination at index 0.
    cb = [item for sublist in cb for item in sublist][::-1]
    return cb
