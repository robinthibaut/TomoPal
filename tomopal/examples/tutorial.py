import numpy as np


def data_read(
    file: str = None,
    start: int = 0,
    end: int = None,
    step: int = 1,
    delimiter: str = None,
):
    """
    Reads data from a file. It needs to be a text file and the data needs to be separated by the same character,
    specified by the delimiter.
    :param file: str: File path, such as 'data.txt'.
    :param start: int: Starting line, default is 0.
    :param end: int: Ending line, default is None (last line).
    :param step: int: Step, default is 1 (every line).
    :param delimiter: str: Delimiter, default is None (space).
    :return: Data contained in file. np.array if data can be converted to float, else list.
    """
    with open(file, "r") as fr:  # Open file
        lines = fr.readlines()[start:end:step]  # Read lines
        try:  # Try to convert to float
            op = np.array(
                [list(map(float, line.split(delimiter))) for line in lines],
                dtype=object,
            )
        except ValueError:  # If not, keep as string.
            op = [line.split(delimiter) for line in lines]
    return op  # Return data


raw_data = data_read("/Users/robin/PycharmProjects/TomoPal/tomopal/crtomopy/data/demo/demo_data.dat", start=1)