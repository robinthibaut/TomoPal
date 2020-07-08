import inspect
import os


def hello():
    previous_frame = inspect.currentframe().f_back
    (filename, line_number,
     function_name, lines, index) = inspect.getframeinfo(previous_frame)
    main_dir = os.path.dirname(os.path.dirname(filename))
    return main_dir
