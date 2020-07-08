import os
from dataclasses import dataclass


@dataclass
class Directories:
    """Define main directories"""
    main_dir = os.path.dirname(os.path.abspath(__file__))
