# This file contains mainly enums and other
# pre-defined variables to assist the main program

from Enum import IntEnum        # All enums defined in this project will be inenum since it is has the greatest compatibility


class Filter_type(IntEnum):
    """docstring forFilter_type."""
    Grey = 0
    Gaussian = 1
    HSV = 2
    Bilateral = 3
    Threshold = 4
