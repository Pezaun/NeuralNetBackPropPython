__author__ = 'gabriel'

import numpy as np

class Instance(object):
    """
    Represents a instance of training.
    """
    def __init__(self):
        self.attributes = np.array([])
        self.output_values = np.array([])

    def __str__(self):
        result = ""
        for i in self.attributes:
            result += i
            result += "\n"
        for i in self.output_values:
            result += i
            result += "\n"
        return result
