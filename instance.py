__author__ = 'gabriel'

import numpy as np

class Instance(object):
    """
    Represents a instance of training.
    """
    def __init__(self):
        self.attributes = np.array([])
        self.input = np.array([])
        self.output_values = np.array([])

    def normalize(self):
        normvet = np.vectorize(self.norm)
        self.attributes = normvet(self.attributes)

    def norm(self, v):
        self.input = self.attributes
        vmax = self.attributes.max()
        vmin = self.attributes.min()
        mmsum = (vmax + vmin) / 2.
        mmsub = (vmax - vmin) / 2.
        return (v - mmsum) / mmsub

    def __str__(self):
        return str(self.attributes) + "\n" + str(self.output_values)

def main():
    pass

if __name__ == "__main__":
    inst = Instance()
    inst.attributes = np.array([[3,0,10,0,1,1,0]])
    inst.normalize()
    print inst