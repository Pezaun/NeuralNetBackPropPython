from __builtin__ import range
from sqlalchemy.dialects.postgresql.base import array

__author__ = 'gabriel'

import numpy as np
import time


class NeuralNet(object):
    """
    The implementation of MLP with back propagation learning algorithm.
    """
    def __init__(self, input_values, architecture, weights = None):
        self.x = input_values
        self.z = 0.0
        self.sigma = 0.0
        self.architecture = architecture
        self.layers = list()
        self.layers.append(input_values)
        self.vsig = np.vectorize(NeuralNet.sigmoid)
        if weights is None:
            self.theta = self.start_weights(architecture)
        else:
            self.theta = weights

    @staticmethod
    def start_weights(arch):
        w = list()
        for i in range(len(arch) - 1):
            w.append(np.random.uniform(-0.1, 0.1, [arch[i + 1], arch[i]]))
        return w

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            tmp = (self.layers[i] * self.theta[i].T).sum(axis=0)
            tmp = self.vsig(tmp)
            tmp = tmp.reshape((tmp.shape[0], 1))
            self.layers.append(tmp)

    @staticmethod
    def signal(v):
        if v >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.e ** - v)

    def __str__(self):
        result = ""
        ct_layer = 1
        for l in self.layers:
            result += "Layer " + str(ct_layer) + ":\n"
            result += str(l) + "\n\n"
            ct_layer += 1
        return result

def main():
    x = np.array([[1], [1], [1], [0],[0],[0],[0],[1],[1],[0],[1]])
    theta = np.array([[[-1.3, 0.4, -0.6, -0.2]]])
    # ann = NeuralNet(x, [4, 1], theta)
    ann = NeuralNet(x, [11, 2, 5, 5, 5, 1])
    t = time.time() * 1000
    ann.feed_forward()
    print ann
    t = (time.time() * 1000) - t
    print t
if __name__ == "__main__":
    main()
