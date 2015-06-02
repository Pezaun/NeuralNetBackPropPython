from __builtin__ import range
from sqlalchemy.dialects.postgresql.base import array

__author__ = 'gabriel'

import numpy as np


class NeuralNet(object):
    """
    The implementation of MLP with back propagation learning algorithm.
    """
    def __init__(self, input_values, architecture, weights = None):
        self.x = input_values
        self.z = 0.0
        self.sigma = 0.0
        self.architecture = architecture
        self.layers = range(len(architecture))
        self.vsig = np.vectorize(NeuralNet.sigmoid)
        if weights is None:
            self.theta = self.start_weights(architecture)
        else:
            self.theta = weights

    @staticmethod
    def start_weights(arch):
        w = list()
        for i in range(len(arch) - 1):
            w.append(np.random.uniform(-0.1,0.1, [arch[i + 1], arch[i]]))
        return w

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            print "..........."
            print self.x
            print "-----------"
            print self.theta[i]
            print "@@@@@@@@@@@"
            self.layers[i] = np.array(self.x * self.theta[i].T).sum(axis=0)
            print self.layers[i]
            self.layers[i] = self.vsig(self.layers[i])
            self.x = self.layers[i]
            print "==========="
            print self.x

    @staticmethod
    def signal(v):
        if v >= 0:
            return 1
        else:
            return 0

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.e ** - v)

    def __str__(self):
        result = ""
        for l in self.layers:
            result += str(l) + "\n"
        return result

def main():
    x = np.array([[1], [0], [0], [1]])
    theta = np.array([[-0.5, 0.4, -0.6, 0.6],[-0.5, 0.4, -0.6, 0.6]])

    print theta.shape
    # print theta
    #ann = NeuralNet(x, [4, 2], theta)
    ann = NeuralNet(x, [4, 2, 1])
    ann.feed_forward()
    # print "-----------"
    # print ann

if __name__ == "__main__":
    main()
