__author__ = 'gabriel'

import numpy as np


class NeuralNet(object):
    """
    The implementation of MLP with back propagation learning algorithm.
    """
    def __init__(self, input_values, weights, layers):
        self.x = input_values
        self.theta = weights
        self.z = 0.0
        self.sigma = 0.0
        self.layers = range(layers)
        self.vsig = np.vectorize(NeuralNet.sigmoid)

    def feed_forward(self):
        for i in range(len(self.layers)):
            self.layers[i] = self.vsig((self.x * self.theta.T).sum(axis=0))
            self.layers[i][0] = 1
            self.x = self.layers[i]

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
    theta = np.array([[-0.5, 0.4, -0.6, 0.6], [0.5, 0.2, -0.3, 0.3], [0.5, 0.2, -0.3, 0.3], [0.5, 0.2, -0.3, 0.3]])
    ann = NeuralNet(x, theta, 2)
    ann.feed_forward()
    print ann

if __name__ == "__main__":
    main()
