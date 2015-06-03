__author__ = 'gabriel'

import numpy as np
import time
import instance


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
        self.s = list()
        self.z = list()
        self.layers.append(input_values)
        self.vsig = np.vectorize(NeuralNet.sigmoid)
        if weights is None:
            self.theta = self.start_weights(architecture)
        else:
            self.theta = weights

    def instances(self, inst):
        self.instances_data = inst

    @staticmethod
    def start_weights(arch):
        w = list()
        for i in range(len(arch) - 1):
            w.append(np.random.uniform(-0.1, 0.1, [arch[i + 1], arch[i]]))
        return w

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            tmp = (self.layers[i] * self.theta[i].T).sum(axis=0)
            self.s.append(tmp)
            tmp = self.vsig(tmp)
            self.z.append(tmp)
            tmp = tmp.reshape((tmp.shape[0], 1))
            self.layers.append(tmp)

    def back_propagate(self):
        for i in range(len(self.layers) - 1, -1, -1):
            print "Camada..."
            print self.layers[i]
            if i > 0:
                print "Pesos..."
                print self.theta[i - 1]

    @staticmethod
    def signal(v):
        if v >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.e ** - v)

    def sigmoid_derivative(self, v):
        return (1 - v)

    def __str__(self):
        result = ""
        ct_layer = 1
        for l in self.layers:
            result += "Layer " + str(ct_layer) + ":\n"
            result += str(l) + "\n\n"
            ct_layer += 1
        return result

    def print_s(self):
        for s in self.s:
            print s

    def print_z(self):
        for z in self.z:
            print z

def main():
    x = np.array([[1], [1], [1], [0], [0], [0], [0], [1], [1], [0], [1], [1]])
    theta = np.array([[[-1.3, 0.4, -0.6, -0.2, -1.3, 0.4, -0.6, -0.2, 0.9, 0.2, 0.3, 0.1]]])
    # ann = NeuralNet(x, [12, 1], theta)
    ann = NeuralNet(x, [12, 6, 5, 5, 5, 4, 1])
    t = time.time() * 1000
    ann.feed_forward()
    # ann.back_propagate()

    ann.print_s()
    print
    ann.print_z()
    print
    print ann
    # i = instance.Instance()
    # t = (time.time() * 1000) - t
    # print t
if __name__ == "__main__":
    main()
