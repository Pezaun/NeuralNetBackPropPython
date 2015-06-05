__author__ = 'gabriel'

import numpy as np
import time
from instance import Instance

class NeuralNet(object):
    """
    The implementation of MLP with back propagation learning algorithm.
    """
    def __init__(self, architecture, weights = None):
        self.learning_rate = 0.4
        self.instance_data = Instance()
        self.x = range(len(architecture))
        self.architecture = architecture
        self.layers = range(len(architecture))
        self.w_sum = range(len(architecture))
        self.activation = range(len(architecture))
        self.error = range(len(architecture))
        self.vsig = np.vectorize(NeuralNet.sigmoid)
        np.random.seed(1)
        if weights is None:
            self.weights = NeuralNet.start_weights(architecture)
        else:
            self.weights = weights

    def instances(self, inst):
        self.instance_data = inst
        self.layers[0] = inst.attributes
        self.x[0] = inst.attributes
        self.activation[0] = inst.attributes

    @staticmethod
    def start_weights(arch):
        w = list()
        for i in range(len(arch) - 1):
            w.append(np.random.uniform(-0.01, 0.01, [arch[i], arch[i + 1]]))
        return w

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            tmp = self.layers[i].dot(self.weights[i])
            self.w_sum[i + 1] = tmp
            self.activation[i + 1] = self.vsig(tmp)
            self.layers[i + 1] = self.activation[i + 1]

    def back_propagate(self):
        first = True
        for i in range(len(self.layers) - 1, 0, -1):
            out = self.activation[i]
            if first:
                # Output layer error calculation
                term1 = np.ones(self.activation[i].shape) - out
                term2 = (self.instance_data.output_values - out)
                out_error = (term1 * term2) * out
                self.error[i] = out_error
                # Output layer weights update
                out = self.activation[i - 1]
                self.weights[i - 1] = (self.weights[i - 1].T + (self.learning_rate * out_error.T * out)).T
                first = False
            else:
                term1 = np.ones(self.activation[i].shape) - out
                term2 = self.error[i + 1].dot(self.weights[i].T)
                out_error = term1 * term2 * out
                self.error[i] = out_error
                term1 = self.weights[i - 1] + (self.learning_rate * out_error * self.activation[i - 1].T)
                self.weights[i - 1] = term1

    @staticmethod
    def signal(v):
        if v >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.e ** - v)

    @staticmethod
    def sigmoid_derivative(v):
        return NeuralNet.sigmoid(v) * (1 - NeuralNet.sigmoid(v))

    def __str__(self):
        result = ""
        ct_layer = 1
        for l in self.layers:
            result += "Layer " + str(ct_layer) + ":\n"
            result += str(l) + "\n\n"
            ct_layer += 1
        return result

    def print_s(self):
        for s in self.w_sum:
            print s

    def print_z(self):
        for z in self.activation:
            print z

def main():
    x = np.array([[1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1]])
    y = np.array([[1,0,0,0]])
    inst = Instance()
    inst.attributes = x
    inst.output_values = y
    ann = NeuralNet([48,96,4])
    ann.instances(inst)

    t = time.time() * 1000
    for i in range(10):
        ann.feed_forward()
        ann.back_propagate()
    t = (time.time() * 1000) - t
    print ann
    print "Time: " + str(t)

if __name__ == "__main__":
    main()
