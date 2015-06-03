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
        self.dvsig = np.vectorize(NeuralNet.sigmoid_derivative)
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
            w.append(np.random.uniform(-0.1, 0.1, [arch[i + 1], arch[i]]))
        return w

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            #tmp = (self.layers[i] * self.theta[i].T).sum(axis=0)
            tmp = self.layers[i].T.dot(self.weights[i].T).T
            self.w_sum[i + 1] = tmp
            self.activation[i + 1] = self.vsig(tmp)
            #self.error[i + 1] = self.dvsig(tmp)
            # tmp = tmp.reshape((tmp.shape[0], 1))
            self.layers[i + 1] = self.activation[i + 1]

    def back_propagate(self):
        first = True
        for i in range(len(self.layers) - 1, -1, -1):
            out = self.activation[i]
            if first:
                # Output layer error calculation
                term1 = np.ones(self.activation[i].shape) - out
                term2 = (self.instance_data.output_values.T - out)
                out_error = (term1 * term2) * out
                # Output layer weights update
                out = self.activation[i - 1]
                self.weights[i - 1] = ((out_error.T * out * self.learning_rate) + self.weights[i - 1].T).T
                first = False
            else:
                pass



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
    x = np.array([[0],[1],[1],[1],[0]])
    y = np.array([[0, 1, 0]])
    inst = Instance()
    inst.attributes = x
    inst.output_values = y
    ann = NeuralNet([5,10,3])
    ann.instances(inst)

    for i in range(10):
        ann.feed_forward()
        ann.back_propagate()

    print ann

if __name__ == "__main__":
    main()
