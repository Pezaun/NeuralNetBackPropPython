__author__ = 'gabriel'

import numpy as np
import time
from instance import Instance
from random import shuffle

class NeuralNet(object):
    """
    The implementation of MLP with back propagation learning algorithm.
    """
    def __init__(self, architecture, bias=False, weights=None):
        self.learning_rate = 0.2
        self.momentum = 0.1
        self.first_back = True
        self.bias = bias
        self.instance_data = Instance()
        self.x = range(len(architecture))
        self.architecture = architecture
        self.layers = range(len(architecture))
        self.activation = range(len(architecture))
        self.error = range(len(architecture))
        self.vsig = np.vectorize(NeuralNet.sigmoid)
        self.instances_list = list()
        np.random.seed(1)
        self.weights_old = NeuralNet.start_weights(architecture)
        if weights is None:
            self.weights = NeuralNet.start_weights(architecture)
        else:
            self.weights = weights
        self.bias_weights = self.start_bias_weights(architecture)

    def train(self, epochs):
        while epochs > 0:
            epochs -= 1
            for i in self.instances_list:
                self.instance(i)
                self.feed_forward()
                self.back_propagate()
            shuffle(self.instances_list)

    def instances(self, inst):
        self.instances_list = inst

    def instance(self, inst):
        self.instance_data = inst
        self.layers[0] = inst.attributes
        self.x[0] = inst.attributes
        self.activation[0] = inst.attributes

    @staticmethod
    def start_weights(arch):
        w = list()
        for i in range(len(arch) - 1):
            w.append(np.random.uniform(-0.1, 0.1, (arch[i], arch[i + 1])))
        return w

    def start_bias_weights(self, arch):
        b = list()
        if self.bias is True:
            for i in range(len(arch) - 1):
                b.append(np.random.uniform(-0.1, 0.1, (1, arch[i + 1])))
        else:
            for i in range(len(arch) - 1):
                b.append(np.zeros((1, arch[i + 1])))
        return b

    def feed_forward(self):
        for i in range(len(self.architecture) - 1):
            tmp = self.layers[i].dot(self.weights[i]) + self.bias_weights[i]
            self.activation[i + 1] = self.vsig(tmp)
            self.layers[i + 1] = self.activation[i + 1]

    def back_propagate(self):
        first = True
        momentum_weights = np.zeros(self.weights[len(self.layers) - 2].shape)
        for i in range(len(self.layers) - 1, 0, -1):
            out = self.activation[i]
            if self.first_back is False:
                momentum_weights = self.momentum * (self.weights[i - 1] - self.weights_old[i - 1])
                self.weights_old[i - 1] = self.weights[i - 1]

            if first:
                term1 = np.ones(out.shape) - out
                term2 = (self.instance_data.output_values - out)
                out_error = (term1 * term2) * out
                self.error[i] = out_error
                out = self.activation[i - 1]
                term3 = self.weights[i - 1] + (self.learning_rate * out_error.T * out).T
                first = False
            else:
                term1 = np.ones(self.activation[i].shape) - out
                term2 = self.error[i + 1].dot(self.weights[i].T)
                out_error = term1 * term2 * out
                self.error[i] = out_error
                term3 = self.weights[i - 1] + (self.learning_rate * out_error * self.activation[i - 1].T)

            self.weights[i - 1] = term3 + momentum_weights
            if self.bias is True:
                    self.bias_weights[i - 1] = self.bias_weights[i - 1] + self.learning_rate * out_error
            self.first_back = False

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

def main():
    inst1 = Instance()
    inst1.attributes = np.array([[1,1]])
    inst1.output_values = np.array([[0]])

    inst2 = Instance()
    inst2.attributes = np.array([[0,1]])
    inst2.output_values = np.array([[1]])

    inst3 = Instance()
    inst3.attributes = np.array([[1,0]])
    inst3.output_values = np.array([[1]])

    inst4 = Instance()
    inst4.attributes = np.array([[0,0]])
    inst4.output_values = np.array([[0]])

    ann = NeuralNet([2, 2, 1], True)
    ann.learning_rate = 0.5
    ann.momentum = 0.2

    instances = list()
    instances.append(inst1)
    instances.append(inst2)
    instances.append(inst3)
    instances.append(inst4)

    ann.instances(instances)
    t = time.time() * 1000
    ann.train(1700)
    t = (time.time() * 1000) - t

    ann.instance(inst1)
    ann.feed_forward()
    print ann

    ann.instance(inst2)
    ann.feed_forward()
    print ann

    ann.instance(inst3)
    ann.feed_forward()
    print ann

    ann.instance(inst4)
    ann.feed_forward()
    print ann

    print "Time: " + str(t)

    print ann.weights
    print ann.weights_old

if __name__ == "__main__":
    main()
