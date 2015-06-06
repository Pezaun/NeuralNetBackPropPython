__author__ = 'gabriel'

import numpy as np
import time
from instance import Instance
from random import shuffle

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
        self.instances_list = list()
        np.random.seed(1)
        if weights is None:
            self.weights = NeuralNet.start_weights(architecture)
        else:
            self.weights = weights

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
            w.append(np.random.uniform(-0.1, 0.1, [arch[i], arch[i + 1]]))
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
    inst1 = Instance()
    inst1.attributes = np.array([[1,0]])
    inst1.output_values = np.array([[1]])

    inst2 = Instance()
    inst2.attributes = np.array([[0,1]])
    inst2.output_values = np.array([[1]])

    inst3 = Instance()
    inst3.attributes = np.array([[1,1]])
    inst3.output_values = np.array([[0]])

    inst4 = Instance()
    inst4.attributes = np.array([[0,0]])
    inst4.output_values = np.array([[0]])



    ann = NeuralNet([2,4,1])
    ann.instance(inst1)

    t = time.time() * 1000
    for i in range(1500):
        ann.instance(inst1)
        ann.feed_forward()
        ann.back_propagate()
        ann.instance(inst2)
        ann.feed_forward()
        ann.back_propagate()
        ann.instance(inst3)
        ann.feed_forward()
        ann.back_propagate()
        ann.instance(inst4)
        ann.feed_forward()
        ann.back_propagate()

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

if __name__ == "__main__":
    main()
