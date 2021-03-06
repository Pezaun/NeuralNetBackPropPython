from instance import Instance
from neural import NeuralNet
import sys

__author__ = 'gabriel'

import numpy as np

def main():
    f = "/Users/gabriel/Desktop/instances.txt"
    base_code = ord("A")
    with open(f) as f:
        lines = f.readlines()
    s = int(lines[0]) * int(lines[1])
    n = int(lines[2])
    l = 4

    instances = list()

    while n > 0:
        n -= 1
        pattern = np.zeros((1, s))
        pattern_class = np.zeros((1, int(lines[2])))
        pt_index = 0
        inst = Instance()
        for i in range(int(lines[1])):
            for c in lines[l].strip():
                pattern[0][pt_index] = 1 if c == "#" else 0
                pt_index += 1
            l += 1
        pattern_class[0][ord(lines[l].strip()) - base_code] = 1
        l += 1
        inst.attributes = pattern
        inst.output_values = pattern_class
        inst.normalize()
        instances.append(inst)

    ann = NeuralNet([35, 35, 26], True)
    ann.learning_rate = 0.5
    ann.momentum = 0.4
    ann.instances(instances)
    ann.train(55)

    correct = 0
    for i in instances:
        ann.instance(i)
        ann.feed_forward()
        f = chr(65 + np.argmax(i.output_values))
        print "F={}".format(f)
        f_hat = chr(65 + np.argmax(ann.output))
        print "F'={}".format(f_hat)

        if f == f_hat:
            correct += 1
        else:
            print "ERROR!"
        print
    print str(correct / float(len(instances)) * 100) + " %"



if __name__ == "__main__":
    main()