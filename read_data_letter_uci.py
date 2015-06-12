from instance import Instance
from neural import NeuralNet

__author__ = 'gabriel'
import numpy as np

f = "/Users/gabriel/Desktop/letter-recognition.data_backup.csv"

instances = []
classes = set()

input_values = np.loadtxt(open(f), dtype=str,delimiter=",",skiprows=1)
print "Reading Dataset..."
for element in input_values:
    classes.add(element[0])


for element in input_values:
    letter = element[0]

    out = np.zeros((1, 26))
    out[0, ord(letter) - 65] = 1

    element = np.delete(element, 0)
    element = element.astype(np.float)
    element = element.reshape((1, element.shape[0]))

    inst = Instance()
    inst.attributes = element
    inst.output_values = out
    inst.normalize()

    instances.append(inst)
print "...Ready!"
print "Instantiate neuralnet..."
ann = NeuralNet([16, 32, 26], True)
ann.learning_rate = 0.09
ann.momentum = 0.44
ann.instances(instances)
print "...Ready!"
print "Generate model..."
ann.train(50, True)
print "...Ready!"

print ann

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
