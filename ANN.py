import numpy as np
import random
import pickle
import math

class node:

    def __init__(self, lastlayer = None):
        self.lastlayer = lastlayer
        self.collector = 0.0
        self.connections = []
        #self.weights = [0] * len(self.connections)
        #print(self.weights)
        if self.connections != None:
            self.weights = [random.random() for i in range(len(self.connections))]

        # for i in range(len(self.connections)):
        #     self.weights.append(random.random())
        # #print(random.random())

net_structure = np.array([4, 2, 1], dtype=np.int64)
output_layer = None
net = []
l_rate = 0.1
def sigmoidal(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def sigmoidal_deriv(output): #Part of Back Propagation algorithm
    return output * (1.0 - output)

def feed_forward(input_data):
    #print("Input Layer:")
    for i in range(len(input_data)):
        if i < len(net[0]):
            net[0][i].collector = input_data[i]
            #print(net[0][i].collector)

    for layer in net:
        for n in layer:
            if n.lastlayer is not None:
                n.collector = 0.0
                for index in range(len(n.connections)):
                    c = n.connections[index]
                    w = n.weights[index]
                    n.collector += (c.collector * w)
                n.collector = sigmoidal(n.collector)

def compute_error(input_data):
    expected = input_data[-1]
    output = [n.collector for n in net[-1]]
    error = (expected - output[0]) ** 2
    return error

def update_weights(n, c, error_gradient, lr):
    index = n.connections.index(c)
    n.weights[index] += c.collector * error_gradient * lr
    c.collector = n.weights[index] * error_gradient * sigmoidal(n.collector)

def back_propagation(input_data):
    global output_layer
    #compute_error(input_data)
    for layer in reversed(net):
        for n in layer:
            if n.lastlayer is not None:
                error_gradient = sigmoidal_deriv(n.collector) * compute_error(input_data)
                for index in range(len(n.connections)):
                    c = n.connections[index]
                    c.error = n.weights[index] * error_gradient  # Update error of connected nodes
                    update_weights(n, c, error_gradient, lr=l_rate)
def activation(network):
    global output_layer
    for i in net_structure:
        first_layer = []
        for j in range(i):
            first_layer.append(node(lastlayer=output_layer))
            if output_layer is not None:
                first_layer[j].weights = [random.random() for i in range(len(output_layer))]
            #print("first layer: ", first_layer)
        net.append(first_layer)
        output_layer = first_layer

def train(input_data, lr, epoch, target_error):
    for e in range(epoch + 1):
        total_error = 0.0
        for index in input_data:
            feed_forward(index[:-1])
            error = compute_error(index)
            total_error += error ** 2
            back_propagation(index)
        total_error /= len(input_data)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, lr, total_error))
        if(total_error <= target_error):
            print("Target Error reached error=%.3f" % total_error)
            break


def main():
    global output_layer
    print("Network Structure: ", net_structure)
    activation(net)
    print("network: ", net)
    print("output layer: ",output_layer) #Output layer aka last layer
    print()
    print("Inputs: ")
    input_data = np.loadtxt('input.csv', delimiter=',', dtype=int)
    #print(len(input_data[0])-1)
    print(input_data, "\n")

    print("Train:")
    train(input_data, lr=l_rate, epoch=10, target_error=0.05)

    print("\nOutput Layer: ")

    print(net[-1][0].collector)
    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)

main()