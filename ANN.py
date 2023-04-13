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
            self.weights = [random.random()] * len(self.connections)

        for i in range(len(self.connections)):
            self.weights.append(random.random())
        #print(random.random())

net_structure = np.array([4, 2, 1])
output_layer = None
net = []
def sigmoidal(activation):
    return 1.0 / (1.0 + math.exp(-activation))
def sigmoidal_deriv(output):
    return output * (1.0 - output)

def feed_forward(input_data):
    for i in range(len(input_data)):
        if i < len(net[0]):
            net[0][i].collector = input_data[i]
            print(net[0][i].collector)

    for layer in net:
        for n in layer:
            if n.lastlayer is not None:
                n.collector = 0.0
                for index in range(len(n.connections)):
                    c = n.connections[index]
                    w = n.weights[index]
                    n.collector += (c.collector * w)
                n.collector = sigmoidal(n.collector)
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


def main():
    global output_layer
    print("Network Structure: ", net_structure)
    #net_structure = np.array([4,2,1])
    #output_layer = None
    #net = []
    #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors
    # for i in net_structure:
    #     first_layer = []
    #     for j in range(i):
    #         first_layer.append(node(lastlayer=output_layer))
    #         #print("first layer: ", first_layer)
    #     net.append(first_layer)
    #     output_layer = first_layer
    activation(net)
    print("network: ", net)
    print("output layer: ",output_layer) #Output layer aka last layer
    print()

    print("Inputs:")
    input_data = np.loadtxt('input.csv', delimiter=',', dtype=int)
    print(input_data)

    feed_forward(input_data)

    # for i in range(len(input_data)):
    #     if i < len(net[0]):
    #         net[0][i].collector = input_data[i]
    #         print(net[0][i].collector)



    print()
    print("Output Layer: ")

    #print(len(input_data))
    #print(sum(input_data))


    # for i in np.nditer(input_data):
    #     #print(i)
    #     for n in net[i]:
    #         #print(n)
    #         #print(net[i-1])
    #         n.connections = net[i-1] # assign n.connections to net[i-1] to avoid summnation errors
    #         #print(n.connections)
    #
    #         if n.connections:
    #             n.weights = [random.random()] * len(n.connections)
    #         for index in range(len(n.connections)):
    #             #print(c)
    #             c = n.connections[index]
    #             w = n.weights[index]
    #             n.collector = n.collector + (c.collector * w)

    print(net[2][0].collector)
    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)


main()