import numpy as np
import random
import pickle
import math
class Node:

    def __init__(self, lastlayer = None):
        self.lastlayer = lastlayer
        self.collector = 0.0
        self.connections = []
        self.weights = []

net_structure = np.array([4,2,1])
output_layer = None
net = []

def sigmoidal(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def sigmoidal_deriv(output):
    return output * (1.0 - output)

def initialize_network(network_structure):
    global output_layer
    for i in network_structure:
        layer = []
        for j in range(i):
            layer.append(Node(lastlayer=output_layer))
            if output_layer is not None:
                layer[-1].connections = output_layer
                layer[-1].weights = [{'weights': random.random()} for i in range(len(output_layer))]
        if output_layer is not None:
            print(layer[-1].weights)
            output_weights = [{'output layer weights': random.random()} for i in range(network_structure[2])]
            print("Output Layer Weights: ", output_weights)
            output_layer = output_weights
        net.append(layer)
        output_layer = layer
    return net

#Used to visualize adding in weights for each connection in each layer
# print("Weights and network Initialization:")
# random.seed(1)
# net = initialize_network(net_structure)
# for layer in net:
#     for node in layer:
#         if hasattr(node, 'weights') and node.weights:
#             print(node.weights)
# print()

def activation(neuron):
    sum_of_weights = 0.0
    for index in range(len(neuron.connections)):
        con = neuron.connections[index]
        weight = neuron.weights[index]['weights'] # access value from dictionary
        sum_of_weights += con.collector * weight
    return sigmoidal(sum_of_weights)

def forward_propagation(network):
    for layer in network:
        for node in layer:
            node.collector = activation(node)
    return network[-1][0].collector

#Test case for forward propagation:
input_values = np.array([0.5982372, 0.000348347, 0.223456, 0.938743784])
net = initialize_network(net_structure)

# Assign input values to the collector of input layer nodes
for i in range(len(input_values)):
    net[0][i].collector = input_values[i]

output = forward_propagation(net)
print("Input Values: ", input_values)
print("Output: ", output)

# def main():
#     global output_layer
#     print("Network Structure:")
#     print(net_structure)
#
#     net = initialize_network(net_structure)
#     #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors
#
#     print("network: ", net)
#     print("output layer: ",output_layer) #Output layer aka last layer
#     print()
#
#     print("Inputs:")
#     input_data = np.loadtxt('input.csv', delimiter=',', dtype=int)
#     print(input_data)
#
#     for i in range(len(input_data)):
#         if i < len(net[0]):
#             net[0][i].collector = input_data[i]
#             print(net[0][i].collector)
#
#     print()
#     print("Output Layer: ")
#
#     #print(len(input_data))
#     #print(sum(input_data))
#
#
#     for i in np.nditer(input_data):
#         #print(i)
#         for n in net[i]:
#             #print(n)
#             #print(net[i-1])
#             n.connections = net[i-1] # assign n.connections to net[i-1] to avoid summnation errors
#             #print(n.connections)
#
#             if n.connections:
#                 n.weights = [random.random()] * len(n.connections)
#             for index in range(len(n.connections)):
#                 #print(c)
#                 c = n.connections[index]
#                 w = n.weights[index]
#                 n.collector = n.collector + (c.collector * w)
#
#     print(net[2][0].collector)
#     # with open('doneANN.pickle', 'wb') as handle:
#     #     pickle.dump(net, handle)
#
#
# main()