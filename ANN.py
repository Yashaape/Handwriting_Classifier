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
        self.delta = 0.0

net_structure = np.array([4,2,1])
input_data = np.loadtxt('input.csv', delimiter=',')
output_layer = None
net = []

def sigmoidal(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def sigmoidal_deriv(output):
    return output * (1.0 - sigmoidal(output))

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
    #print(net)
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
        #print(sum_of_weights)
    #print(sigmoidal(sum_of_weights))
    return sigmoidal(sum_of_weights)

def forward_propagation(network):
    for layer in network:
        for node in layer:
            node.collector = activation(node)
    return network[-1][0].collector

#Test case for forward propagation:
# input_values = np.array([0.5982372, 0.000348347, 0.223456, 0.938743784])
# print("Original Weights:")
# net = initialize_network(net_structure)
# print("\nTesting for forward propagation:")
# # Assign input values to the collector of input layer nodes
# for i in range(len(input_values)):
#     net[0][i].collector = input_values[i]
#
# output = forward_propagation(net)
# print("Input Values: ", input_values)
# print("Output: ", output)

def back_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network) - 1:
            #for output layer
            for j in range(len(layer)):
                node = layer[j]
                output = node.collector
                error = expected - output
                errors.append(error)
        else:
            #for hidden layers
            for j in range(len(layer)):
                node = layer[j]
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron.weights[j]['weights'] * neuron.delta
                errors.append(error)

        for j in range(len(layer)):
            node = layer[j]
            delta = errors[j] * sigmoidal_deriv(node.collector)
            node.delta = delta

def update_weights(network, lr):
    for i in range(len(network)):
        layer = network[i]
        for node in layer:
            for j in range(len(node.connections)):
                con = node.connections[j]
                weight = node.weights[j]
                weight['weights'] += lr * node.delta * con.collector

#Test Case for backpropagation:
# expected = 0.8
# l_rate = 0.1
# back_propagation(net, expected)
# update_weights(net, l_rate)
# print("\nUpdated Weights:")
# for layer in net:
#     for node in layer:
#         if hasattr(node, 'weights') and node.weights:
#             print(node.weights)


def train_network(network, train, lr, n_epochs, target_error):
    epoch = 0
    while epoch < n_epochs:
        error_sum = 0.0
        for data in train:
            input_values = data['input']
            expected_output = data['output']

            #Forward Prop:
            for i in range(len(input_values)):
                network[0][i].collector = input_values[i]
            output = forward_propagation(network)

            #calculate error:
            error = np.mean((expected_output - output)**2)
            error_sum += error

            #back prop:
            back_propagation(network, expected_output)

            #update weights:
        update_weights(network, lr)

        #calculate average error for this epoch
        avg_error = error_sum / len(train)
        print("Epoch: {}, l_rate: {}, Error: {}".format(epoch, lr, avg_error))

        # Check if target error is reached:
        if avg_error <= target_error:
            print("Target error reached: {}".format(target_error))
            break
        epoch +=1
    #epoch +=1
    if epoch == n_epochs:
        print("Maximum number of epochs reached. Training stopped.")

# Define the test case
#net_structure = np.array([4, 2, 1])
# train_data = [{'input': np.array([0.5982372, 0.000348347, 0.223456, 0.938743784]), 'output': 0.8}]
# # print("train: ", train_data)
# # # Initialize the network
# net = initialize_network(net_structure)
#
# # Train the network
# lr = 0.1
# n_epochs = 100
# target_error = 0.05
#
# print("Training the network...")
# train_network(net, train_data, lr, n_epochs, target_error)

# Test the trained network
# input_values = np.array([0.5982372, 0.000348347, 0.223456, 0.938743784])
# output = forward_propagation(net)
# print("Input Values: ", input_values)
# print("Output: ", output)

def main():
    global output_layer
    print("Network Structure:")
    print(net_structure)

    net = initialize_network(net_structure)
    #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors

    print("network: ", net)
    print("output layer: ",output_layer) #Output layer aka last layer
    print()

    print("Inputs:")
    print(input_data, "\n")
    train_data = []
    print("Training.......................................................")
    for row in input_data:
        X = row[:4]
        Y = row[4] # not sure if this is correct
        train_data.append({'input': np.array(X), 'output': 1})

    #print(forward_propagation(net))
    train_network(net, train_data, lr=0.1, n_epochs=200, target_error=0.05)

    print("Output Layer: ")
    print(net[-1][0].collector)
    #print(len(input_data))
    #print(sum(input_data))



    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)


main()