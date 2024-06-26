
import numpy as np
import random
import pandas as pd
import pickle
import sqlite3
import math

np.random.seed(42)

class Node:

    def __init__(self, lastlayer = None):
        self.lastlayer = lastlayer
        self.collector = 0.0
        self.connections = []
        self.weights = []
        self.delta = 0.0
        if lastlayer is not None:
            self.connections = lastlayer
            self.weights = [{'weights': np.random.rand()} for _ in range
                (len(lastlayer))]  # Used np.random.normal because I was having issues with exploding gradients


net_structure = np.array([784, 400, 200, 60, 1])


# input_data = []
# output_layer = None
# net = []

def get_data(letter):
    conn = sqlite3.connect('hw.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hw_data WHERE letter = {} ORDER BY RANDOM() limit 1000".format(letter))
    data = cursor.fetchall()
    #print(data)
    conn.close()
    return data

#get_data("0")

def create_train_test_datasets(data, train_percentage):
    np.random.shuffle(data)
    train_data = data[:int(len(data) * train_percentage)]
    test_data = data[int(len(data) * train_percentage):]

    return train_data, test_data

def sigmoidal(activation):
    return 1.0 / (1.0 + np.exp(-activation))

def sigmoidal_deriv(output):
    return sigmoidal(output) * (1.0 - sigmoidal(output))

def initialize_network(network_structure):
    global output_layer
    net = []
    output_layer = None
    for i in network_structure:
        layer = []
        for _ in range(i):
            layer.append(Node(lastlayer=output_layer))
            if output_layer is not None:
                layer[-1].connections = output_layer
                layer[-1].weights = [{'weights': np.random.rand()} for _ in range(len(output_layer))]
        if output_layer is not None:
            print(layer[-1].weights)
            output_weights = [{'output layer weights': np.random.rand()} for _ in range(network_structure[-1])]
            print("Output Layer Weights: ", output_weights)
            output_layer = output_weights
        net.append(layer)
        output_layer = layer
    # print(net)
    return net


def activation(neuron):
    sum_of_weights = 0.0
    for index in range(len(neuron.connections)):
        con = neuron.connections[index]
        weight = neuron.weights[index]['weights'] # access value from dictionary
        # neuron.collector = sigmoidal(sum_of_weights)
        sum_of_weights += con.collector * weight
        # print(sum_of_weights)
    # print(sigmoidal(sum_of_weights))
    return sigmoidal(sum_of_weights)

def forward_propagation(network):
    for layer in network:
        for node in layer:
            node.collector = activation(node)
    return network[-1][0].collector


def back_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network) - 1:
            # for output layer
            for j in range(len(layer)):
                node = layer[j]
                output = node.collector
                error = expected - output
                errors.append(error)
        else:
            # for hidden layers
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


def train_network(network, train, test_letter, lr, n_epochs, target_error):
    for epoch in range(n_epochs):
        error_sum = 0.0
        for data in train:
            letter = data[0]
            input_values = data[1:]
            expected_output = 1 if letter == test_letter else 0

            # forward prop:
            for i in range(len(input_values)):
                network[0][i].collector = input_values[i]
            output = forward_propagation(network)
            # print(output)
            # calculate error:
            # print(expected_output)
            error = (expected_output - output) ** 2
            error_sum += error

            # Back prop:
            back_propagation(network, expected_output)

            # update weights:
            update_weights(network, lr)
        # print(error_sum)
        # calculate average error per epoch:
        avg_error = error_sum / len(train)
        print("Epoch: {}, l_rate: {}, Error: {}".format(epoch, lr, avg_error))
        if avg_error <= target_error:
            print(f"Target error ({target_error}) reached after {epoch + 1} epochs")
            break

        # reset error sum
        error_sum = 0.0
        if (epoch + 1) == n_epochs:
            print("Maximum number of epochs reached. Training stopped.")
def predict(network, input_data, expected_output):
    for i in range(len(input_data)):
        network[0][i].collector = input_data[i]
    output = forward_propagation(network)
    predicted_output = round(output)
    print("Expected = {}, Predicted = {}".format(expected_output, predicted_output))
    return predicted_output


def model_A():
    print("Traning on the letter A....\n")
    data = get_data("0")
    train_data, test_data = create_train_test_datasets(data, 0.2)
    print(len(train_data))
    print(len(test_data))
    net = initialize_network(net_structure)
    print(forward_propagation(net))
    train_network(net, train_data, test_letter=0, lr=0.001, n_epochs=2500, target_error=0.01)
    print(forward_propagation(net))


def model_B():
    print("\nTraning on the letter B....\n")
    data = get_data("1")
    train_data, test_data = create_train_test_datasets(data, 0.2)
    print(len(train_data))
    print(len(test_data))
    net = initialize_network(net_structure)
    print(forward_propagation(net))
    train_network(net, train_data, test_letter=1, lr=0.001, n_epochs=2500, target_error=0.01)
    print(forward_propagation(net))


def model_C():
    print("\nTraning on the letter C....\n")
    data = get_data("2")
    train_data, test_data = create_train_test_datasets(data, 0.2)
    print(len(train_data))
    print(len(test_data))
    net = initialize_network(net_structure)
    print(forward_propagation(net))
    train_network(net, train_data, test_letter=2, lr=0.001, n_epochs=2500, target_error=0.01)
    print(forward_propagation(net))


def model_D():
    print("\nTraning on the letter D....\n")
    data = get_data("3")
    train_data, test_data = create_train_test_datasets(data, 0.2)
    print(len(train_data))
    print(len(test_data))
    net = initialize_network(net_structure)
    print(forward_propagation(net))
    train_network(net, train_data, test_letter=3, lr=0.001, n_epochs=2500, target_error=0.01)
    print(forward_propagation(net))


def model_Z():
    print("\nTraning on the letter Z....\n")
    data = get_data("25")
    train_data, test_data = create_train_test_datasets(data, 0.2)
    print(len(train_data))
    print(len(test_data))
    net = initialize_network(net_structure)
    print(forward_propagation(net))
    train_network(net, train_data, test_letter=25, lr=0.001, n_epochs=2500, target_error=0.01)
    print(forward_propagation(net))


#Note: For better training, a learning rate of 0.0001 seems to be best, it's just very long
#      To save time I used .001 as it still shows that the error is decreasing

def main():
    net = initialize_network(net_structure)
    model_A()
    model_B()
    model_C()
    model_D()
    model_Z()

    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)

main()