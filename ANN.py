import numpy as np

class node:
    def __int__(self):
        self.collector = 0.0
        self.connections = []

    def __str__(self):
        str_out = self.collector

net = []
network = "4,2,1"
output_layer = None

network = network.split(',')
network_struct = np.array([network]).T
print(network_struct)

with open('input.txt', 'r') as num:
    first_layer = np.array([num.read().split(',')]).astype(float)
    print(first_layer)

for i in network_struct:
    net.append([int(i)])
    print(net)

for i in range(first_layer.shape[0]):
    net.append(network_struct[0][i])
    print(net)

