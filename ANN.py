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



N_struct = []
with open('network.txt', 'r') as Net:
    #print(Net.read())


    # N_struct = Net.read().split(',')
    # print(N_struct)

    tmp = Net.read().split(',')
    for t in tmp:
        N_struct.append(int(t.strip()))

    Network = []
    for i in N_struct:
        for j in range(i):
            tmp.append([])
    print(tmp)

# with open('input.txt') as num:
#     Inputs = np.array(num.read().splitlines())
#     print(Inputs)
input_layer = []
with open('input.txt', 'r') as num:
    first_layer = num.read().split(',')
    print(first_layer)
    for i in range(len(first_layer)):
        N_struct[0][i]