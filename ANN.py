import numpy as np

class node:
    def __int__(self):
        self.collector = 0.0
        self.connections = []

net = []
network = "4,2,1"
output_layer = None
network = network.split(',')
network_struct = np.array([network]).T
#print(network_struct)
print("Network Structure:")
for i in network_struct:
    net.append(int(i))
    print(net)
print()

input= []
print("Inputs:")
with open('input.txt', 'r') as num:
    first_layer = np.array(num.read().split(',')).T
    #print(first_layer, '\n')
    for i in first_layer:
        input.append(float(i))
        print(input)


