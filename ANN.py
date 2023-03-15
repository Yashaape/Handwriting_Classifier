import numpy as np

class node:
    def __init__(self, lastlayer=None):
        self.lastlayer = lastlayer
        self.collector = 0.0
        self.connections = []

def main():
    net_structure = []
    network = "4,2,1"
    output_layer = None
    network = network.split(',')
    network = np.array([network]).T
    #print(network)
    print("Network Structure:")
    for i in network:
        net_structure.append(int(i))
        print(net_structure)

    net = []
    nd = node(output_layer)
    for i in net_structure:
        tmp = []
        for j in range(i):
            tmp.append(nd)
        net.append(tmp)
        output_layer = tmp
        print(net)
        print(output_layer)
    print()

    input= []
    print("Inputs:")
    with open('input.txt', 'r') as num:
        first_layer = np.array(num.read().split(',')).T
        #print(first_layer, '\n')
        for i in first_layer:
            input.append(float(i))
            print(input)
    #print(type(input))
    #print(len(input))
    #connections:

    for i in range(len(input)):
        net[0][i].collector = input[i]
        print(net)

    print()
    print("Sum: ") #doesn't actually provide the correct sum
    print(len(input))
    for i in range(1, len(input)-1, 1):
        for n in net[i]:
            for c in n.connections:
                n.collector = n.collector + c.collector
            print(n.collector)

    print(net[2][0].collector)

main()