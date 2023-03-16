import numpy as np

class node:
    def __init__(self):
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
    nd = node()
    for i in net_structure:
        tmp = []
        for j in range(i):
            tmp.append(nd)
        net.append(tmp)
        output_layer = tmp
    print(net)
    print()
    print(output_layer)
    print()

    input_data = []
    print("Inputs:")
    with open('input.txt', 'r') as num:
        first_layer = np.array(num.read().split(','))
        #print(first_layer, '\n')
        for i in first_layer:
            input_data.append(float(i))
            print(input_data)
    #print(input_data)
    #print(type(input))
    #print(len(input))
    #connections:

    for i in range(len(input_data)):
        net[0][i].collector = input_data[i]
        #print(net[0][i].collector)

    print()
    print("Sum: ") #doesn't actually provide the correct sum correct sum should be: 46

    print(len(input_data))
    print(sum(input_data))
    for i in range(1,len(input_data) - 1, 1):
        #print(i)
        for n in net[i]:
            #print(n)
            #print(net[i-1])
            for c in net[i-1]:
                #print(c)
                n.collector = n.collector + c.collector

    print(net[2][0].collector)
main()