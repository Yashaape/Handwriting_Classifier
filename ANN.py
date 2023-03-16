import numpy as np

class node:
    def __init__(self, lastlayer = None):
        self.lastlayer = lastlayer
        self.collector = 0.0
        self.connections = []

def main():
    print("Network Structure:")
    net_structure = np.array([4,2,1], dtype=np.int32)
    print(net_structure)
    output_layer = None
    net = []
    #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors
    for i in net_structure:
        first_layer = []
        for j in range(i):
            first_layer.append(node(lastlayer=output_layer))
        net.append(first_layer)
        output_layer = first_layer
    print("network: ", net)
    print("output layer: ",output_layer) #Output layer aka last layer
    print()


    # net = []
    # nd = node()
    # for i in net_structure:
    #     tmp = []
    #     for j in range(i):
    #         tmp.append(nd)
    #     net.append(tmp)
    #     output_layer = tmp
    # print(net)
    # print()
    # print(output_layer)
    # print()
    print("Inputs:")
    input_data = np.loadtxt('input.txt', delimiter=',', dtype=np.float64)
    print(input_data)


    # with open('input.txt', 'r') as num:
    #     first_layer = np.array(num.read().split(','))
    #     #print(first_layer, '\n')
    #     for i in first_layer:
    #         input_data.append(float(i))
    #         print(input_data)
    #print(input_data)
    #print(type(input))
    #print(len(input))
    #connections:

    for i in range(len(input_data)):
        net[0][i].collector = input_data[i]
        print(net[0][i].collector)

    print()
    print("Output Layer: ")

    #print(len(input_data))
    #print(sum(input_data))
    for i in range(1,len(input_data) - 1, 1):
        #print(i)
        for n in net[i]:
            #print(n)
            #print(net[i-1])
            for c in net[i-1]: # replace n.connetions, causes summation to be incorrect
                #print(c)
                n.collector = n.collector + c.collector

    print(net[2][0].collector)
main()