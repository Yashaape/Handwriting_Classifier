import numpy as np
import random
import pickle
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
def main():
    print("Network Structure:")
    net_structure = np.array([4,2,1])
    print(net_structure)
    output_layer = None
    net = []
    #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors
    for i in net_structure:
        first_layer = []
        for j in range(i):
            first_layer.append(node(lastlayer=output_layer))
            #print("first layer: ", first_layer)
        net.append(first_layer)
        output_layer = first_layer
    print("network: ", net)
    print("output layer: ",output_layer) #Output layer aka last layer
    print()

    print("Inputs:")
    input_data = np.loadtxt('input.txt', delimiter=',')
    print(input_data)

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
            n.connections = net[i-1] # assign n.connections to net[i-1] to avoid summnation errors
            #print(n.connections)

            if n.connections:
                n.weights = [random.random()] * len(n.connections)
            for index in range(len(n.connections)):
                #print(c)
                c = n.connections[index]
                w = n.weights[index]
                n.collector = n.collector + (c.collector * w)

    print(net[2][0].collector)
    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)


main()