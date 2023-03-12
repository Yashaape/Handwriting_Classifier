import numpy as np


class node:
    def __int__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


net = []

output_layer = None
collector = 0.0

with open('input.txt') as num:
    network = num.read().splitlines()

print(network)
