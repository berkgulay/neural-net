import numpy as np

class OutputLayer:
    def __init__(self,output_size):
        self.z = np.zeros((output_size,1))
        self.a = np.zeros((output_size,1))
        self.prime = 1
        self.func = 'sig'
        self.error = np.zeros((output_size,1))
        self.neuron_size = output_size