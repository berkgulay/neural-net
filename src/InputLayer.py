import numpy as np

class InputLayer:
    def __init__(self,input_size,next_layer_neuron_size):
        self.a = np.zeros((input_size,1))
        self.weight = np.random.rand(next_layer_neuron_size,input_size+1) / 1000
        self.accum = np.zeros(np.shape(self.weight))

    def clear_accum(self):
        self.accum = np.zeros(np.shape(self.weight))