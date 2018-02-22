import numpy as np

class HiddenLayer:
    def __init__(self,neuron_size,next_layer_neuron_size,func):
        self.z = np.zeros((neuron_size,1))
        self.a = np.zeros((neuron_size,1))
        self.prime = np.zeros((neuron_size,1))
        self.func = func
        self.weight = np.random.rand(next_layer_neuron_size,neuron_size+1) / 1000
        self.error = np.zeros((neuron_size,1))
        self.accum = np.zeros(np.shape(self.weight))
        self.neuron_size = neuron_size

    def clear_accum(self):
        self.accum = np.zeros(np.shape(self.weight))
