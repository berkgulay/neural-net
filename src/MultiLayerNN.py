import numpy as np
import matplotlib.pyplot as plt
import InputLayer
import HiddenLayer
import OutputLayer

class MultiLayerNN:

    def __init__(self,train_data,train_data_labels,class_num,epoch_num,alpha,reg_lambda,hidden_layers_descriptor=[]):
        self.train_data = self.__normalize(train_data)
        self.labels = train_data_labels
        self.class_num = class_num
        self.hid_descriptor = hidden_layers_descriptor
        self.network = []
        self.loss_list = []
        self.epoch = epoch_num
        self.alpha = alpha
        self.reg_lambda = reg_lambda

        #Network arch. creation
        self.network.append(self.create_output_layer())
        for layer in self.hid_descriptor[::-1]:
            self.network.insert(0,self.create_hidden_layer(layer[0],self.network[0].neuron_size,layer[1]))
        self.network.insert(0,self.create_input_layer(self.network[0].neuron_size))

    def __normalize(self,value):
        return value / 255

    def __sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
    def __sigmoid_prime(self,a):
        return a * (1-a)

    def __tan(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def __tan_prime(self,a):
        return 1 - np.square(a)

    def __reLU(self,x):
        return np.maximum(x,0)
    def __reLU_prime(self,x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def __vector_y(self,correct_class):
        res = np.zeros(self.class_num, dtype=int)
        res[correct_class] = 1
        return np.reshape(res,(np.shape(res)[0],1))

    def create_input_layer(self,next_layer_neuron_size):
        input_layer = InputLayer.InputLayer(np.shape(self.train_data)[1],next_layer_neuron_size)
        return input_layer
    def create_hidden_layer(self,neuron_size,next_layer_neuron_size,func):
        hidden_layer = HiddenLayer.HiddenLayer(neuron_size,next_layer_neuron_size,func)
        return hidden_layer
    def create_output_layer(self):
        output_layer = OutputLayer.OutputLayer(self.class_num)
        return output_layer

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(np.arange(0, self.epoch), self.loss_list)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.show()

    def predict(self,test_data):
        test_data = self.__normalize(test_data)

        result_list = []
        for t in np.arange(0,len(test_data)):
            self.network[0].a = np.reshape(test_data[t], (np.shape(test_data[t])[0], 1))
            self.forward_pass()
            result_list.append((t,np.argmax(self.network[-1].a)))

        return result_list

    def load_weights(self,weight_model):
        self.network[0].weight = np.load(weight_model)

    def train(self):
        input_size = len(self.train_data)
        for epoch in np.arange(0, self.epoch):
            epoch_loss = []

            for m in np.arange(0,input_size):
                self.network[0].a = np.reshape(self.train_data[m],(np.shape(self.train_data[m])[0],1))
                self.forward_pass()

                loss = self.calculate_loss(self.labels[m])
                epoch_loss.append(loss)

                self.back_prop()

            # Here used derivative to optimize weights in each layer
            for layer in self.network[-2::-1]:
                layer.weight += -self.alpha * self.calculate_D_Reg(layer.accum, layer.weight, input_size)
                layer.clear_accum()  # Cleared accumulator

            self.loss_list.append(sum(epoch_loss))

        np.save('model',self.network[0].weight)

    # Here calculate D(Derivative with regularization by accumulators)
    def calculate_D_Reg(self,accumulator,weight,m):
        D = np.zeros(np.shape(accumulator))
        for i in np.arange(0,len(accumulator)):
            for j in np.arange(0,len(accumulator[i])):
                if(j==0):
                    D[i][j]= accumulator[i][j] / m
                else:
                    D[i][j] = (accumulator[i][j] + (self.reg_lambda*weight[i][j])) / m   #Regularization added

        return D

    def forward_pass(self):

        for l in np.arange(0,len(self.network)-1):
            self.network[l+1].z = np.dot(self.network[l].weight,np.vstack(([1],self.network[l].a)))
            if(self.network[l+1].func == 'sig'):
                self.network[l+1].a = self.__sigmoid(self.network[l+1].z)
                self.network[l+1].prime = self.__sigmoid_prime(self.network[l+1].a)
            elif(self.network[l+1].func == 'tan'):
                self.network[l+1].a = self.__tan(self.network[l+1].z)
                self.network[l+1].prime = self.__tan_prime(self.network[l+1].a)
            else: #ReLU
                self.network[l+1].a = self.__reLU(self.network[l+1].z)
                self.network[l+1].prime = self.__reLU_prime(self.network[l+1].z)

    def calculate_loss(self,true_label):
        out_layer = self.network[-1]
        out_layer.error = out_layer.a - self.__vector_y(true_label)

        return np.sum(out_layer.error ** 2)

    def back_prop(self):

        for l in np.arange(len(self.network) - 1,1,-1):
            self.network[l-1].error = np.delete(np.dot(self.network[l-1].weight.T, self.network[l].error),0,0) * self.network[l-1].prime
            self.network[l-1].accum += np.dot(self.network[l].error , np.vstack(([1],self.network[l-1].a)).T)

        self.network[0].accum += np.dot(self.network[1].error , np.vstack(([1],self.network[0].a)).T) #Input Layer