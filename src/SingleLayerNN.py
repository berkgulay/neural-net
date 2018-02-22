import numpy as np
import matplotlib.pyplot as plt
import math

class SingleLayerNN:
    def __init__(self,train_data,train_data_labels,class_num,epoch_num,alpha,batch_size):
        self.X = self.__normalize(train_data)
        self.labels = train_data_labels
        self.class_num = class_num
        self.W = np.random.rand(self.X.shape[1]+1,class_num) / 1000
        self.epoch = epoch_num
        self.alpha = alpha
        self.batch_size = batch_size
        self.loss_list = []

    def __normalize(self,value):
        return value / 255

    def __sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def __tan(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def __reLU(self,x):
        return np.maximum(x,0)

    def __vector_y(self,correct_class):
        res = np.zeros(self.class_num, dtype=int)
        res[correct_class] = 1
        return res


    def train(self):
        X = np.c_[np.ones((self.X.shape[0])), self.X]

        for epoch in np.arange(0,self.epoch):
            epoch_loss = []

            for b in np.arange(0,X.shape[0],self.batch_size):
                batch_X = X[b:b+self.batch_size]
                batch_labels = self.labels[b:b+self.batch_size]

                predictions = self.__sigmoid(batch_X.dot(self.W))
                batch_label_vectors = []
                for true_val in batch_labels:
                    batch_label_vectors.append(self.__vector_y(true_val))

                errors = predictions - batch_label_vectors

                loss = np.sum(errors ** 2)
                epoch_loss.append(loss)

                gradient = batch_X.T.dot(errors) / batch_X.shape[0]
                self.W += -self.alpha * gradient

            self.loss_list.append(sum(epoch_loss))

        np.save('model',self.W) #Saves weight model to project directory

    def predict(self,test_data):
        test_data = self.__normalize(test_data)
        test_x = np.c_[np.ones((test_data.shape[0])), test_data]

        pred = self.__sigmoid(test_x.dot(self.W))
        result_list = []

        for i in np.arange(0,len(pred)):
            result_list.append((i,pred[i].tolist().index(max(pred[i]))))

        return result_list

    def load_weights(self,weight_numpy):
        self.W = np.load(weight_numpy)

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(np.arange(0, self.epoch), self.loss_list)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.show()

    def visual_params(self):
        for i in range(0,self.class_num):
            img = np.reshape(self.W[1:, i], (28, 28))
            plt.imshow(img, cmap='gray')
            plt.show()

