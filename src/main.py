import numpy as np
import argparse
import SingleLayerNN


ap = argparse.ArgumentParser()
ap.add_argument("-data_path", "--train_data_path", type=str, default='./train-data.npy',
	help="train data file path")
ap.add_argument("-label_path", "--train_data_labels", type=str, default='./train-label.npy',
	help="train data labels file path")
ap.add_argument("-e", "--epochs", type=int, default=150,
	help="number of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=16,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

# Data files path will be taken as arg
train_data = np.concatenate((np.load('./hw3_data/train-data.npy'),np.load('./hw3_data/validation-data.npy')),axis=0)
train_label = np.concatenate((np.load('./hw3_data/train-label.npy'),np.load('./hw3_data/validation-label.npy')),axis=0)

single_layer_NN = SingleLayerNN.SingleLayerNN(train_data,train_label,10,args["epochs"],args["alpha"],args["batch_size"])
single_layer_NN.train()

predictions = single_layer_NN.predict(np.load('./hw3_data/test-data.npy'))

for i in predictions:
	print(str(i[0]+1)+','+str(i[1]))
