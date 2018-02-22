import numpy as np
import argparse
import SingleLayerNN

ap = argparse.ArgumentParser()
ap.add_argument("-data_path", "--train_data_path", type=str, default='./train-data.npy',
	help="train data file path")
ap.add_argument("-label_path", "--train_data_labels", type=str, default='./train-label.npy',
	help="train data labels file path")
ap.add_argument("-e", "--epochs", type=int, default=100,
	help="number of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.02,
	help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=64,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

# Data files path will be taken as arg
train_data = np.load(args['train_data_path'])
train_label = np.load(args['train_data_labels'])

single_layer_NN = SingleLayerNN.SingleLayerNN(train_data,train_label,10,args["epochs"],args["alpha"],args["batch_size"])
single_layer_NN.train()
single_layer_NN.plot_loss()
