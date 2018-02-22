import numpy as np
import argparse
import MultiLayerNN

ap = argparse.ArgumentParser()
ap.add_argument("-data_path", "--test_data", type=str, default='hw3_data/validation-data.npy',
	help="test data file path")
ap.add_argument("-model_path", "--weight_path", type=str, default='./model.npy',
	help="weight model file path")
args = vars(ap.parse_args())


multi_layer_NN = MultiLayerNN.MultiLayerNN(np.random.rand(5,5),np.random.rand(5,1),10,1,1,1)

multi_layer_NN.load_weights(args['weight_path']) #load numpy formatted weights(model) to Neural Net(Single Layer) directly
test_data = np.load(args['test_data'])
predictions = multi_layer_NN.predict(test_data)

print(predictions) # print result list which made by (image number, prediction class) tuples