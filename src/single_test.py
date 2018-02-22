import numpy as np
import argparse
import SingleLayerNN

ap = argparse.ArgumentParser()
ap.add_argument("-data_path", "--test_data", type=str, default='./test-data.npy',
	help="test data file path")
ap.add_argument("-model_path", "--weight_path", type=str, default='./model',
	help="weight model file path")
args = vars(ap.parse_args())


single_layer_NN = SingleLayerNN.SingleLayerNN(np.random.rand(5,5),np.random.rand(5,1),10,1,1,1)

single_layer_NN.load_weights(args['weight_path']) #load numpy formatted weights(model) to Neural Net(Single Layer) directly
single_layer_NN.visual_params() #show visualized version of weights
test_data=np.load(args['test_data'])
predictions = single_layer_NN.predict(test_data)

print(predictions) # print result list which made by (image number, prediction class) tuples