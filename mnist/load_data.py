

from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import numpy as np

def LoadData(fname='mnist.pkl.gz'):

	'''Load the data from the mnist pkl. The pkl makes it simple.'''

	with gzip.open(fname,'rb') as f:
		
		try:
			dataset = pickle.load(f, encoding='lantin1')
		except:
			dataset = pickle.load(f)

	return dataset

def OneHot(j,n=10):

	'''Returns a vector of len = n with j = 1 and all others = 0'''

	e = np.zeros((n,1))
	e[j] = 1
	return e


def GetData(fname='mnist.pkl.gz'):

	'''Reshapes the data after loading so that the format is 

	(x,y) with x a column vector and y a one_hot vector for training and int for valid and test.

	'''

	training, valid, test = LoadData(fname)

	## Just do it one by one
	## reshaping and recasting each.
	## Training
	training_inputs = [np.reshape(x, (784, 1)) for x in training[0]]
	training_results = [OneHot(y) for y in training[1]]
	training_data = zip(training_inputs, training_results)

	## Validation
	validation_inputs = [np.reshape(x, (784, 1)) for x in valid[0]]
	validation_data = zip(validation_inputs, valid[1])

	## Test
	test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
	test_data = zip(test_inputs, test[1])

	return (training_data, validation_data, test_data)

if __name__ == "__main__":

	train, valid, test = LoadData()
	X,Y = train
	print(X.shape)
	print(Y.shape)