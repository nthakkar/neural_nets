from __future__ import print_function

import numpy as np
import six.moves.cPickle as pickle
import load_data

class NeuralNetwork(object):

	'''Multi-layer neural network classifier with sigmoid neurons.
	Following http://neuralnetworksanddeeplearning.com/

	Training is done with backpropagation and sgd. 
	'''

	def __init__(self, sizes):

		'''Sizes is a list of layer sizes. Connectivity is full.'''

		## Global geometric properties
		self.num_layers = len(sizes)
		self.sizes = sizes

		## Storage and randome initialization for the parameters
		## biases are a list of np.arrays with shape (size,1)
		## weights are a list of np.arrays with shape (size_l+1,size_l)
		## Weights are initialized with a normal distribution with variance inversely 
		## proportional to the number of inputs to avoid initialization with saturated 
		## neurons.
		self.biases = [np.random.randn(k,1) for k in sizes[1:]]
		self.weights = [np.random.randn(k,j)/np.sqrt(j) for j,k in zip(sizes[:-1],sizes[1:])]


	######################################################################################################
	# Cost and activation function related methods
	######################################################################################################

	def sigmoid(self,z):

		'''Activation function.'''

		return 1./(1.+np.exp(-z))

	def sigmoid_prime(self,z):

		'''Derivative of the activation function.'''

		return self.sigmoid(z)*(1.-self.sigmoid(z))

	def cross_entropy_cost(self, a, y):

		'''Cross entropy cost function, stablized with np.nan_to_num'''

		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	def cross_entropy_delta(self, z, output_activations, y):

		'''error for the cross entropy cost function at the output layer L.'''

		return (output_activations - y)

	def l2_cost(self, a, y):

		'''l2 cost function'''

		return 0.5*np.linalg.norm(a-y)**2

	def l2_delta(self, z, output_activations, y):

		'''error for the L2 cost function at the output layer L.'''

		return (output_activations - y)*self.sigmoid_prime(z)

	######################################################################################################
	# Training related functions
	######################################################################################################

	def feed_forward(self,x):

		'''Return network output with input x'''

		## iterate forward through the network
		for b,w in zip(self.biases, self.weights):
			x = self.sigmoid(np.dot(w,x)+b)
		return x

	def backward_propagation(self, x, y):

		'''Takes an input x and true classification y and returns a tuple (nabla_b, nabla_w)
		which represent the gradient C_x
		'''

		## Initialize a list of np.zeros with the same
		## structure as self.biases and self.weights
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		## Feed forward.
		## I do not use the feed_forward function since
		## I want to store a bunch of intermediate quantities
		## Initial activation is just the input. As I go, I store
		## a_j (the activations at each layer) and z_j, the weighted
		## inputs at each layer.
		activation = x
		activations = [x]
		zs = []
		for b,w in zip(self.biases, self.weights):

			## Compute weighted input at this layer and store
			z = np.dot(w, activation) + b
			zs.append(z)

			## Activate and store, overwriting the previous activation
			activation = self.sigmoid(z)
			activations.append(activation)

		## Now backpropagate. Use the backprop equations to estimate the 
		## error (delta) at the last layer first. Then move backwards to 
		## collect terms in the chain rule
		## Last is first
		delta = self.cross_entropy_delta(zs[-1],activations[-1],y)

		## These specify the updates for the biases and 
		## weights of the final layer
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		## Now loop backwards through the layers and compute delta
		for l in range(2, self.num_layers):

			## Get the weighted input from the previous layer
			z = zs[-l]
			sp = self.sigmoid_prime(z)

			## Compute and update delta
			delta = np.dot(self.weights[-l+1].transpose(), delta)*sp

			## Update gradients
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def sgd_training(self, training_data, epochs, mini_batch_size, learning_rate, weight_decay=0., valid_data=None):

		'''training and valid data are lists of (x,y) tuples. Valid data computes a validation score at
		each epoch if provided.

		weight_decay is used for L2 regularization in self.update_mini_batch, default is zero.

		'''

		## Figure out how long the loops are
		if valid_data: 
			n_valid = len(valid_data)
			best_score = 0.
		n = len(training_data)


		for j in range(epochs):

			## Construct the mini_batches by random shuffling
			np.random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

			## Train on each minibatch
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, learning_rate, weight_decay, n)

			if valid_data:
				n_correct = self.evaluate(valid_data)
				score = (100.*n_correct)/n_valid
				print("Epoch {0}: {1}/{2} ~ {3} %".format(j, n_correct,n_valid,score))
				if score > best_score:

					## Save the best parameters
					## Probably pickling is a better option here
					best_score = score
					best_biases = self.biases
					best_weights = self.weights
			else:
				print("Epoch {0} complete".format(j))

		if valid_data:
			print("After {0} epochs, best score was {1} %".format(epochs, best_score))
			self.biases = best_biases
			self.weights = best_weights

	def update_mini_batch(self, mini_batch, learning_rate, weight_decay, n):

		'''Wrapper for the back prop algorithms interface with the mini batch generator.'''

		## Make storage for the gradients 
		## that mimics the self.biases and self.weights
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		## Go point-to-point in the minibatch. In principle, this part
		## should get vectorized to handle a full minibatch all at once.
		for x,y in mini_batch:

			## Get the contribution to nabla
			delta_nabla_b, delta_nabla_w = self.backward_propagation(x,y)

			## Update accordingly
			nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

		## Finally, actually update the weights.
		## If I wanted to regularlize, this would be the place to do it.
		self.weights = [(1.-learning_rate*(weight_decay/n))*w-(learning_rate/len(mini_batch))*nw 
						for w,nw in zip(self.weights, nabla_w)]
		self.biases = [b-(learning_rate/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

	######################################################################################################
	# Testing/prediction related methods
	######################################################################################################

	def evaluate(self, test_data):

		'''Return the performance on a test_data set assuming arg_max on the output activations
		is the classification.'''

		test_results = [1*(np.argmax(self.feed_forward(x))==y) for x,y in test_data]
		return sum(test_results)

	######################################################################################################
	# Saving state and loading
	######################################################################################################

	def save(self, filename='network.pkl'):

		'''Save the neural network to the file filename.'''

		## Construct a dictionary for pickling
		data = {'sizes': self.sizes, 'weights': self.weights, 'biases': self.biases}
		
		## Pickle the dictionary
		with open(filename, 'w') as f:
			pickle.dump(data, f)

def LoadNetwork(filename='network.pkl'):

	'''External function, ouputs a network class from fname'''

	## Unpickle the file
	data = pickle.load(open(filename))

	## Parse
	network = NeuralNetwork(data['sizes'])
	network.weights = data['weights']
	network.biases = data['biases']

	return network

if __name__=="__main__":

	## Get the data
	train, valid, test = load_data.GetData()

	## Train the model
	model = NeuralNetwork([28*28, 30, 10])
	model.sgd_training(train, epochs=30, mini_batch_size=10, 
					learning_rate=0.5, weight_decay=5.,valid_data=valid)







