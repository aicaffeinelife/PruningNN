import sys, os 
import numpy as np 
import copy
import random
from DataLoader import  MNISTLoader 
import pprint

# A module to train and evaluate 
# a network based on the given sizes 

# for pruning we need to save the 'state' of
# the neural network and then load it for pruning.

def sigmoid_forward(x):
	return 1./(1.+np.exp(-x))


def sigmoid_grad(x):
	return x*(1-x)

def error_grad(scores, target):
	return target-scores


class Network(object):
	def __init__(self, sizes):
		super(Network, self).__init__()
		self.sizes = sizes 
		self.num_layers = len(self.sizes)
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
		# self.init_weights = copy.deepcopy(self.weights) # initial array of weights for karnin. 
		self.slist = [np.zeros(w.shape) for w in self.weights] # the sensitivity list. 
		self.grads = [] # stored as list of list of grads of each layer. 


	def feedforward(self, x):
		"""
		Compute the affine forward for each layer.
		y = \sigma(X.w + b) 
		"""
		for w,b in zip(self.weights, self.biases):
			x = sigmoid_forward(np.dot(w,x) + b)

		return x



	def train(self, x_train, y_train, lr, num_iters, batch_size, test_data=None):
		"""
		Trains a neural network using SGD(no momentum). 
		"""
		train_len = len(y_train)
		init_weight = copy.deepcopy(self.weights)
		np.savez('init_weights.npz', init_weight)

		for it in range(num_iters):
			idx = random.sample(range(train_len), batch_size)
			x_batch = x_train[idx]
			y_batch = y_train[idx]

			nabla_w = [np.zeros_like(w) for w in self.weights]
			nabla_b = [np.zeros_like(b) for b in self.biases]

			# calculate gradients over a small batch 
			for x,y in zip(x_batch, y_batch): 
				delta_b, delta_w = self.backprop(x,y)
				nabla_b = [b+db for b,db in zip(nabla_b, delta_b)]
				nabla_w = [w+dw for w,dw in zip(nabla_w, delta_w)]

			
			self.weights = [w-(lr/batch_size)*nw for w,nw in zip(self.weights, nabla_w)]
			self.biases  = [b-(lr/batch_size)*nb for b,nb in zip(self.biases, nabla_b)]
			diff = [np.square(nw)/lr for nw in nabla_w]
			self.slist = [s+d for s,d in zip(self.slist, diff)]           
			print("Iteration:{}".format(it))

		np.savez('checkpoint.npz', self.weights, self.biases, self.slist)


	# def update_mini_batch(self,mini_batch, lr):
	# 	nabla_b = [np.zeros_like(b) for b in self.biases]
	# 	nabla_w = [np.zeros_like(w) for w in self.weights]

	# 	for x,y in mini_batch:
	# 		delta_b, delta_w = self.backprop(x,y)
	# 		nabla_b = [b+db for b,db in zip(nabla_b, delta_b)]
	# 		nabla_w = [w+dw for w,dw in zip(nabla_w, delta_w)]

	# 	init_weights = copy.deepcopy(self.weights) # copy the initial weights
	# 	self.weights = [w - (lr/len(mini_batch))*nw for w, dw in zip(self.weights, nabla_w)]
	# 	self.biases  = [b - (lr/len(mini_batch))*nb for b,db in zip(self.biases, nabla_b)]

	# 	up_i = self.update_slist(init_weights, self.weights, nabla_w ,lr) 
	# 	self.slist = [s+u for s,u in zip(self.slist, up_i)]



	def backprop(self, x, y):
		"""
		Forward and backward pass of the network.
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x 
		activation_lst = [x]
		net_lst = [] 

		for b,w in zip(self.biases, self.weights):
			# print("w shape: {}".format(w.shape))
			net = np.dot(w, activation) + b 
			net_lst.append(net)
			activation = sigmoid_forward(net) 
			activation_lst.append(activation) 

		delta = error_grad(activation_lst[-1], y)
		nabla_b[-1] = delta 
		nabla_w[-1] = np.dot(delta, activation_lst[-2].transpose())

		for l in range(2,1,self.num_layers):
			net_i = net_lst[-l]
			net_diff = sigmoid_grad(net_i)
			delta = np.dot(self.weights[-l-1].transpose(), delta)*net_diff
			nabla_w[-l] = np.dot(delta, activation_lst[-l-1].transpose())
			nabla_b[-l] = delta 

		return (nabla_b, nabla_w)


	# def update_slist(self, lr, grad_w):
	# 	"""
	# 	Update the sensisitvity list 
	# 	"""
	# 	shp_lst = [np.dot(gw.T, gw).shape for gw in grad_w]
	# 	# print(shp_lst)
	# 	slist = [np.zeros(sp) for sp in shp_lst]
	# 	print(self.slist)
	# 	prod_lst = [np.dot(gw.T, gw)/lr for gw in grad_w]
	# 	slist = [s+p for s,p in zip(self.slist, prod_lst)]
		

		
	def evaluate(self, test_data, test_labels ):
		res = [self.feedforward(x) for x in test_data]

		return sum([int(np.argmax(res[i])== np.argmax(test_labels[i])) for i in range(len(res))])/float(len(res))





# if __name__ == '__main__':
# 	ml = MNISTLoader()
# 	labels, data = ml.load_data()
# 	sz_list = [784, 100, 10]
# 	net = Network(sz_list)
# 	print(len(net.weights))
# 	assert(net.weights[0].shape == (100,784)), "weights of hidden layer do not match"
# 	assert(net.weights[1].shape == (10,100)), "weights of the output layer do not match"

# 	lr = 1e-2
# 	batch_size = 5
# 	num_iters =  5 
# 	# print("sensisitvity list before:{}".format(net.slist))
# 	net.train(data, labels,lr,num_iters,batch_size)
# 	print("sensisitvity list after:{}".format(net.slist))










	








		