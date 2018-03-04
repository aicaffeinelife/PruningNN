import numpy as np 
import sys, os 
from DataLoader import MNISTLoader 

def sigmoid_forward(x):
	return 1./(1.+np.exp(-x))


def sigmoid_grad(x):
	return x*(1-x)


def affine_forward(x,w,b):
	out = np.dot(x,w) + b 
	cache  = (x,w,b)
	return out, cache

def affine_backward(dout, cache):
	x,w,b = cache 
	dx = np.dot(dout, w.T)
	dw = np.dot(x.T, dout)
	db = np.sum(dout, axis=0)
	return dx, dw, db 


def softmax(x):
	y = np.exp(x)
	y = y/(np.sum(x, axis=1, keepdims=True))
	return y 


class FNN(object):
	"""
	Implements a fully connected net 
	with N inputs, H hidden layers, C classes
	with backward and forward passes.
	Activation used: sigmoid 
	"""
	def __init__(self, mode='train', **kwargs):
		super(FNN, self).__init__()
        self.input_dims = kwargs.pop("input_dims")
        self.hidden_dims = kwargs.pop("hidden_dims") # hidden dims is a list.
        self.num_hidden = kwargs.pop("num_hidden",1)
        self.num_classes = kwargs.pop("num_classes",10)
        self.initializer = kwargs.pop("initializer","gaussian") 
        self.lr = kwargs.pop("learning_rate", 1e-2)
        self.n_layers = 1 + self.num_hidden
        self.mode = mode
        self.params = self._gen_params(self.num_hidden, self.input_dims, self.hidden_dims, self.num_classes)
        print(self.params)
        self.grads = {} 
        self.fwd_cache = None


	def _gen_params(self, num_hidden, ip_dims, hidden_dims, nc):
		param_dict = {}
		param_dict['W0'] = np.random.randn(ip_dims*hidden_dims[0])
		param_dict['b0'] = np.zeros(hidden_dims[0])

		for i  in range(len(hidden_dims)):
			if (i+1)%len(hidden_dims) == 0:
				param_dict['W'+ str(i+1)] = np.random.randn(hidden_dims[i], nc)
				param_dict['b'+ str(i+1)] = np.zeros(nc)
			else:
				param_dict['W'+ str(i+1)] = np.random.randn(hidden_dims[i], hidden_dims[i+1])
				param_dict['b'+ str(i+1)] = np.random.zeros(hidden_dims[i+1])

		return param_dict 



	def forward(self, X):
		"""
		Runs a forward pass of a training example through a network
		returns a loss and cache if in train mode else scores. 
		"""
	
		cache = ()
		scores = None
		w0, b0 = self.params['W0'], self.params['b0']
		h1,cache_ip = affine_forward(X,w0,b0)
		h1_act = sigmoid_forward(h1)
		hidden_cache = []
		for i in range(self.num_hidden):
			w_i = self.params['W'+str(i+1)]
			b_i = self.params['b'+str(i+1)]
			if i == 0:	
				h2, cache_ip = affine_forward(h1_act, w_i, b_i)
				h2_act = sigmoid(h2)
				hidden_cache.append(cache_ip)
			elif (i+1)%self.num_hidden == 0:
				 scores_pre, cache = affine_forward(hidden_cache[-1][0], w_i, b_i)
                 scores = sigmoid_forward(scores_pre)

			else:
				h_i, cache_i = affine_forward(hidden_cache[i-1][0], w_i, b_i)
				h_i_act = sigmoid(h_i)
				hidden_cache.append(cache_i)

		print(scores.shape)
		if mode == 'test':
			return scores 
		else:
			self.fwd_cache = (cache_ip, hidden_cache, cache)
			return scores 


	def backprop(self, scores, target=None):
		"""
		Calculates the error and backprops this to 
		every neuron.  
		"""
        grads = {}
        hidden_grads = {}
		err = scores - target 
		dout = sigmoid_grad(scores)*err 
		cache_ip, hidden_cache, cache = self.fwd_cache 
		dhout_act, dwout, dbout = affine_backward(dout, cache)
		dhout = sigmoid_grad(dhout_act)
		grads['W'+str(self.hidden_dims)] = dwout
		grads['b'+str(self.hidden_dims)] = dbout 
		hidden_grads['H'+str(self.hidden_dims)] = dhout
		for i  in range(self.hidden_dims-1, -1, -1):
			if i == 0:
				dip_act, dW0, db0 = affine_backward(hidden_grads['H'+str(i+1)], cache_ip)
				dip = sigmoid_grad(dip_act)
				grads['W'+str(i)] = dW0 
				grads['b'+str(i)] = db0 
			else:
				dh_i_act, dw_i, db_i = affine_backward(hidden_grads['H'+str(i+1)], hidden_cache[i-1])
				dh  = sigmoid_grad(dh_i_act)
				grads['W'+str(i)] = dw_i 
				grads['b'+str(i)] = db_i 
                hidden_grads['H'+str(i)] = dh 
            self.grads  = grads




if __name__ == '__main__':
    mnist_loader = MNISTLoader()
    label, data = mnist_loader.load_data()
    fnn = FNN(input_dims=784, hidden_dims=[100])

     


		











