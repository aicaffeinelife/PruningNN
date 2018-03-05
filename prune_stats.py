import numpy as np 
import sys, os 
from DataLoader import MNISTLoader 
from network1 import Network
import matplotlib.pyplot as plt 


# Some utils to study effect of pruning 


# def recommend_prune_idx(sen_lst, weight_lst, threshold=1e-2):
# 	"""
# 	return a list of weight indices that can be pruned 
# 	and the actual weights per layer that are below 
# 	threshold. 
# 	"""
# 	prune_ids = [] 


# 	for l in range(len(weight_lst)):
# 		sl = sen_lst[l]
# 		sl = sl[np.logical_not(np.isnan(sl))]
# 		idx = np.zeros(sen_lst[l].shape)
# 		# weight = weight_lst[l]
# 		idx = np.where(sl<threshold)
# 		# print("idx of layer-{}:{}".format(l, idx))
# 		prune_ids.append(idx) 
# 	return prune_ids


def one_hot_encode(y):
	cats = np.unique(y)
	y_cat = np.zeros((y.shape[0], cats.shape[0]))
	y_cat[np.arange(y.shape[0]), y] = 1

	return y_cat 

	

def prune_weights(sen_lst, weight_lst, threshold=1e-2):
	"""
	prune the weights and evaluate the accuracy. 
	"""
	for l in range(len(weight_lst)):
		print("Weight shape pre-pruning:{}".format(weight_lst[l].shape))

		wl = weight_lst[l]
		sl = sen_lst[l]
		idx = np.zeros(sl.shape)
		sl = sl[np.logical_not(np.isnan(sl))]
		idx = np.where(sl<threshold)
		prw = np.take(wl, idx)
		print("Weight shape ater-pruning:{}".format(prw.shape))




def evaluate(net, checkpoint, test_data, test_labels):
	"""
	Evaluate a net based on the weights and biases
	"""
	ckpts = np.load(checkpoint)
	print(ckpts.files)
	weights = ckpts['arr_0']
	biases = ckpts['arr_1']
	cache = (net.weights, net.biases)
	net.weights = weights 
	net.biases = biases 
	acc = net.evaluate(test_data, test_labels)
	print("accuracy:{}".format(acc))







def main():
	mloader = MNISTLoader()
	tml = MNISTLoader('test')
	labels, data = mloader.load_data()
	test_labels, test_data = tml.load_data()
	labels_unq = one_hot_encode(np.asarray(labels))
	test_labels_unq = one_hot_encode(np.asarray(test_labels))
	sizes = [784,100,10] # the size of the network
	net = Network(sizes)
	lr = 0.1
	batch_size = 200
	num_iters = 1000
	net.train(data,labels, lr, num_iters, batch_size)
	ckpt_f = np.load('checkpoint.npz')
	iwf = np.load('init_weights.npz')
	iw = iwf['arr_0']
	slist = ckpt_f['arr_2']
	fw = ckpt_f['arr_0']
	delta = [wf/(wf-wi) for wf, wi in zip(fw, iw)]
	slist_f = [s*d for s,d in zip(slist, delta)]
	print(slist_f)
	evaluate(net, 'checkpoint.npz',test_data, test_labels_unq)

	# evaluate(net, 'checkpoint.npz', test_data, test_labels)

	# prune_weights(net.slist, net.weights)
	# prune_idx = recommend_prune_idx(net.slist, net.weights)
	# print(prune_idx)





if __name__ == '__main__':
	main()


