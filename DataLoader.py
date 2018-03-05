from __future__ import print_function 
import sys, os
import struct 
from array import array
import numpy as np 



class MNISTLoader(object):
    """
    Reads and loads the MNIST dataset for training. 
    """
    def __init__(self, mode='train'):
        if mode == 'train':
            self.data_file = os.path.join(os.getcwd(),'data', 'train-images-idx3-ubyte')
            self.label_file = os.path.join(os.getcwd(),'data', 'train-labels-idx1-ubyte')
        else:
            self.data_file = os.path.join(os.getcwd(),'data','t10k-images-idx3-ubyte')
            self.label_file = os.path.join(os.getcwd(),'data','t10k-labels-idx1-ubyte')  
    
    def load_data(self):
        lf = open(self.label_file, 'rb')
        mnubr, size = struct.unpack('>II', lf.read(8))
        labels = np.fromfile(lf, dtype=np.uint8)
        labels_np = np.asarray(labels)

        imf = open(self.data_file, 'rb')
        magic, num_imgs, rows, cols = struct.unpack('>IIII', imf.read(16))
        img = np.fromfile(imf, dtype=np.uint8).reshape(len(labels), rows*cols,1)
        return labels, img 

        








        