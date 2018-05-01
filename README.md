# PruningNN

A Python module to build Karnin Pruning and test it on MNIST Data. Karnin Pruning builds up a shadow list of square of gradients over learning rate during training. This list then can be examined to prune the weights which have a lower sensitivity value than a pre-set threshold. 



To run the script simply run `python network1.py` and look at the logs. It's not pretty but it works. 
