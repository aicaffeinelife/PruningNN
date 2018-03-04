from __future__ import print_function
import sys, os 
from config_parser import ConfigParser
import pytest


def test_parser(fname, field_name):
    assert(fname), "File not at correct location"
    cfgp  = ConfigParser(fname)
    data = cfgp.extract_data(field_name)
    assert(data != None)
	




if __name__ == '__main__':
	fname = "/home/akulshr/ANN/three_layer.json"
	test_parser(fname, "hyperparams")
