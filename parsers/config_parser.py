from __future__ import print_function
import sys, os 
import json



class ConfigParser(object):
    """
    A simple module to parse a config
    file for creating a neural network
    """
    def __init__(self, cfg_file='cfg.json'):
	    super(ConfigParser, self).__init__()
	    self.fname = cfg_file
	    self.data = self._parse_json()



    def _parse_json(self):
	    _data = json.load(open(self.fname))
	    return _data


    def extract_data_by_field(self, field_name=''):
        assert(field_name!=''), "A field name needs to be specified"
        try: 
            data_field = self.data[field_name]
        except KeyError:
            print("The given field doesn't exist in the data")
        return data_field

    def extract_data(self):
        return self.data













