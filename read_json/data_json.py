# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import sys
import numpy as np
import json as js

class load_image_from_json(object):

    def __init__(self, json_file):
        self.json_file = json_file
    
    def __del__(self):
        pass
    
    def js_load(self):
        f = open(self.json_file, 'r')
        js_data = js.load(f)
        return js_data


if __name__ == "__main__":
    all_data = load_image_from_json('0(6015).json').js_load()
    for data in all_data:
        print (data['image_id'])
        print (data['keypoint']['human1'])