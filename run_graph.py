#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:42:07 2019

@author: amith
"""
from miniFlow import *

inputs , weights , bias = Input() , Input() , Input()

f= Linear(inputs , weights , bias)
X_ = np.array([[-1. , -2.],[-1 , -2]])
W_ = np.array([[2. , -3],[2. , -3]])
b_ = np.array([-3. , -5])
feed_dict = {
        inputs:X_,
        weights:,
        bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f , graph)

print(output)


