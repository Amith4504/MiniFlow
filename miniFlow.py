#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:18:39 2019

@author: amith

Creating neural networks differentiable graphs and forward propogation 
"""
""" inbound nodes and outbound nodes   """

import numpy as np

class Node(object):
    def __init__(self,inbound_nodes=[]):
        # nodes from which this nodes receives the values
        self.inbound_nodes = inbound_nodes
        # nodes to which this nodes passes the outputs  list of Node objects
        self.outbound_nodes = []
          
        #Keys are inputs to this node and thier values are the partials of this node with respect to that
        # input
        self.gradients = {}
        
        #for each inbound_node , add the current node as its outbound node
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        #a calculated value    
        self.value = None
        
    def forward(self):
        raise NotImplementedError
        """
        Forward propogation 
        Compute the output value based on inbound_nodes and store 
        the result in self.value
        """
        
    def backward(self):
      
      raise NotImplementedError
        
""" inheritance of class Node --> Input"""
class Input(Node):
    def __init__(self):
        Node.__init__(self)
        # an input node has no inbound nodes
        
    def forward(self,value =None):
        pass
        # overwrite the value if one is passed in
        # input nodes do not calculate anything
        
    def backward(self):
      # An Input node has no inputs => grad = 0
      # the key self is reference to this object
      self.gradients = {self : 0}
      #weights and biases may be inputs so you need to sum the gradients from op gradients
      for n in self.outbound_nodes:
        grad_cost = n.gradients[self]
        self.gradients[self]  += grad_cost * 1
        
class Linear(Node):
    def __init__(self,X,W,b):
        Node.__init__(self,[X,W,b])
    
    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        output_na = np.dot(X,W) + b
        #sig = 1./(1.+ np.exp(-1*output_na))
        self.value = output_na
        
    def backward(self):
      """
      Calculates the gradient based ont the output values
      """
      self.gradients = {n : np.zeros_like(n.value) for n in self.inbound_nodes}
      
      """
      Cycle through the outputs 
      
      """
      for n in self.outbound_nodes:
        grad_cost = n.gradients[self]
        
        # Partial of the loss wrt this node's inputs
        self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost , self.inbound_nodes[1].value.T)
        # Partial of the loss wrt this node's weights
        self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T , grad_cost)
        # Partial of the loss wrt this node's bias
        self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost  , axis = 0 , keepdims = False)
        

class Sigmoid(Node):
  """
  a sigmoid is a part of its own derivative 
  used in both forward and backward prop
  
  """
  def __init__(self , node):
    Node.__init__(self , [node])
    
  def sigmoid(self , x):
    
    sigmoid_result = 1./ (1 + np.exp(-1 * x))
    
    return sigmoid_result
  
  def forward(self):
    x = self.inbound_nodes[0].value
    self.value = self.sigmoid(x)
    
  def backward(self):
    """
    Gradient of the sigmoid function
    
    """
    self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
    
    """
    Cycle through the outputs. The gradient changes depending on each output . Thus gradients are summed overall 
    outputs
    """
    for n in self.outbound_nodes:
      grad_cost = n.gradients[self]
      sigmoid = self.value
      self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
    
    
    
class MSE(Node):
  
  def __init__(self , y , a):
    Node.__init__(self , [y , a])
    
  
  def forward(self):
    """
    Reshaping y ,a to avoid possible errors
    
    """
    y = self.inbound_nodes[0].value.reshape(-1 , 1)
    a = self.inbound_nodes[1].value.reshape(-1 , 1)
    
    self.m = self.inbound_nodes[0].value.shape[0]
    
    self.diff = y-a
    
    self.value = np.mean(self.diff**2)  # MSE Node value contains MSE
    
  
  def backward(self):
    """
    Calculating the gradient of the cost
    
    """
    self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
    self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
    


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.{} + {} = {}(according to miniflow)".format(feed_dict[x] , feed_dict[y] , 

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    
    """
    
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value      


def forward_and_backward(graph):
  # graph after topological sort
  for n in graph:
    n.forward()
    
    
  for n in graph[::-1]:
    n.backward()
    
    
def sgd_update(trainables , learning_rate = 1e-2):
  """
  Arguements: 
      trainables : A list of Input Nodes representing weight/biases
  """
    
  for variable in trainables:
    partial  = variable.gradients[variable]
    variable.value -= learning_rate * partial
    
  pass