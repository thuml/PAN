import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
class AdversarialLayer(torch.autograd.Function):
  def __init__(self, max_iter):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter
    self.lamb = 2.0
    
  def forward(self, input):
    self.iter_num += 1
    return input * 1.0

  def backward(self, gradOutput):
    coeff = np.float(self.lamb * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
    return -coeff * gradOutput

class RMANLayer(torch.autograd.Function):
  def __init__(self, input_dim_list=[], output_dim=1024):
    self.input_num = len(input_dim_list)
    self.output_dim = output_dim
    self.random_matrix = [Variable(torch.randn(input_dim_list[i], output_dim)) for i in xrange(self.input_num)]
    for val in self.random_matrix:
      val.requires_grad = False
  def forward(self, input_list):
    return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in xrange(self.input_num)]
    return_list[0] = return_list[0] / float(self.output_dim)
    return return_list
  def cuda(self):
    self.random_matrix = [val.cuda() for val in self.random_matrix]

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input

  def backward(self, gradOutput):
    return 0 * gradOutput

