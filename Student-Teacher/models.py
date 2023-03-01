import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

class LinearNN(nn.Module):
	def __init__(self,inp_size,out_size=1):
		super(LinearNN,self).__init__()
		self.inp_size = inp_size
		self.out_size = out_size
		self.fc = nn.Linear(self.inp_size,self.out_size)

	def forward(self, x):
		out = self.fc(x)
		return out

	def zero_gradients(self):
		if self.fc.weight.grad is not None:
			self.fc.weight.grad.data.zero_()
			self.fc.bias.grad.data.zero_()


class MultiLayerNN(nn.Module):
	def __init__(self,inp_size,num_layers=1,hidden_dim=None,non_linearity=None,out_size=1):
		super(MultiLayerNN,self).__init__()
		self.inp_size = inp_size
		assert num_layers>=1, "At least one layer required"
		self.num_layers = num_layers
		if hidden_dim is not None:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = inp_size
		if non_linearity is not None:
			assert non_linearity=='relu', NotImplementedError 
			self.non_linearity = nn.ReLU()
		else:
			self.non_linearity = None
		self.out_size = out_size
		if self.num_layers>1:
			self.layers = [nn.Linear(self.inp_size,self.hidden_dim)]
			if self.non_linearity is not None:
				self.layers.append(nn.ReLU())
			for l in range(1,self.num_layers-1):
				self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim))
				if self.non_linearity is not None:
					self.layers.append(nn.ReLU())
			self.layers.append(nn.Linear(self.hidden_dim,self.out_size))
		else:
			self.layers = [nn.Linear(self.inp_size,self.out_size)]
		self.fc = nn.Sequential(*self.layers)

	def forward(self, x):
		out = self.fc(x)
		return out