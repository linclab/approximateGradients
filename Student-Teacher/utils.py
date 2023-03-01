import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
torch.set_default_tensor_type(torch.DoubleTensor)
from tqdm import tqdm
from models import LinearNN, MultiLayerNN

def save_all_figs_to_disk(folder='Results'):
	for figname in plt.get_figlabels():
		plt.figure(figname)
		figname_save = figname.replace(' ','_')
		figname_save = figname_save.replace('_-_','_')
		plt.savefig(os.path.join(folder,figname_save))

def create_data_labels(inp_dim=10,num_data=100,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	X = torch.rand((num_data,inp_dim))
	X.requires_grad_ = False
	teacher_net = LinearNN(inp_size=inp_dim)
	with torch.no_grad():
		Y = teacher_net(X)
	# loss_fn = torch.nn.MSELoss()
	return inp_dim,num_data,X,Y,teacher_net


def get_projection_matrix(inp_dim=10,projection_dim=20,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	expansion_dim = inp_dim if inp_dim>projection_dim else projection_dim
	with torch.no_grad():
		A = torch.rand((expansion_dim,expansion_dim))
		A = A+A.T
		l,v = torch.linalg.eig(A)
		v = torch.real(v)
	W = v[:inp_dim+projection_dim-expansion_dim]
	if inp_dim>projection_dim:
		W = W.T
	return W


def load_student_net(inp_dim,projection_weights=None,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = LinearNN(inp_size=inp_dim)
	if projection_weights is not None:
		student_net.fc.weight.data = torch.matmul(student_net.fc.weight.data,projection_weights)
		student_net.inp_size = projection_weights.shape[-1]
	return student_net


def get_loss(net,X,y,loss_fn):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(device)
	X = X.to(device)
	y = y.to(device)
	with torch.no_grad():
		Y_hat = net(X)
		loss = loss_fn(Y_hat,y)
	return loss.item()


def gradient_descent_one_step(net,lr,bias=0.0,variance=0.0,grad_noise=None,no_update=False,relative=False):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(device)
	params = list(net.parameters())
	grad_list = []
	for p in params:
		if p.requires_grad:
			grad_list.append(p.grad.data.flatten())
	grad_vector = torch.cat(grad_list)
	if relative:
		grad_norm = torch.norm(grad_vector).item()
		bias = bias*grad_norm
	# MAYBE BIAS SHOULD BE DIVIDED BY GRAD_VECTOR SIZE TO CONTROL FOR DIFFERENT NETWORK SIZE -- No it should not (we are looking at bias in synaptic wt. update)!
	# bias = bias/np.sqrt(len(grad_vector))
	if grad_noise is None:
		grad_noise = torch.normal(mean=-bias*torch.ones(grad_vector.size()),std=np.sqrt(variance)*torch.ones(grad_vector.size())).to(grad_vector.device)
	else:
		grad_noise -= bias
	# NOTE: -ve sign in bias above because this will be added to the gradient and thereafter subtracted from current wt.
	# Hence this has reverse effect of the theory assumption!
	# tqdm.write("{} {}".format(torch.norm(grad_vector).item(),torch.norm(grad_noise)))
	new_grad_vector = grad_vector+grad_noise.to(device)
	param_pointer = 0
	for p in params:
		if p.requires_grad:
			p_size = p.flatten().size(0)
			new_grad = new_grad_vector[param_pointer:param_pointer+p_size].view(p.size())
			p.grad.data = new_grad
			param_pointer += p_size

	if not no_update:
		for p in params:
			if p.requires_grad:
				p.data -= lr*p.grad
	return (grad_vector, grad_noise)

	
def train_one_step(net,X,y,loss_fn,lr,bias=0.0,variance=0.0,grad_noise=None,verbose=True,no_update=False):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(device)
	X = X.to(device)
	y = y.to(device)
	net.zero_grad()
	Y_hat = net(X)
	loss = loss_fn(Y_hat,y)
	loss.backward()
	grad = gradient_descent_one_step(net,lr=lr,bias=bias,variance=variance,grad_noise=grad_noise,no_update=no_update)
	if verbose:
		print("Loss 1: {}".format(loss.item()))
	with torch.no_grad():
		Y_hat = net(X)
		loss_val = loss_fn(Y_hat,y)
		if verbose:
			print("Loss 2: {}".format(loss_val.item()))
	return grad, loss.item()-loss_val.item()


def calculate_hessian(net,X,y,loss_fn):
	params_list = [p.view(-1) for p in net.parameters()]
	def loss_function_hess(W_b):
		out_size = y.size(-1)
		return loss_fn(torch.nn.functional.linear(X,W_b[:-out_size].view(out_size,-1),W_b[-out_size:]),y)
	grad2_W = torch.autograd.functional.hessian(loss_function_hess,torch.hstack(params_list))
	return grad2_W


def check_loss_decrease_eq(grad,grad2,lr,loss_dec,bias=0.0,variance=0.0,tol=None,verbose=True):
	if tol is None:
		tol = 1e-8
	grad_decrease = ((-torch.norm(grad)**2)+bias*torch.sum(grad))*lr
	grad2_decrease = 0.5*(torch.sum(grad.unsqueeze(1)*torch.matmul(grad2,grad.unsqueeze(1))) \
						-bias*torch.sum(grad*torch.sum(grad2+grad2.transpose(0,1),dim=1)) \
						+(bias**2)*torch.sum(grad2) \
						+(variance**2)*torch.trace(grad2)/grad2.shape[0])*(lr**2)
	if verbose:
		print(loss_dec,(-grad_decrease-grad2_decrease).numpy(),loss_dec-(-grad_decrease-grad2_decrease).numpy())
		# print(grad_decrease.numpy(),grad2_decrease.numpy())
	assert np.allclose(loss_dec,(-grad_decrease-grad2_decrease).numpy(),atol=tol), "Some error in calculation!"
	return loss_dec-(-grad_decrease-grad2_decrease).numpy()


def get_uncorrupted_loss_decrease(grad,grad2,lr):
	grad_decrease = (-torch.norm(grad)**2)*lr
	grad2_decrease = 0.5*torch.sum(grad.unsqueeze(1)*torch.matmul(grad2,grad.unsqueeze(1)))*(lr**2)
	return (-grad_decrease-grad2_decrease).numpy()


def calculate_sampling_bias_correction_term(grad,grad2,noise_mean,lr,bias=0.0):
	grad_decrease_correction = torch.sum(grad*noise_mean)*lr
	grad2_decrease_correction = 0.5*(-torch.sum(noise_mean*torch.matmul(grad2+grad2.T,grad)) \
								 +bias*torch.sum(noise_mean*torch.sum(grad2+grad2.T,dim=1)))*lr**2
	# print("correction",noise_mean.shape,grad.shape,grad2.shape,grad_decrease_correction,grad2_decrease_correction,-grad_decrease_correction-grad2_decrease_correction)
	# print(noise_mean,grad,grad2)
	return (-grad_decrease_correction-grad2_decrease_correction).numpy()
								 

def get_optimal_lr_value(grad,grad2,bias=0.0,variance=0.0,verbose=True):
	grad_decrease_coeff = (-torch.norm(grad)**2)+bias*torch.sum(grad)
	grad2_decrease_coeff = (torch.sum(grad.unsqueeze(1)*torch.matmul(grad2,grad.unsqueeze(1))) \
							-bias*torch.sum(grad*torch.sum(grad2+grad2.transpose(0,1),dim=1)) \
							+(bias**2)*torch.sum(grad2) \
							+(variance**2)*torch.trace(grad2)/grad2.shape[0])
	# If Loss=ax+0.5*bx^2 <==> x* = -a/b
	if verbose:
		print(bias,variance,grad_decrease_coeff,grad2_decrease_coeff)
	optimal_lr = -grad_decrease_coeff/(grad2_decrease_coeff+1e-9)
	return optimal_lr.item()


def generate_random_noise_sequence(repeats,inp_dim,variance,seed=7):
	# torch.manual_seed(seed)
	# np.random.seed(seed)
	# torch.random.seed()
	noise_vectors = torch.randn((repeats-1,1+inp_dim))
	noise_vectors = (variance)*torch.div(noise_vectors,torch.norm(noise_vectors,dim=1).unsqueeze(1))
	noise_vectors = torch.vstack([noise_vectors,-noise_vectors.sum(dim=0)])    # to ensure that sum of random noise vectors is 0
	return noise_vectors


def create_NL_data_labels(inp_dim=10,num_data=100,num_layers=1,non_linearity=None,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	X = torch.rand((num_data,inp_dim))
	X.requires_grad_ = False
	teacher_net = MultiLayerNN(inp_size=inp_dim,num_layers=num_layers,hidden_dim=inp_dim,non_linearity=non_linearity)
	with torch.no_grad():
		Y = teacher_net(X)
	return inp_dim,num_data,X,Y,teacher_net

def load_multilayer_student_net(inp_dim,num_layers=1,hidden_dim=None,non_linearity=None,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = MultiLayerNN(inp_size=inp_dim,num_layers=num_layers,hidden_dim=hidden_dim,non_linearity=non_linearity)
	return student_net

def copy_from_model_to_model(net1,net2):
	assert type(net1).__name__==type(net2).__name__, "Cannot copy params from {} to {}".format(type(net1).__name__,type(net2).__name__)
	reference_params_list = list(net1.parameters())
	new_params_list = list(net2.parameters())
	assert len(reference_params_list)==len(new_params_list), "Param lists ({} vs {}) don't match!".format(len(reference_params_list),len(new_params_list))
	for pidx,p in enumerate(new_params_list):
		p.data = reference_params_list[pidx].data.clone()
	
def check_model_params_equal(net1,net2):
	if type(net1).__name__!=type(net2).__name__:
		return False
	reference_params_list = list(net1.parameters())
	check_params_list = list(net2.parameters())
	if len(reference_params_list)!= len(check_params_list):
		return False
	for pidx,p in enumerate(check_params_list):
		if p.data.shape!=reference_params_list[pidx].data.shape:
			return False
		if not torch.allclose(p.data,reference_params_list[pidx].data):
			return False
	return True
