import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from tqdm import tqdm
from tqdm.contrib.itertools import product
from linclab_utils import plot_utils
plot_utils.linclab_plt_defaults(font="Arial", fontdir=os.path.join(os.path.expanduser('~'),'Projects/fonts'))

class LinearNN(nn.Module):
	def __init__(self,inp_size,out_size=1):
		super(LinearNN,self).__init__()
		self.inp_size = inp_size
		self.out_size = out_size
		self.fc = nn.Linear(self.inp_size,self.out_size,bias=False)

	def forward(self, x):
		out = self.fc(x)
		return out

	def zero_gradients(self):
		if self.fc.weight.grad is not None:
			self.fc.weight.grad.data.zero_()
			self.fc.bias.grad.data.zero_()

def create_data_labels(inp_dim=10,num_data=100,seed=13,Sigma_arr=None):
	torch.manual_seed(seed)
	np.random.seed(seed)
	if Sigma_arr is None: Sigma_arr=[1]*inp_dim
	# X = torch.rand((num_data,inp_dim))
	X = gen_dataset(inp_dim=inp_dim,num_data=num_data,Sigma_arr=Sigma_arr,verbose=False,seed=seed)
	X.requires_grad_ = False
	teacher_net = LinearNN(inp_size=inp_dim)
	with torch.no_grad():
		Y = teacher_net(X)
	# loss_fn = torch.nn.MSELoss()
	return inp_dim,num_data,X,Y,teacher_net

def gen_dataset(inp_dim, num_data=100, Sigma_arr=None, alpha=1.0, beta=1.1, seed=13, verbose=True):
	np.random.seed(seed)
	torch.manual_seed(seed)
	num_samples = num_data
	# assert num_samples <= inp_dim, "Not operating in N < d regime!"
	V, _ = np.linalg.qr(np.random.randn(inp_dim, inp_dim))
	if Sigma_arr is None:
	  Sigma = np.diag(np.sqrt(np.array([i**-alpha for i in range(1, inp_dim+1)]).reshape(inp_dim)))
	  # Sigma = np.diag(np.sqrt(np.array([(i**-alpha)/(np.log(i+1)**beta) for i in range(1, dim+1)]).reshape(dim)))
	  Sigma *= 100./np.sqrt(np.sum(Sigma[:num_samples,:num_samples]**2))
	else:
	  if 'ndarray' in Sigma_arr.__class__.__name__: Sigma_arr = Sigma_arr.tolist()
	  assert len(Sigma_arr)==inp_dim, "Sigma_arr ({}) should have same length as inp_dim ({})".format(len(Sigma_arr),inp_dim)
	  Sigma_arr.extend([0]*(num_samples-inp_dim))
	  Sigma = np.diag(np.array(Sigma_arr).astype('float64'))
	# z = np.random.randn(dim, num_samples)     # Doing a slight change here - Arna
	z,_ = np.linalg.qr(np.random.randn(num_samples,num_samples))
	# z_tr = z[:,:ntrain]
	# z_te = z[:,ntrain:ntrain+ntest]
	X = V @ Sigma[:inp_dim,:num_samples] @ z.T
	# train = V @ Sigma[:,:ntrain] @ z_tr.T
	# test = V @ Sigma[:,:ntest] @ z_te.T
	# print(np.sum(Sigma**2))
	if verbose:
		print(np.sum(np.diag(X @ X.T)))
	# print(num_samples,V.shape,Sigma.shape,z.shape,X.shape)
	return torch.from_numpy(X.T)

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
		#   print(v.shape)
		W = W.T
	return W

def load_student_net(inp_dim,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = LinearNN(inp_size=inp_dim)
	return student_net

def get_loss(net,X,y,loss_fn):
	with torch.no_grad():
		Y_hat = net(X)
		loss = loss_fn(Y_hat,y)
	return loss.item()

def gradient_descent_one_step(net,lr,bias=0.0,variance=0.0,grad_noise=None,no_update=False):
	params = list(net.parameters())
	grad_list = []
	for p in params:
		grad_list.append(p.grad.data.flatten())
	grad_vector = torch.cat(grad_list)
	accum_wt_change = 0
	accum_bias_change = 0
	bias = bias*grad_vector.norm()
	if grad_noise is None:
		grad_noise = torch.normal(mean=-bias*torch.ones(grad_vector.size()),std=np.sqrt(variance)*torch.ones(grad_vector.size())).to(grad_vector.device)
	else:
		grad_noise = (variance*grad_vector.norm())*torch.div(grad_noise,torch.norm(grad_noise))
		grad_noise -= bias
	# NOTE: -ve sign in bias above because this will be added to the gradient and thereafter subtracted from current wt.
	# Hence this has reverse effect of the theory assumption!
	new_grad_vector = grad_vector+grad_noise
	param_pointer = 0
	for p in params:
		p_size = p.flatten().size(0)
		new_grad = new_grad_vector[param_pointer:param_pointer+p_size].view(p.size())
		p.grad.data = new_grad
		param_pointer += p_size

	if not no_update:
		for p in params:
			p.data -= lr*p.grad
		# net.fc.weight.data -= lr*net.fc.weight.grad
		# net.fc.bias.data -= lr*net.fc.bias.grad
	return (grad_vector, grad_noise)
  
def train_one_step(net,X,y,loss_fn,lr,bias=0.0,variance=0.0,grad_noise=None,verbose=True,no_update=False):
	net.zero_grad()
	Y_hat = net(X)
	loss = loss_fn(Y_hat,y)
	if verbose:
		# print("Weight 1: {}".format(net.fc.weight.data))
		print("Loss 1: {}".format(loss.item()))
	loss.backward()
	grad = gradient_descent_one_step(net,lr=lr,bias=bias,variance=variance,grad_noise=grad_noise,no_update=no_update)

	with torch.no_grad():
		Y_hat = net(X)
		loss_val = loss_fn(Y_hat,y)
		if verbose:
			# print("Weight 2: {}".format(net.fc.weight.data))
			print("Loss 2: {}".format(loss_val.item()))
	return grad, loss.item()-loss_val.item()

def calculate_hessian(net,X,y,loss_fn):
	params_list = [p.view(-1) for p in net.parameters()]
	def loss_function_hess(W_b):
		return loss_fn(torch.nn.functional.linear(X,W_b[:-1].view(1,-1),W_b[-1:]),y)
	def loss_function_hess_nobias(W_b):
		return loss_fn(torch.nn.functional.linear(X,W_b.view(1,-1),torch.Tensor([0])),y)
	if net.fc.bias:
		grad2_W = torch.autograd.functional.hessian(loss_function_hess,torch.hstack(params_list))
	else:
		grad2_W = torch.autograd.functional.hessian(loss_function_hess_nobias,torch.hstack(params_list))
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
		print(grad_decrease,grad2_decrease)
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

def generate_random_noise_sequence(repeats,inp_dim,variance,seed=7):
	# torch.manual_seed(seed)
	# np.random.seed(seed)
	# torch.random.seed()
	noise_vectors = torch.randn((repeats-1,1+inp_dim))
	noise_vectors = (variance)*torch.div(noise_vectors,torch.norm(noise_vectors,dim=1).unsqueeze(1))
	noise_vectors = torch.vstack([noise_vectors,-noise_vectors.sum(dim=0)])    # to ensure that sum of random noise vectors is 0
	return noise_vectors

def plot_colormap(data_dict,eigs0_arr_plot,eigsN_arr_plot,variance=1,lr=0.01):
	data_arr = np.empty((len(eigs0_arr_plot.keys()),len(eigsN_arr_plot.keys())))
	data_arr[:] = np.nan
	for idx0,e0 in enumerate(eigs0_arr_plot.keys()):
		for idx1,eN in enumerate(eigsN_arr_plot.keys()):
			# print(idx0,e0,idx1,eN)
			try:
				data_arr[idx1,idx0] = data_dict[e0][eN]
			except:
				pass

	variance = variance
	plt.figure(figsize=(15,8))
	plt.imshow(data_arr,cmap='coolwarm',vmin=0,vmax=1,origin='lower')
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=15)
	cbar.set_label('Empirical Descent probability',rotation=270,fontsize=20,labelpad=30)
	eigs0_array_np = np.array(list(eigs0_arr_plot.keys()))
	eigsN_array_np = np.array(list(eigsN_arr_plot.keys()))
	plt.xticks(np.arange(0,len(eigs0_array_np),3),eigs0_array_np[::3],fontsize=15)
	plt.yticks(np.arange(0,len(eigsN_array_np),3),eigsN_array_np[::3],fontsize=15)
	plt.xlabel(r'$\lambda_1$',fontsize=20)
	plt.ylabel(r'$\lambda_N$',fontsize=20)
	theory_lim = (2./lr)*(1./(1+variance))   # variance is added relative to grad norm in gradient_one_step function
	xlim0, xlim1 = plt.xlim()
	slope = ((xlim1-xlim0)/(np.sqrt(list(eigs0_arr_plot.keys())[-1])-np.sqrt(list(eigs0_arr_plot.keys())[0])))
	theory_lim_pos = slope*(np.sqrt(theory_lim)-np.sqrt(list(eigs0_arr_plot.keys())[0])) + xlim0
	plt.axvline(x=theory_lim_pos,ls=':',color='k',lw=5,label='Theoretical limit')
	plt.axhline(y=theory_lim_pos,ls=':',color='k',lw=5)
	plt.legend(fontsize=15)
	plt.title('INR = {}'.format(variance),fontsize=24)

def run_experiment_theorem_validation(lr=0.01,repeats=10,inp_dim=10,num_data=100,seeds=100,eig0=5,eigN=1):
	Sigma_arr = np.linspace(eigN,eig0,inp_dim)
	frob_norm_factor_approx = np.sqrt(np.sum(4*Sigma_arr**4))*lr
	num_params_teacher = inp_dim
	theoretical_limit_approx = (1/(1+1.5*frob_norm_factor_approx))*(1/np.sqrt(num_params_teacher))
	bias_INR_arr = np.linspace(0,(1+int(np.sqrt(inp_dim)))*theoretical_limit_approx,40)
	loss_fn_sum = nn.MSELoss(reduction='sum')

	pos_loss_decrease_arr = []

	for bias in tqdm(bias_INR_arr):
		pos_loss_decrease_count = 0
		for r in range(1,1+repeats):
			inp_dim,num_data,X,Y,teacher_net = create_data_labels(inp_dim=inp_dim,num_data=num_data,seed=r,Sigma_arr=Sigma_arr)  #inp_dim=10,num_data=100
			for sidx in range(seeds):
				student_net = load_student_net(inp_dim=inp_dim,seed=sidx)
				# assert not torch.allclose(teacher_net.fc.weight,student_net.fc.weight), "Teacher and Student have the same weight!"
				if torch.allclose(teacher_net.fc.weight,student_net.fc.weight): pos_loss_decrease_count +=1; continue;
				grad2_W = calculate_hessian(net=student_net,X=X,y=Y,loss_fn=loss_fn_sum)
				(grad_true,grad_corruption), loss_decrease = train_one_step(net=student_net,X=X,y=Y,loss_fn=loss_fn_sum,lr=lr,
																			bias=bias,variance=0.0,grad_noise=None,verbose=False)
				pos_loss_decrease_count += (loss_decrease>=0)
		pos_loss_decrease_arr.append(pos_loss_decrease_count/(seeds*repeats))

	num_params = sum(p.numel() for p in student_net.parameters())
	frob_norm_factor = grad2_W.norm('fro')*lr
	lim_frob_term = (1+1.5*frob_norm_factor)
	lim_frob_term_detail = (1+frob_norm_factor+np.sqrt(1+4*frob_norm_factor+frob_norm_factor**2))/2
	theoretical_limit = (1/lim_frob_term_detail)*(1/np.sqrt(num_params))
	theoretical_limit_stricter = theoretical_limit*(num_params**0.25)
	return bias_INR_arr,pos_loss_decrease_arr,theoretical_limit, theoretical_limit_stricter

def plot_result(bias_INR_arr,pos_loss_decrease_arr,theoretical_limit,theoretical_limit_stricter):
	plt.figure(figsize=(9,6))
	# plt.plot(bias_INR_arr,pos_loss_decrease_arr,lw=2,marker='o')
	plt.scatter(x=bias_INR_arr,y=pos_loss_decrease_arr,c=pos_loss_decrease_arr,marker='o',cmap='coolwarm',vmin=0,vmax=1)
	plt.axvline(x=theoretical_limit,color='k',ls='--',lw=2) #,label='Theoretical Limit')
	plt.axvline(x=theoretical_limit_stricter,color='gray',ls='--',lw=2) #,label='Theoretical Limit')
	# plt.axhline(y=1.0,color='k',ls='--')
	plt.ylabel('Empirical Likelihood of descent',fontsize=24)
	plt.xlabel(r'$\frac{b}{\|\nabla_w \mathcal{L} \|}$',fontsize=24)
	# plt.legend(fontsize=20)

if __name__=='__main__':
	lr = 0.001
	inp_dim = 1
	num_data = 1000

	bias_INR_arr, prob_loss_decrease_arr, theoretical_lim, theoretical_lim_stricter = run_experiment_theorem_validation(lr=lr,repeats=20,inp_dim=inp_dim,
																							num_data=num_data,seeds=20,
																							eig0=5,eigN=5)

	plot_result(bias_INR_arr,prob_loss_decrease_arr,theoretical_lim,theoretical_lim_stricter)
	plt.show()
