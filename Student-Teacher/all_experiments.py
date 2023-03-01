import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
torch.set_default_tensor_type(torch.DoubleTensor)
from tqdm import tqdm
# from tqdm.contrib import itertools
import itertools
from models import LinearNN, MultiLayerNN
from utils import *

def biased_training_linear(dataset_inits=5,random_inits=5,inp_dim=25,projection_dim=None,num_data=100,epochs=10,lr=0.01,bias=0.1):
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	# breakpoint()
	for d,ridx in itertools.product(range(dataset_inits),range(random_inits)):
	# for d,ridx in tqdm(itertools.product(range(dataset_inits),range(random_inits))):
		inp_dim,num_data,X,Y,teacher_net = create_data_labels(inp_dim=inp_dim,num_data=num_data,seed=d)
		if projection_dim is None or projection_dim==inp_dim:
			W_projection = None
		else:
			W_projection = get_projection_matrix(inp_dim=inp_dim,projection_dim=projection_dim,seed=d)
			X = torch.matmul(X,W_projection)
			# inp_dim = projection_dim
		# breakpoint()
		# covar = X.T@X
		# print(covar.shape,torch.sum(covar))
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		grad_norms_arr = []
		student_net_control = load_student_net(inp_dim=inp_dim,projection_weights=W_projection,seed=ridx)
		student_net = load_student_net(inp_dim=inp_dim,projection_weights=W_projection,seed=ridx)
		assert not check_model_params_equal(teacher_net,student_net), "Teacher and Student have the same weight!"
		loss0 = get_loss(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss())
		tr_loss_arr.append(loss0)
		tr_loss_arr_control.append(loss0)
		tr_loss_decrease_arr_control.append(loss0)
		for epoch in range(epochs):
			copy_from_model_to_model(student_net_control,student_net)
			_, loss_decrease = train_one_step(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=bias,verbose=False)
			tr_loss_arr.append(get_loss(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss()))
			(grad,_), loss_decrease_control = train_one_step(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,verbose=False)
			tr_loss_decrease_arr_control.append(loss_decrease_control)
			tr_loss_arr_control.append(get_loss(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss()))
			grad_norms_arr.append(torch.norm(grad).item())

		tr_loss_arr = np.array(tr_loss_arr)
		tr_loss_arr_inits.append(tr_loss_arr)
		tr_loss_arr_control = np.array(tr_loss_arr_control)
		tr_loss_arr_control_inits.append(tr_loss_arr_control)
		tr_loss_decrease_arr_control = np.array(tr_loss_decrease_arr_control)
		tr_loss_decrease_arr_control_inits.append(tr_loss_decrease_arr_control)
		grad_norms_inits.append(grad_norms_arr)

	tr_loss_arr_inits = np.array(tr_loss_arr_inits)
	tr_loss_arr_control_inits = np.array(tr_loss_arr_control_inits)
	tr_loss_decrease_arr_control_inits = np.array(tr_loss_decrease_arr_control_inits)
	grad_norms_inits = np.array(grad_norms_inits)
	return tr_loss_arr_inits, tr_loss_arr_control_inits, tr_loss_decrease_arr_control_inits, grad_norms_inits


def noisy_training_linear(dataset_inits=5,random_inits=5,inp_dim=25,projection_dim=None,num_data=100,epochs=10,lr=0.01,variance=0.1):
	repeats = 50
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	# breakpoint()
	for d,ridx in itertools.product(range(dataset_inits),range(random_inits)):
	# for d,ridx in tqdm(itertools.product(range(dataset_inits),range(random_inits))):
		inp_dim,num_data,X,Y,teacher_net = create_data_labels(inp_dim=inp_dim,num_data=num_data,seed=d)
		if projection_dim is None or projection_dim==inp_dim:
			W_projection = None
			projection_dim = inp_dim
		else:
			W_projection = get_projection_matrix(inp_dim=inp_dim,projection_dim=projection_dim,seed=d)
			X = torch.matmul(X,W_projection)
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		grad_norms_arr = []
		student_net_control = load_student_net(inp_dim=inp_dim,projection_weights=W_projection,seed=ridx)
		student_net = load_student_net(inp_dim=inp_dim,projection_weights=W_projection,seed=ridx)
		assert not check_model_params_equal(teacher_net,student_net), "Teacher and Student have the same weight!"
		loss0 = get_loss(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss())
		tr_loss_arr.append(loss0)
		tr_loss_arr_control.append(loss0)
		tr_loss_decrease_arr_control.append(loss0)
		for epoch in range(epochs):
			mean_new_loss = 0
			grad_noise_vectors = generate_random_noise_sequence(repeats=repeats,inp_dim=projection_dim,variance=variance)
			for r in range(repeats):
				copy_from_model_to_model(student_net_control,student_net)	
				_, loss_decrease = train_one_step(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,
												variance=variance,grad_noise=grad_noise_vectors[r],verbose=False)
				mean_new_loss += get_loss(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss())
			mean_new_loss = mean_new_loss/repeats
			tr_loss_arr.append(mean_new_loss)
			(grad,_), loss_decrease_control = train_one_step(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,verbose=False)
			tr_loss_decrease_arr_control.append(loss_decrease_control)
			tr_loss_arr_control.append(get_loss(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss()))
			grad_norms_arr.append(torch.norm(grad).item())

		tr_loss_arr = np.array(tr_loss_arr)
		tr_loss_arr_inits.append(tr_loss_arr)
		tr_loss_arr_control = np.array(tr_loss_arr_control)
		tr_loss_arr_control_inits.append(tr_loss_arr_control)
		tr_loss_decrease_arr_control = np.array(tr_loss_decrease_arr_control)
		tr_loss_decrease_arr_control_inits.append(tr_loss_decrease_arr_control)
		grad_norms_inits.append(grad_norms_arr)

	tr_loss_arr_inits = np.array(tr_loss_arr_inits)
	tr_loss_arr_control_inits = np.array(tr_loss_arr_control_inits)
	tr_loss_decrease_arr_control_inits = np.array(tr_loss_decrease_arr_control_inits)
	grad_norms_inits = np.array(grad_norms_inits)
	return tr_loss_arr_inits, tr_loss_arr_control_inits, tr_loss_decrease_arr_control_inits, grad_norms_inits


def vary_bias_over_training_linear(inp_dim=25,num_data=100,epochs=10):
	plt.figure('Impact of varying bias over training - LinearNN',figsize=(16,11))
	lr = 2e-2 #0.02
	bias_range = np.linspace(0.2,0.5,3) #[0.01,0.1,0.5]
	# bias_range = [0.5]
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)
	# plt.figure(); ax1 = plt.axes(); 
	grad_norms_bias_arr = []
	tr_loss_bias_arr = []
	tr_loss_control_bias_arr = []
	tr_loss_decrease_control_bias_arr = []
	for bidx,bias in enumerate(tqdm(bias_range)):
		#try random initializations
		dataset_inits = 40
		random_inits = 40
		res = biased_training_linear(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,num_data=num_data,
									epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		plt.sca(ax1)
		if bidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',label='Unbiased gradient')
		for pidx in range(tr_loss_arr_inits.shape[1]-1):
			if pidx==0:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--',label='Bias = {:.2f}'.format(bias))
			else:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--')
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
		# 	color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		plt.sca(ax3)
		if bidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',label='Unbiased gradient')
		for pidx in range(tr_loss_arr_inits.shape[1]-1):
			if pidx==0:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--',label='Bias = {:.2f}'.format(bias))
			else:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--')
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
		# 	color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		plt.sca(ax2)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)/tr_loss_decrease_arr_control_inits
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		plt.sca(ax4)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		if bidx==0:
			tr_loss_control_bias_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_bias_arr.append(tr_loss_decrease_arr_control_inits)
		tr_loss_bias_arr.append(tr_loss_arr_inits)
		grad_norms_bias_arr.append(grad_norms_inits)

	tr_loss_bias_arr = np.array(tr_loss_bias_arr)
	tr_loss_control_bias_arr = np.array(tr_loss_control_bias_arr)
	tr_loss_decrease_control_bias_arr = np.array(tr_loss_decrease_control_bias_arr)
	grad_norms_bias_arr = np.array(grad_norms_bias_arr)
	plt.sca(ax1)
	# plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)
	plt.grid('on')
	plt.xlim(right=0.5+epochs)

	plt.sca(ax2)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to bias in gradient',fontsize=16)
	plt.legend(fontsize=15)
	plt.grid('on')

	plt.sca(ax3)
	# plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)
	plt.grid('on')

	plt.sca(ax4)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to bias in gradient',fontsize=16)
	plt.legend(fontsize=15)
	plt.grid('on')

	plt.figure('Impact of bias vs grad norm - LinearNN',figsize=(16,6))
	error_loss_bias_arr = (tr_loss_bias_arr-tr_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]/tr_loss_decrease_control_bias_arr[...,1:]
	# print(tr_loss_decrease_control_bias_arr[...,1:].shape,error_loss_bias_arr.shape,grad_norms_bias_arr.shape)
	plt.subplot(121)
	for bidx,bias in enumerate(bias_range):
		# sc_plot_x = 1./grad_norms_bias_arr[bidx].flatten()
		sc_plot_x = grad_norms_bias_arr[bidx].flatten()
		sc_plot_y = error_loss_bias_arr[bidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[bidx%len(color_range)],marker='o',alpha=0.4,
			label='Bias = {:.2f}'.format(bias))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)

	error_loss_bias_arr = (tr_loss_bias_arr-tr_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]
	plt.subplot(122)
	for bidx,bias in enumerate(bias_range):
		# sc_plot_x = 1./grad_norms_bias_arr[bidx].flatten()
		sc_plot_x = grad_norms_bias_arr[bidx].flatten()
		sc_plot_y = error_loss_bias_arr[bidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[bidx%len(color_range)],marker='o',alpha=0.4,
			label='Bias = {:.2f}'.format(bias))  
		# plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
		plt.scatter(sc_plot_x[sc_plot_y<0],(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.axhline(y=0,color='k',ls='--')
	# plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	# plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)


def vary_netSize_fixed_bias_over_training_linear(inp_dim=25,num_data=1000,bias=0.1):
	plt.figure('Impact of varying network size with fixed bias over training - LinearNN',figsize=(19,6))
	lr = 1e-2 #0.02
	epochs = 10
	# bias_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.append(np.array([inp_dim//2]),np.linspace(inp_dim,5*inp_dim,6).astype(int)) #[0.01,0.1,0.5]
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	# ax3 = plt.subplot(223)
	# ax4 = plt.subplot(224)
	grad_norms_bias_arr = []
	tr_loss_bias_arr = []
	tr_loss_control_bias_arr = []
	tr_loss_decrease_control_bias_arr = []
	for sidx,size in enumerate(tqdm(size_range)):
		#try random initializations
		dataset_inits = 20
		random_inits = 50
		res = biased_training_linear(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,projection_dim=size,
									num_data=num_data,epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		plt.sca(ax1)
		if size==inp_dim:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',marker='x',label='Unbiased gradient')
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		# plt.sca(ax3)
		# if sidx==0:
		# 	plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
		# 		y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
		# 		color='k',marker='x',label='Unbiased gradient')
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
		# 	color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		plt.sca(ax2)
		# error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)/tr_loss_decrease_arr_control_inits
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
		# 	color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		# plt.sca(ax4)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=np.nanmean(error_loss_arr_inits,axis=0),yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		tr_loss_control_bias_arr.append(tr_loss_arr_control_inits)
		tr_loss_decrease_control_bias_arr.append(tr_loss_decrease_arr_control_inits)
		tr_loss_bias_arr.append(tr_loss_arr_inits)
		grad_norms_bias_arr.append(grad_norms_inits)

	tr_loss_bias_arr = np.array(tr_loss_bias_arr)
	tr_loss_control_bias_arr = np.array(tr_loss_control_bias_arr)
	tr_loss_decrease_control_bias_arr = np.array(tr_loss_decrease_control_bias_arr)
	grad_norms_bias_arr = np.array(grad_norms_bias_arr)
	plt.sca(ax1)
	plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	# plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	# plt.xlabel('Epochs',fontsize=14)
	# plt.title('Increase in loss due to bias in gradient',fontsize=16)
	# plt.legend(fontsize=15)

	# plt.sca(ax3)
	# plt.yscale('log')
	# plt.ylabel('Loss',fontsize=14)
	# plt.xlabel('Epochs',fontsize=14)
	# plt.title('Loss vs epochs',fontsize=16)
	# plt.legend(fontsize=15)

	# plt.sca(ax4)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to bias in gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.figure('Impact of fixed bias vs grad norm varying network size - LinearNN',figsize=(16,6))
	error_loss_bias_arr = (tr_loss_bias_arr-tr_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]/tr_loss_decrease_control_bias_arr[...,1:]
	# print(tr_loss_decrease_control_bias_arr[...,1:].shape,error_loss_bias_arr.shape,grad_norms_bias_arr.shape)
	plt.subplot(121)
	for sidx,size in enumerate(size_range):
		# sc_plot_x = 1./grad_norms_bias_arr[sidx].flatten()
		sc_plot_x = grad_norms_bias_arr[sidx].flatten()
		sc_plot_y = error_loss_bias_arr[sidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='NetSize = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)

	error_loss_bias_arr = (tr_loss_bias_arr-tr_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]
	plt.subplot(122)
	for sidx,size in enumerate(size_range):
		# sc_plot_x = 1./grad_norms_bias_arr[sidx].flatten()
		sc_plot_x = grad_norms_bias_arr[sidx].flatten()
		sc_plot_y = error_loss_bias_arr[sidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='NetSize = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)


def vary_variance_over_training_linear(inp_dim=25,num_data=100):
	plt.figure('Impact of varying variance over training - LinearNN',figsize=(16,11))
	lr = 2e-1 #0.02
	epochs = 10
	variance_range = np.linspace(0.1,0.6,3) #[0.01,0.1,0.5]
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)
	grad_norms_variance_arr = []
	tr_loss_variance_arr = []
	tr_loss_control_variance_arr = []
	tr_loss_decrease_control_variance_arr = []
	for vidx,variance in enumerate(tqdm(variance_range)):
		#try random initializations
		dataset_inits = 20
		random_inits = 20
		res = noisy_training_linear(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,num_data=num_data,
									epochs=epochs,lr=lr,variance=variance)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		plt.sca(ax1)
		if vidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',label='True gradient')
		for pidx in range(tr_loss_arr_inits.shape[1]-1):
			if pidx==0:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[vidx%len(color_range)],ls='--',label='Variance = {:.2f}'.format(variance))
			else:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[vidx%len(color_range)],ls='--')
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
		# 	color=color_range[vidx%len(color_range)],label='Variance = {:.2f}'.format(variance))

		plt.sca(ax3)
		if vidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',label='True gradient')
		for pidx in range(tr_loss_arr_inits.shape[1]-1):
			if pidx==0:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[vidx%len(color_range)],ls='--',label='Variance = {:.2f}'.format(variance))
			else:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[tr_loss_arr_control_inits[:,pidx].mean(axis=0),tr_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,tr_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0])],
							color=color_range[vidx%len(color_range)],ls='--')
		# plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
		# 	y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
		# 	color=color_range[vidx%len(color_range)],label='Variance = {:.2f}'.format(variance))

		plt.sca(ax2)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)/tr_loss_decrease_arr_control_inits
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[vidx%len(color_range)],label='Variance = {:.2f}'.format(variance))

		plt.sca(ax4)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[vidx%len(color_range)],label='Variance = {:.2f}'.format(variance))

		if vidx==0:
			tr_loss_control_variance_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_variance_arr.append(tr_loss_decrease_arr_control_inits)
		tr_loss_variance_arr.append(tr_loss_arr_inits)
		grad_norms_variance_arr.append(grad_norms_inits)

	tr_loss_variance_arr = np.array(tr_loss_variance_arr)
	tr_loss_control_variance_arr = np.array(tr_loss_control_variance_arr)
	tr_loss_decrease_control_variance_arr = np.array(tr_loss_decrease_control_variance_arr)
	grad_norms_variance_arr = np.array(grad_norms_variance_arr)
	plt.sca(ax1)
	# plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to variance in gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax3)
	# plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax4)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to variance in gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.figure('Impact of variance vs grad norm - LinearNN',figsize=(16,6))
	error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
	error_loss_variance_arr = error_loss_variance_arr[...,1:]/tr_loss_decrease_control_variance_arr[...,1:]
	# print(tr_loss_decrease_control_variance_arr[...,1:].shape,error_loss_variance_arr.shape,grad_norms_variance_arr.shape)
	plt.subplot(121)
	for vidx,variance in enumerate(variance_range):
		# sc_plot_x = 1./grad_norms_variance_arr[vidx].flatten()
		sc_plot_x = grad_norms_variance_arr[vidx].flatten()
		sc_plot_y = error_loss_variance_arr[vidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[vidx%len(color_range)],marker='o',alpha=0.4,
			label='Variance = {:.2f}'.format(variance))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[vidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)

	error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
	error_loss_variance_arr = error_loss_variance_arr[...,1:]
	plt.subplot(122)
	for vidx,variance in enumerate(variance_range):
		# sc_plot_x = 1./grad_norms_variance_arr[vidx].flatten()
		sc_plot_x = grad_norms_variance_arr[vidx].flatten()
		sc_plot_y = error_loss_variance_arr[vidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[vidx%len(color_range)],marker='o',alpha=0.4,
			label='Variance = {:.2f}'.format(variance))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[vidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)


def vary_netSize_fixed_variance_over_training_linear(inp_dim=25,num_data=100,variance=0.1):
	plt.figure('Impact of varying network size with fixed variance over training - LinearNN',figsize=(16,11))
	lr = 1e-3 #0.02
	epochs = 10
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.linspace(inp_dim,5*inp_dim,5).astype(int) #[0.01,0.1,0.5]
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)
	grad_norms_variance_arr = []
	tr_loss_variance_arr = []
	tr_loss_control_variance_arr = []
	tr_loss_decrease_control_variance_arr = []
	for sidx,size in enumerate(tqdm(size_range)):
		#try random initializations
		dataset_inits = 20
		random_inits = 20
		res = noisy_training_linear(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,projection_dim=size,
									num_data=num_data,epochs=epochs,lr=lr,variance=variance)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		plt.sca(ax1)
		if sidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',marker='x',label='True gradient')
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		plt.sca(ax3)
		if sidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',marker='x',label='True gradient')
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=tr_loss_arr_inits.mean(axis=0),yerr=tr_loss_arr_inits.std(axis=0)/np.sqrt(tr_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		plt.sca(ax2)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)/tr_loss_decrease_arr_control_inits
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		plt.sca(ax4)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[sidx%len(color_range)],label='NetSize = {}'.format(size))

		if sidx==0:
			tr_loss_control_variance_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_variance_arr.append(tr_loss_decrease_arr_control_inits)
		tr_loss_variance_arr.append(tr_loss_arr_inits)
		grad_norms_variance_arr.append(grad_norms_inits)

	tr_loss_variance_arr = np.array(tr_loss_variance_arr)
	tr_loss_control_variance_arr = np.array(tr_loss_control_variance_arr)
	tr_loss_decrease_control_variance_arr = np.array(tr_loss_decrease_control_variance_arr)
	grad_norms_variance_arr = np.array(grad_norms_variance_arr)
	plt.sca(ax1)
	plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to variance in gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax3)
	plt.yscale('log')
	plt.ylabel('Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax4)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in loss due to variance in gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.figure('Impact of fixed variance vs grad norm varying network size - LinearNN',figsize=(16,6))
	error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
	error_loss_variance_arr = error_loss_variance_arr[...,1:]/tr_loss_decrease_control_variance_arr[...,1:]
	# print(tr_loss_decrease_control_variance_arr[...,1:].shape,error_loss_variance_arr.shape,grad_norms_variance_arr.shape)
	plt.subplot(121)
	for sidx,size in enumerate(size_range):
		# sc_plot_x = 1./grad_norms_variance_arr[sidx].flatten()
		sc_plot_x = grad_norms_variance_arr[sidx].flatten()
		sc_plot_y = error_loss_variance_arr[sidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='NetSize = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)

	error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
	error_loss_variance_arr = error_loss_variance_arr[...,1:]
	plt.subplot(122)
	for sidx,size in enumerate(size_range):
		# sc_plot_x = 1./grad_norms_variance_arr[sidx].flatten()
		sc_plot_x = grad_norms_variance_arr[sidx].flatten()
		sc_plot_y = error_loss_variance_arr[sidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='NetSize = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs norm of gradient',fontsize=16)
	plt.legend(fontsize=15)


def vary_netSize_vary_variance_over_training_linear(inp_dim=25,num_data=100):
	plt.figure('Impact of varying network size with varying variance over training - LinearNN',figsize=(12,6))
	lr = 1e-3 #0.02
	epochs = 5
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.linspace(inp_dim//2,5*inp_dim,10).astype(int) #[0.01,0.1,0.5]
	variance_range = np.linspace(0.1,0.4,4)
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(111)
	for vidx,variance in enumerate(tqdm(variance_range)):
		grad_norms_variance_arr = []
		tr_loss_variance_arr = []
		tr_loss_control_variance_arr = []
		tr_loss_decrease_control_variance_arr = []
		for sidx,size in enumerate(tqdm(size_range)):
			#try random initializations
			dataset_inits = 10
			random_inits = 10
			res = noisy_training_linear(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,projection_dim=size,
										num_data=num_data,epochs=epochs,lr=lr,variance=variance)
			tr_loss_arr_inits = res[0]
			tr_loss_arr_control_inits = res[1]
			tr_loss_decrease_arr_control_inits = res[2]
			grad_norms_inits = res[3]

			tr_loss_control_variance_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_variance_arr.append(tr_loss_decrease_arr_control_inits)
			tr_loss_variance_arr.append(tr_loss_arr_inits)
			grad_norms_variance_arr.append(grad_norms_inits)

		tr_loss_variance_arr = np.array(tr_loss_variance_arr)
		tr_loss_control_variance_arr = np.array(tr_loss_control_variance_arr)
		tr_loss_decrease_control_variance_arr = np.array(tr_loss_decrease_control_variance_arr)
		grad_norms_variance_arr = np.array(grad_norms_variance_arr)
		
		plt.sca(ax1)
		# error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
		# error_loss_variance_arr = error_loss_variance_arr[...,1:]/tr_loss_decrease_control_variance_arr[...,1:]
		# # print(tr_loss_decrease_control_variance_arr[...,1:].shape,error_loss_variance_arr.shape,grad_norms_variance_arr.shape)
		# error_loss_variance_arr_total = np.nanmean(error_loss_variance_arr,axis=-1)
		# plt.errorbar(x=size_range,y=np.nanmean(error_loss_variance_arr_total,axis=-1),
		# 			yerr=np.nanstd(error_loss_variance_arr_total,axis=-1)/np.sqrt(error_loss_variance_arr_total.shape[-1]),
		# 			color=color_range[vidx%len(color_range)],label='Variance={:.2f}'.format(variance))

		error_loss_variance_arr = (tr_loss_variance_arr-tr_loss_control_variance_arr)
		error_loss_variance_arr = error_loss_variance_arr[...,1:]
		# plt.sca(ax2)
		error_loss_variance_arr_total = np.nanmean(error_loss_variance_arr,axis=-1)
		plt.errorbar(x=size_range,y=np.nanmean(error_loss_variance_arr_total,axis=-1),
					yerr=np.nanstd(error_loss_variance_arr_total,axis=-1)/np.sqrt(error_loss_variance_arr_total.shape[-1]),
					color=color_range[vidx%len(color_range)],label='Variance={:.2f}'.format(variance))
		
	plt.sca(ax1)
	# plt.axvline(x=inp_dim,color='k',ls=':',label='Teacher Net')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	# plt.yscale('log')
	# plt.xlabel('Network size',fontsize=14)
	# plt.xscale('log')
	# plt.title('Impact of variance vs Network size',fontsize=16)
	# plt.legend(fontsize=15)

	# plt.sca(ax2)
	plt.axvline(x=inp_dim,color='k',ls=':',label='Teacher Net')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Network size',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)


if __name__=='__main__':
	run_local = True
	# vary_bias_over_training_linear(inp_dim=25,num_data=500)
	# vary_netSize_fixed_bias_over_training_linear(inp_dim=25,num_data=4000,bias=0.2)

	# vary_variance_over_training_linear(inp_dim=15,num_data=500)
	# vary_netSize_fixed_variance_over_training_linear(inp_dim=25,num_data=100,variance=0.05)
	vary_netSize_vary_variance_over_training_linear(inp_dim=25,num_data=500)
	plt.grid('on')
	if run_local:
		plt.show()
	else:
		save_all_figs_to_disk()