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

def noisy_training_multiLayer(dataset_inits=5,random_inits=5,inp_dim=25,teacher_layers=4,non_linearity=None,num_data=100,
							student_layers=4,student_hidden_dim=25,epochs=10,lr=0.01,variance=0.01):
	repeats = 15
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	# breakpoint()
	for d,ridx in itertools.product(range(dataset_inits),range(random_inits)):
	# for d,ridx in tqdm(itertools.product(range(dataset_inits),range(random_inits))):
		inp_dim,num_data,X,Y,teacher_net = create_NL_data_labels(inp_dim=inp_dim,num_layers=teacher_layers,
																non_linearity=non_linearity,num_data=num_data,seed=d)
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		grad_norms_arr = []
		student_NL_net = load_multilayer_student_net(inp_dim=inp_dim,num_layers=student_layers,hidden_dim=student_hidden_dim,
													non_linearity=non_linearity,seed=ridx)
		student_NL_net_control = load_multilayer_student_net(inp_dim=inp_dim,num_layers=student_layers,hidden_dim=student_hidden_dim,
													non_linearity=non_linearity,seed=ridx)

		assert not check_model_params_equal(teacher_net,student_NL_net), "Teacher and Student have the same weight!"
		loss0 = get_loss(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss())
		tr_loss_arr.append(loss0)
		tr_loss_arr_control.append(loss0)
		tr_loss_decrease_arr_control.append(loss0)
		for epoch in range(epochs):
			mean_new_loss = 0
			grad_noise_vectors = generate_random_noise_sequence(repeats=repeats,inp_dim=sum(p.numel() for p in student_NL_net.parameters())-1,variance=variance)
			for r in range(repeats):
				copy_from_model_to_model(student_NL_net_control,student_NL_net)	
				_, loss_decrease = train_one_step(net=student_NL_net,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,
												variance=variance,grad_noise=grad_noise_vectors[r],verbose=False)
				mean_new_loss += get_loss(net=student_NL_net,X=X,y=Y,loss_fn=nn.MSELoss())
			mean_new_loss = mean_new_loss/repeats
			tr_loss_arr.append(mean_new_loss)
			(grad,_), loss_decrease_control = train_one_step(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,variance=0.0,verbose=False)
			tr_loss_decrease_arr_control.append(loss_decrease_control)
			tr_loss_arr_control.append(get_loss(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss()))
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


def biased_training_multiLayer(dataset_inits=5,random_inits=5,inp_dim=25,teacher_layers=4,non_linearity=None,num_data=100,
							student_layers=4,student_hidden_dim=25,epochs=10,lr=0.01,bias=0.1):
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	# breakpoint()
	for d,ridx in itertools.product(range(dataset_inits),range(random_inits)):
	# for d,ridx in tqdm(itertools.product(range(dataset_inits),range(random_inits))):
		inp_dim,num_data,X,Y,teacher_net = create_NL_data_labels(inp_dim=inp_dim,num_layers=teacher_layers,
																non_linearity=non_linearity,num_data=num_data,seed=d)
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		grad_norms_arr = []
		student_NL_net = load_multilayer_student_net(inp_dim=inp_dim,num_layers=student_layers,hidden_dim=student_hidden_dim,
													non_linearity=non_linearity,seed=ridx)
		student_NL_net_control = load_multilayer_student_net(inp_dim=inp_dim,num_layers=student_layers,hidden_dim=student_hidden_dim,
													non_linearity=non_linearity,seed=ridx)
		assert not check_model_params_equal(teacher_net,student_NL_net), "Teacher and Student have the same weight!"
		loss0 = get_loss(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss())
		tr_loss_arr.append(loss0)
		tr_loss_arr_control.append(loss0)
		tr_loss_decrease_arr_control.append(loss0)
		for epoch in range(epochs):
			copy_from_model_to_model(student_NL_net_control,student_NL_net)
			_, loss_decrease = train_one_step(net=student_NL_net,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=bias,variance=0.0,verbose=False)
			tr_loss_arr.append(get_loss(net=student_NL_net,X=X,y=Y,loss_fn=nn.MSELoss()))
			(grad,_), loss_decrease_control = train_one_step(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,variance=0.0,verbose=False)
			tr_loss_decrease_arr_control.append(loss_decrease_control)
			tr_loss_arr_control.append(get_loss(net=student_NL_net_control,X=X,y=Y,loss_fn=nn.MSELoss()))
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


def vary_netSize_fixed_variance_over_training_multiLayer_linear(inp_dim=25,teacher_layers=4,num_data=100,variance=0.01):
	res_error_loss_total_arr = []
	res_params_arr = []
	res_size_arr = []
	teacher_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=teacher_layers,hidden_dim=inp_dim,non_linearity=None)
	teacher_params = sum(p.numel() for p in teacher_net_dummy.parameters())
	plt.figure('Impact of varying network size with fixed variance over training - Multilayer LinearNN',figsize=(12,6))
	lr = 1e-3 #0.02
	epochs = 5
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.linspace(inp_dim//2,4*inp_dim,5).astype(int) #[0.01,0.1,0.5]
	layer_range = np.linspace(2,8,4).astype(int)
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	for lidx,layer in enumerate(tqdm(layer_range)):
		grad_norms_layer_arr = []
		tr_loss_layer_arr = []
		tr_loss_control_layer_arr = []
		tr_loss_decrease_control_layer_arr = []
		params_range = []
		for sidx,size in enumerate(tqdm(size_range)):
			student_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=layer,hidden_dim=size,non_linearity=None)
			params_range.append(sum(p.numel() for p in student_net_dummy.parameters()))
			#try random initializations
			dataset_inits = 10
			random_inits = 10
			res = noisy_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
										non_linearity=None,num_data=num_data,student_layers=layer,student_hidden_dim=size,epochs=epochs,lr=lr,variance=variance)
			tr_loss_arr_inits = res[0]
			tr_loss_arr_control_inits = res[1]
			tr_loss_decrease_arr_control_inits = res[2]
			grad_norms_inits = res[3]

			tr_loss_control_layer_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_layer_arr.append(tr_loss_decrease_arr_control_inits)
			tr_loss_layer_arr.append(tr_loss_arr_inits)
			grad_norms_layer_arr.append(grad_norms_inits)

		tr_loss_layer_arr = np.array(tr_loss_layer_arr)
		tr_loss_control_layer_arr = np.array(tr_loss_control_layer_arr)
		tr_loss_decrease_control_layer_arr = np.array(tr_loss_decrease_control_layer_arr)
		grad_norms_layer_arr = np.array(grad_norms_layer_arr)
		# plt.figure('Impact of variance vs grad norm - LinearNN',figsize=(16,6))
		plt.sca(ax1)
		error_loss_layer_arr = (tr_loss_layer_arr-tr_loss_control_layer_arr)
		# error_loss_layer_arr = error_loss_layer_arr[...,1:]/tr_loss_decrease_control_layer_arr[...,1:]
		error_loss_layer_arr = error_loss_layer_arr[...,1:]
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		error_loss_layer_arr_total = np.nanmean(error_loss_layer_arr,axis=-1)
		plt.errorbar(x=params_range,y=np.nanmean(error_loss_layer_arr_total,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total,axis=-1)/np.sqrt(error_loss_layer_arr_total.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		error_loss_layer_arr = (tr_loss_layer_arr-tr_loss_control_layer_arr)
		error_loss_layer_arr = error_loss_layer_arr[...,1:]
		plt.sca(ax2)
		error_loss_layer_arr_total = np.nanmean(error_loss_layer_arr,axis=-1)
		plt.errorbar(x=size_range,y=np.nanmean(error_loss_layer_arr_total,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total,axis=-1)/np.sqrt(error_loss_layer_arr_total.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		res_error_loss_total_arr.append(error_loss_layer_arr_total)
		res_params_arr.append(params_range)
		res_size_arr.append(size_range)
		
	plt.sca(ax1)
	plt.axvline(x=teacher_params,color='k',ls=':',label='Teacher net')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	# plt.xlabel('Network width',fontsize=14)
	plt.xlabel('#Network parameters',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	plt.axvline(x=inp_dim,color='k',ls=':',label='Teacher net')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Network width',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.suptitle('Teacher Net (input_dim={},layers={},hidden_dim={}), variance={}'.format(inp_dim,teacher_layers,inp_dim,variance),fontsize=20)
	return res_error_loss_total_arr, res_params_arr, res_size_arr, layer_range


def vary_netSize_fixed_variance_over_training_multiLayer_nonlinear(inp_dim=25,teacher_layers=4,num_data=100,variance=0.01):
	res_error_loss_total_arr = []
	res_params_arr = []
	res_size_arr = []
	teacher_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=teacher_layers,hidden_dim=inp_dim,non_linearity='relu')
	teacher_params = sum(p.numel() for p in teacher_net_dummy.parameters())
	plt.figure('Impact of varying network size with fixed variance over training - Multilayer ReluNN',figsize=(12,6))
	lr = 1e-3 #0.02
	epochs = 5
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.linspace(inp_dim//2,4*inp_dim,5).astype(int) #[0.01,0.1,0.5]
	layer_range = np.linspace(2,8,4).astype(int)
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	for lidx,layer in enumerate(tqdm(layer_range)):
		grad_norms_layer_arr = []
		tr_loss_layer_arr = []
		tr_loss_control_layer_arr = []
		tr_loss_decrease_control_layer_arr = []
		params_range = []
		for sidx,size in enumerate(tqdm(size_range)):
			student_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=layer,hidden_dim=size,non_linearity='relu')
			params_range.append(sum(p.numel() for p in student_net_dummy.parameters()))
			#try random initializations
			dataset_inits = 10
			random_inits = 10
			res = noisy_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
										non_linearity='relu',num_data=num_data,student_layers=layer,student_hidden_dim=size,epochs=epochs,lr=lr,variance=variance)
			tr_loss_arr_inits = res[0]
			tr_loss_arr_control_inits = res[1]
			tr_loss_decrease_arr_control_inits = res[2]
			grad_norms_inits = res[3]

			tr_loss_control_layer_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_layer_arr.append(tr_loss_decrease_arr_control_inits)
			tr_loss_layer_arr.append(tr_loss_arr_inits)
			grad_norms_layer_arr.append(grad_norms_inits)

		tr_loss_layer_arr = np.array(tr_loss_layer_arr)
		tr_loss_control_layer_arr = np.array(tr_loss_control_layer_arr)
		tr_loss_decrease_control_layer_arr = np.array(tr_loss_decrease_control_layer_arr)
		grad_norms_layer_arr = np.array(grad_norms_layer_arr)
		# plt.figure('Impact of variance vs grad norm - LinearNN',figsize=(16,6))
		plt.sca(ax1)
		error_loss_layer_arr = (tr_loss_layer_arr-tr_loss_control_layer_arr)
		# error_loss_layer_arr = error_loss_layer_arr[...,1:]/tr_loss_decrease_control_layer_arr[...,1:]
		error_loss_layer_arr = error_loss_layer_arr[...,1:]
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		error_loss_layer_arr_total = np.nanmean(error_loss_layer_arr,axis=-1)
		plt.errorbar(x=params_range,y=np.nanmean(error_loss_layer_arr_total,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total,axis=-1)/np.sqrt(error_loss_layer_arr_total.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		error_loss_layer_arr = (tr_loss_layer_arr-tr_loss_control_layer_arr)
		error_loss_layer_arr = error_loss_layer_arr[...,1:]
		plt.sca(ax2)
		error_loss_layer_arr_total = np.nanmean(error_loss_layer_arr,axis=-1)
		plt.errorbar(x=size_range,y=np.nanmean(error_loss_layer_arr_total,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total,axis=-1)/np.sqrt(error_loss_layer_arr_total.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		res_error_loss_total_arr.append(error_loss_layer_arr_total)
		res_params_arr.append(params_range)
		res_size_arr.append(size_range)
		
	plt.sca(ax1)
	plt.axvline(x=teacher_params,color='k',ls=':',label='Teacher net')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	# plt.xlabel('Network width',fontsize=14)
	plt.xlabel('#Network parameters',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	plt.axvline(x=inp_dim,color='k',ls=':',label='Teacher net')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Network width',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.suptitle('Relu Teacher Net (input_dim={},layers={},hidden_dim={}), variance={}'.format(inp_dim,teacher_layers,inp_dim,variance),fontsize=20)
	return res_error_loss_total_arr, res_params_arr, res_size_arr, layer_range


def vary_netSize_fixed_variance_over_training_multiLayer_relu_v_linear(inp_dim=25,teacher_layers=4,num_data=100,variance=0.01):
	teacher_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=teacher_layers,hidden_dim=inp_dim,non_linearity=None)
	teacher_params = sum(p.numel() for p in teacher_net_dummy.parameters())
	res_linear = vary_netSize_fixed_variance_over_training_multiLayer_linear(inp_dim=inp_dim,teacher_layers=teacher_layers,num_data=num_data,variance=variance)
	assert len(res_linear[0])==len(res_linear[1]) and len(res_linear[1])==len(res_linear[2]), "res_linear has unequal number of elements"
	res_relu = vary_netSize_fixed_variance_over_training_multiLayer_nonlinear(inp_dim=inp_dim,teacher_layers=teacher_layers,num_data=num_data,variance=variance)
	assert len(res_relu[0])==len(res_relu[1]) and len(res_relu[1])==len(res_relu[2]), "res_relu has unequal number of elements"
	assert len(res_linear[0])==len(res_relu[0]), "res_linear and res_relu have unequal number of elements"
	plt.figure('Impact of varying network size with fixed variance over training - Multilayer NN (Relu v Linear)',figsize=(16,11))
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	for lidx in range(len(res_linear[0])):
		error_loss_layer_arr_total_linear = res_linear[0][lidx]
		params_range_linear =  res_linear[1][lidx]
		size_range_linear = res_linear[2][lidx]

		error_loss_layer_arr_total_relu = res_relu[0][lidx]
		params_range_relu =  res_relu[1][lidx]
		size_range_relu = res_relu[2][lidx]

		plt.sca(ax1)
		plt.errorbar(x=params_range_linear,y=np.nanmean(error_loss_layer_arr_total_linear,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total_linear,axis=-1)/np.sqrt(error_loss_layer_arr_total_linear.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(res_linear[3][lidx])) 	# ,label='Layers={} (linear)'.format(res_linear[3][lidx])
		plt.errorbar(x=params_range_relu,y=np.nanmean(error_loss_layer_arr_total_relu,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total_relu,axis=-1)/np.sqrt(error_loss_layer_arr_total_relu.shape[-1]),
					color=color_range[lidx%len(color_range)],ls='--') 	# ,label='Layers={} (relu)'.format(res_relu[3][lidx])

		plt.sca(ax2)
		plt.errorbar(x=size_range_linear,y=np.nanmean(error_loss_layer_arr_total_linear,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total_linear,axis=-1)/np.sqrt(error_loss_layer_arr_total_linear.shape[-1]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(res_linear[3][lidx])) 	# ,label='Layers={} (linear)'.format(res_linear[3][lidx])
		plt.errorbar(x=size_range_relu,y=np.nanmean(error_loss_layer_arr_total_relu,axis=-1),
					yerr=np.nanstd(error_loss_layer_arr_total_relu,axis=-1)/np.sqrt(error_loss_layer_arr_total_relu.shape[-1]),
					color=color_range[lidx%len(color_range)],ls='--') 	# ,label='Layers={} (relu)'.format(res_relu[3][lidx])

	plt.sca(ax1)
	plt.axvline(x=teacher_params,color='k',ls=':',label='Teacher net')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	# plt.xlabel('Network width',fontsize=14)
	plt.xlabel('#Network parameters',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	plt.axvline(x=inp_dim,color='k',ls=':',label='Teacher net')
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Network width',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of variance vs Network size',fontsize=16)
	plt.legend(fontsize=15)

	plt.suptitle('Relu(--)/Linear(-) Teacher Net (input_dim={},layers={},hidden_dim={}), variance={}'.format(inp_dim,teacher_layers,inp_dim,variance),fontsize=20)


def vary_netSize_fixed_bias_over_training_multiLayer_linear(inp_dim=25,teacher_layers=4,num_data=100,bias=0.01):
	teacher_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=teacher_layers,hidden_dim=inp_dim,non_linearity=None)
	teacher_params = sum(p.numel() for p in teacher_net_dummy.parameters())
	lr = 1e-2 #0.02
	epochs = 15
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.append(np.array([inp_dim//2]),np.linspace(inp_dim,3*inp_dim,5).astype(int)) #[0.01,0.1,0.5]
	layer_range = np.linspace(teacher_layers//2,3*teacher_layers,6).astype(int)
	color_range = ['purple','blue','green','gold','orange','red','brown']
	plt.figure('Impact of varying network size with fixed bias over training - Multilayer LinearNN',figsize=(16,11))
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)

	plt.figure('Impact of varying network size with fixed bias over training - Multilayer LinearNN absolute',figsize=(16,11))
	absax1 = plt.subplot(221)
	absax2 = plt.subplot(222)
	absax3 = plt.subplot(223)
	absax4 = plt.subplot(224)

	error_loss_layers_arr = []
	for lidx,layer in enumerate(tqdm(layer_range)):
		# params_range = []
		# student_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=layer,hidden_dim=inp_dim,non_linearity=None)
		# params_range.append(sum(p.numel() for p in student_net_dummy.parameters()))
		#try random initializations
		dataset_inits = 10
		random_inits = 20
		res = biased_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
									non_linearity=None,num_data=num_data,student_layers=layer,student_hidden_dim=inp_dim,epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		# error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(ax1)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		plt.sca(ax2)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[lidx%len(color_range)],marker='o',alpha=0.4,
			label='Layers = {}'.format(layer))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[lidx%len(color_range)],marker='x',alpha=0.4)

		error_loss_layers_arr.append(error_loss_arr_inits)


		error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(absax1)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		plt.sca(absax2)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[lidx%len(color_range)],marker='o',alpha=0.4,
			label='Layers = {}'.format(layer))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[lidx%len(color_range)],marker='x',alpha=0.4)

	error_loss_size_arr = []
	for sidx,size in enumerate(tqdm(size_range)):
		#try random initializations
		dataset_inits = 10
		random_inits = 20
		res = biased_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
									non_linearity=None,num_data=num_data,student_layers=teacher_layers,student_hidden_dim=size,epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		# error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(ax3)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[sidx%len(color_range)],label='Width={}'.format(size))

		plt.sca(ax4)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='Width = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)

		error_loss_size_arr.append(error_loss_arr_inits)


		error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(absax3)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[sidx%len(color_range)],label='Width={}'.format(size))

		plt.sca(absax4)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='Width = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
		

	plt.sca(ax1)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network depth',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax1)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network depth',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax2)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax3)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network width',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax3)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network width',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax4)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)
	plt.suptitle('Teacher Net (input_dim={},layers={},hidden_dim={}), bias={}'.format(inp_dim,teacher_layers,inp_dim,bias),fontsize=20)

	plt.sca(absax4)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)
	plt.suptitle('Teacher Net (input_dim={},layers={},hidden_dim={}), bias={}'.format(inp_dim,teacher_layers,inp_dim,bias),fontsize=20)
	return error_loss_layers_arr, layer_range, error_loss_size_arr, size_range


def vary_netSize_fixed_bias_over_training_multiLayer_nonlinear(inp_dim=25,teacher_layers=4,num_data=100,bias=0.01):
	teacher_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=teacher_layers,hidden_dim=inp_dim,non_linearity='relu')
	teacher_params = sum(p.numel() for p in teacher_net_dummy.parameters())
	lr = 1e-2 #0.02
	epochs = 15
	# variance_range = np.linspace(0.05,0.5,3) #[0.01,0.1,0.5]
	size_range = np.append(np.array([inp_dim//2]),np.linspace(inp_dim,3*inp_dim,5).astype(int)) #[0.01,0.1,0.5]
	layer_range = np.linspace(teacher_layers//2,3*teacher_layers,6).astype(int)
	color_range = ['purple','blue','green','gold','orange','red','brown']
	plt.figure('Impact of varying network size with fixed bias over training - Multilayer ReluNN',figsize=(16,11))
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)

	plt.figure('Impact of varying network size with fixed bias over training - Multilayer ReluNN absolute',figsize=(16,11))
	absax1 = plt.subplot(221)
	absax2 = plt.subplot(222)
	absax3 = plt.subplot(223)
	absax4 = plt.subplot(224)

	error_loss_layers_arr = []
	for lidx,layer in enumerate(tqdm(layer_range)):
		#try random initializations
		dataset_inits = 10
		random_inits = 20
		res = biased_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
									non_linearity='relu',num_data=num_data,student_layers=layer,student_hidden_dim=inp_dim,epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		# error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		plt.sca(ax1)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		plt.sca(ax2)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[lidx%len(color_range)],marker='o',alpha=0.4,
			label='Layers = {}'.format(layer))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[lidx%len(color_range)],marker='x',alpha=0.4)

		error_loss_layers_arr.append(error_loss_arr_inits)


		error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(absax1)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[lidx%len(color_range)],label='Layers={}'.format(layer))

		plt.sca(absax2)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[lidx%len(color_range)],marker='o',alpha=0.4,
			label='Layers = {}'.format(layer))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[lidx%len(color_range)],marker='x',alpha=0.4)	

	error_loss_size_arr = []
	for sidx,size in enumerate(tqdm(size_range)):
		# params_range = []
		# student_net_dummy = MultiLayerNN(inp_size=inp_dim,num_layers=layer,hidden_dim=inp_dim,non_linearity='relu')
		# params_range.append(sum(p.numel() for p in student_net_dummy.parameters()))
		#try random initializations
		dataset_inits = 10
		random_inits = 20
		res = biased_training_multiLayer(dataset_inits=dataset_inits,random_inits=random_inits,inp_dim=inp_dim,teacher_layers=teacher_layers,
									non_linearity='relu',num_data=num_data,student_layers=teacher_layers,student_hidden_dim=size,epochs=epochs,lr=lr,bias=bias)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		grad_norms_inits = res[3]

		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		# error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(ax3)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[sidx%len(color_range)],label='Width={}'.format(size))

		plt.sca(ax4)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='Width = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)

		error_loss_size_arr.append(error_loss_arr_inits)


		error_loss_arr_inits = np.abs(error_loss_arr_inits)#/tr_loss_decrease_arr_control_inits
		
		plt.sca(absax3)
		# print(tr_loss_decrease_control_layer_arr[...,1:].shape,error_loss_layer_arr.shape,grad_norms_layer_arr.shape)
		plt.errorbar(x=np.linspace(1,error_loss_arr_inits.shape[1],error_loss_arr_inits.shape[1]),
					y=np.nanmean(error_loss_arr_inits,axis=0),
					yerr=np.nanstd(error_loss_arr_inits,axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
					color=color_range[sidx%len(color_range)],label='Width={}'.format(size))

		plt.sca(absax4)
		sc_plot_x = grad_norms_inits.flatten()
		sc_plot_y = error_loss_arr_inits[...,1:].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[sidx%len(color_range)],marker='o',alpha=0.4,
			label='Width = {}'.format(size))  
		plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[sidx%len(color_range)],marker='x',alpha=0.4)
		

	plt.sca(ax1)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network depth',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax1)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network depth',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax2)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax3)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network width',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(absax3)
	plt.axhline(y=0,color='k',ls='--')
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	# plt.yscale('log')
	plt.xlabel('Epochs',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs Network width',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax4)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'$\Delta Loss$',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)
	plt.suptitle('Relu Teacher Net (input_dim={},layers={},hidden_dim={}), bias={}'.format(inp_dim,teacher_layers,inp_dim,bias),fontsize=20)

	plt.sca(absax4)
	# plt.ylabel(r'$\frac{\Delta Loss}{True\_Loss\_Decrease}$',fontsize=18)
	plt.ylabel(r'|$\Delta Loss$|',fontsize=18)
	plt.yscale('log')
	plt.xlabel('Gradient norm',fontsize=14)
	plt.xscale('log')
	plt.title('Impact of bias vs gradient norm',fontsize=16)
	plt.legend(fontsize=15)
	plt.suptitle('Relu Teacher Net (input_dim={},layers={},hidden_dim={}), bias={}'.format(inp_dim,teacher_layers,inp_dim,bias),fontsize=20)
	return error_loss_layers_arr, layer_range, error_loss_size_arr, size_range


if __name__=='__main__':
	run_local = True
	# vary_netSize_fixed_variance_over_training_multiLayer_linear(inp_dim=25,teacher_layers=3,num_data=1000,variance=0.01)
	# vary_netSize_fixed_variance_over_training_multiLayer_nonlinear(inp_dim=25,teacher_layers=3,num_data=1000,variance=0.01)
	vary_netSize_fixed_variance_over_training_multiLayer_relu_v_linear(inp_dim=25,teacher_layers=4,num_data=5000,variance=0.1)

	# vary_netSize_fixed_bias_over_training_multiLayer_linear(inp_dim=25,teacher_layers=4,num_data=20000,bias=0.2)
	# run_local = False
	# vary_netSize_fixed_bias_over_training_multiLayer_nonlinear(inp_dim=25,teacher_layers=4,num_data=2000,bias=0.2)
	# plt.grid('on')
	if run_local:
		plt.show()
	else:
		save_all_figs_to_disk()
