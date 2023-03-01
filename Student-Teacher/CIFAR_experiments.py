import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as torchmodels
from tqdm import tqdm
from utils import *
torch.set_default_tensor_type(torch.FloatTensor)
import argparse
from pyhessian import hessian
import gc, copy

dataset_folder = '/network/datasets/cifar10.var/cifar10_torchvision/'
batch_size_train = 5000
batch_size_test = 5000

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=dataset_folder, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

trainset_eval = torchvision.datasets.CIFAR10(
    root=dataset_folder, train=True, download=False, transform=transform_test)
trainloader_eval = torch.utils.data.DataLoader(
    trainset_eval, batch_size=batch_size_train, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=dataset_folder, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

def get_torchvision_model(model_class=torchmodels.vgg11,pretrained=False):
	assert model_class in [torchmodels.vgg11,torchmodels.vgg13,torchmodels.vgg16,torchmodels.vgg19,
							torchmodels.vgg11_bn,torchmodels.vgg13_bn,torchmodels.vgg16_bn,torchmodels.vgg19_bn], NotImplementedError
	model = model_class(pretrained=pretrained)
	input_lastLayer = model.classifier[-1].in_features
	model.classifier[-1] = nn.Linear(input_lastLayer,10)
	return model

def calculate_trace_hessian(net,dataloader=trainloader):
	net.eval()
	criterion = nn.CrossEntropyLoss()
	trace_hessian_values_arr = []
	for (inputs,labels) in dataloader:
		net.zero_grad()
		hessian_comp = hessian(net, criterion, data=(inputs, labels), cuda=True)
		trace_hessian = hessian_comp.trace(maxIter=20)
		trace_hessian_values_arr.extend(trace_hessian)
	trace_hessian_per_param = np.mean(trace_hessian)/sum(p.numel() for p in net.parameters())
	return trace_hessian_per_param

def train_GD_epoch(net,lr=1e-3,bias=0.0,variance=0.0,grad_noise=None,no_update=False):
	net.train()
	net.zero_grad()
	criterion = nn.CrossEntropyLoss()
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if 'cuda' in device.type:
		net = net.cuda()
		net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
		torch.backends.cudnn.benchmark = True
	running_loss = 0.0
	for i, data in enumerate(trainloader):
		inputs, labels = data
		if 'cuda' in device.type:
			inputs = inputs.cuda()
			labels = labels.cuda()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		running_loss += loss.item()
	# params = list(net.parameters())
	# for p in params:
	# 	p.data -= lr*p.grad
	grad = gradient_descent_one_step(net,lr=lr,bias=bias,variance=variance,grad_noise=grad_noise,no_update=no_update)
	return grad, running_loss/len(trainloader)

def evaluate_accuracy(net,dataloader):
	net.eval()
	criterion = nn.CrossEntropyLoss()
	test_loss = 0
	correct = 0
	total = 0
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if 'cuda' in device.type:
		net = net.cuda()
		net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
		torch.backends.cudnn.benchmark = True
	with torch.no_grad():
		for batch_idx, (inputs,labels) in enumerate(dataloader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net(inputs)
			loss = criterion(outputs,labels)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
	test_loss /= len(dataloader)
	acc = 100.*correct/total
	return test_loss, acc

def biased_training_VGG(random_inits=5,model_class=torchmodels.vgg11,pretrained=False,epochs=10,lr=1e-3,bias=0.02):
	dict_folder = os.path.join('CIFAR_results','Plot_metadata','Bias')
	dict_fname = '{}{}.npy'.format(model_class.__name__,'_pretrained' if pretrained else '')
	try:
		results_dict = np.load(os.path.join(dict_folder,dict_fname),allow_pickle=True).item()
		print("Writing to old file ==> {}".format(os.path.join(dict_folder,dict_fname)))
	except:
		print("Writing to new file ==> {}".format(os.path.join(dict_folder,dict_fname)))
		results_dict = {}
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	te_loss_arr_control_inits = []
	te_loss_arr_inits = []
	te_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	for ridx in tqdm(range(random_inits)):
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		te_loss_arr_control = []
		te_loss_decrease_arr_control = []
		te_loss_arr = []
		grad_norms_arr = []
		torch.manual_seed(ridx)
		np.random.seed(ridx)
		net_control = get_torchvision_model(model_class=model_class,pretrained=pretrained)
		torch.manual_seed(ridx)
		np.random.seed(ridx)
		net = get_torchvision_model(model_class=model_class,pretrained=pretrained)
		tr_loss0,_ = evaluate_accuracy(net=net_control,dataloader=trainloader_eval)
		te_loss0,_ = evaluate_accuracy(net=net_control,dataloader=testloader)
		tr_loss_arr.append(tr_loss0)
		tr_loss_arr_control.append(tr_loss0)
		tr_loss_decrease_arr_control.append(tr_loss0)
		te_loss_arr.append(te_loss0)
		te_loss_arr_control.append(te_loss0)
		te_loss_decrease_arr_control.append(te_loss0)
		
		for epoch in range(epochs):
			copy_from_model_to_model(net_control,net)
			_, tr_loss = train_GD_epoch(net=net,lr=lr,bias=bias)
			# _, loss_decrease = train_one_step(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=bias,verbose=False)
			tr_loss,_ = evaluate_accuracy(net=net,dataloader=trainloader_eval)
			te_loss,_ = evaluate_accuracy(net=net,dataloader=testloader)
			tr_loss_arr.append(tr_loss)
			te_loss_arr.append(te_loss)
			# tr_loss_arr.append(get_loss(net=student_net,X=X,y=Y,loss_fn=nn.MSELoss()))
			(grad,_), tr_loss_control = train_GD_epoch(net=net_control,lr=lr,bias=0.0)
			# (grad,_), loss_decrease_control = train_one_step(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss(),lr=lr,bias=0.0,verbose=False)
			tr_loss_control,_ = evaluate_accuracy(net=net_control,dataloader=trainloader_eval)
			te_loss_control,_ = evaluate_accuracy(net=net_control,dataloader=testloader)
			tr_loss_decrease_arr_control.append(tr_loss_arr_control[-1]-tr_loss_control)
			te_loss_decrease_arr_control.append(te_loss_arr_control[-1]-te_loss_control)
			# tr_loss_decrease_arr_control.append(loss_decrease_control)
			tr_loss_arr_control.append(tr_loss_control)
			te_loss_arr_control.append(te_loss_control)
			# tr_loss_arr_control.append(get_loss(net=student_net_control,X=X,y=Y,loss_fn=nn.MSELoss()))
			grad_norms_arr.append(torch.norm(grad).item())

		tr_loss_arr = np.array(tr_loss_arr)
		tr_loss_arr_inits.append(tr_loss_arr)
		tr_loss_arr_control = np.array(tr_loss_arr_control)
		tr_loss_arr_control_inits.append(tr_loss_arr_control)
		tr_loss_decrease_arr_control = np.array(tr_loss_decrease_arr_control)
		tr_loss_decrease_arr_control_inits.append(tr_loss_decrease_arr_control)
		te_loss_arr = np.array(te_loss_arr)
		te_loss_arr_inits.append(te_loss_arr)
		te_loss_arr_control = np.array(te_loss_arr_control)
		te_loss_arr_control_inits.append(te_loss_arr_control)
		te_loss_decrease_arr_control = np.array(te_loss_decrease_arr_control)
		te_loss_decrease_arr_control_inits.append(te_loss_decrease_arr_control)
		grad_norms_inits.append(grad_norms_arr)

	tr_loss_arr_inits = np.array(tr_loss_arr_inits)
	tr_loss_arr_control_inits = np.array(tr_loss_arr_control_inits)
	tr_loss_decrease_arr_control_inits = np.array(tr_loss_decrease_arr_control_inits)
	te_loss_arr_inits = np.array(te_loss_arr_inits)
	te_loss_arr_control_inits = np.array(te_loss_arr_control_inits)
	te_loss_decrease_arr_control_inits = np.array(te_loss_decrease_arr_control_inits)
	grad_norms_inits = np.array(grad_norms_inits)
	res = tr_loss_arr_inits, tr_loss_arr_control_inits, tr_loss_decrease_arr_control_inits, te_loss_arr_inits, te_loss_arr_control_inits, te_loss_decrease_arr_control_inits, grad_norms_inits
	res_key = 'bias={:.2f}'.format(bias)
	if res_key in results_dict.keys():
			results_dict[res_key].append(res)
	else:
		results_dict[res_key] = [res]
	np.save(os.path.join(dict_folder,dict_fname),results_dict)
	return res

def vary_bias_over_training_VGG(model_class=torchmodels.vgg11,pretrained=False):
	dict_folder = os.path.join('CIFAR_results','Plot_metadata')
	dict_fname = 'vary_bias_over_training_VGG'
	try:
		results_dict = np.load(os.path.join(dict_folder,dict_fname),allow_pickle=True).item()
	except:
		print("Writing to new file ==> {}".format(os.path.join(dict_folder,dict_fname)))
		results_dict = {}
	plt.figure('Impact of varying bias over training - {} {}'.format(model_class.__name__,'(pretrained)' if pretrained else ''),figsize=(16,11))
	lr = 1e-3
	epochs = 10 #10 4
	bias_range = np.linspace(0.01,0.03,3) #[0.01,0.1,0.5]
	color_range = ['purple','blue','green','gold','orange','red','brown']
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)
	grad_norms_bias_arr = []
	tr_loss_bias_arr = []
	tr_loss_control_bias_arr = []
	tr_loss_decrease_control_bias_arr = []
	te_loss_bias_arr = []
	te_loss_control_bias_arr = []
	te_loss_decrease_control_bias_arr = []
	for bidx,bias in enumerate(tqdm(bias_range)):
		#try random initializations
		random_inits = 5 #5 2
		res = biased_training_VGG(random_inits=random_inits,model_class=model_class,pretrained=pretrained,
									epochs=epochs,lr=lr,bias=bias)
		res_key = '{}{} bias={:.2f}'.format(model_class.__name__,'(pretrained)' if pretrained else '',bias)
		if res_key in results_dict.keys():
			results_dict[res_key].append(res)
		else:
			results_dict[res_key] = [res]
		np.save(os.path.join(dict_folder,dict_fname),results_dict)
		tr_loss_arr_inits = res[0]
		tr_loss_arr_control_inits = res[1]
		tr_loss_decrease_arr_control_inits = res[2]
		te_loss_arr_inits = res[3]
		te_loss_arr_control_inits = res[4]
		te_loss_decrease_arr_control_inits = res[5]
		grad_norms_inits = res[6]

		plt.sca(ax1)
		if bidx==0:
			plt.errorbar(x=np.linspace(0,tr_loss_arr_control_inits.shape[1]-1,tr_loss_arr_control_inits.shape[1]),
				y=tr_loss_arr_control_inits.mean(axis=0),yerr=tr_loss_arr_control_inits.std(axis=0)/np.sqrt(tr_loss_arr_control_inits.shape[0]),
				color='k',label='True gradient')
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

		plt.sca(ax2)
		if bidx==0:
			plt.errorbar(x=np.linspace(0,te_loss_arr_control_inits.shape[1]-1,te_loss_arr_control_inits.shape[1]),
				y=te_loss_arr_control_inits.mean(axis=0),yerr=te_loss_arr_control_inits.std(axis=0)/np.sqrt(te_loss_arr_control_inits.shape[0]),
				color='k',label='True gradient')
		for pidx in range(te_loss_arr_inits.shape[1]-1):
			if pidx==0:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[te_loss_arr_control_inits[:,pidx].mean(axis=0),te_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,te_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(te_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--',label='Bias = {:.2f}'.format(bias))
			else:
				plt.errorbar(x=np.linspace(pidx,pidx+1,2),
							y=[te_loss_arr_control_inits[:,pidx].mean(axis=0),te_loss_arr_inits[:,pidx+1].mean(axis=0)],
							yerr=[0,te_loss_arr_inits[:,pidx+1].std(axis=0)/np.sqrt(te_loss_arr_inits.shape[0])],
							color=color_range[bidx%len(color_range)],ls='--')

		plt.sca(ax3)
		error_loss_arr_inits = (tr_loss_arr_inits-tr_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,tr_loss_arr_inits.shape[1]-1,tr_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		plt.sca(ax4)
		error_loss_arr_inits = (te_loss_arr_inits-te_loss_arr_control_inits)
		plt.errorbar(x=np.linspace(0,te_loss_arr_inits.shape[1]-1,te_loss_arr_inits.shape[1]),
			y=error_loss_arr_inits.mean(axis=0),yerr=error_loss_arr_inits.std(axis=0)/np.sqrt(error_loss_arr_inits.shape[0]),
			color=color_range[bidx%len(color_range)],label='Bias = {:.2f}'.format(bias))

		if bidx==0:
			tr_loss_control_bias_arr.append(tr_loss_arr_control_inits)
			tr_loss_decrease_control_bias_arr.append(tr_loss_decrease_arr_control_inits)
			te_loss_control_bias_arr.append(te_loss_arr_control_inits)
			te_loss_decrease_control_bias_arr.append(te_loss_decrease_arr_control_inits)
		tr_loss_bias_arr.append(tr_loss_arr_inits)
		grad_norms_bias_arr.append(grad_norms_inits)
		te_loss_bias_arr.append(te_loss_arr_inits)
		grad_norms_bias_arr.append(grad_norms_inits)

	tr_loss_bias_arr = np.array(tr_loss_bias_arr)
	tr_loss_control_bias_arr = np.array(tr_loss_control_bias_arr)
	tr_loss_decrease_control_bias_arr = np.array(tr_loss_decrease_control_bias_arr)
	te_loss_bias_arr = np.array(te_loss_bias_arr)
	te_loss_control_bias_arr = np.array(te_loss_control_bias_arr)
	te_loss_decrease_control_bias_arr = np.array(te_loss_decrease_control_bias_arr)
	grad_norms_bias_arr = np.array(grad_norms_bias_arr)
	plt.sca(ax1)
	# plt.yscale('log')
	plt.ylabel('Train set Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Training set Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax2)
	# plt.yscale('log')
	plt.ylabel('Test set Loss',fontsize=14)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Test set Loss vs epochs',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax3)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta$Train Loss',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in training set loss due to biased gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.sca(ax4)
	plt.axhline(y=0,color='k',ls='--')
	plt.ylabel(r'$\Delta$Test Loss',fontsize=18)
	plt.xlabel('Epochs',fontsize=14)
	plt.title('Increase in test set loss due to biased gradient',fontsize=16)
	plt.legend(fontsize=15)

	plt.figure('Impact of bias vs grad norm - {} {}'.format(model_class.__name__,'(pretrained)' if pretrained else ''),figsize=(16,6))
	error_loss_bias_arr = (tr_loss_bias_arr-tr_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]
	plt.subplot(121)
	for bidx,bias in enumerate(bias_range):
		sc_plot_x = grad_norms_bias_arr[bidx].flatten()
		sc_plot_y = error_loss_bias_arr[bidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[bidx%len(color_range)],marker='o',alpha=0.4,
			label='Bias = {:.2f}'.format(bias))  
		# plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
		plt.scatter(sc_plot_x[sc_plot_y<0],(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta$Train Loss',fontsize=18)
	plt.axhline(y=0,color='k',ls='--')
	# plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	# plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient (Train)',fontsize=16)
	plt.legend(fontsize=15)

	error_loss_bias_arr = (te_loss_bias_arr-te_loss_control_bias_arr)
	error_loss_bias_arr = error_loss_bias_arr[...,1:]
	plt.subplot(122)
	for bidx,bias in enumerate(bias_range):
		sc_plot_x = grad_norms_bias_arr[bidx].flatten()
		sc_plot_y = error_loss_bias_arr[bidx].flatten()
		plt.scatter(sc_plot_x[sc_plot_y>=0],sc_plot_y[sc_plot_y>=0],s=7,color=color_range[bidx%len(color_range)],marker='o',alpha=0.4,
			label='Bias = {:.2f}'.format(bias))  
		# plt.scatter(sc_plot_x[sc_plot_y<0],np.abs(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
		plt.scatter(sc_plot_x[sc_plot_y<0],(sc_plot_y[sc_plot_y<0]),s=7,color=color_range[bidx%len(color_range)],marker='x',alpha=0.4)
	plt.ylabel(r'$\Delta$Test Loss',fontsize=18)
	plt.axhline(y=0,color='k',ls='--')
	# plt.yscale('log')
	plt.xlabel('Norm of gradient',fontsize=14)
	# plt.xscale('log')
	plt.title('Impact of bias vs norm of gradient (Test)',fontsize=16)
	plt.legend(fontsize=15)
	

def noisy_training_VGG(random_inits=5,model_class=torchmodels.vgg11,pretrained=False,epochs=5,lr=1e-3,variance=0.5):
	dict_folder = os.path.join('CIFAR_results','Plot_metadata','Variance')
	dict_fname = '{}{}.npy'.format(model_class.__name__,'_pretrained' if pretrained else '')
	try:
		results_dict = np.load(os.path.join(dict_folder,dict_fname),allow_pickle=True).item()
		print("Writing to old file ==> {}".format(os.path.join(dict_folder,dict_fname)))
	except:
		print("Writing to new file ==> {}".format(os.path.join(dict_folder,dict_fname)))
		results_dict = {}
	repeats = 10 #20
	tr_loss_arr_control_inits = []
	tr_loss_arr_inits = []
	tr_loss_decrease_arr_control_inits = []
	te_loss_arr_control_inits = []
	te_loss_arr_inits = []
	te_loss_decrease_arr_control_inits = []
	grad_norms_inits = []
	hessian_trace_inits = []
	# breakpoint()
	for ridx in range(random_inits):
		tr_loss_arr_control = []
		tr_loss_decrease_arr_control = []
		tr_loss_arr = []
		te_loss_arr_control = []
		te_loss_decrease_arr_control = []
		te_loss_arr = []
		grad_norms_arr = []
		hessian_trace_arr = []
		torch.manual_seed(ridx)
		np.random.seed(ridx)
		net_control = get_torchvision_model(model_class=model_class,pretrained=pretrained)
		torch.manual_seed(ridx)
		np.random.seed(ridx)
		net = get_torchvision_model(model_class=model_class,pretrained=pretrained)
		tr_loss0,_ = evaluate_accuracy(net=net_control,dataloader=trainloader_eval)
		te_loss0,_ = evaluate_accuracy(net=net_control,dataloader=testloader)
		tr_loss_arr.append(tr_loss0)
		tr_loss_arr_control.append(tr_loss0)
		tr_loss_decrease_arr_control.append(tr_loss0)
		te_loss_arr.append(te_loss0)
		te_loss_arr_control.append(te_loss0)
		te_loss_decrease_arr_control.append(te_loss0)
		for epoch in tqdm(range(epochs)):
			mean_new_tr_loss = 0
			mean_new_te_loss = 0
			grad_noise_vectors = generate_random_noise_sequence(repeats=repeats,inp_dim=sum(p.numel() for p in net.parameters())-1,variance=variance)
			for r in range(repeats):
				gc.collect()
				copy_from_model_to_model(net_control,net)	
				_, tr_loss = train_GD_epoch(net=net,lr=lr,bias=0.0,variance=variance,grad_noise=grad_noise_vectors[r])
				tr_loss,_ = evaluate_accuracy(net=net,dataloader=trainloader_eval)
				mean_new_tr_loss += tr_loss
				te_loss,_ = evaluate_accuracy(net=net,dataloader=testloader)
				mean_new_te_loss += te_loss
			mean_new_tr_loss = mean_new_tr_loss/repeats
			mean_new_te_loss = mean_new_te_loss/repeats
			tr_loss_arr.append(mean_new_tr_loss)
			te_loss_arr.append(mean_new_te_loss)
			hessian_trace_val_param = calculate_trace_hessian(net_control)
			(grad,_), tr_loss_control = train_GD_epoch(net=net_control,lr=lr,bias=0.0,variance=0.0)
			tr_loss_control,_ = evaluate_accuracy(net=net_control,dataloader=trainloader_eval)
			te_loss_control,_ = evaluate_accuracy(net=net_control,dataloader=testloader)
			# print(hessian_trace_val_param, mean_new_tr_loss-tr_loss_control, mean_new_te_loss-te_loss_control)
			tr_loss_decrease_arr_control.append(tr_loss_arr_control[-1]-tr_loss_control)
			te_loss_decrease_arr_control.append(te_loss_arr_control[-1]-te_loss_control)
			tr_loss_arr_control.append(tr_loss_control)
			te_loss_arr_control.append(te_loss_control)
			grad_norms_arr.append(torch.norm(grad).item())
			hessian_trace_arr.append(hessian_trace_val_param)

		tr_loss_arr = np.array(tr_loss_arr)
		tr_loss_arr_inits.append(tr_loss_arr)
		tr_loss_arr_control = np.array(tr_loss_arr_control)
		tr_loss_arr_control_inits.append(tr_loss_arr_control)
		tr_loss_decrease_arr_control = np.array(tr_loss_decrease_arr_control)
		tr_loss_decrease_arr_control_inits.append(tr_loss_decrease_arr_control)
		te_loss_arr = np.array(te_loss_arr)
		te_loss_arr_inits.append(te_loss_arr)
		te_loss_arr_control = np.array(te_loss_arr_control)
		te_loss_arr_control_inits.append(te_loss_arr_control)
		te_loss_decrease_arr_control = np.array(te_loss_decrease_arr_control)
		te_loss_decrease_arr_control_inits.append(te_loss_decrease_arr_control)
		grad_norms_arr = np.array(grad_norms_arr)
		grad_norms_inits.append(grad_norms_arr)
		hessian_trace_arr = np.array(hessian_trace_arr)
		hessian_trace_inits.append(hessian_trace_arr)

	tr_loss_arr_inits = np.array(tr_loss_arr_inits)
	tr_loss_arr_control_inits = np.array(tr_loss_arr_control_inits)
	tr_loss_decrease_arr_control_inits = np.array(tr_loss_decrease_arr_control_inits)
	te_loss_arr_inits = np.array(te_loss_arr_inits)
	te_loss_arr_control_inits = np.array(te_loss_arr_control_inits)
	te_loss_decrease_arr_control_inits = np.array(te_loss_decrease_arr_control_inits)
	grad_norms_inits = np.array(grad_norms_inits)
	hessian_trace_inits = np.array(hessian_trace_inits)
	res = tr_loss_arr_inits, tr_loss_arr_control_inits, tr_loss_decrease_arr_control_inits, te_loss_arr_inits, te_loss_arr_control_inits, te_loss_decrease_arr_control_inits, grad_norms_inits, hessian_trace_inits
	res_key = 'variance={:.2f}'.format(variance)
	if res_key in results_dict.keys():
			results_dict[res_key].append(res)
	else:
		results_dict[res_key] = [res]
	np.save(os.path.join(dict_folder,dict_fname),results_dict)
	return res

# print("vgg11 untrained: ",evaluate_accuracy(torchmodels.vgg11(pretrained=False),trainloader_eval),evaluate_accuracy(torchmodels.vgg11(pretrained=False),testloader))
# print("vgg11 pretrained: ",evaluate_accuracy(torchmodels.vgg11(pretrained=True),trainloader_eval),evaluate_accuracy(torchmodels.vgg11(pretrained=True),testloader))
# print("vgg13 untrained: ",evaluate_accuracy(torchmodels.vgg13(pretrained=False),trainloader_eval),evaluate_accuracy(torchmodels.vgg13(pretrained=False),testloader))
# print("vgg13 pretrained: ",evaluate_accuracy(torchmodels.vgg13(pretrained=True),trainloader_eval),evaluate_accuracy(torchmodels.vgg13(pretrained=True),testloader))
# print("vgg16 untrained: ",evaluate_accuracy(torchmodels.vgg16(pretrained=False),trainloader_eval),evaluate_accuracy(torchmodels.vgg16(pretrained=False),testloader))
# print("vgg16 pretrained: ",evaluate_accuracy(torchmodels.vgg16(pretrained=True),trainloader_eval),evaluate_accuracy(torchmodels.vgg16(pretrained=True),testloader))
# print("vgg19 untrained: ",evaluate_accuracy(torchmodels.vgg19(pretrained=False),trainloader_eval),evaluate_accuracy(torchmodels.vgg19(pretrained=False),testloader))
# print("vgg19 pretrained: ",evaluate_accuracy(torchmodels.vgg19(pretrained=True),trainloader_eval),evaluate_accuracy(torchmodels.vgg19(pretrained=True),testloader))

# epochs = 25
# vgg11_untrained = torchmodels.vgg11(pretrained=False)
# input_lastLayer = vgg11_untrained.classifier[-1].in_features
# vgg11_untrained.classifier[-1] = nn.Linear(input_lastLayer,10)
# print("vgg11 untrained: ",evaluate_accuracy(vgg11_untrained,trainloader_eval),evaluate_accuracy(vgg11_untrained,testloader))
# for epoch in tqdm(range(epochs)):
# 	train_GD_epoch(vgg11_untrained)
# print(f"vgg11 untrained (trained {epochs} epochs): ",evaluate_accuracy(vgg11_untrained,trainloader_eval),evaluate_accuracy(vgg11_untrained,testloader))
# vgg11_pretrained = torchmodels.vgg11(pretrained=True)
# input_lastLayer = vgg11_pretrained.classifier[-1].in_features
# vgg11_pretrained.classifier[-1] = nn.Linear(input_lastLayer,10)
# print("vgg11 pretrained: ",evaluate_accuracy(vgg11_pretrained,trainloader_eval),evaluate_accuracy(vgg11_pretrained,testloader))
# for epoch in tqdm(range(epochs)):
# 	train_GD_epoch(vgg11_pretrained)
# print(f"vgg11 pretrained (trained {epochs} epochs): ",evaluate_accuracy(vgg11_pretrained,trainloader_eval),evaluate_accuracy(vgg11_pretrained,testloader))

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="Running bias-variance experiments for different VGG models on CIFAR10")
	parser.add_argument("--model_class",type=int,default=11,help="Should be one of 11,13,16 or 19")
	parser.add_argument("--random_inits",type=int,default=5,help="Number of random seeds")
	parser.add_argument("--pretrained",type=str,default="True",help="use pretrained model weights or random weights")
	parser.add_argument("--batchnorm",type=str,default="False",help="use batchnorm in model or not")
	args = parser.parse_args()
	argsdict = args.__dict__
	print(argsdict)
	run_local = False
	model_class_num = argsdict["model_class"]
	assert model_class_num in [11,13,16,19], "model_class should be 11, 13, 16 or 19 because other VGGs don't exist!"
	if model_class_num==11:
		if argsdict["batchnorm"]=="True":
			model_class = torchmodels.vgg11_bn
		else:
			model_class = torchmodels.vgg11
	elif model_class_num==13:
		if argsdict["batchnorm"]=="True":
			model_class = torchmodels.vgg13_bn
		else:
			model_class = torchmodels.vgg13
	elif model_class_num==16:
		if argsdict["batchnorm"]=="True":
			model_class = torchmodels.vgg16_bn
		else:
			model_class = torchmodels.vgg16
	else:
		if argsdict["batchnorm"]=="True":
			model_class = torchmodels.vgg19_bn
		else:
			model_class = torchmodels.vgg19
	pretrained = True if argsdict["pretrained"]=="True" else False
	# res = biased_training_VGG(random_inits=argsdict["random_inits"],model_class=model_class,pretrained=pretrained,epochs=10,lr=1e-3,bias=0.02)
	# vary_bias_over_training_VGG(model_class=model_class,pretrained=pretrained)
	res = noisy_training_VGG(random_inits=argsdict["random_inits"],model_class=model_class,pretrained=pretrained,epochs=3,variance=20,lr=1e-4)
	if run_local:
		plt.show()
	else:
		save_all_figs_to_disk(folder='CIFAR_results')
