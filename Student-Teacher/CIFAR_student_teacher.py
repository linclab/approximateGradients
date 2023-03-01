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
# from pyhessian import hessian
import gc, copy
import CIFAR_VGG_models.cifar_vgg as cifar_vgg

batch_size_train = 5000
batch_size_test = 5000

class CIFAR_TeacherDataset(torch.utils.data.Dataset):
	def __init__(self,folder='CIFAR_VGG_models',train=True,random_net=False):
		super(CIFAR_TeacherDataset, self).__init__()
		self.folder = folder
		self.train = train
		fname_X = os.path.join(self.folder,'{}teacher_cifar_X_{}.pt'.format('random_' if random_net else '','train' if self.train else 'test'))
		fname_Y = os.path.join(self.folder,'{}teacher_cifar_Y_{}.pt'.format('random_' if random_net else '','train' if self.train else 'test'))
		self.X = torch.load(fname_X)
		self.Y = torch.load(fname_Y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self,idx):
		return self.X[idx], self.Y[idx]

class studentVGG(nn.Module):
	def __init__(self,arch='vgg11',pretrained=False):
		super(studentVGG,self).__init__()
		self.arch = arch
		self.pretrained = pretrained
		orig_model = torchmodels.__dict__[self.arch](pretrained=self.pretrained)
		self.vgg_model = cifar_vgg.VGG(orig_model.features)
		del orig_model

	def forward(self, x) -> torch.Tensor:
		out = self.vgg_model(x)
		return out

def load_student_net(arch='vgg11',pretrained=False,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = studentVGG(arch=arch,pretrained=pretrained)
	return student_net

def get_loss(net,dataloader,loss_fn):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(device)
	tot_loss = 0
	for i,data in enumerate(dataloader):
		x,y = data
		x = x.to(device)
		y = y.to(device)
		with torch.no_grad():
			y_hat = net(x)
		loss = loss_fn(y_hat,y)
		tot_loss += loss.item()*x.size(0)
	# breakpoint()
	tot_loss /= len(dataloader.dataset)
	return tot_loss

def train_GD_epoch(net,dataloader,loss_fn,lr=1e-3,bias=0.0,variance=0.0,grad_noise=None,no_update=False):
	# net.train()
	net.eval()
	net.zero_grad()
	# criterion = nn.CrossEntropyLoss()
	criterion = loss_fn
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if 'cuda' in device.type:
		net = net.cuda()
		# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
		# torch.backends.cudnn.benchmark = True
	running_loss = 0.0
	for i, data in enumerate(dataloader):
		x,y = data
		if 'cuda' in device.type:
			x = x.cuda()
			y = y.cuda()
		outputs = net(x)
		loss = criterion(outputs, y)
		loss.backward()
		running_loss += loss.item()*x.size(0)
		# break
	# params = list(net.parameters())
	# for p in params:
	# 	p.data -= lr*p.grad
	breakpoint()
	grad = gradient_descent_one_step(net,lr=lr,bias=bias,variance=variance,grad_noise=grad_noise,no_update=no_update)
	return grad, running_loss/len(dataloader.dataset)

def biased_training(arch,pretrained,train_dataloader,val_dataloader,seed_start=1,seeds=20,loss_fn=nn.MSELoss(reduction='mean'),epochs=5,lr=1e-3,bias=0.01,verbose=False):
	all_tr_loss_arr = []
	all_tr_loss_control_arr = []
	all_val_loss_arr = []
	all_val_loss_control_arr = []
	all_grad_norm_arr = []
	for sidx in tqdm(range((seed_start-1)*seeds,seeds*seed_start)):
		tqdm.write('{}'.format(sidx))
		student_net = load_student_net(arch=arch,pretrained=pretrained,seed=sidx)
		student_net_control = load_student_net(arch=arch,pretrained=pretrained,seed=sidx)
		assert check_model_params_equal(student_net,student_net_control), "Init problem for student nets!"
		loss_function = loss_fn
		random_tr_loss = get_loss(net=student_net,dataloader=train_dataloader,loss_fn=loss_function)
		tr_loss_arr = [random_tr_loss]
		tr_loss_control_arr = [random_tr_loss]
		random_val_loss = get_loss(net=student_net,dataloader=val_dataloader,loss_fn=loss_function)
		val_loss_arr = [random_val_loss]
		val_loss_control_arr = [random_val_loss]
		grad_norm_arr = []
		if loss_fn.reduction=='sum':
			use_lr = lr/(X.size(0)*Y.size(-1))
		else:
			use_lr = lr
		for epoch in range(epochs):
			copy_from_model_to_model(student_net_control,student_net)
			(grad,noise),loss = train_GD_epoch(net=student_net,dataloader=train_dataloader,loss_fn=loss_function,lr=use_lr,bias=bias)
			tr_loss_arr.append(get_loss(net=student_net,dataloader=train_dataloader,loss_fn=loss_function))
			val_loss_arr.append(get_loss(net=student_net,dataloader=val_dataloader,loss_fn=loss_function))
			# tqdm.write("angle: {:.4f}, grad_norm: {:.4f} ({:.4f}), noise_norm: {:.4f}".format(torch.acos(torch.sum(grad*noise)/(torch.norm(grad)*torch.norm(noise)))*180/np.pi,torch.norm(grad),torch.sum(grad),torch.norm(noise)))
			# breakpoint()
			(grad_control,_),loss_control = train_GD_epoch(net=student_net_control,dataloader=train_dataloader,loss_fn=loss_function,lr=use_lr,bias=0.0)
			tr_loss_control_arr.append(get_loss(net=student_net_control,dataloader=train_dataloader,loss_fn=loss_function))
			val_loss_control_arr.append(get_loss(net=student_net_control,dataloader=val_dataloader,loss_fn=loss_function))
			grad_norm_arr.append(torch.norm(grad_control.cpu()))
		all_tr_loss_arr.append(tr_loss_arr)
		all_tr_loss_control_arr.append(tr_loss_control_arr)
		all_val_loss_arr.append(val_loss_arr)
		all_val_loss_control_arr.append(val_loss_control_arr)
		all_grad_norm_arr.append(grad_norm_arr)
	return all_tr_loss_arr,all_tr_loss_control_arr,all_val_loss_arr,all_val_loss_control_arr, all_grad_norm_arr

def noisy_training(arch,pretrained,train_dataloader,val_dataloader,seed_start=1,seeds=20,repeats=20,loss_fn=nn.MSELoss(reduction='mean'),epochs=4,lr=1e-3,variance=0.01,verbose=False):
	all_tr_loss_arr = []
	all_tr_loss_control_arr = []
	all_val_loss_arr = []
	all_val_loss_control_arr = []
	all_grad_norm_arr = []
	for sidx in tqdm(range((seed_start-1)*seeds,seeds*seed_start)):
		tqdm.write('{}'.format(sidx))
		student_net = load_student_net(arch=arch,pretrained=pretrained,seed=sidx)
		student_net_control = load_student_net(arch=arch,pretrained=pretrained,seed=sidx)
		assert check_model_params_equal(student_net,student_net_control), "Init problem for student nets!"
		loss_function = loss_fn
		random_tr_loss = get_loss(net=student_net,dataloader=train_dataloader,loss_fn=loss_function)
		tr_loss_arr = [random_tr_loss]
		tr_loss_control_arr = [random_tr_loss]
		random_val_loss = get_loss(net=student_net,dataloader=val_dataloader,loss_fn=loss_function)
		val_loss_arr = [random_val_loss]
		val_loss_control_arr = [random_val_loss]
		grad_norm_arr = []
		if loss_fn.reduction=='sum':
			use_lr = lr/(X.size(0)*Y.size(-1))
		else:
			use_lr = lr
		for epoch in range(epochs):
			mean_tr_loss = 0
			mean_val_loss = 0
			grad_noise_sum = 0
			for ridx in range(repeats):
				if ridx!=repeats-1:
					grad_noise_vec = torch.randn(sum(p.numel() for p in student_net.parameters()))
					grad_noise_vec = variance*grad_noise_vec/torch.norm(grad_noise_vec)
					grad_noise_sum += grad_noise_vec
				else:
					grad_noise_vec = -grad_noise_sum
				copy_from_model_to_model(student_net_control,student_net)
				(grad,noise),loss = train_GD_epoch(net=student_net,dataloader=train_dataloader,loss_fn=loss_function,lr=use_lr,variance=variance,grad_noise=grad_noise_vec)
				mean_tr_loss += get_loss(net=student_net,dataloader=train_dataloader,loss_fn=loss_function)
				mean_val_loss += get_loss(net=student_net,dataloader=val_dataloader,loss_fn=loss_function)
				# mean_loss += get_loss(net=student_net,inputs=X,targets=Y,loss_fn=loss_function)
			mean_tr_loss /= repeats
			mean_val_loss /= repeats
			tr_loss_arr.append(mean_tr_loss)
			val_loss_arr.append(mean_val_loss)
			(grad_control,_),loss_control = train_GD_epoch(net=student_net_control,dataloader=train_dataloader,loss_fn=loss_function,lr=use_lr,bias=0.0,variance=0.0)
			tr_loss_control_arr.append(get_loss(net=student_net_control,dataloader=train_dataloader,loss_fn=loss_function))
			val_loss_control_arr.append(get_loss(net=student_net_control,dataloader=val_dataloader,loss_fn=loss_function))
			grad_norm_arr.append(torch.norm(grad_control.cpu()))
		all_tr_loss_arr.append(tr_loss_arr)
		all_tr_loss_control_arr.append(tr_loss_control_arr)
		all_val_loss_arr.append(val_loss_arr)
		all_val_loss_control_arr.append(val_loss_control_arr)
		all_grad_norm_arr.append(grad_norm_arr)
	return all_tr_loss_arr,all_tr_loss_control_arr,all_val_loss_arr,all_val_loss_control_arr, all_grad_norm_arr


def plot_results_delta_loss(all_tr_loss_arr,all_tr_loss_control_arr,all_val_loss_arr,all_val_loss_control_arr,bias=0.0,variance=0.0,fig=None):
	tr_loss_arr_np = np.array(all_tr_loss_arr)
	tr_loss_control_arr_np = np.array(all_tr_loss_control_arr)
	delta_tr_loss = tr_loss_arr_np-tr_loss_control_arr_np
	delta_tr_loss_mean = np.mean(delta_tr_loss,axis=0)
	delta_tr_loss_sterr = np.std(delta_tr_loss,axis=0)/np.sqrt(delta_tr_loss.shape[0])
	if fig is not None:
		plt.figure(fig.number)
	plt.errorbar(x=np.arange(1,delta_tr_loss.shape[1]),y=delta_tr_loss_mean[1:],yerr=delta_tr_loss_sterr[1:],color='b',ls='-',label='Bias = {:.3f}, variance = {:.3f} (Train)'.format(bias,variance))
	val_loss_arr_np = np.array(all_val_loss_arr)
	val_loss_control_arr_np = np.array(all_val_loss_control_arr)
	delta_val_loss = val_loss_arr_np-val_loss_control_arr_np
	delta_val_loss_mean = np.mean(delta_val_loss,axis=0)
	delta_val_loss_sterr = np.std(delta_val_loss,axis=0)/np.sqrt(delta_val_loss.shape[0])
	plt.errorbar(x=np.arange(1,delta_val_loss.shape[1]),y=delta_val_loss_mean[1:],yerr=delta_val_loss_sterr[1:],color='r',ls=':',label='Bias = {:.3f}, variance = {:.3f} (Val)'.format(bias,variance))
	plt.axhline(y=0.0,color='k',ls='--')
	plt.legend()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Running student-teacher for CIFAR dataset")
	parser.add_argument("--bias",type=float,default=0.0,help="Bias value")
	parser.add_argument("--variance",type=float,default=0.0,help="Variance value")
	parser.add_argument("--random_teacher",type=str,default='False',help="Random or CIFAR-trained teacher network")
	parser.add_argument("--arch",type=str,default='vgg11',help="VGG architecture")
	parser.add_argument("--pretrained",type=str,default='False',help="pretrained or untrained")
	parser.add_argument("--seed_start",type=int,default=1,help="seed start (int)")
	parser.add_argument("--seeds",type=int,default=10,help="number of seeds")
	parser.add_argument("--save_dir",type=str,default='CIFAR_results/Student_Teacher_metadata',help="seed start (int)")
	args = parser.parse_args()
	argsdict = args.__dict__
	print(argsdict)

	bias = argsdict['bias']
	variance = argsdict['variance']
	random_net = True if argsdict['random_teacher']=='True' else False
	arch = argsdict['arch']
	pretrained = True if argsdict['pretrained']=='True' else False
	seed_start = argsdict['seed_start']
	seeds = argsdict['seeds']
	save_dir = argsdict['save_dir']
	assert not (bias==0.0 and variance==0.0), "Bias and variance are both zero"
	if bias!=0:
		save_dir = os.path.join(save_dir,'Bias{}{}'.format('_random' if random_net else '','_negative' if bias<0 else ''))
	elif variance!=0:
		save_dir = os.path.join(save_dir,'Variance{}'.format('_random' if random_net else ''))
		
	train_dataset = CIFAR_TeacherDataset(train=True,random_net=random_net)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=False)
	val_dataset = CIFAR_TeacherDataset(train=False,random_net=random_net)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size_test,shuffle=False)
	# all_tr_loss_arr, all_tr_loss_control_arr, all_val_loss_arr, all_val_loss_control_arr, all_grad_norm_arr = biased_training(arch='vgg11',pretrained=False,train_dataloader=train_dataloader,val_dataloader=val_dataloader,seeds=10,bias=bias)
	if bias!=0: 
		all_tr_loss_arr, all_tr_loss_control_arr, all_val_loss_arr, all_val_loss_control_arr, all_grad_norm_arr = biased_training(
																													arch=arch,
																													pretrained=pretrained,
																													train_dataloader=train_dataloader,
																													val_dataloader=val_dataloader,
																													seed_start=seed_start,seeds=seeds,
																													bias=bias)
	if variance!=0:
		all_tr_loss_arr, all_tr_loss_control_arr, all_val_loss_arr, all_val_loss_control_arr, all_grad_norm_arr = noisy_training(
																													arch=arch,
																													pretrained=pretrained,
																													train_dataloader=train_dataloader,
																													val_dataloader=val_dataloader,
																													seed_start=seed_start,seeds=seeds,
																													variance=variance)
	# breakpoint()
	# plot_results_delta_loss(all_tr_loss_arr,all_tr_loss_control_arr,all_val_loss_arr,all_val_loss_control_arr,bias=bias,variance=variance)
	save_dict = {'tr_loss_arr': np.array(all_tr_loss_arr), 'tr_loss_control_arr': np.array(all_tr_loss_control_arr),
				'val_loss_arr': np.array(all_val_loss_arr), 'val_loss_control_arr': np.array(all_val_loss_control_arr),
				'grad_norm_arr': np.array(all_grad_norm_arr)}
	np.save(os.path.join(save_dir,'{}_{}_{}'.format(arch,'pretrained' if pretrained else 'untrained',seed_start)),save_dict)


