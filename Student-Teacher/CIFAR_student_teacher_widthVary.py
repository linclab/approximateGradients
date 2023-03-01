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


# class studentVGG(nn.Module):
# 	def __init__(self,arch='vgg11',pretrained=False):
# 		super(studentVGG,self).__init__()
# 		self.arch = arch
# 		self.pretrained = pretrained
# 		orig_model = torchmodels.__dict__[self.arch](pretrained=self.pretrained)
# 		self.vgg_model = cifar_vgg.VGG(orig_model.features)
# 		del orig_model

# 	def forward(self, x) -> torch.Tensor:
# 		out = self.vgg_model(x)
# 		return out

cfg = {
	'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class flexible_VGG(nn.Module):
	def __init__(self, vgg_name, downsampling_factor, init_weights=True, num_classes=10, mini_exp=False):
		super(flexible_VGG, self).__init__()
		if len(vgg_name.split("_"))>1:
			self.vgg_name = vgg_name.split("_")[0]
			self.batch_norm = True
		else:
			self.vgg_name = vgg_name
			self.batch_norm = False
		self.downsampling_factor = 2**int(np.log2(int(downsampling_factor)))
		assert self.downsampling_factor<=32, "Cannot set downsampling_factor greater than 32"
		self.num_classes = num_classes
		self.features = self._make_layers(cfg[self.vgg_name])
		self.classifier = self._make_classifier()
		if init_weights:
			self._initialize_weights()
		self.mini_exp = mini_exp
		if self.mini_exp:
			for pidx,params in enumerate(self.parameters()):
				if pidx<len(list(self.parameters()))-2:
					params.requires_grad=False
		self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
	
	def forward(self, x) -> torch.Tensor:
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out
	
	def _make_layers(self, cfg) -> nn.Sequential:
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				if self.batch_norm:
					layers += [nn.Conv2d(in_channels, x//self.downsampling_factor, kernel_size=3, padding=1),
							   nn.BatchNorm2d(x//self.downsampling_factor),
							   nn.ReLU(inplace=True)]
				else:
					layers += [nn.Conv2d(in_channels, x//self.downsampling_factor, kernel_size=3, padding=1),
							   nn.ReLU(inplace=True)]
				in_channels = x//self.downsampling_factor
		# layers += [nn.AdaptiveAvgPool2d((7, 7))]	# No AdaptiveAvgPool in teacher network!
		return nn.Sequential(*layers)

	def _initialize_weights(self) -> None:
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				# nn.init.normal_(m.weight, 0, 0.01)
				nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
	
	def _make_classifier(self):
		classifier = nn.Sequential(
			nn.Linear(512//self.downsampling_factor, 512//self.downsampling_factor),
			nn.ReLU(True),
			# nn.Dropout(),
			nn.Linear(512//self.downsampling_factor, 512//self.downsampling_factor),
			nn.ReLU(True),
			# nn.Dropout(),
			nn.Linear(512//self.downsampling_factor, self.num_classes),
		)
		return classifier

# def get_model(model_class=torchmodels.vgg11,downsampling_factor=1):
# 	assert model_class in [torchmodels.vgg11,torchmodels.vgg13,torchmodels.vgg16,torchmodels.vgg19,
# 							torchmodels.vgg11_bn,torchmodels.vgg13_bn,torchmodels.vgg16_bn,torchmodels.vgg19_bn], NotImplementedError
# 	model_num = model_class.__name__.split('_')[0].split('vgg')[-1]
# 	batch_norm = True if len(model_class.__name__.split('_'))>1 else False
# 	model_name = 'VGG{}{}'.format(model_num,'_bn' if batch_norm else '')
# 	model = flexible_VGG(vgg_name=model_name,downsampling_factor=downsampling_factor)
# 	return model

def load_student_net(arch='vgg11',downsampling_factor=1,seed=13):
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = flexible_VGG(vgg_name=arch,downsampling_factor=downsampling_factor)
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
	# breakpoint()
	grad = gradient_descent_one_step(net,lr=lr,bias=bias,variance=variance,grad_noise=grad_noise,no_update=no_update)
	return grad, running_loss/len(dataloader.dataset)

def biased_training(arch,downsampling_factor,train_dataloader,val_dataloader,seed_start=1,seeds=20,loss_fn=nn.MSELoss(reduction='mean'),epochs=5,lr=1e-3,bias=0.01,verbose=False):
	all_tr_loss_arr = []
	all_tr_loss_control_arr = []
	all_val_loss_arr = []
	all_val_loss_control_arr = []
	all_grad_norm_arr = []
	for sidx in tqdm(range((seed_start-1)*seeds,seeds*seed_start)):
		tqdm.write('{}'.format(sidx))
		student_net = load_student_net(arch=arch,downsampling_factor=downsampling_factor,seed=sidx)
		student_net_control = load_student_net(arch=arch,downsampling_factor=downsampling_factor,seed=sidx)
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

def noisy_training(arch,downsampling_factor,train_dataloader,val_dataloader,seed_start=1,seeds=20,repeats=20,loss_fn=nn.MSELoss(reduction='mean'),epochs=4,lr=1e-3,variance=0.01,verbose=False):
	all_tr_loss_arr = []
	all_tr_loss_control_arr = []
	all_val_loss_arr = []
	all_val_loss_control_arr = []
	all_grad_norm_arr = []
	for sidx in tqdm(range((seed_start-1)*seeds,seeds*seed_start)):
		tqdm.write('{}'.format(sidx))
		student_net = load_student_net(arch=arch,downsampling_factor=downsampling_factor,seed=sidx)
		student_net_control = load_student_net(arch=arch,downsampling_factor=downsampling_factor,seed=sidx)
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
	parser.add_argument("--downsampling_factor",type=int,default=1,help="Width downsampling factor")
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
	downsampling_factor = argsdict['downsampling_factor']
	seed_start = argsdict['seed_start']
	seeds = argsdict['seeds']
	save_dir = argsdict['save_dir']
	assert not (bias==0.0 and variance==0.0), "Bias and variance are both zero"
	if bias!=0:
		save_dir = os.path.join(save_dir,'Bias{}_widthVary'.format('_random' if random_net else ''))
	elif variance!=0:
		save_dir = os.path.join(save_dir,'Variance{}_widthVary'.format('_random' if random_net else ''))

	train_dataset = CIFAR_TeacherDataset(train=True,random_net=random_net)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=False)
	val_dataset = CIFAR_TeacherDataset(train=False,random_net=random_net)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size_test,shuffle=False)
	if bias!=0: 
		all_tr_loss_arr, all_tr_loss_control_arr, all_val_loss_arr, all_val_loss_control_arr, all_grad_norm_arr = biased_training(
																													arch=arch,
																													downsampling_factor=downsampling_factor,
																													train_dataloader=train_dataloader,
																													val_dataloader=val_dataloader,
																													seed_start=seed_start,seeds=seeds,
																													bias=bias)
	if variance!=0:
		all_tr_loss_arr, all_tr_loss_control_arr, all_val_loss_arr, all_val_loss_control_arr, all_grad_norm_arr = noisy_training(
																													arch=arch,
																													downsampling_factor=downsampling_factor,
																													train_dataloader=train_dataloader,
																													val_dataloader=val_dataloader,
																													seed_start=seed_start,seeds=seeds,
																													variance=variance)
	# breakpoint()
	# plot_results_delta_loss(all_tr_loss_arr,all_tr_loss_control_arr,all_val_loss_arr,all_val_loss_control_arr,bias=bias,variance=variance)
	save_dict = {'tr_loss_arr': np.array(all_tr_loss_arr), 'tr_loss_control_arr': np.array(all_tr_loss_control_arr),
				'val_loss_arr': np.array(all_val_loss_arr), 'val_loss_control_arr': np.array(all_val_loss_control_arr),
				'grad_norm_arr': np.array(all_grad_norm_arr)}
	np.save(os.path.join(save_dir,'{}_{}_{}'.format(arch,downsampling_factor,seed_start)),save_dict)
