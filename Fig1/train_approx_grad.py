from typing import List
import os, random

# Import torch and torchvision
import torch as ch
import torchvision
import cifar_vgg

# Import ffcv packages
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
	RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

# Import torch cuda autocast packages (for fp16 training) & other torch.nn packages
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, Adam

from tqdm import tqdm

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	ch.manual_seed(seed)
	ch.cuda.manual_seed(seed)
	ch.backends.cudnn.benchmark = False
	ch.backends.cudnn.deterministic = True

def generate_random_noise_sequence(repeats,inp_dim,variance):
	noise_vectors = ch.randn((repeats-1,1+inp_dim))
	noise_vectors = (variance)*ch.div(noise_vectors,ch.norm(noise_vectors,dim=1).unsqueeze(1))
	noise_vectors = ch.vstack([noise_vectors,-noise_vectors.sum(dim=0)])    # to ensure that sum of random noise vectors is 0
	return noise_vectors

def get_grad_vector(net):
	params = list(net.parameters())
	grad_list = []
	for p in params:
		if p.requires_grad:
			grad_list.append(p.grad.data.flatten())
	grad_vector = ch.cat(grad_list)
	return params, grad_vector

def corrupt_gradient(net,bias=0.0,variance=0.0,grad_noise=None,relative=True,no_update=True):
	params, grad_vector = get_grad_vector(net)
	if relative:
		grad_norm = ch.norm(grad_vector).item()
		variance *= grad_norm
		bias = bias*grad_norm
		if grad_noise is not None and ch.norm(grad_noise)>0:
			grad_noise = (variance)*ch.div(grad_noise,ch.norm(grad_noise))
	# We don't divide bias by grad_vector size because we are assuming bias in individual synaptic wt. update!
	
	if grad_noise is None:
		breakpoint()
		grad_noise = ch.normal(mean=-bias*ch.ones(grad_vector.size()),std=np.sqrt(variance)*ch.ones(grad_vector.size())).to(grad_vector.device)
	else:
		grad_noise -= bias
	# NOTE: -ve sign in bias above because this will be added to the gradient and thereafter subtracted from current wt.
	# Hence this has reverse effect of the theory assumption!
	# breakpoint()
	# tqdm.write("{:.3f} {:.3f}".format(grad_vector.norm().item(), grad_noise.norm().item()))
	device = grad_vector.device
	new_grad_vector = grad_vector+grad_noise.to(device)
	param_pointer = 0
	for p in params:
		if p.requires_grad:
			p_size = p.flatten().size(0)
			new_grad = new_grad_vector[param_pointer:param_pointer+p_size].view(p.size())
			p.grad.data = new_grad
			param_pointer += p_size

	# if not no_update:
	# 	for p in params:
	# 		if p.requires_grad:
	# 			p.data -= lr*p.grad
	return (grad_vector, grad_noise)

# def gradient_descent_one_step(net,lr,bias=0.0,variance=0.0,grad_noise=None,no_update=False,relative=False):
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	net = net.to(device)
# 	params = list(net.parameters())
# 	grad_list = []
# 	for p in params:
# 		if p.requires_grad:
# 			grad_list.append(p.grad.data.flatten())
# 	grad_vector = torch.cat(grad_list)
# 	if relative:
# 		grad_norm = torch.norm(grad_vector).item()
# 		bias = bias*grad_norm
# 	# MAYBE BIAS SHOULD BE DIVIDED BY GRAD_VECTOR SIZE TO CONTROL FOR DIFFERENT NETWORK SIZE -- No it should not (we are looking at bias in synaptic wt. update)!
# 	# bias = bias/np.sqrt(len(grad_vector))
# 	if grad_noise is None:
# 		grad_noise = torch.normal(mean=-bias*torch.ones(grad_vector.size()),std=np.sqrt(variance)*torch.ones(grad_vector.size())).to(grad_vector.device)
# 	else:
# 		grad_noise -= bias
# 	# NOTE: -ve sign in bias above because this will be added to the gradient and thereafter subtracted from current wt.
# 	# Hence this has reverse effect of the theory assumption!
# 	# tqdm.write("{} {}".format(torch.norm(grad_vector).item(),torch.norm(grad_noise)))
# 	new_grad_vector = grad_vector+grad_noise.to(device)
# 	param_pointer = 0
# 	for p in params:
# 		if p.requires_grad:
# 			p_size = p.flatten().size(0)
# 			new_grad = new_grad_vector[param_pointer:param_pointer+p_size].view(p.size())
# 			p.grad.data = new_grad
# 			param_pointer += p_size

# 	if not no_update:
# 		for p in params:
# 			if p.requires_grad:
# 				p.data -= lr*p.grad
# 	return (grad_vector, grad_noise)

set_seed(42)

save_dir = 'checkpoints_relative_no_lrd'
ffcv_datadir = os.path.join('/network/projects/_groups/linclab_users/ffcv/','ffcv_datasets') 	# 
dataset = 'cifar10'
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

BATCH_SIZE = 5000
seeds = 3

# Define the dataloaders
# ==================================================================================================================================
loaders = {}
for name in ['train', 'test']:
	# initialize the labels pipeline
	label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
	# initialize the Image pipeline
	image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

	# Add image transforms and normalization
	# if name == 'train':
	# 	image_pipeline.extend([
	# 		RandomHorizontalFlip(),
	# 		RandomTranslate(padding=2),
	# 		Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
	# 	])
	# Add the conversion to cuda and torch Tensor transformations (same for train and validation/test) 
	image_pipeline.extend([
		ToTensor(),
		ToDevice('cuda:0', non_blocking=True),
		ToTorchImage(),
		Convert(ch.float16),
		torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])

	# Create loaders
	loaders[name] = Loader(f'{ffcv_datadir}/{dataset}/{name}.beton',
							batch_size=BATCH_SIZE,
							num_workers=2, 			# Change the number of workers here if you are using 1 cpu	
							order=OrderOption.RANDOM,
							drop_last=(name == 'train'),
							pipelines={'image': image_pipeline,
									   'label': label_pipeline})
# ==================================================================================================================================

def load_model():
	model = torchvision.models.vgg16(pretrained=True)
	# model = torchvision.models.vgg11(pretrained=True)
	model.classifier[6] = ch.nn.Linear(model.classifier[6].in_features,10)

	model = model.to(memory_format=ch.channels_last).cuda()
	return model

# Training and evaluation functions
# ==================================================================================================================================
def train(model,opt,scheduler,loss_fn,dataloader,scaler,bias=0,variance=0,grad_noise=None):
	model.train()
	opt.zero_grad(set_to_none=True)
	for ims, labs in dataloader:
		with autocast():
			out = model(ims)
			loss = loss_fn(out, labs)/len(dataloader)

		scaler.scale(loss).backward()
	(grad,grad_noise) = corrupt_gradient(net=model,bias=bias,variance=variance,grad_noise=grad_noise)
	scaler.step(opt)
	scaler.update()
	# scheduler.step()
	return grad_noise

def eval(model,dataloader,loss_fn):
	model.eval()
	with ch.no_grad():
		total_correct, total_num, total_loss = 0., 0., 0.
		for ims, labs in dataloader:
			with autocast():
				# out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
				out = model(ims)
				total_loss += loss_fn(out, labs).item()/len(dataloader)
				total_correct += out.argmax(1).eq(labs).sum().cpu().item()
				total_num += ims.shape[0]
	return total_correct / total_num * 100, total_loss
# ==================================================================================================================================

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Running bias-variance exploratory experiments for different VGG models on CIFAR10")
	parser.add_argument("--bias",type=float,default=0.0,help="Bias value")
	parser.add_argument("--variance",type=float,default=0.0,help="Variance value")
	parser.add_argument("--epochs",type=int,default=50,help="Number of training epochs")
	parser.add_argument("--logfreq",type=int,default=2,help="Logging frequency")
	args = parser.parse_args()
	print(args)

	bias = args.bias
	variance = args.variance
	res_dict = {}
	for seed in range(seeds):
		set_seed(seed)
		res_dict[seed] = {}
		noise_running_mean = None
		model = load_model()
		num_params = sum(p.numel() for p in model.parameters())
		# print(num_params)
		# Setup your optimization parameters -- taken from ffcv example
		# ==================================================================================================================================
		EPOCHS = args.epochs #100
		lr = 0.02 # 0.02#/100
		weight_decay = 0.0 #5e-4
		momentum = 0.9
		opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		# iters_per_epoch = 50000 // BATCH_SIZE
		# lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
		# 						[0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
		# 						[0, 1, 0])
		# scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
		scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=200)
		scaler = GradScaler()
		loss_fn = CrossEntropyLoss(label_smoothing=0.1)

		log_epochs = args.logfreq #5
		# ==================================================================================================================================
		print("Training model for {} epochs with {} optimizer, lr={}, wd={:.1e}; bias={}, variance={}".format(
									EPOCHS,opt.__class__.__name__,lr,weight_decay,bias,variance))
		train_acc_arr = []
		train_loss_arr = []
		test_acc_arr = []
		test_loss_arr = []
		progress_bar = tqdm(range(1,1+EPOCHS))

		for ep in progress_bar:
			if ep<EPOCHS:
				noise_vector = ch.normal(mean=0,std=ch.ones(num_params)) #ch.randn(num_params)
				noise_vector = (variance)*ch.div(noise_vector,ch.norm(noise_vector))
			else:
				noise_vector = -noise_running_mean
			if noise_running_mean is None:
				noise_running_mean = noise_vector
			else:	
				noise_running_mean += noise_vector
			grad_corruption = train(model=model,opt=opt,scheduler=scheduler,loss_fn=loss_fn,dataloader=loaders['train'],scaler=scaler,
									bias=bias,variance=variance,grad_noise=noise_vector)
			if ep%log_epochs==0:
				train_acc,train_loss = eval(model=model,dataloader=loaders['train'],loss_fn=loss_fn)
				test_acc,test_loss = eval(model=model,dataloader=loaders['test'],loss_fn=loss_fn)
				progress_bar.set_description("Test accuracy after {} epochs: {:.2f}".format(ep,test_acc))
				train_acc_arr.append(train_acc)
				train_loss_arr.append(train_loss)
				test_acc_arr.append(test_acc)
				test_loss_arr.append(test_loss)

		try:
			print(f'Accuracy: {total_correct / total_num * 100:.1f}%')
		except:
			pass
		print(train_loss_arr)
		print(test_loss_arr)
		print(train_acc_arr)
		print(test_acc_arr)
		print(np.sum(np.array(train_acc_arr)),np.sum(np.array(test_acc_arr)))
		res_dict[seed]['train_acc_arr'] = train_acc_arr
		res_dict[seed]['train_loss_arr'] = train_loss_arr
		res_dict[seed]['test_acc_arr'] = test_acc_arr
		res_dict[seed]['test_loss_arr'] = test_loss_arr
	save_fname = 'vgg16_bias_{}_variance_{}.npy'.format(bias,variance)
	np.save(os.path.join(save_dir,save_fname),res_dict,allow_pickle=True)
