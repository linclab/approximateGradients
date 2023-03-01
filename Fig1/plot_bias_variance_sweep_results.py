import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
import os, glob
from linclab_utils import plot_utils
from tqdm import tqdm
from sklearn.metrics import r2_score

plot_utils.linclab_plt_defaults(font="Arial",fontdir=os.path.expanduser('~')+"/Projects/fonts") 	# Run locally, not from cluster

dataset = 'cifar10'
ckpt_dir = 'checkpoints_relative_no_lrd'

def get_tick_positions(locs,ticks,vals):
	slope = (locs[-1]-locs[0])/(ticks[-1]-ticks[0])
	pos = slope*(vals-ticks[0]) + locs[0]
	return pos

def plot_colorplot(variance_arr,bias_arr,attribute_dict,attribute_label,vmin=None,vmax=None,cmap='seismic'):
	plotting_arr = np.empty((len(variance_arr),len(bias_arr)))
	plotting_arr[:] = np.nan
	for pidx,pdim in enumerate(variance_arr):
		for lidx,lamda in enumerate(bias_arr):
			try:
				if type(attribute_dict[pdim][lamda]) is list:
					plotting_arr[pidx,lidx] = np.mean(np.array(attribute_dict[pdim][lamda]))
				else:
					plotting_arr[pidx,lidx] = attribute_dict[pdim][lamda]
			except:
				pass
	plt.figure()
	cmap.set_under('gold')
	plt.imshow(plotting_arr,cmap=cmap,vmin=vmin,vmax=vmax,origin='lower')
	# cbar = plt.colorbar()
	cbar = plt.colorbar(extend='min')
	cbar.set_label('Accuracy',rotation=270,labelpad=25,fontsize=25)
	plt.title(attribute_label,fontsize=27)
	# breakpoint()
	plt.yticks(np.arange(0,len(variance_arr),3),variance_arr[np.arange(0,len(variance_arr),3)])
	locs, _ = plt.yticks()
	locs1 = locs[1:]
	ticks1 = np.log10(variance_arr[np.arange(0,len(variance_arr),3)][1:])
	# vals = np.linspace(0,8,5)
	vals = np.linspace(-2,2,3)
	pos = get_tick_positions(locs=locs1,ticks=ticks1,vals=vals)
	new_locs = np.insert(pos,0,locs[0])
	new_ticks = np.insert(10**vals,0,variance_arr[0])
	string_labels = []
	for i in range(len(new_ticks)):
		if new_ticks[i]==0.0:
			string_labels.append('0.0')
		else:
			string_labels.append(r"$10^{%d}$" % (np.log10(new_ticks[i])))
	plt.yticks(new_locs,string_labels)

	# plt.gca().set_yticklabels(FormatStrFormatter('%.1e').format_ticks(new_ticks))
	# breakpoint()
	plt.ylabel('Variance',fontsize=23)
	plt.xticks(np.arange(0,len(bias_arr),4),bias_arr[np.arange(0,len(bias_arr),4)])
	locs, _ = plt.xticks()
	locs1 = locs[1:]
	ticks1 = np.log10(bias_arr[np.arange(0,len(bias_arr),4)][1:])
	# vals = np.linspace(-2,2,3)
	vals = np.linspace(-6,-2,3)
	pos = get_tick_positions(locs=locs1,ticks=ticks1,vals=vals)
	new_locs = np.insert(pos,0,locs[0])
	new_ticks = np.insert(10**vals,0,variance_arr[0])
	string_labels = []
	for i in range(len(new_ticks)):
		if new_ticks[i]==0.0:
			string_labels.append('0.0')
		else:
			string_labels.append(r"$10^{%d}$" % (np.log10(new_ticks[i])))
	plt.xticks(new_locs,string_labels)
	# plt.gca().set_xticklabels(FormatStrFormatter('%.1e').format_ticks(new_ticks))
	plt.xlabel('Bias',fontsize=23)

def plot_scatterplot(attribute1_dict,attribute1_label,attribute2_dict,attribute2_label,filter_dict=None,vmin=None,vmax=None,hmin=None,hmax=None):
	plt.figure()
	markers_arr = ['o','^','x','s','d','v','*']
	for pidx,pdim in enumerate(attribute1_dict.keys()):
		attr1_arr = []
		attr1_err_arr = []
		attr2_arr = []
		attr2_err_arr = []
		for lidx,lamda in enumerate(attribute1_dict[pdim].keys()):
			if filter_dict is not None and np.nanmean(np.array(filter_dict[pdim][lamda]))<R2_thresh: continue
			if type(attribute1_dict[pdim][lamda]) is list:
				attr1_arr.append(np.mean(np.array(attribute1_dict[pdim][lamda])))
				attr1_err_arr.append(np.std(np.array(attribute1_dict[pdim][lamda])))
			else:
				attr1_arr.append(attribute1_dict[pdim][lamda])
				attr1_err_arr.append(0)
			if type(attribute2_dict[pdim][lamda]) is list:
				attr2_arr.append(np.mean(np.array(attribute2_dict[pdim][lamda])))
				attr2_err_arr.append(np.std(np.array(attribute2_dict[pdim][lamda])))
			else:
				attr2_arr.append(attribute2_dict[pdim][lamda])
				attr2_err_arr.append(0)
	
		# plt.scatter(attr1_arr,attr2_arr,marker=markers_arr[pidx%len(markers_arr)],label='pdim={}'.format(int(pdim)))
		plt.errorbar(x=attr1_arr,y=attr2_arr,xerr=attr1_err_arr,yerr=attr2_err_arr,marker=markers_arr[pidx%len(markers_arr)],ls='',label='pdim={}'.format(int(pdim)))
	plt.title('{} vs {}'.format(attribute1_label,attribute2_label))
	plt.xlabel('{}'.format(attribute1_label))
	plt.ylabel('{}'.format(attribute2_label))
	plt.legend()
	plt.ylim([vmin,vmax])
	plt.xlim([hmin,hmax])


def plot_surfaceplot(variance_arr,bias_arr,attribute_dict,attribute_label,vmin=None,vmax=None,cmap='seismic'):
	plotting_arr = np.empty((len(variance_arr),len(bias_arr)))
	plotting_arr[:] = np.nan
	x_zero = np.nan
	y_zero = np.nan
	for pidx,pdim in enumerate(variance_arr):
		for lidx,lamda in enumerate(bias_arr):
			try:
				if type(attribute_dict[pdim][lamda]) is list:
					plotting_arr[pidx,lidx] = np.mean(np.array(attribute_dict[pdim][lamda]))
				else:
					plotting_arr[pidx,lidx] = attribute_dict[pdim][lamda]
				if lamda==0.0:
					x_zero = lidx
				if pdim==0.0:
					y_zero = pidx
			except:
				pass
	X,Y = np.meshgrid(np.arange(len(bias_arr)),np.arange(len(variance_arr)))
	# breakpoint()
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	surf = ax.plot_surface(X,Y,plotting_arr,cmap=cmap,linewidth=0,antialiased=False,vmin=vmin,vmax=vmax)
	# plt.imshow(plotting_arr,cmap=cmap,vmin=vmin,vmax=vmax)
	fig.colorbar(surf,ax=ax)
	plt.title(attribute_label)
	plt.yticks(np.arange(0,len(variance_arr),4),variance_arr[np.arange(0,len(variance_arr),4)])
	plt.gca().set_yticklabels(FormatStrFormatter('%.1e').format_ticks(variance_arr[np.arange(0,len(variance_arr),4)]))
	plt.gca().yaxis.set_label_coords(0.5,-0.3)
	# plt.ylabel('Variance')
	plt.xticks(np.arange(0,len(bias_arr),5),bias_arr[np.arange(0,len(bias_arr),5)])
	plt.gca().set_xticklabels(FormatStrFormatter('%.1e').format_ticks(bias_arr[np.arange(0,len(bias_arr),5)]))
	plt.gca().xaxis.set_label_coords(0.5,-0.3)
	# plt.xlabel('Bias')
	ax.scatter(x_zero,y_zero,attribute_dict[0.0][0.0],marker='o',color='k')

files = glob.glob(os.path.join(ckpt_dir,'*'))
sorting_order = [i[0] for i in sorted(enumerate(files),key=lambda x: float(os.path.basename(x[1]).split('bias_')[-1].split('_')[0])+1e9*float(os.path.basename(x[1]).split('variance_')[-1].split('.npy')[0]))]
files_sorted = [files[idx] for idx in sorting_order]

bias_arr = {}
variance_arr = {}
final_train_accuracy_dict = {}
final_test_accuracy_dict = {}
best_test_accuracy_dict = {}
train_accuracy_AUC_dict = {}
test_accuracy_AUC_dict = {}
for fidx,file in enumerate(tqdm(files_sorted)):
	try:
		bias_val = float(os.path.basename(file).split('bias_')[-1].split('_')[0])
		variance_val = float(os.path.basename(file).split('variance_')[-1].split('.npy')[0])
		if bias_val<0: continue
		bias_arr[bias_val]=True
		variance_arr[variance_val]=True
		if variance_val not in final_train_accuracy_dict.keys():
			final_train_accuracy_dict[variance_val] = {}
		if variance_val not in final_test_accuracy_dict.keys():
			final_test_accuracy_dict[variance_val] = {}
		if variance_val not in best_test_accuracy_dict.keys():
			best_test_accuracy_dict[variance_val] = {}
		if variance_val not in train_accuracy_AUC_dict.keys():
			train_accuracy_AUC_dict[variance_val] = {}
		if variance_val not in test_accuracy_AUC_dict.keys():
			test_accuracy_AUC_dict[variance_val] = {}

		acc_file = np.load(file,allow_pickle=True).item()
		if bias_val not in final_train_accuracy_dict[variance_val].keys():
			final_train_accuracy_dict[variance_val][bias_val] = []
			final_test_accuracy_dict[variance_val][bias_val] = []
			best_test_accuracy_dict[variance_val][bias_val] = []
			train_accuracy_AUC_dict[variance_val][bias_val] = []
			test_accuracy_AUC_dict[variance_val][bias_val] = []

		for sidx in acc_file.keys():
			train_acc_arr = np.array(acc_file[sidx]['train_acc_arr'])
			test_acc_arr = np.array(acc_file[sidx]['test_acc_arr'])
			final_train_accuracy_dict[variance_val][bias_val].append(train_acc_arr[-1])
			final_test_accuracy_dict[variance_val][bias_val].append(test_acc_arr[-1])
			best_test_accuracy_dict[variance_val][bias_val].append(test_acc_arr.max())
			train_accuracy_AUC_dict[variance_val][bias_val].append(train_acc_arr.sum())
			test_accuracy_AUC_dict[variance_val][bias_val].append(test_acc_arr.sum())
		
	except:
		breakpoint()
		pass


bias_arr = np.array(list(bias_arr.keys()))
bias_arr.sort()
variance_arr = np.array(list(variance_arr.keys()))
variance_arr.sort()
plot_colorplot(variance_arr,bias_arr,final_train_accuracy_dict,"Train",cmap='coolwarm',vmin=80,vmax=85)
plot_colorplot(variance_arr,bias_arr,final_test_accuracy_dict,"Test",cmap='coolwarm',vmin=80,vmax=83)
plot_colorplot(variance_arr,bias_arr,best_test_accuracy_dict,"Best test accuracy",cmap='coolwarm',vmin=50,vmax=85)
plot_colorplot(variance_arr,bias_arr,train_accuracy_AUC_dict,"Train accuracy AUC",cmap='coolwarm',vmin=1000,vmax=1800)
plot_colorplot(variance_arr,bias_arr,test_accuracy_AUC_dict,"Test accuracy AUC",cmap='coolwarm',vmin=1000,vmax=1800)

plt.show()

