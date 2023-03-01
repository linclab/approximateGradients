import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

metadata_folder = os.path.join('CIFAR_results','Plot_metadata')

def set_boxplot_colors(boxplot,color1='cyan',edgecolor1='navy',color2='gold',edgecolor2='red',spreadcolor='black'):
	for eidx,elem in enumerate(boxplot['medians']):
		if eidx%2:
			elem.set(color=edgecolor2,linewidth=2)
		else:
			elem.set(color=edgecolor1,linewidth=2)
	for element in ['whiskers', 'caps']:
		for eidx,elem in enumerate(boxplot[element]):
			if eidx%4>=2:
				elem.set(color=spreadcolor)
			else:
				elem.set(color=spreadcolor)
	for bidx,box in enumerate(boxplot['boxes']): 
		if bidx%2: 
			box.set(color=edgecolor2,facecolor=color2)
		else:
			box.set(color=edgecolor1,facecolor=color1)

def plot_bias_results(bias=0.02,plot_abs=True):
	data_folder = os.path.join(metadata_folder,'Bias')
	# files = list(set(glob(os.path.join(data_folder,'vgg*.npy'))) - set(glob(os.path.join(data_folder,'vgg*_bn*.npy'))))
	files = glob(os.path.join(data_folder,'vgg*_bn*.npy'))
	files.sort()
	fig1 = plt.figure()
	tr_delta_loss_arr = []
	fig2 = plt.figure()
	te_delta_loss_arr = []
	fig3 = plt.figure()
	grad_norm_arr = []
	fname_arr =[]
	xtick_arr = []
	pos_arr = []
	for fidx,file in enumerate(files):
		fname = os.path.basename(file).split(".")[0]
		dict_data = np.load(file,allow_pickle=True).item()
		data_arr = dict_data['bias={:.2f}'.format(bias)]		
		tr_delta_loss = data_arr[0][0][:,1:] - data_arr[0][1][:,1:]
		te_delta_loss = data_arr[0][3][:,1:] - data_arr[0][4][:,1:]
		grad_norm = data_arr[0][6]
		if plot_abs:
			tr_delta_loss_arr.append(np.abs(tr_delta_loss).flatten())
			te_delta_loss_arr.append(np.abs(te_delta_loss).flatten())
		else:
			tr_delta_loss_arr.append(tr_delta_loss.flatten())
			te_delta_loss_arr.append(te_delta_loss.flatten())
		grad_norm_arr.append(np.abs(grad_norm).flatten())
		if 'pretrained' not in fname:
			fname_arr.append(fname)
			xtick_arr.append((fidx//2)*4+0.5)
		pos_arr.append(int((fidx//2)*4+fidx%2))
	plt.figure(fig1.number)
	bp0 = plt.boxplot(tr_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False,notch=True)
	# breakpoint()
	set_boxplot_colors(bp0)
	plt.xticks(xtick_arr,fname_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Training Set Error for bias={:.2f}'.format(bias))
	plt.figure(fig2.number)
	bp1 = plt.boxplot(te_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False,notch=True)
	set_boxplot_colors(bp1)
	plt.xticks(xtick_arr,fname_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Test Set Error for bias={:.2f}'.format(bias))
	plt.figure(fig3.number)
	bp2 = plt.boxplot(grad_norm_arr,positions=pos_arr,patch_artist=True,showfliers=False,notch=True)
	set_boxplot_colors(bp2)
	plt.xticks(xtick_arr,fname_arr)
	plt.ylabel(r'$\left|\nabla_w F\right|$')
	plt.yscale('log')
	plt.title('Norm of gradient')

def plot_variance_results(variance=20,plot_abs=True):
	data_folder = os.path.join(metadata_folder,'Variance')
	# files = list(set(glob(os.path.join(data_folder,'vgg*.npy'))) - set(glob(os.path.join(data_folder,'vgg*_bn*.npy'))))
	# files = glob(os.path.join(data_folder,'vgg*_bn*.npy'))
	files = glob(os.path.join(data_folder,'test_vgg*_bn*.npy'))
	files.sort()
	fig1 = plt.figure()
	tr_delta_loss_arr = []
	fig2 = plt.figure()
	te_delta_loss_arr = []
	fig3 = plt.figure()
	grad_norm_arr = []
	hessian_trace_arr = []
	fname_arr =[]
	xtick_arr = []
	pos_arr = []
	for fidx,file in enumerate(files):
		fname = os.path.basename(file).split(".")[0]
		dict_data = np.load(file,allow_pickle=True).item()
		data_arr = dict_data['variance={:.2f}'.format(variance)]		
		tr_delta_loss = data_arr[0][0][:,1:] - data_arr[0][1][:,1:]
		te_delta_loss = data_arr[0][3][:,1:] - data_arr[0][4][:,1:]
		grad_norm = data_arr[0][6]
		try:
			hessian_trace = data_arr[0][7]
			hessian_trace_arr.append(hessian_trace.flatten())
		except:
			pass
		if plot_abs:
			tr_delta_loss_arr.append(np.abs(tr_delta_loss).flatten())
			te_delta_loss_arr.append(np.abs(te_delta_loss).flatten())
		else:
			tr_delta_loss_arr.append(tr_delta_loss.flatten())
			te_delta_loss_arr.append(te_delta_loss.flatten())
		grad_norm_arr.append(grad_norm.flatten())
		if 'pretrained' not in fname:
			fname_arr.append(fname)
			xtick_arr.append((fidx//2)*4+0.5)
		pos_arr.append(int((fidx//2)*4+fidx%2))
	plt.figure(fig1.number)
	bp0 = plt.boxplot(tr_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	# breakpoint()
	set_boxplot_colors(bp0)
	plt.xticks(xtick_arr,fname_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Training Set Error for variance={:.2f}'.format(variance))
	plt.figure(fig2.number)
	bp1 = plt.boxplot(te_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	set_boxplot_colors(bp1)
	plt.xticks(xtick_arr,fname_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Test Set Error for variance={:.2f}'.format(variance))
	plt.figure(fig3.number)
	bp2 = plt.boxplot(grad_norm_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	set_boxplot_colors(bp2)
	plt.xticks(xtick_arr,fname_arr)
	plt.ylabel(r'$\left|\nabla_w F\right|$')
	plt.yscale('log')
	plt.title('Norm of gradient')
	if len(hessian_trace_arr)>0:
		fig4 = plt.figure()
		bp3 = plt.boxplot(hessian_trace_arr,positions=pos_arr,patch_artist=True,showfliers=False)
		set_boxplot_colors(bp3)
		plt.xticks(xtick_arr,fname_arr)
		plt.ylabel(r'$Tr\left[\nabla_w^2 F\right]$')
		plt.yscale('log')
		plt.title('Trace of Hessian')

def plot_variance_results_widthVary(variance=20,plot_abs=True):
	data_folder = os.path.join(metadata_folder,'Variance')
	files = list(set(glob(os.path.join(data_folder,'widthVary_vgg*.npy'))) - set(glob(os.path.join(data_folder,'widthVary_vgg*_bn*.npy'))))
	# files = glob(os.path.join(data_folder,'widthVary_vgg*_bn*.npy'))
	files.sort()
	fig1 = plt.figure()
	tr_delta_loss_arr = []
	fig2 = plt.figure()
	te_delta_loss_arr = []
	fig3 = plt.figure()
	grad_norm_arr = []
	hessian_trace_arr = []
	fname_arr =[]
	net_width_arr =[]
	xtick_arr = []
	pos_arr = []
	for fidx,file in enumerate(files):
		fname = os.path.basename(file).split(".")[0]
		dict_data = np.load(file,allow_pickle=True).item()
		all_data_arr = dict_data['variance={:.2f}'.format(variance)]
		for kidx,k in enumerate(all_data_arr.keys()):
			data_arr = all_data_arr[k]
			tr_delta_loss = data_arr[0][0][:,1:] - data_arr[0][1][:,1:]
			te_delta_loss = data_arr[0][3][:,1:] - data_arr[0][4][:,1:]
			grad_norm = data_arr[0][6]
			try:
				hessian_trace = data_arr[0][7]
				hessian_trace_arr.append(hessian_trace.flatten())
			except:
				pass
			if plot_abs:
				tr_delta_loss_arr.append(np.abs(tr_delta_loss).flatten())
				te_delta_loss_arr.append(np.abs(te_delta_loss).flatten())
			else:
				tr_delta_loss_arr.append(tr_delta_loss.flatten())
				te_delta_loss_arr.append(te_delta_loss.flatten())
			grad_norm_arr.append(grad_norm.flatten())
			net_width_arr.append(k)
			xtick_arr.append(kidx)
			pos_arr.append(kidx)
		if 'pretrained' not in fname:
			fname_arr.append(fname)
			# xtick_arr.append((fidx//2)*4+0.5)
		# pos_arr.append(int((fidx//2)*4+fidx%2))
	plt.figure(fig1.number)
	bp0 = plt.boxplot(tr_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	set_boxplot_colors(bp0,color1='cyan',edgecolor1='navy',color2='cyan',edgecolor2='navy')
	plt.xticks(xtick_arr,net_width_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Training Set Error for variance={:.2f}'.format(variance))
	plt.figure(fig2.number)
	bp1 = plt.boxplot(te_delta_loss_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	set_boxplot_colors(bp1,color1='cyan',edgecolor1='navy',color2='cyan',edgecolor2='navy')
	plt.xticks(xtick_arr,net_width_arr)
	if plot_abs: 
		plt.ylabel('|$\Delta$Loss|')
		plt.yscale('log')
	else:
		plt.ylabel('$\Delta$Loss')
	plt.title('Impact on Test Set Error for variance={:.2f}'.format(variance))
	plt.figure(fig3.number)
	bp2 = plt.boxplot(grad_norm_arr,positions=pos_arr,patch_artist=True,showfliers=False)
	set_boxplot_colors(bp2,color1='cyan',edgecolor1='navy',color2='cyan',edgecolor2='navy')
	plt.xticks(xtick_arr,net_width_arr)
	plt.ylabel(r'$\left|\nabla_w F\right|$')
	plt.yscale('log')
	plt.title('Norm of gradient')
	if len(hessian_trace_arr)>0:
		fig4 = plt.figure()
		bp3 = plt.boxplot(hessian_trace_arr,positions=pos_arr,patch_artist=True,showfliers=False)
		set_boxplot_colors(bp3,color1='cyan',edgecolor1='navy',color2='cyan',edgecolor2='navy')
		plt.xticks(xtick_arr,net_width_arr)
		plt.ylabel(r'$\frac{Tr\left[\nabla_w^2 F\right]}{N}$')
		plt.yscale('log')
		plt.title('Trace of Hessian')

if __name__=='__main__':
	plot_bias_results(plot_abs=True)
	# plot_variance_results(variance=0.02, plot_abs=False)
	# plot_variance_results_widthVary(variance=0.2, plot_abs=True)
	plt.show()
