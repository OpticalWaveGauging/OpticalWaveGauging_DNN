## compile_results.py 
## A script to collate all model results and make plots of model-observation mismatch
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from glob import glob
import shutil

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy.io import savemat

from utils import *
	
#==============================================================	
if __name__ == '__main__':	

	# load the user configs
	with open(os.getcwd()+os.sep+'conf'+os.sep+'config.json') as f:    
		config = json.load(f)

	print(config)
	# config variables
	imsize    = int(config["img_size"])
	num_epochs = int(config["num_epochs"]) ##100
	test_size = float(config["test_size"])
	batch_size = int(config["batch_size"])
	height_shift_range = float(config["height_shift_range"])
	width_shift_range = float(config["width_shift_range"])
	rotation_range = float(config["rotation_range"])
	samplewise_std_normalization = config["samplewise_std_normalization"]
	horizontal_flip = config["horizontal_flip"]
	vertical_flip = config["vertical_flip"]
	samplewise_center = config["samplewise_center"]
	shear_range = float(config["shear_range"])
	zoom_range = float(config["zoom_range"])
	steps_per_epoch = int(config["steps_per_epoch"])
	dropout_rate = float(config["dropout_rate"])
	epsilon = float(config["epsilon"])
	min_lr = float(config["min_lr"])
	factor = float(config["factor"])
	input_image_format = config["input_image_format"]
	input_csv_file = config["input_csv_file"]
	category = config["category"] ##'H'
	fill_mode = config["fill_mode"]
	
	
	IMG_SIZE = (imsize, imsize) ##(128, 128) 

	base_dir = os.getcwd()+os.sep+'train'


	df = pd.read_csv(os.path.join(base_dir, input_csv_file)) ##'training-dataset.csv'))
																																									 
	df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
													'images',  
													'{}.png'.format(x)))
																 
	df['exists'] = df['path'].map(os.path.exists)
	print(df['exists'].sum(), 'images found of', df.shape[0], 'total')

	if category == 'H':
		mean = df['H'].mean() 
		div = 2*df['H'].std() 
		df['zscore'] = df['H'].map(lambda x: (x-mean)/div)
	elif category == 'T':
		mean = df['T'].mean() 
		div = 2*df['T'].std() 
		df['zscore'] = df['T'].map(lambda x: (x-mean)/div)			
	else:
		print("Unknown category: "+str(category))
		print("Fix config file, exiting now ...")
		import sys
		sys.exit()
				
	df.dropna(inplace = True)

	if category == 'H':
		df['category'] = pd.cut(df['H'], 10)
	else:
		df['category'] = pd.cut(df['T'], 8)

	new_df = df.groupby(['category']).apply(lambda x: x.sample(2000, replace = True) 
														  ).reset_index(drop = True)
	print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])

	train_df, valid_df = train_test_split(new_df, 
									   test_size = test_size, #0.33, 
									   random_state = 2018,
									   stratify = new_df['category'])
	print('train', train_df.shape[0], 'validation', valid_df.shape[0])


	im_gen = ImageDataGenerator(samplewise_center=samplewise_center, ##True, 
								  samplewise_std_normalization=samplewise_std_normalization, ##True, 
								  horizontal_flip = horizontal_flip, ##False, 
								  vertical_flip = vertical_flip, ##False, 
								  height_shift_range = height_shift_range, ##0.1, 
								  width_shift_range = width_shift_range, ##0.1, 
								  rotation_range = rotation_range, ##10, 
								  shear_range = shear_range, ##0.05,
								  fill_mode = fill_mode, ##'reflect', #'nearest',
								  zoom_range= zoom_range) ##0.2)

	train_gen = gen_from_df(im_gen, train_df, 
								 path_col = 'path',
								y_col = 'zscore', 
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = batch_size) ##64)

	valid_gen = gen_from_df(im_gen, valid_df, 
								 path_col = 'path',
								y_col = 'zscore', 
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = batch_size) ##64) 
							
	test_X, test_Y = next(gen_from_df(im_gen, 
								   valid_df, 
								 path_col = 'path',
								y_col = 'zscore', 
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = len(df))) #1000)) 


	t_x, t_y = next(train_gen)


	out ={}
	out['mean'] = mean
	out['div'] = div
	out['text_X'] = test_X
	out['text_Y'] = test_Y
	out['df_T'] = df['waveperiod']
	out['df_H'] = df['waveheight']
	out['df_zscore'] = df['zscore']
	out['df_category'] = [] 

	root = os.getcwd()+os.sep+'im128'+os.sep+'res_snap'


	for epics in [num_epochs]: ##[20, 50, 100]:
		print("Epoch: "+str(epics))
		
		for batch_size in [16,32,64,128]: 
		
			print("Batch: "+str(batch_size))
			archs = {'1':MobileNet, '2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}
			counter=1

			for arch in archs:
				print("Model: "+arch)

				train_gen.batch_size = batch_size

				if arch=='4':
					if category=='H':
						infiles  = sorted(glob(os.path.normpath(root+os.sep+str(epics)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'*.hdf5')))					
					else:
						infiles  = sorted(glob(os.path.normpath(root+os.sep+str(epics)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'*.hdf5')))	
					tmp = shutil.copyfile(infiles[0], 'mymodel.hdf5')		
					with open(tmp, "ab") as data1, open(infiles[1], "rb") as file2, open(infiles[2], "rb") as file3:
						data1.write(file2.read())
						data1.write(file3.read())
					weight_path='mymodel.hdf5'
				else:
					if category=='H':			
						weight_path = sorted(glob(os.path.normpath(root+os.sep+str(epics)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'*.hdf5')))[0]	
					else:
						weight_path = sorted(glob(os.path.normpath(root+os.sep+str(epics)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'*.hdf5')))[0]
						
				base_model = archs[arch](input_shape =  t_x.shape[1:], include_top = False, weights = None)	

				owg = Sequential()
				owg.add(BatchNormalization(input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)))
				owg.add(base_model)
				owg.add(BatchNormalization())
				owg.add(GlobalAveragePooling2D())
				owg.add(Dropout(dropout_rate))
				owg.add(Dense(1, activation = 'linear' ))  
				owg.load_weights(weight_path)

				out['yhat_epochs'+str(epics)+'_batchsize'+str(batch_size)+'_model'+arch] = div*owg.predict(test_X, batch_size = batch_size, verbose = True)+mean
				counter+=1
				os.remove(weight_path)

	E = []			
	for epics in [num_epochs]: ##20, 50, 100]:
		print(epics)
		fig, axs = plt.subplots(4, 4)
		fig.subplots_adjust(hspace=0.5, wspace=0.3)

		counter1=0	
		
		for batch_size in [16,32,64,128]: 
			print(batch_size)
			archs = {'1':MobileNet, '2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}
			
			if batch_size==16:
			   letters='abcd'
			elif batch_size==32:
			   letters='efgh'
			elif batch_size==64:
			   letters='ijkl'		
			else:
			   letters='mnop'
			   
			counter2=0		
			   
			for arch in archs:
				print(arch)
				
				pred_Y = np.squeeze(out['yhat_epochs'+str(epics)+'_batchsize'+str(batch_size)+'_model'+arch])		
				E.append(pred_Y)

				out['rms_epochs'+str(epics)+'_batchsize'+str(batch_size)+'_model'+arch] = np.sqrt(np.nanmean((pred_Y - test_Y)**2))
				out['rsq_epochs'+str(epics)+'_batchsize'+str(batch_size)+'_model'+arch] = np.min(np.corrcoef(test_Y, pred_Y))**2

				axs[counter1, counter2].plot(test_Y, pred_Y, 'k.', label = 'estimated', markersize=2)
				axs[counter1, counter2].plot(test_Y, test_Y, 'r--', label = 'observed', lw=0.5)
				if counter1==3:
				   if counter2==0:
					  if category=='H':
						axs[counter1, counter2].set_xlabel('Observed H (m)', fontsize=6)
						axs[counter1, counter2].set_ylabel(r'Estimated H (m)', fontsize=6)				  
					  else:
						axs[counter1, counter2].set_xlabel('Observed T (s)', fontsize=6)
						axs[counter1, counter2].set_ylabel(r'Estimated T (s)', fontsize=6)
				if category=='H':
					string = r'RMS (m): '+str(np.sqrt(np.nanmean((pred_Y - test_Y)**2)))[:4] + '  R$^2$: '+str(np.min(np.corrcoef(test_Y, pred_Y))**2)[:4]
					axs[counter1, counter2].set_title(letters[counter2]+') '+string, fontsize=4, loc='left')
					axs[counter1, counter2].set_xlim(0, 3)
					axs[counter1, counter2].set_ylim(0,3)				
				else:			
					string = r'RMS (s): '+str(np.sqrt(np.nanmean((pred_Y - test_Y)**2)))[:4] + '  R$^2$: '+str(np.min(np.corrcoef(test_Y, pred_Y))**2)[:4]
					axs[counter1, counter2].set_title(letters[counter2]+') '+string, fontsize=4, loc='left')
					axs[counter1, counter2].set_xlim(7, 24)
					axs[counter1, counter2].set_ylim(7,24)
				plt.setp(axs[counter1, counter2].get_xticklabels(), fontsize=6)
				plt.setp(axs[counter1, counter2].get_yticklabels(), fontsize=6)
				axs[counter1, counter2].set_aspect(1.0)
				counter2 += 1
				
			counter1+=1
		if category=='H':
			plt.savefig('IR_H_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')	
		else:
			plt.savefig('IR_T_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')
		plt.close('all')
		del fig			
				
				
	out['pred_Y_ensemble'] = np.mean(np.vstack(E), axis=0)
	out['rms_ensemble'] = np.sqrt(np.nanmean((out['pred_Y_ensemble'] - test_Y)**2))


	if category=='H':
		savemat('IR_all_model_preds_height_'+str(IMG_SIZE[0])+'.mat', out)
	else:			
		savemat('IR_all_model_preds_period_'+str(IMG_SIZE[0])+'.mat', out)


