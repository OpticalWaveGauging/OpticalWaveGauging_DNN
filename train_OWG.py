#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## train_OWG.py 
## A script to train optical wave gauges for 4 models and 4 batch sizes
## GPU with 8+GB memory is recommended
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import numpy as np 
import pandas as pd 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import zipfile
import sys, getopt, os, gc
import time, datetime, json
from glob import glob
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0' ##use GPU. 0=first, 1=second, etc

from utils import *
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

## the original implementation used MobileNetV2
## however, for an unknown reason, keras implementation of MobileNetV2 in Tensorflow 2 is not giving the same result as in Tensorflow 1
## so below I have replaced MobileNetV2 with DenseNet201 which is comparable accuracy
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf


#==============================================================	
## script starts here
if __name__ == '__main__':

	# start time
	print ("start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
	start = time.time()
	argv = sys.argv[1:]
	try:
	   opts, args = getopt.getopt(argv,"h:c:")
	except getopt.GetoptError:
	   print('python train_OWG.py -c configfile.json')
	   sys.exit(2)
	for opt, arg in opts:
	   if opt == '-h':
	      print('Example usage: python3 train_OWG.py -c config_IR_H.json')
	      sys.exit()
	   elif opt in ("-c"):
	      configfile = arg
	
	# load the user configs
	with open(os.getcwd()+os.sep+'config'+os.sep+configfile) as f:    
		config = json.load(f)

	# start time
	print ("start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
	start = time.time()
	
#	# exploit multiple GPUs if available
#	strategy = tf.distribute.MirroredStrategy()
#	print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	print(config)
	# config variables
	imsize    = int(config["img_size"])
	num_epochs = int(config["num_epochs"]) 
	test_size = float(config["test_size"])
	height_shift_range = float(config["height_shift_range"])
	width_shift_range = float(config["width_shift_range"])
	rotation_range = float(config["rotation_range"])
	samplewise_std_normalization = config["samplewise_std_normalization"]
	horizontal_flip = config["horizontal_flip"]
	vertical_flip = config["vertical_flip"]
	samplewise_center = config["samplewise_center"]
	shear_range = float(config["shear_range"])
	zoom_range = float(config["zoom_range"])
	dropout_rate = float(config["dropout_rate"])
	epsilon = float(config["epsilon"])
	min_lr = float(config["min_lr"])
	factor = float(config["factor"])
	input_image_format = config["input_image_format"]
	input_csv_file = config["input_csv_file"]
	category = config["category"] 
	fill_mode = config["fill_mode"]
	prc_lower_withheld = config['prc_lower_withheld'] 
	prc_upper_withheld = config['prc_upper_withheld'] 
	 
	base_dir = os.path.normpath(os.getcwd()) 
	
	## download files and unzip
	if input_csv_file=='IR-training-dataset.csv':
		print('Downloading IR imagery ...')
		print('... file is ~2GB - takes a while')
		url = 'https://drive.google.com/file/d/1ljkY4akD8O8ShLyOywTnyQl7GsH3KIJ3/view?usp=sharing'
		image_dir = 'IR_images'+os.sep+'data'
		if not os.path.isdir(os.path.join(base_dir,image_dir)):
			file_id = '1ljkY4akD8O8ShLyOywTnyQl7GsH3KIJ3'
			destination = 'IR_images.zip'
			download_file_from_google_drive(file_id, destination)
			print('download complete ... unzipping')	
			zip_ref = zipfile.ZipFile(destination, 'r')
			zip_ref.extractall(os.getcwd()+os.sep+'train')
			zip_ref.close()
			os.remove(destination)
		
	elif input_csv_file=='snap-training-dataset.csv':
		print('Downloading nearshore imagery ...')
		print('... file is ~1GB - takes a while')
		#url = 'https://drive.google.com/file/d/1QqPUbgXudZSDFXH2VaP30TQYR6PY0acM/view?usp=sharing'
		url = 'https://drive.google.com/file/d/1TVnuPnrbIhtv0y7BXpiXmJMElf7KnSXx/view?usp=sharing'
		image_dir = 'snap_images'+os.sep+'data'
		if not os.path.isdir(os.path.join(base_dir,image_dir)):
			file_id = '1TVnuPnrbIhtv0y7BXpiXmJMElf7KnSXx' #'1QqPUbgXudZSDFXH2VaP30TQYR6PY0acM'
			destination = 'snap_images.zip'
			download_file_from_google_drive(file_id, destination)	
			print('download complete ... unzipping')	
			zip_ref = zipfile.ZipFile(destination, 'r')
			zip_ref.extractall(os.getcwd()+os.sep+'train')
			zip_ref.close()
			os.remove(destination)		
	elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
		image_dir = 'snap'+os.sep+'data'
		
	IMG_SIZE = (imsize, imsize) 

    # call the utils.py function get_and_tidy_df            	
	new_df, df = get_and_tidy_df(base_dir, input_csv_file, image_dir, category)
		
	## making subsets of data based on prc_lower_withheld and prc_upper_withheld
	if (prc_lower_withheld>0) & (prc_upper_withheld>0):
		up = np.percentile(new_df[category], 100-prc_upper_withheld)
		low = np.percentile(new_df[category], prc_lower_withheld)
		extreme_df = new_df.loc[(new_df[category] < low) | (new_df[category] > up)]
		new_df = new_df.loc[(new_df[category] >= low) & (new_df[category] <= up)]
	elif (prc_lower_withheld>0) & (prc_upper_withheld==0):
		low = np.percentile(new_df[category], prc_lower_withheld)
		extreme_df = new_df.loc[(new_df[category] < low)]
		new_df = new_df.loc[(new_df[category] >= low)]
	elif (prc_lower_withheld==0) & (prc_upper_withheld>0):
		up = np.percentile(new_df[category], 100-prc_upper_withheld)
		extreme_df = new_df.loc[(new_df[category] > up)]
		new_df = new_df.loc[(new_df[category] <= up)]

	print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])
				
	train_df, valid_df = train_test_split(new_df, 
									   test_size = test_size, 
									   random_state = 2018,
									   stratify = new_df['category'])
	print('train', train_df.shape[0], 'validation', valid_df.shape[0])

	im_gen = ImageDataGenerator(samplewise_center=samplewise_center,
								  samplewise_std_normalization=samplewise_std_normalization,
								  horizontal_flip = horizontal_flip,  
								  vertical_flip = vertical_flip, 
								  height_shift_range = height_shift_range, 
								  width_shift_range = width_shift_range, 
								  rotation_range = rotation_range,
								  shear_range = shear_range, 
								  fill_mode = fill_mode,
								  zoom_range= zoom_range) 

    # call the utils.py function gen_from_def            	
	train_X, train_Y = gen_from_def(IMG_SIZE, train_df, image_dir, category, im_gen)
	test_X, test_Y = gen_from_def(IMG_SIZE, valid_df, image_dir, category, im_gen)
	
	if (prc_lower_withheld>0) | (prc_upper_withheld>0):	
	    ex_X, ex_Y = gen_from_def(IMG_SIZE, extreme_df, image_dir, category, im_gen)

    							
	## loop through 4 different batch sizes
	for batch_size in [128]: #[16,32,64,128]: #
		print ("[INFO] Batch size = "+str(batch_size))
		
		## this is the original list of models
		##archs = {'1':MobileNet, '2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}		
		## however, for an unknown reason, keras implementation of MobileNetV2 in Tensoeflow 2 is not giving the same result as in Tensorflow 1, so below I have replaced MobileNetV2 with DenseNet201 which is comparable accuracy
		#archs = {'1':MobileNet, '2':DenseNet201, '3':InceptionV3, '4':InceptionResNetV2}
		#counter =1
		archs = {'2':DenseNet201, '3':InceptionV3}
		counter =2
						
		## loop through 4 different base models
		for arch in archs:
			print("==========================================================")
			print("==========================================================")
			print("==========================================================")
			print(arch)	

            # call the utils.py function get_weights_path            				
			weights_path = get_weights_path(input_csv_file, category, counter, imsize, batch_size, num_epochs)
											
			model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
									 save_best_only=True, mode='min', save_weights_only = True)

			reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)
			earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=25) 
			callbacks_list = [model_checkpoint, earlystop, reduceloss_plat]	
			
			print ("[INFO] Training optical wave gauge")
			    
			base_model = archs[arch](input_shape =  (IMG_SIZE[0], IMG_SIZE[1],1), include_top = False, weights = None)			
			OWG = Sequential()
			OWG.add(BatchNormalization(input_shape = (IMG_SIZE[0], IMG_SIZE[1],1)))
			OWG.add(base_model)
			OWG.add(BatchNormalization())
			OWG.add(GlobalAveragePooling2D())
			OWG.add(Dropout(dropout_rate)) 
			OWG.add(Dense(1, activation = 'linear' )) 

			OWG.compile(optimizer = 'rmsprop', loss = 'mse', #adam
								   metrics = [mae_metric])

			OWG.summary()	
			history = OWG.fit(train_X, train_Y, batch_size=batch_size, 
			                  validation_data = (test_X, test_Y),
			                  epochs=num_epochs, callbacks = callbacks_list)

			OWG.load_weights(weights_path)
			# serialize model to JSON	
			model_json = OWG.to_json()
			with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
			    json_file.write(model_json)						

			print ("[INFO] Testing optical wave gauge")
					
			if category=='zscore':
			    # or if the model predicts zscores - recover value using pop. mean and standard dev.			
			    pred_Y = div*np.squeeze(OWG.predict(test_X, batch_size = batch_size, verbose = True))+mean
			    if (prc_lower_withheld>0) | (prc_upper_withheld>0):	
			        pred_extreme_Y = div*np.squeeze(OWG.predict(ex_X, batch_size = batch_size, verbose = True))+mean		
			else:			
			    pred_Y = np.squeeze(OWG.predict(test_X, batch_size = batch_size, verbose = True))
			    if (prc_lower_withheld>0) | (prc_upper_withheld>0):	
			        pred_extreme_Y = np.squeeze(OWG.predict(ex_X, batch_size = batch_size, verbose = True))

			print ("[INFO] Creating plots ")
			
			fig, ax1 = plt.subplots(1,1, figsize = (6,6))
			ax1.plot(test_Y, pred_Y, 'b.', label = 'predictions')
			ax1.plot(test_Y, test_Y, 'k-', label = 'actual')
			if (prc_lower_withheld>0) | (prc_upper_withheld>0):
			    ax1.plot(ex_Y, pred_extreme_Y, 'r*', label = 'extreme predictions')
						
			ax1.legend()
			if category == 'H':			
				ax1.set_xlabel('Actual H (m)')
				ax1.set_ylabel('Predicted H (m)')
				if input_csv_file=='IR-training-dataset.csv':										
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_IR.png', dpi=300, bbox_inches='tight')
				elif input_csv_file=='snap-training-dataset.csv':
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_nearshore.png', dpi=300, bbox_inches='tight')			
				elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_oblique.png', dpi=300, bbox_inches='tight')						
			else:
				ax1.set_xlabel('Actual T (s)')
				ax1.set_ylabel('Predicted T (s)')
				if input_csv_file=='IR-training-dataset.csv':										
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_IR.png', dpi=300, bbox_inches='tight')
				elif input_csv_file=='snap-training-dataset.csv':
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_nearshore.png', dpi=300, bbox_inches='tight')					
				elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
					plt.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_oblique.png', dpi=300, bbox_inches='tight')				
			
			plt.close('all')
			
			
			fig = plt.figure(figsize = (6,6))
			plt.subplot(121)
			# summarize history for accuracy
			plt.plot(history.history['mae_metric'])
			plt.plot(history.history['val_mae_metric'])
			plt.ylabel('Mean absolute error')
			plt.xlabel('Epoch')
			plt.legend(['train', 'test'], loc='upper left')
			
			plt.subplot(122)
			# summarize history for loss
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.legend(['train', 'test'], loc='upper left')

			if input_csv_file=='IR-training-dataset.csv':													
				outfile = os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+category+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_'+category+'_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'_loss_acc_curves_IR.png'
			elif input_csv_file=='snap-training-dataset.csv':
				outfile = os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+category+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_'+category+'_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'_loss_acc_curves_nearshore.png'	
			elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
				outfile = os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+category+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_'+category+'_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'_loss_acc_curves_oblique.png'					
			plt.savefig(outfile, dpi=300, bbox_inches='tight')		
			plt.close('all')
			print('Made '+outfile)			
			
			rand_idx = np.random.choice(range(test_X.shape[0]), 9)
			fig, m_axs = plt.subplots(3, 3, figsize = (16, 32))
			for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):
			  c_ax.imshow(test_X[idx, :,:,0], cmap = 'gray')
			  if category == 'H':
			     c_ax.set_title('H: %0.3f\nPredicted H: %0.3f' % (test_Y[idx], pred_Y[idx]))
			  else:
			     c_ax.set_title('T: %0.3f\nPredicted T: %0.3f' % (test_Y[idx], pred_Y[idx]))
			  
			  c_ax.axis('off')

			if category == 'H':	
				if input_csv_file=='IR-training-dataset.csv':													
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_IR.png', dpi=300, bbox_inches='tight')
				elif input_csv_file=='snap-training-dataset.csv':
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_nearshore.png', dpi=300, bbox_inches='tight')
				elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveheight_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_oblique.png', dpi=300, bbox_inches='tight')					
			else:
				if input_csv_file=='IR-training-dataset.csv':													
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_IR.png', dpi=300, bbox_inches='tight')
				elif input_csv_file=='snap-training-dataset.csv':
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_nearshore.png', dpi=300, bbox_inches='tight')				
				elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
					fig.savefig(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_waveperiod_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_oblique.png', dpi=300, bbox_inches='tight')	
				
			counter += 1	
			print ("[INFO] Finished training")
			gc.collect()
				
	# end time
	end = time.time()
	print ("end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


#			# Call the distribution scope context manager
#			with strategy.scope():
#			    from tensorflow.keras.metrics import mean_absolute_error
#			    # mean absolute error
#			    def mae_metric(in_gt, in_pred):
#			        return mean_absolute_error(in_gt, in_pred)

#                # call the utils.py function get_weights_path            				
#			    weights_path = get_weights_path(input_csv_file, category, counter, imsize, batch_size, num_epochs)
#											    
#			    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
#									     save_best_only=True, mode='min', save_weights_only = True)

#			    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)
#			    earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=25) 
#			    callbacks_list = [model_checkpoint, earlystop, reduceloss_plat]	
#			    
#			    print ("[INFO] Training optical wave gauge")

#			    base_model = archs[arch](input_shape =  (IMG_SIZE[0], IMG_SIZE[1],1), include_top = False, weights = None)
#			    OWG = Sequential()
#			    OWG.add(BatchNormalization(input_shape = (IMG_SIZE[0], IMG_SIZE[1],1)))
#			    OWG.add(base_model)
#			    OWG.add(BatchNormalization())
#			    OWG.add(GlobalAveragePooling2D())
#			    OWG.add(Dropout(dropout_rate)) 
#			    OWG.add(Dense(1, activation = 'linear' )) 

#			    OWG.compile(optimizer = 'rmsprop', loss = 'mse', #adam
#								       metrics = [mae_metric])

#			    history = OWG.fit(train_X, train_Y, batch_size=batch_size, 
#			                  validation_data = (test_X, test_Y),
#			                  epochs=num_epochs, callbacks = callbacks_list)

#			with strategy.scope():	
#			    # load the new model weights							  
#			    OWG.load_weights(weights_path)


#	df = pd.read_csv(os.path.join(base_dir, input_csv_file))
#	if input_csv_file=='IR-training-dataset.csv':
#		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))+".png"
#	elif input_csv_file=='snap-training-dataset.csv':
#		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))
#	elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
#		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))+".jpg"
#		
#	df = df.rename(index=str, columns={" H": "H", " T": "T"})   
#	
#	if category == 'H':
#		mean = df['H'].mean() 
#		div = df['H'].std() 
#		df['zscore'] = df['H'].map(lambda x: (x-mean)/div)
#	elif category == 'T':
#		mean = df['T'].mean() 
#		div = df['T'].std() 
#		df['zscore'] = df['T'].map(lambda x: (x-mean)/div)			
#	else:
#		print("Unknown category: "+str(category))
#		print("Fix config file, exiting now ...")
#		sys.exit()
#	
#	df.dropna(inplace = True)
#	try:
#		df = df.sort_values(by='time', axis=0)
#	except:
#		df = df.sort_values(by='id', axis=0)

#	if category == 'H':
#		df['category'] = pd.cut(df['H'], 10)
#	else:
#		df['category'] = pd.cut(df['T'], 8)
#		
#	df['index1'] = df.index

#	if input_csv_file=='IR-training-dataset.csv':
#		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)
#	elif input_csv_file=='snap-training-dataset.csv':
#		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)
#	elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
#		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)


#from keras.applications.mobilenet import MobileNet

#try:
#   from keras.applications.mobilenet_v2 import MobileNetV2
#except:
#   from keras.applications.mobilenetv2 import MobileNetV2

#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

#from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
#from keras.models import Sequential
#from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#from keras.preprocessing.image import ImageDataGenerator


#	test_generator = im_gen.flow_from_dataframe(dataframe=valid_df,
#								  directory=image_dir,
#								  x_col="path",
#								  y_col=category,
#								  target_size=IMG_SIZE,
#								  batch_size=len(valid_df),
#								  color_mode = 'grayscale',
#								  shuffle=False,
#								  class_mode='raw')
#								  	
#	test_X, test_Y = next(test_generator)
	
#	ex_generator = im_gen.flow_from_dataframe(dataframe=extreme_df,
#								  directory=image_dir,
#								  x_col="path",
#								  y_col=category,
#								  target_size=IMG_SIZE,
#								  batch_size=len(extreme_df),
#								  color_mode = 'grayscale',
#								  shuffle=False,
#								  class_mode='raw')
#								  	
#	ex_X, ex_Y = next(ex_generator)

#	test_X, test_Y = next(gen_from_df(im_gen, 
#								   valid_df, 
#								 path_col = 'path',
#								y_col = category, 
#								target_size = IMG_SIZE,
#								 color_mode = 'grayscale',
#								batch_size = len(train_df))) 	
									
#	train_X, train_Y = next(gen_from_df(im_gen, 
#								   train_df, 
#								 path_col = 'path',
#								y_col = category,
#								target_size = IMG_SIZE,
#								 color_mode = 'grayscale',
#								batch_size = len(train_df))) 								
	
#	ex_X, ex_Y = next(gen_from_df(im_gen, 
#								   extreme_df, 
#								 path_col = 'path',
#								y_col = category, #'zscore', 
#								target_size = IMG_SIZE,
#								 color_mode = 'grayscale',
#								batch_size = len(extreme_df)))
	
			
			# out ={}
			# out['rms'] = np.sqrt(np.nanmean((pred_Y - test_Y)**2))
			# out['rsq'] = np.min(np.corrcoef(test_Y, pred_Y))**2
			# out['rms_ex'] = np.sqrt(np.nanmean((ex_Y - pred_extreme_Y)**2))
			# out['rsq_ex'] = np.min(np.corrcoef(ex_Y, pred_extreme_Y))**2
			# out['y'] = test_Y
			# out['yhat'] = pred_Y
			# out['yhat_extreme'] = pred_extreme_Y
			# out['test_X'] = test_X
			# out['test_Y'] = test_Y
			# out['extreme_X'] = ex_X
			# out['extreme_Y'] = ex_Y
			# out['history_train_mae'] = history.history['mae_metric']
			# out['history_val_mae'] = history.history['val_mae_metric']
			# out['history_train_loss'] = history.history['loss']
			# out['history_val_loss'] = history.history['val_loss']
			
			# if input_csv_file=='IR-training-dataset.csv':
			    # savemat(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+category+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_IR.mat', out, do_compression=True)
			# else:
			    # savemat(os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+category+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'im'+str(IMG_SIZE[0])+'_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(batch_size)+'batch_nearshore.mat', out, do_compression=True)		    
			    
			# list all data in history
			#print(history.history.keys())	
