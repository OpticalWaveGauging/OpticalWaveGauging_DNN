## train_regress_height.py 
## A script to train optical wave height gauges for 4 models and 4 batch sizes
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import numpy as np 
import pandas as pd 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
from glob import glob
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, reduceloss_plateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def mae_metric(in_gt, in_pred):
    return mean_absolute_error(div*in_gt, div*in_pred)

def gen_from_df(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
	
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse', **dflow_args)
									
    df_gen.filenames = in_df[path_col].values
	
    df_gen.classes = np.stack(in_df[y_col].values)
	
    df_gen.samples = in_df.shape[0]
	
    df_gen.n = in_df.shape[0]
	
    df_gen._set_index_array()
	
    df_gen.directory = '' 
    return df_gen	

#==============================================================	
if __name__ == '__main__':
		
	IMG_SIZE = (128, 128) 

	num_epochs = 100 #20, 50
	
	base_dir = os.path.normpath(os.getcwd()+os.sep+'train') 

	for batch_size in [16,32,64,128]: 
		print ("[INFO] Batch size = "+str(batch_size))
		
		archs = {'1':MobileNet, '2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}

		counter =1

		for arch in archs:
			print(arch)

			df = pd.read_csv(os.path.join(base_dir, 'training-dataset.csv'))
																																									 
			df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
																	 'categorical',  
																	 '{}.png'.format(x)))														 
																	 
			df['exists'] = df['path'].map(os.path.exists)
			print(df['exists'].sum(), 'images found of', df.shape[0], 'total')

			mean = df['H'].mean() 
			div = 2*df['H'].std() 
			df['zscore'] = df['H'].map(lambda x: (x-mean)/div) 
			df.dropna(inplace = True)
			df.sample(3)

			df['category'] = pd.cut(df['H'], 10)

			new_df = df.groupby(['category']).apply(lambda x: x.sample(2000, replace = True) 
																  ).reset_index(drop = True)
			print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])

			train_df, valid_df = train_test_split(new_df, 
											   test_size = 0.33, 
											   random_state = 2018,
											   stratify = new_df['category'])
			print('train', train_df.shape[0], 'validation', valid_df.shape[0])


			im_gen = ImageDataGenerator(samplewise_center=True, 
										  samplewise_std_normalization=True, 
										  horizontal_flip = False, 
										  vertical_flip = False, 
										  height_shift_range = 0.1, 
										  width_shift_range = 0.1, 
										  rotation_range = 10, 
										  shear_range = 0.05,
										  fill_mode = 'reflect', #'nearest',
										  zoom_range=0.2)

			train_gen = gen_from_df(im_gen, train_df, 
										 path_col = 'path',
										y_col = 'zscore', 
										target_size = IMG_SIZE,
										 colour_mode = 'grayscale',
										batch_size = 64)

			valid_gen = gen_from_df(im_gen, valid_df, 
										 path_col = 'path',
										y_col = 'zscore', 
										target_size = IMG_SIZE,
										 colour_mode = 'grayscale',
										batch_size = 64) 
									
			test_X, test_Y = next(gen_from_df(im_gen, 
										   valid_df, 
										 path_col = 'path',
										y_col = 'zscore', 
										target_size = IMG_SIZE,
										 colour_mode = 'grayscale',
										batch_size = 1000)) 


			t_x, t_y = next(train_gen)
		   
			train_gen.batch_size = batch_size

			weights_path="im"+str(IMG_SIZE[0])+"_waveheight_weights_model"+str(counter)+"_"+str(num_epochs)+"epoch_"+str(train_gen.batch_size)+"batch.best.hdf5" 

			model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
									 save_best_only=True, mode='min', save_weights_only = True)


			reduceloss_plat = reduceloss_plateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
			earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=15) 
			callbacks_list = [model_checkpoint, earlystop, reduceloss_plat]	

			base_model = archs[arch](input_shape =  t_x.shape[1:], include_top = False, weights = None)

			print ("[INFO] Training optical wave gauge")
			
			OWG = Sequential()
			OWG.add(BatchNormalization(input_shape = t_x.shape[1:]))
			OWG.add(base_model)
			OWG.add(BatchNormalization())
			OWG.add(GlobalAveragePooling2D())
			OWG.add(Dropout(0.5))
			OWG.add(Dense(1, activation = 'linear' )) 

			OWG.compile(optimizer = 'adam', loss = 'mse',
								   metrics = [mae_metric])

			OWG.summary()

			OWG.fit_generator(train_gen, validation_data = (test_X, test_Y), 
										  epochs = num_epochs, steps_per_epoch= 100, 
										  callbacks = callbacks_list)

			OWG.load_weights(weights_path)

			pred_Y = div*OWG.predict(test_X, batch_size = train_gen.batch_size, verbose = True)+mean
			test_Y = div*test_Y+mean

			fig, ax1 = plt.subplots(1,1, figsize = (6,6))
			ax1.plot(test_Y, pred_Y, 'k.', label = 'predictions')
			ax1.plot(test_Y, test_Y, 'r-', label = 'actual')
			ax1.legend()
			ax1.set_xlabel('Actual H (m)')
			ax1.set_ylabel('Predicted H (m)')

			plt.savefig('im'+str(IMG_SIZE[0])+'_waveheight_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch.png', dpi=300, bbox_inches='tight')
			plt.close('all')

			rand_idx = np.random.choice(range(test_X.shape[0]), 9)
			fig, m_axs = plt.subplots(3, 3, figsize = (16, 32))
			for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):
			  c_ax.imshow(test_X[idx, :,:,0], cmap = 'gray')

			  c_ax.set_title('H: %0.3f\nPredicted H: %0.3f' % (test_Y[idx], pred_Y[idx]))
			  c_ax.axis('off')
			fig.savefig('im'+str(IMG_SIZE[0])+'_waveheight_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch.png', dpi=300, bbox_inches='tight')

			counter += 1

   
   
