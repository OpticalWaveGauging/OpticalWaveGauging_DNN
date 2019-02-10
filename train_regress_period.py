
## Daniel Buscombe, Nov-Dec 2018

import numpy as np 
import pandas as pd 
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from skimage.io import imread
import os
from glob import glob
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

#base_dir = '/home/ddb265/waves/train' 
#base_dir = r'C:\Users\ddb265\github_clones\IRwaves\train'
base_dir = os.path.normpath(os.getcwd()+os.sep+'train') ## fir colab


def mae(in_gt, in_pred):
    return mean_absolute_error(div*in_gt, div*in_pred)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen	
	
IMG_SIZE = (128, 128) #(256, 256)

##train

num_epochs = 100 #20, 50

for batch_size in [16,32,64,128]:
	
    archs = {'1':MobileNet, '2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}
    #archs = {'2':MobileNetV2, '3':InceptionV3, '4':InceptionResNetV2}
    counter = 1#4
    
    for arch in archs:
	    print(arch)

	    df = pd.read_csv(os.path.join(base_dir, 'training-dataset.csv'))
																																							     
	    df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
															     'categorical',  
															     '{}.png'.format(x)))														 
															     
	    df['exists'] = df['path'].map(os.path.exists)
	    print(df['exists'].sum(), 'images found of', df.shape[0], 'total')

	    mean = df['T'].mean() 
	    div = 2*df['T'].std() 
	    df['zscore'] = df['T'].map(lambda x: (x-mean)/div) 
	    df.dropna(inplace = True)
	    df.sample(3)

	    df['category'] = pd.cut(df['T'], 8) #10)

	    new_df = df.groupby(['category']).apply(lambda x: x.sample(2000, replace = True) 
														      ).reset_index(drop = True)
	    print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])

	    train_df, valid_df = train_test_split(new_df, 
									       test_size = 0.33, 
									       random_state = 2018,
									       stratify = new_df['category'])
	    print('train', train_df.shape[0], 'validation', valid_df.shape[0])


	    core_idg = ImageDataGenerator(samplewise_center=True, 
								      samplewise_std_normalization=True, 
								      horizontal_flip = False, 
								      vertical_flip = False, 
								      height_shift_range = 0.1, 
								      width_shift_range = 0.1, 
								      rotation_range = 10, 
								      shear_range = 0.05,
								      fill_mode = 'reflect', #'nearest',
								      zoom_range=0.2)

	    train_gen = flow_from_dataframe(core_idg, train_df, 
								     path_col = 'path',
								    y_col = 'zscore', 
								    target_size = IMG_SIZE,
								     color_mode = 'grayscale',
								    batch_size = 64)

	    valid_gen = flow_from_dataframe(core_idg, valid_df, 
								     path_col = 'path',
								    y_col = 'zscore', 
								    target_size = IMG_SIZE,
								     color_mode = 'grayscale',
								    batch_size = 64) 
							    
	    # we can use much larger batches for evaluation
	    # used a fixed dataset for evaluating the algorithm
	    test_X, test_Y = next(flow_from_dataframe(core_idg, 
								       valid_df, 
								     path_col = 'path',
								    y_col = 'zscore', 
								    target_size = IMG_SIZE,
								     color_mode = 'grayscale',
								    batch_size = 1000)) # one big batch #500


	    t_x, t_y = next(train_gen)
       
	    train_gen.batch_size = batch_size

	    weight_path="im"+str(IMG_SIZE[0])+"_waveperiod_weights_model"+str(counter)+"_"+str(num_epochs)+"epoch_"+str(train_gen.batch_size)+"batch.best.hdf5" 

	    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
							     save_best_only=True, mode='min', save_weights_only = True)


	    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
	    early = EarlyStopping(monitor="val_loss", 
					      mode="min", 
					      patience=15) # probably needs to be more patient, but kaggle time is limited
	    callbacks_list = [checkpoint, early, reduceLROnPlat]	

	    base_model = archs[arch](input_shape =  t_x.shape[1:], 
								     include_top = False, 
								     weights = None)

	    #base_model = MobileNet(input_shape =  t_x.shape[1:], 
	    #                              include_top = False, 
	    #                              weights = None)
	    model = Sequential()
	    model.add(BatchNormalization(input_shape = t_x.shape[1:]))
	    model.add(base_model)
	    model.add(BatchNormalization())
	    model.add(GlobalAveragePooling2D())
	    model.add(Dropout(0.5))
	    model.add(Dense(1, activation = 'linear' )) # linear is what 16bit did

	    model.compile(optimizer = 'adam', loss = 'mse',
						       metrics = [mae])

	    model.summary()

	    model.fit_generator(train_gen, validation_data = (test_X, test_Y), 
								      epochs = num_epochs, steps_per_epoch= 100, 
								      callbacks = callbacks_list)



	    model.load_weights(weight_path)

	    pred_Y = div*model.predict(test_X, batch_size = train_gen.batch_size, verbose = True)+mean
	    test_Y = div*test_Y+mean

	    fig, ax1 = plt.subplots(1,1, figsize = (6,6))
	    ax1.plot(test_Y, pred_Y, 'r.', label = 'predictions')
	    ax1.plot(test_Y, test_Y, 'b-', label = 'actual')
	    ax1.legend()
	    ax1.set_xlabel('Actual T (s)')
	    ax1.set_ylabel('Predicted T (s)')

	    plt.savefig('im'+str(IMG_SIZE[0])+'_waveperiod_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch.png', dpi=300, bbox_inches='tight')
	    plt.close('all')

	    rand_idx = np.random.choice(range(test_X.shape[0]), 9)
	    fig, m_axs = plt.subplots(3, 3, figsize = (16, 32))
	    for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):
	      c_ax.imshow(test_X[idx, :,:,0], cmap = 'gray')

	      c_ax.set_title('T: %0.3f\nPredicted T: %0.3f' % (test_Y[idx], pred_Y[idx]))
	      c_ax.axis('off')
	    #fig.savefig('trained_img_predictions_Oct_20epoch32batch.png', dpi = 300)
	    fig.savefig('im'+str(IMG_SIZE[0])+'_waveperiod_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch.png', dpi=300, bbox_inches='tight')
	    counter += 1

   
   
