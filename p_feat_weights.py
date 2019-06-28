
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from skimage.io import imread
from glob import glob
import os, shutil

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications import inception_resnet_v2
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Model, Sequential
import keras.backend as K

def rescale(x, NewMin, NewMax):
    OldRange = (np.max(x) - np.min(x))  
    NewRange = (NewMax - NewMin)  
    return (((x - np.min(x)) * NewRange) / OldRange) + NewMin

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x	
	
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
	
base_dir = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN\train'

IMG_SIZE = (2048, 2448)

imagetype = 'snap'
image_dir = 'snap'
csv_file = 'Nearshore-Training-Oblique-cam2-snap.csv'

category = 'T' 
weight_path = r'C:\Users\ddb265\github_clones\OWG_DNN_res\200epoch\T\model4\batch16\waveperiod_weights_model4_16batch.best.IR.hdf5'

# category = 'H' 
# weight_path = r'C:\Users\ddb265\github_clones\OWG_DNN_res\200epoch\H\model4\batch16\waveheight_weights_model4_16batch.best.IR.hdf5'

df = pd.read_csv(os.path.join(base_dir, csv_file))

df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))+".jpg"
				 
df['exists'] = df['path'].map(os.path.exists)
print(df['exists'].sum(), 'images found of', df.shape[0], 'total')
df = df.rename(index=str, columns={" H": "H", " T": "T"})   

if category=='H':
	#mean = df['waveheight'].mean() 
	#div = 2*df['waveheight'].std() 
	#df['zscore'] = df['waveheight'].map(lambda x: (x-mean)/div) 
	df.dropna(inplace = True)
	df.sample(3)
	df['category'] = pd.cut(df['H'], 8)

elif category == 'T':
	#mean = df['waveperiod'].mean() 
	#div = 2*df['waveperiod'].std() 
	#df['zscore'] = df['waveperiod'].map(lambda x: (x-mean)/div) 
	df.dropna(inplace = True)
	df.sample(3)
	df['category'] = pd.cut(df['T'], 8)


new_df = df.groupby(['category']).apply(lambda s: s.sample(1)).reset_index(drop = True)
print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])

F = []
for k in new_df['path']:
	F.append(k)
			


core_idg = ImageDataGenerator(samplewise_center=True, 
							  samplewise_std_normalization=True, 
							  horizontal_flip = False, 
							  vertical_flip = False, 
							  height_shift_range = 0, 
							  width_shift_range = 0, 
							  rotation_range = 0, 
							  shear_range = 0,
							  fill_mode = 'reflect', #'nearest',
							  zoom_range=0)
							  
train_gen = flow_from_dataframe(core_idg, new_df, 
							 path_col = 'path',
							y_col = category, #'zscore', 
							target_size = IMG_SIZE,
							 color_mode = 'grayscale',
							batch_size = 8) #10)
							
							
model1 = InceptionResNetV2(weights='imagenet', include_top=False)
layer_name = 'conv_7b' #'conv_7b_bn'

# #create a section of the model to output the layer we want
model1 = Model(model1.input, model1.get_layer(layer_name).output)


#infiles = glob(r'C:\Users\ddb265\github_clones\NNwaves\im128\res_snap\100epoch\H\model4\batch128\*.hdf5')
# infiles = glob(r'C:\Users\ddb265\github_clones\NNwaves\im128\res_snap\100epoch\T\model4\batch128\*.hdf5')

# tmp = shutil.copyfile(infiles[0], 'mymodel.hdf5')		
# with open(tmp, "ab") as data1, open(infiles[1], "rb") as file2, open(infiles[2], "rb") as file3:
   # data1.write(file2.read())
   # data1.write(file3.read())
# weight_path='mymodel.hdf5'

	
t_x, t_y = next(train_gen)


base_model = InceptionResNetV2(input_shape =  t_x.shape[1:],
							 include_top = False, 
							 weights = None)	
							 
owg = Sequential()
owg.add(BatchNormalization(input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)))
owg.add(base_model)
owg.add(BatchNormalization())
owg.add(GlobalAveragePooling2D())
owg.add(Dropout(0.5))
owg.add(Dense(1, activation = 'linear' )) # linear is what 16bit did   
owg.load_weights(weight_path)	

#create a section of the model to output the layer we want
model2 = owg.get_layer('inception_resnet_v2')

model2 = Model(owg.get_layer('inception_resnet_v2').get_input_at(0), owg.get_layer('inception_resnet_v2').get_layer(layer_name).output)
			
counter=0
c= 0

FM1 = []
FM2 = []
for file in F:
	print(c)
	#open and preprocess the cat image
	Image = image.load_img(file)#, target_size=image_size)

	Image = np.expand_dims(Image,axis=0)
	Image = preprocess_input(Image)

	#get the layer outputs
	features1 = model1.predict(Image)
	
	channel = len(features1)

	Image = image.load_img(file)#, target_size=image_size)

	featureMap = deprocess_image(features1[:,:,:,channel])[0]
	FM1.append(featureMap)
	
	ax=plt.subplot(8,3,counter+1)
	plt.imshow(Image, cmap='gray')
	ax.set_xticks([]) 
	ax.set_yticks([])
	if category=='H':
		ax.set_ylabel('H = '+str(new_df[category][c])+' m', fontsize=4)
	else:
		ax.set_ylabel('T = '+str(new_df[category][c])+' s', fontsize=4)	
	if counter==0:
	   plt.title('Image', fontsize=6)
	
	ax=plt.subplot(8,3,counter+2)	
	plt.imshow(np.log(featureMap/255), vmin=-1, vmax=0, cmap='gray') #np.abs(-np.log(featureMap/np.max(featureMap))), cmap='gray')
	ax.set_xticks([]) 
	ax.set_yticks([]) 	
	if counter==0:
	   plt.title('Imagenet weights', fontsize=6)
		
	try:	
		input = np.expand_dims(imread(file), axis=0)
		features2 = model2.predict(np.expand_dims(input.transpose(1,2,0), axis=0))
	except:
		input = np.expand_dims(np.expand_dims(imread(file)[:,:,0],axis=0),axis=3)
		input = preprocess_input(input)
		#get the layer outputs
		features2 = model2.predict(input)
	
	#channel = len(features2)

	featureMap2 = np.max(features2[:,:,:,:][0], axis=2)
	featureMap2 = featureMap2/np.max(featureMap2)	
	featureMap2[featureMap2>.25] = np.min(featureMap2)
	featureMap2 = rescale(featureMap2, 0, 1)
	
	featureMap2 = scipy.signal.medfilt(featureMap2)
	
	#featureMap2 = deprocess_image(features2[:,:,:,channel])[0]
	FM2.append(featureMap2)

	ax=plt.subplot(8,3,counter+3)	
	#plt.imshow(np.log(featureMap2/255), vmin=-1, vmax=0, cmap='gray')
	plt.imshow(featureMap2, vmin=0.1, vmax=0.5, cmap='gray')	
	ax.set_xticks([]) 
	ax.set_yticks([]) 
	if counter==0:
	   plt.title('Custom weights', fontsize=6)
			
	counter += 3
	c += 1

#plt.savefig('NN_cats_H.png', dpi=1200, bbox_inches='tight')	
#plt.savefig('NN_cats_T.png', dpi=1200, bbox_inches='tight')
plt.savefig('oblique_cats_T.png', dpi=1200, bbox_inches='tight')
#plt.savefig('oblique_cats_H.png', dpi=1200, bbox_inches='tight')	
plt.close('all')		


