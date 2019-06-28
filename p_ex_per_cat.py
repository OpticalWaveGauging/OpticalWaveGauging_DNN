
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from skimage.io import imread
from glob import glob
import os, shutil

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, accuracy_score, f1_score


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

cat = 'H'
#cat = 'T'

image_dir = 'snap'
csv_file = 'Nearshore-Training-Oblique-cam2-snap.csv'


df = pd.read_csv(os.path.join(base_dir, csv_file))

df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))+".jpg"
				 
df['exists'] = df['path'].map(os.path.exists)
print(df['exists'].sum(), 'images found of', df.shape[0], 'total')
df = df.rename(index=str, columns={" H": "H", " T": "T"})   
df['time'] = [int(k.split(os.sep)[-1].split('.')[0]) for k in df.path]
df = df.sort_values(by='time', axis=0)

if cat == 'H':
	df.dropna(inplace = True)
	df.sample(3)
	df['category'] = pd.cut(df[cat], 10)

elif cat == 'T': 
	df.dropna(inplace = True)
	df.sample(3)
	df['category'] = pd.cut(df[cat], 8)

	
X = []
for k in range(8):
	tmp = df.groupby('category').apply(lambda s: s.sample(1))
	for k in tmp['path']:
		X.append(imread(k))

		
if cat=='H':
		
	fig, m_axs = plt.subplots(8, 8, figsize = (16, 16))

	counter=1
	for (idx, c_ax) in zip(np.arange(len(X)), m_axs.flatten()):
		c_ax.imshow(X[idx], cmap = 'gray')
		c_ax.axis('off')
		if counter==1:
		   c_ax.set_title('0.39 - 0.61 m', fontsize=10)
		if counter==2:
		   c_ax.set_title('0.61 - 0.82 m', fontsize=10)	
		if counter==3:
		   c_ax.set_title('0.82 - 1.04 m', fontsize=10)	   
		if counter==4:
		   c_ax.set_title('1.04 - 1.25 m', fontsize=10)	   
		if counter==5:
		   c_ax.set_title('1.69 - 1.9 m', fontsize=10)	   
		if counter==6:
		   c_ax.set_title('1.9 - 2.12 m', fontsize=10)
		if counter==7:
		   c_ax.set_title('2.12 - 2.34 m', fontsize=10)	
		if counter==8:
		   c_ax.set_title('2.34 - 2.56 m', fontsize=10)		   
		   
		counter += 1
		
	fig.savefig(cat+'_ex_per_cat-oblique.png', dpi = 600)	
	del fig
	plt.close('all')

else:

	fig, m_axs = plt.subplots(8, 8, figsize = (16, 16))

	counter=1
	for (idx, c_ax) in zip(np.arange(len(X)), m_axs.flatten()):
		c_ax.imshow(X[idx], cmap = 'gray')
		c_ax.axis('off')
		if counter==1:
		   c_ax.set_title('7.1 - 9.1 s', fontsize=10)
		if counter==2:
		   c_ax.set_title('9.1 - 11.1 s', fontsize=10)	
		if counter==3:
		   c_ax.set_title('11.1 - 13.1 s', fontsize=10)	   
		if counter==4:
		   c_ax.set_title('13.1 - 15.1 s', fontsize=10)	   
		if counter==5:
		   c_ax.set_title('15.1 - 17.1 s', fontsize=10)	   
		if counter==6:
		   c_ax.set_title('17.1 - 19.1 s', fontsize=10)
		if counter==7:
		   c_ax.set_title('19.1 - 21.1 s', fontsize=10)	
		if counter==8:
		   c_ax.set_title('21.1 - 23.1 s', fontsize=10)		   
		   
		counter += 1
		
	fig.savefig(cat+'_ex_per_cat-oblique.png', dpi = 600)	
	del fig
	plt.close('all')

	
	
	
	