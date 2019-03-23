## test_OWG_folder.py 
## A script to test a model on independent data
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import numpy as np 
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import model_from_json
from imageio import imread
from keras.preprocessing.image import ImageDataGenerator
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.metrics import mean_absolute_error

def mae_metric(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)

#==============================================================	
## script starts here
if __name__ == '__main__':

    #==============================================================
    ## user inputs
    with open(os.getcwd()+os.sep+'conf'+os.sep+'config_test.json') as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["im_size"])
    category = config["category"] 
    input_csv_file = config["input_csv_file"]   
    image_direc = config["image_direc"]   
    weights_path = config["weights_path"]
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"]  
    
    IMG_SIZE = (im_size, im_size) ##(128, 128)
    #==============================================================
     
    # load json and create model
    json_file = open(weights_path.replace('.hdf5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    OWG = model_from_json(loaded_model_json)
    # load weights into new model
    OWG.load_weights(weights_path)
    print("Loaded model from disk")

    OWG.compile(optimizer = 'adam', loss = 'mse',metrics = [mae_metric]) 

    df = pd.read_csv(input_csv_file) 
    df['path'] = df['id'].map(lambda x: os.path.join(image_direc,'{}'.format(x))) ##base_dir
    df.dropna(inplace = True)
    df = df.sort_values(by='time', axis=0)

    train_df, valid_df = train_test_split(df, 
								       test_size = 0.9999, 
								       random_state = 2018,
								       shuffle=False,
								       stratify = None) 
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])

    im_gen = ImageDataGenerator(samplewise_center=samplewise_center, #True, 
							      samplewise_std_normalization=samplewise_std_normalization, #True, 
							      horizontal_flip = False, 
							      vertical_flip = False, 
							      height_shift_range = 0, 
							      width_shift_range = 0, 
							      rotation_range = 0,  
							      shear_range = 0, 
							      fill_mode = 'reflect', 
							      zoom_range= 0) 
							      
    test_X, test_Y = next(gen_from_df(im_gen, 
							       valid_df, 
                                 shuffle = False,
							     path_col = 'path',
							    y_col = category, #'zscore', 
							    target_size = IMG_SIZE,
							     color_mode = 'grayscale',
							    batch_size = len(valid_df))) 
												    
    pred_Y = OWG.predict(test_X, batch_size = 1, verbose = True)
    fig, ax1 = plt.subplots(1,1, figsize = (6,6))
    ax1.plot(test_Y, pred_Y, 'k.', label = 'predictions')
    ax1.plot(test_Y, test_Y, 'r-', label = 'actual')
    ax1.legend()
    ax1.set_xlabel('Actual H (m)')
    ax1.set_ylabel('Predicted H (m)')
    plt.savefig('test_model_'+category+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')    
 
    
    
    
    

