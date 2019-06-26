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
from glob import glob
import zipfile

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.metrics import mean_absolute_error

def mae_metric(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)

#==============================================================	
## script starts here
if __name__ == '__main__':

    #image_dir = 'snap_images'
    #configfile = 'config_nearshore_H.json'
    #configfile = 'config_nearshore_T.json'
	
    image_dir = 'IR_images'		
    #configfile = 'config_IR_H.json'
    configfile = 'config_IR_T.json'
    #==============================================================
    ## user inputs
    with open(os.getcwd()+os.sep+'config'+os.sep+configfile) as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["img_size"])
    category = config["category"] 
    input_csv_file = config["input_csv_file"]   
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"]  
    num_epochs = int(config["num_epochs"]) ##100
	
    base_dir = os.path.normpath(os.getcwd()+os.sep+'train') 
	    
    IMG_SIZE = (im_size, im_size) ##(128, 128)
    
    prc_lower_withheld = 5
    prc_upper_withheld = 5

    # #==============================================================
    
    df = pd.read_csv(os.path.join(base_dir, input_csv_file))
    if input_csv_file=='snap-training-dataset.csv':
       df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))#+".jpg"
    elif input_csv_file=='IR-training-dataset.csv':
       df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}'.format(x)))+".png"
	   
    df = df.rename(index=str, columns={" H": "H", " T": "T"})   

    df.dropna(inplace = True)

    if input_csv_file=='snap-training-dataset.csv':    
        df['time'] = [int(k.split(os.sep)[-1].split('.')[0]) for k in df.path]
        df = df.sort_values(by='time', axis=0)

    ## making subsets of data based on prc_lower_withheld and prc_upper_withheld
    if (prc_lower_withheld>0) & (prc_upper_withheld>0):
        up = np.percentile(df[category], 100-prc_upper_withheld)
        low = np.percentile(df[category], prc_lower_withheld)
        extreme_df = df.loc[(df[category] < low) | (df[category] > up)]
        df = df.loc[(df[category] >= low) & (df[category] <= up)]
    elif (prc_lower_withheld>0) & (prc_upper_withheld==0):
        low = np.percentile(df[category], prc_lower_withheld)
        extreme_df = df.loc[(df[category] < low)]
        df = df.loc[(df[category] >= low)]
    elif (prc_lower_withheld==0) & (prc_upper_withheld>0):
        up = np.percentile(df[category], 100-prc_upper_withheld)
        extreme_df = df.loc[(new_df[category] > up)]
        df = df.loc[(df[category] <= up)]

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
							       df, 
                                 shuffle = False,
							     path_col = 'path',
							    y_col = category, #'zscore', 
							    target_size = IMG_SIZE,
							     color_mode = 'grayscale',
							    batch_size = len(df)))     


    ex_X, ex_Y = next(gen_from_df(im_gen, 
							   extreme_df, 
							 path_col = 'path',
							y_col = category, #'zscore', 
							target_size = IMG_SIZE,
							 color_mode = 'grayscale',
							batch_size = len(extreme_df)))

    #==============================================================    
    yhat = {}; exyhat = {}
    
    #==============================================================
    for batch_size in [16,32,64,128]: #16
        #counter = 0    
        for counter in range(1,5):
            if category == 'H':
                if input_csv_file=='snap-training-dataset.csv':			
                   weights_path=os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.nearshore.hdf5'
                elif input_csv_file=='IR-training-dataset.csv':
                   weights_path=os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.IR.hdf5'				
            else:
                if input_csv_file=='snap-training-dataset.csv':			
                   weights_path=os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.nearshore.hdf5'
                elif input_csv_file=='IR-training-dataset.csv':
                   weights_path=os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.IR.hdf5'			   
            if not os.path.isfile(weights_path): #counter==4:
                if input_csv_file=='snap-training-dataset.csv':			
                   files = sorted(glob(os.path.dirname(weights_path)+os.sep+'*nearshore*hdf5'))
                elif input_csv_file=='IR-training-dataset.csv':
                   files = sorted(glob(os.path.dirname(weights_path)+os.sep+'*IR*hdf5'))			   
                out_data = b''
                for fn in files:
                    with open(fn, 'rb') as fp:
                        out_data += fp.read()
                with open(weights_path, 'wb') as fp:
                   fp.write(out_data)                   	

            # load json and create model
            print("Creating model")						
            json_file = open(weights_path.replace('.hdf5','.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            OWG = model_from_json(loaded_model_json)
            print("Loading weights into model")			
            # load weights into new model
            OWG.load_weights(weights_path)
            print("Loaded model "+str(counter)+" from disk")

            OWG.compile(optimizer = 'adam', loss = 'mse',metrics = [mae_metric]) 
											    
            yhat['M'+str(counter)+'_B'+str(batch_size)] = OWG.predict(test_X, batch_size = 100, verbose = True) #len(test_X)
            exyhat['M'+str(counter)+'_B'+str(batch_size)] = OWG.predict(ex_X, batch_size = 100, verbose = True)            
            #counter += 1

    #==============================================================

    #==============================================================
    fig = plt.figure(figsize = (16,16))
    labels = 'ABCDEFGHIJKLMNOP'	
    counter = 1
    for model in [1,2,3,4]:
        for batch_size in [16,32,64,128]:
            ## average over batch per model
            pred_Y = yhat['M'+str(model)+'_B'+str(batch_size)] 
		    #(yhat['M'+str(counter)+'_B16']+yhat['M'+str(counter)+'_B32']+yhat['M'+str(counter)+'_B64']+yhat['M'+str(counter)+'_B128'])/4
            pred_Y = np.squeeze(np.asarray(pred_Y))

            pred_exY = exyhat['M'+str(model)+'_B'+str(batch_size)] 
		    #(exyhat['M'+str(counter)+'_B16']+exyhat['M'+str(counter)+'_B32']+exyhat['M'+str(counter)+'_B64']+exyhat['M'+str(counter)+'_B128'])/4
            pred_exY = np.squeeze(np.asarray(pred_exY))
                
            plt.subplot(4,4,counter)
            plt.plot(test_Y, pred_Y, 'k.', markersize=3, label = 'predictions')
            plt.plot(ex_Y, pred_exY, 'bx', markersize=3, label = 'predictions')
            if input_csv_file=='snap-training-dataset.csv':						
               if category=='H':			
                  plt.plot([0.5, 2.75], [0.5, 2.75], 'r-', label = 'actual')
                  plt.xlim(0.25,3); plt.ylim(0.25, 3)
               else:
                  plt.plot([8, 23], [8, 23], 'r-', label = 'actual')
                  plt.xlim(7,24); plt.ylim(7, 24)	
            elif input_csv_file=='IR-training-dataset.csv':						
               if category=='H':			
                  plt.plot([0.25, 5.75], [0.25, 5.75], 'r-', label = 'actual')
                  plt.xlim(0,6); plt.ylim(0, 6)
               else:
                  plt.plot([3, 19], [3, 19], 'r-', label = 'actual')
                  plt.xlim(2,20); plt.ylim(2, 20)				  
            if counter==13:
               if input_csv_file=='snap-training-dataset.csv':						
                  if category=='H':
                     plt.xlabel(r'Actual $H_s$ (m)', fontsize=6)
                     plt.ylabel(r'Predicted $H_s$ (m)', fontsize=6)
                  elif category=='T':
                     plt.xlabel(r'Actual $T_p$ (s)', fontsize=6)
                     plt.ylabel(r'Predicted $T_p$ (s)', fontsize=6)
               elif input_csv_file=='IR-training-dataset.csv':						
                  if category=='H':
                     plt.xlabel(r'Actual $H$ (m)', fontsize=6)
                     plt.ylabel(r'Predicted $H$ (m)', fontsize=6)
                  elif category=='T':
                     plt.xlabel(r'Actual $T$ (s)', fontsize=6)
                     plt.ylabel(r'Predicted $T$ (s)', fontsize=6)				  
            rms = np.sqrt(np.nanmean((pred_Y - test_Y)**2))
            rsq = np.min(np.corrcoef(test_Y, pred_Y))**2
            exrms = np.sqrt(np.nanmean((pred_exY - ex_Y)**2))
            if category=='H':			
               string = r'RMS (m): '+str(rms)[:4] + ' ('+str(exrms)[:4]+'), R$^2$: '+str(rsq)[:4]
            elif category=='T':
               string = r'RMS (s): '+str(rms)[:4] + ' ('+str(exrms)[:4]+'), R$^2$: '+str(rsq)[:4]			
            plt.title(labels[counter-1]+') '+string, fontsize=6, loc='left')
            plt.setp(plt.gca().get_xticklabels(), fontsize=5)				
            plt.setp(plt.gca().get_yticklabels(), fontsize=5)
            counter += 1
    if input_csv_file=='IR-training-dataset.csv':			
       plt.savefig('ensemble_allmodels_'+category+'-IR.png', dpi=300, bbox_inches='tight')	
    elif input_csv_file=='snap-training-dataset.csv':
       plt.savefig('ensemble_allmodels_'+category+'-nearshore.png', dpi=300, bbox_inches='tight')
    plt.close('all') ; del fig
	
	
	