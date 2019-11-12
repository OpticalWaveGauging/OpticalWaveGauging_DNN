#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## compile_results.py 
## A script to test all 16 model and make plots
## showing accuracy as a function of model and batch size
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys, getopt, os, gc
from glob import glob
import zipfile, json
from sklearn.model_selection import train_test_split
import pandas as pd
from imageio import imread

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU
from utils import *

#==============================================================	
## script starts here
if __name__ == '__main__':

    #==============================================================
    ## user inputs
    argv = sys.argv[1:]
    try:
       opts, args = getopt.getopt(argv,"h:i:c:")
    except getopt.GetoptError:
       print('python compile_results.py -w path/to/folder -c config_file.json')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('Example usage: python compile_results.py -c config_nearshore_H.json')
          sys.exit()
       elif opt in ("-i"):
          image_dir = arg
       elif opt in ("-c"):
          configfile = arg
          		  
    #==============================================================
	# load the user configs
    with open(os.getcwd()+os.sep+'config'+os.sep+configfile) as f:    
	    config = json.load(f)
		
    # config variables
    im_size    = int(config["img_size"])
    category = config["category"] 
    input_csv_file = config["input_csv_file"]   
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"]  
    num_epochs = int(config["num_epochs"]) 
    prc_lower_withheld = config['prc_lower_withheld'] 
    prc_upper_withheld = config['prc_upper_withheld']
    test_size = float(config["test_size"])	
    
    
    base_dir = os.path.normpath(os.getcwd())
	    
    IMG_SIZE = (im_size, im_size) 


    ## download files and unzip
    if input_csv_file=='IR-training-dataset.csv':
        image_direc = 'IR_images'+os.sep+'data'
    elif input_csv_file=='snap-training-dataset.csv':
        image_direc = 'snap_images'+os.sep+'data'		
    elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':	
        image_direc = 'snap'+os.sep+'data'

    print ("[INFO] Preparing the data ...")
    # call the utils.py function get_and_tidy_df
    newdf, df = get_and_tidy_df(os.path.normpath(os.getcwd()), input_csv_file, image_direc, category)
		
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

    # call the utils.py function im_gen_noaug            
    im_gen = im_gen_noaug(samplewise_std_normalization, samplewise_center)
    
    train_df, valid_df = train_test_split(newdf, test_size = test_size, random_state = 2018, stratify = newdf['category'])
    print('train', train_df.shape[0], 'validation', valid_df.shape[0])
    
    # call the utils.py function gen_from_def
    train_X, train_Y = gen_from_def(IMG_SIZE, train_df, image_direc, category, im_gen)
    test_X, test_Y = gen_from_def(IMG_SIZE, valid_df, image_direc, category, im_gen)
    
    if (prc_lower_withheld>0) | (prc_upper_withheld>0):
        ex_X, ex_Y = gen_from_def(IMG_SIZE, extreme_df, image_direc, category, im_gen)


    #==============================================================    
    yhat = {}; exyhat = {}
    
    for batch_size in [16,32,64,128]:
        print ("[INFO] Working on batch size = %i ..." % (batch_size))
        for counter in range(1,5):
            # call the utils.py function get_weights_path
            weights_path = get_weights_path(input_csv_file, category, counter, 
			                                im_size, batch_size, num_epochs)                	
            # call the utils.py function load_OWG_json   							    
            OWG = load_OWG_json(weights_path)
            
            yhat['M'+str(counter)+'_B'+str(batch_size)] = OWG.predict(test_X, batch_size = 100, verbose = True)
            if (prc_lower_withheld>0) | (prc_upper_withheld>0):	
                exyhat['M'+str(counter)+'_B'+str(batch_size)] = OWG.predict(ex_X, batch_size = 100, verbose = True)           
            gc.collect()
            
    #==============================================================
    print ("[INFO] Making plots ...")
    #==============================================================
    fig = plt.figure(figsize = (16,16))
    labels = 'ABCDEFGHIJKLMNOP'	
    counter = 1
    for model in [1,2,3,4]:
        for batch_size in [16,32,64,128]:
            pred_Y = yhat['M'+str(model)+'_B'+str(batch_size)] 
            pred_Y = np.squeeze(np.asarray(pred_Y))

            if (prc_lower_withheld>0) | (prc_upper_withheld>0):
                pred_exY = exyhat['M'+str(model)+'_B'+str(batch_size)] 
                pred_exY = np.squeeze(np.asarray(pred_exY))
                
            plt.subplot(4,4,counter)
            plt.plot(test_Y, pred_Y, 'b.', markersize=3, label = 'predictions')
            if (prc_lower_withheld>0) | (prc_upper_withheld>0):
               plt.plot(ex_Y, pred_exY, 'rx', markersize=3, label = 'predictions')
            if input_csv_file=='IR-training-dataset.csv':						
               if category=='H':			
                  plt.plot([0.5, 2.75], [0.5, 2.75], 'k-', label = 'actual')
                  plt.xlim(0.25,3); plt.ylim(0.25, 3)
               else:
                  plt.plot([8, 23], [8, 23], 'k-', label = 'actual')
                  plt.xlim(7,24); plt.ylim(7, 24)	
            else:
               if category=='H':			
                  plt.plot([0.25, 5.75], [0.25, 5.75], 'k-', label = 'actual')
                  plt.xlim(0,6); plt.ylim(0, 6)
               else:
                  plt.plot([3, 19], [3, 19], 'k-', label = 'actual')
                  plt.xlim(2,20); plt.ylim(2, 20)				  
            if counter==13:
               if input_csv_file=='IR-training-dataset.csv':						
                  if category=='H':
                     plt.xlabel(r'Actual $H$ (m)', fontsize=6)
                     plt.ylabel(r'Predicted $H$ (m)', fontsize=6)
                  elif category=='T':
                     plt.xlabel(r'Actual $T$ (s)', fontsize=6)
                     plt.ylabel(r'Predicted $T$ (s)', fontsize=6)
               else:						
                  if category=='H':
                     plt.xlabel(r'Actual $H_s$ (m)', fontsize=6)
                     plt.ylabel(r'Predicted $H_s$ (m)', fontsize=6)
                  elif category=='T':
                     plt.xlabel(r'Actual $T_p$ (s)', fontsize=6)
                     plt.ylabel(r'Predicted $T_p$ (s)', fontsize=6)					 
            rms = np.sqrt(np.nanmean((pred_Y - test_Y)**2))
            rsq = np.min(np.corrcoef(test_Y, pred_Y))**2
            if (prc_lower_withheld>0) | (prc_upper_withheld>0):
                exrms = np.sqrt(np.nanmean((pred_exY - ex_Y)**2))
            if category=='H':
                string = r'RMS (m): '+str(rms)[:4] + ', R$^2$: '+str(rsq)[:4]
            elif category=='T':
                string = r'RMS (s): '+str(rms)[:4] + ', R$^2$: '+str(rsq)[:4]			
            plt.title(labels[counter-1]+') '+string, fontsize=6, loc='left')
            plt.setp(plt.gca().get_xticklabels(), fontsize=5)				
            plt.setp(plt.gca().get_yticklabels(), fontsize=5)
            counter += 1
    if input_csv_file=='IR-training-dataset.csv':			
       plt.savefig('ensemble_allmodels_'+category+'-IR.png', dpi=300, bbox_inches='tight')	
    elif input_csv_file=='snap-training-dataset.csv':
       plt.savefig('ensemble_allmodels_'+category+'-nearshore.png', dpi=300, bbox_inches='tight')
    elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
       plt.savefig('ensemble_allmodels_'+category+'-oblique.png', dpi=300, bbox_inches='tight')
	   
    plt.close('all') ; del fig
	
	

