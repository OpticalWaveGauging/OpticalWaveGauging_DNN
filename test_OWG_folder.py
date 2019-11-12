#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## test_OWG_folder.py 
## A script to test a model on a test data set from a csv
## modify config_test.json with relevant inputs
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import sys, getopt, os, json
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU
from utils import *

#==============================================================	
## script starts here
if __name__ == '__main__':

    #==============================================================
    ## user inputs
    argv = sys.argv[1:]
    try:
       opts, args = getopt.getopt(argv,"h:i:")
    except getopt.GetoptError:
       print('python test_OWG_folder.py -w path/to/folder')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('Example usage: python test_OWG_folder.py -i snap_images/data')
          sys.exit()
       elif opt in ("-i"):
          image_direc = arg
          
    #==============================================================
    ## user inputs
    with open(os.getcwd()+os.sep+'config'+os.sep+'config_test.json') as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["im_size"])
    category = config["category"] 
    input_csv_file = config["input_csv_file"]   
    weights_path = config["weights_path"]
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"]  
    
    IMG_SIZE = (im_size, im_size) 
    #==============================================================

    print ("[INFO] Preparing model...")     
    # load json and create model
    # call the utils.py function load_OWG_json
    OWG = load_OWG_json(os.getcwd()+os.sep+weights_path)

    # call the utils.py function get_and_tidy_df
    _, df = get_and_tidy_df(os.path.normpath(os.getcwd()), input_csv_file, image_direc, category)

    # call the utils.py function im_gen_noaug    
    im_gen = im_gen_noaug(samplewise_std_normalization, samplewise_center)

    # call the utils.py function gen_from_def
    test_X, test_Y = gen_from_def(IMG_SIZE, df, image_direc, category, im_gen)

    print ("[INFO] Predicting ...")     												    
    pred_Y = OWG.predict(test_X, batch_size = 128, verbose = True)
    
    print ("[INFO] Plotting ...")     												        
    fig, ax1 = plt.subplots(1,1, figsize = (6,6))
    ax1.plot(test_Y, pred_Y, 'b.', alpha=0.5, label = 'Estimated')
    ax1.plot(test_Y, test_Y, 'k-', label = 'Observed')
    ax1.legend()
    if category=='H':
       ax1.set_xlabel('Actual H (m)')
       ax1.set_ylabel('Estimated H (m)')
    else:
       ax1.set_xlabel('Actual T (s)')
       ax1.set_ylabel('Estimated T (s)')    
    plt.savefig(image_direc.split(os.sep)[0]+'_test_model_'+category+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')    
	

