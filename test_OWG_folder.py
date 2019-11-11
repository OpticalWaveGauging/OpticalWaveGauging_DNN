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
          print('Example usage: python test_OWG_folder.py -i train/snap_images')
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
	

#    df = pd.read_csv(input_csv_file)
#    base_dir = os.path.normpath(os.getcwd())
#    if input_csv_file=='IR-training-dataset.csv':
#       df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_direc,'{}'.format(x)))+".png"
#    elif input_csv_file=='snap-training-dataset.csv':
#       df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_direc,'{}'.format(x)))
#    elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
#       df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_direc,'{}'.format(x)))+".jpg"
#		    
#    df.dropna(inplace = True)
#    df = df.sort_values(by='time', axis=0)

#    im_gen = ImageDataGenerator(samplewise_center=samplewise_center,  
#							      samplewise_std_normalization=samplewise_std_normalization, 
#							      horizontal_flip = False, 
#							      vertical_flip = False, 
#							      height_shift_range = 0, 
#							      width_shift_range = 0, 
#							      rotation_range = 0,  
#							      shear_range = 0, 
#							      fill_mode = 'reflect', 
#							      zoom_range= 0) 
	
	#    json_file = open(weights_path.replace('.hdf5','.json'), 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    OWG = model_from_json(loaded_model_json)
#    # load weights into new model
#    OWG.load_weights(weights_path)
#    print("Loaded model from disk")

#    OWG.compile(optimizer = 'adam', loss = 'mse',metrics = [mae_metric])  #rmsprop

#    test_generator = im_gen.flow_from_dataframe(dataframe=valid_df,
#								  directory=image_direc,
#								  x_col="path",
#								  y_col=category,
#								  target_size=IMG_SIZE,
#								  batch_size=len(valid_df),
#								  color_mode = 'grayscale',
#								  shuffle=False,
#								  class_mode='raw')
#								  	
#    test_X, test_Y = next(test_generator)
								      
#    test_X, test_Y = next(gen_from_df(im_gen, 
#							       valid_df, 
#                                 shuffle = False,
#							     path_col = 'path',
#							    y_col = category, 
#							    target_size = IMG_SIZE,
#							     color_mode = 'grayscale',
#							    batch_size = len(valid_df))) 	
