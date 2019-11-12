#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## predict_image.py 
## A script to use a model on a single image for prediction
## modify config_test.json with relevant inputs
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import sys, getopt, os
import numpy as np 
import json

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
       print('python predict_image.py -w path/to/image.{jpg/png/tiff}')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('Example usage: python predict_image.py -i snap_images/data/1513706400.cx.snap.jpg')
          sys.exit()
       elif opt in ("-i"):
          image_path = arg

    ##examples:
    #image_path = 'snap_images/data/1513706400.cx.snap.jpg' #H = 0.4      
    #image_path = 'snap_images/data/1516127400.cx.snap.jpg' #H = 1.85
    #image_path = 'snap_images/data/1516401000.cx.snap.jpg' #H = 2.33

    with open(os.getcwd()+os.sep+'config'+os.sep+'config_test.json') as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["im_size"])
    category = config["category"] 
    weights_path = config["weights_path"]
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"] 

    IMG_SIZE = (im_size, im_size)
    #==============================================================
    print ("[INFO] Preparing model...")     
    # load json and create model
    # call the utils.py function load_OWG_json    
    OWG = load_OWG_json(weights_path)

    print ("[INFO] Predicting ...")
    # call the utils.py function pred_1image  
    pred_Y = pred_1image(OWG, image_path, IMG_SIZE, 
                        samplewise_std_normalization, samplewise_center)
    print("====================================")
    print(category+' = '+str(pred_Y)[:5])
    print("====================================")


