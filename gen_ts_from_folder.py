## predict_image.py 
## A script to test a model on a single image
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import sys, getopt, os
import numpy as np 
from keras.models import model_from_json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU
from keras.preprocessing import image
import json

#==============================================================	
## script starts here
if __name__ == '__main__':

    #==============================================================
    ## user inputs
    argv = sys.argv[1:]
    try:
       opts, args = getopt.getopt(argv,"h:i:")
    except getopt.GetoptError:
       print('python predict_image.py -w path/to/folder')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('Example usage: python predict_image.py -i train/snap_images')
          sys.exit()
       elif opt in ("-i"):
          image_path = arg
		  
    #==============================================================
    with open(os.getcwd()+os.sep+'conf'+os.sep+'config_test.json') as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["im_size"])
    category = config["category"] 
    weights_path = config["weights_path"]
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"] 
	file_ext = config["file_ext"]

    IMG_SIZE = (im_size, im_size)
    #==============================================================
     
    # load json and create model
    json_file = open(weights_path.replace('.hdf5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    OWG = model_from_json(loaded_model_json)
    # load weights into new model
    OWG.load_weights(weights_path)

    OWG.compile(optimizer = 'adam', loss = 'mse') 

    T = []; V = []	
	
    for file in glob(image_path+os.sep+'*.'file_ext):

        img = image.load_img(file, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x =  0.21*x[:,:,0] + 0.72*x[:,:,1] + 0.07*x[:,:,2] ##rgb to grey
    
        if samplewise_std_normalization==True:
            x = x/np.std(x)   
        if samplewise_center==True:
            x = x - np.mean(x)    
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=3)
						      
        pred_Y = np.squeeze(OWG.predict(x, batch_size = 1, verbose = False))
        print("====================================")
        print(category+' = '+str(pred_Y)[:5])
        print("====================================")
        t = file.split(os.sep)[-1].split('.')[0]
        print("time is ", t)
        T.append(t)
        V.append(pred_Y)

 
