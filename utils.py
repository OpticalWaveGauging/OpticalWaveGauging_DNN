#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## utils.py 
## common functions for training and testing optical wave gauges 
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import os, requests
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# get a keras image data generator with no augmentation
# but scale the images according to samplewise_std_normalization and samplewise_center
def im_gen_noaug(samplewise_std_normalization, samplewise_center):
    return ImageDataGenerator(samplewise_center=samplewise_center,  
							      samplewise_std_normalization=samplewise_std_normalization, 
							      horizontal_flip = False, 
							      vertical_flip = False, 
							      height_shift_range = 0, 
							      width_shift_range = 0, 
							      rotation_range = 0,  
							      shear_range = 0, 
							      fill_mode = 'reflect', 
							      zoom_range= 0) 


#use a trained and compiled OWG model for prediction on 1 image
def pred_1image(OWG,image_path, IMG_SIZE, samplewise_std_normalization, samplewise_center):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x =  0.21*x[:,:,0] + 0.72*x[:,:,1] + 0.07*x[:,:,2] ##rgb to grey
    
    if samplewise_std_normalization==True:
        x = x/np.std(x)   
    if samplewise_center==True:
        x = x - np.mean(x)    
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=3)
						      
    return np.squeeze(OWG.predict(x, batch_size = 1, verbose = False))
    
# define a weights file path based on user inputs    
def get_weights_path(input_csv_file, category, counter, imsize, batch_size, num_epochs):
	if category == 'H':		
		if input_csv_file=='IR-training-dataset.csv':			
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.IR.hdf5'
		elif input_csv_file=='snap-training-dataset.csv':
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.nearshore.hdf5'	
		elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'H'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.oblique.hdf5'						
	else:
		if input_csv_file=='IR-training-dataset.csv':						
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.IR.hdf5'	
		elif input_csv_file=='snap-training-dataset.csv':
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.nearshore.hdf5'	
		elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
			weights_path=os.getcwd()+os.sep+'im'+str(imsize)+os.sep+'res'+os.sep+str(num_epochs)+'epoch'+os.sep+'T'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model'+str(counter)+'_'+str(batch_size)+'batch.best.oblique.hdf5'
	return weights_path

#load an OWG from json file and load weights and compile, ready for use
# in prediction
def load_OWG_json(weights_path):
    # load json and create model
    print("Creating model")						
    json_file = open(weights_path.replace('.hdf5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    OWG = model_from_json(loaded_model_json)
    #print("Loading weights into model")			
    # load weights into new model
    OWG.load_weights(weights_path)
    #print("Loaded model from disk")

    OWG.compile(optimizer = 'adam', loss = 'mse',metrics = [mae_metric]) 
    return OWG

# generate a pandas datafrom from csv file and categorize for 
# subsequent stratified random sampling
def get_and_tidy_df(base_dir, input_csv_file, image_dir, category):
	df = pd.read_csv(os.path.join(base_dir, input_csv_file))
	if input_csv_file=='IR-training-dataset.csv':
		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
		                                                image_dir,'{}'.format(x)))+".png"
	elif input_csv_file=='snap-training-dataset.csv':
		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
		                                                image_dir,'{}'.format(x)))
	elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
		                                                image_dir,'{}'.format(x)))+".jpg"
		
	df = df.rename(index=str, columns={" H": "H", " T": "T"})   
	
	if category == 'H':
		mean = df['H'].mean() 
		div = df['H'].std() 
		df['zscore'] = df['H'].map(lambda x: (x-mean)/div)
	elif category == 'T':
		mean = df['T'].mean() 
		div = df['T'].std() 
		df['zscore'] = df['T'].map(lambda x: (x-mean)/div)			
	else:
		print("Unknown category: "+str(category))
		print("Fix config file, exiting now ...")
		sys.exit()
	
	df.dropna(inplace = True)
	try:
		df = df.sort_values(by='time', axis=0)
	except:
		df = df.sort_values(by='id', axis=0)

	if category == 'H':
		df['category'] = pd.cut(df['H'], 10)
	else:
		df['category'] = pd.cut(df['T'], 8)
		
	df['index1'] = df.index

	if input_csv_file=='IR-training-dataset.csv':
		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)
	elif input_csv_file=='snap-training-dataset.csv':
		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)
	elif input_csv_file=='Nearshore-Training-Oblique-cam2-snap.csv':
		new_df = df.groupby(['category']).apply(lambda x: x.sample(int(len(df)/2), replace = True)).reset_index(drop = True)
		
	return new_df, df

# mean absolute error
def mae_metric(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)
	
# make a genrator from a dataframe	
def gen_from_def(IMG_SIZE, df, image_dir, category, im_gen):

	gen = im_gen.flow_from_dataframe(dataframe=df,
								  directory=image_dir,
								  x_col="path",
								  y_col=category,
								  target_size=IMG_SIZE,
								  batch_size=len(df),
								  color_mode = 'grayscale',
								  shuffle=False,
								  class_mode='raw')
								  	
	ex_X, ex_Y = next(gen)
	return ex_X, ex_Y
			
# as it says on the tin
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)	
                
                
