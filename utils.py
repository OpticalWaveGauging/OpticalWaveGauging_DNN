## utils.py 
## Utilities for training optical wave gauges 
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import os
import numpy as np
import requests
from keras.metrics import mean_absolute_error

# mean absolute error
def mae_metric(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)

	
def gen_from_df(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
	
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse', **dflow_args)
									
    df_gen.filenames = in_df[path_col].values
	
    df_gen.classes = np.stack(in_df[y_col].values)
	
    df_gen.samples = in_df.shape[0]
	
    df_gen.n = in_df.shape[0]
	
    df_gen._set_index_array()
	
    df_gen.directory = '' 
    return df_gen	
	

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