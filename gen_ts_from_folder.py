#      ▄▄▌ ▐ ▄▌ ▄▄ • 
#▪     ██· █▌▐█▐█ ▀ ▪
# ▄█▀▄ ██▪▐█▐▐▌▄█ ▀█▄
#▐█▌.▐▌▐█▌██▐█▌▐█▄▪▐█
# ▀█▄▀▪ ▀▀▀▀ ▀▪·▀▀▀▀ 
#
## gen_ts_from_folder.py 
## A script to use a model on a folder of images
## and compile a time-series of predictions
## modify config_test.json with relevant inputs
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

# import libraries
import sys, getopt, os
import numpy as np 
import json
from glob import glob
import pandas as pd
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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
       print('python predict_image.py -w path/to/folder')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('Example usage: python predict_image.py -i snap_images/data')
          sys.exit()
       elif opt in ("-i"):
          image_path = arg
          
    if not os.path.isdir(image_path):
        print('Provided image directory apparently does not exist ... exiting')
        sys.exit(1)
    
    with open(os.getcwd()+os.sep+'config'+os.sep+'config_test.json') as f:    
	    config = json.load(f)

    # config variables
    im_size    = int(config["im_size"])
    category = config["category"] 
    weights_path = config["weights_path"]
    samplewise_std_normalization = config["samplewise_std_normalization"]
    samplewise_center = config["samplewise_center"]
    file_ext = config["file_ext"]
    input_csv_file = config["input_csv_file"]   
    
    IMG_SIZE = (im_size, im_size)
    #==============================================================

    # load json and create model
    # call the utils.py function load_OWG_json    
    OWG = load_OWG_json(os.getcwd()+os.sep+weights_path)

    # call the utils.py function get_and_tidy_df
    _, df = get_and_tidy_df(os.path.normpath(os.getcwd()), input_csv_file, image_path, category)
    
    # call the utils.py function im_gen_noaug    
    im_gen = im_gen_noaug(samplewise_std_normalization, samplewise_center)

    print ("[INFO] Reading images ...")     												    
    # call the utils.py function gen_from_def
    test_X, test_Y = gen_from_def(IMG_SIZE, df, image_path, category, im_gen)

    print ("[INFO] Predicting ...")     												    

    if len(test_X)<2:
        print('Insufficient imagery - check your config_test.json file ... exiting')
        sys.exit(1)
        
    V = OWG.predict(test_X, batch_size = 128, verbose = True)
    
    files  = sorted(glob(image_path+os.sep+'*.'+file_ext))
    
    T = [file.split(os.sep)[-1].split('.')[0] for file in files]

    print ("[INFO] Making plots...")
    T = np.array(T, dtype=np.float32)
    V = np.array(V.squeeze(), dtype=np.float32)
     
    # interpolate onto a regular small timestamp
    df = df.sort_values('time')
    x = np.arange(T.min(), T.max(),len(T)*5)
    Vi = np.interp(x,T,V)
    
    # make a dataframe
    if category=='H':
       d = {'dates': [datetime.fromtimestamp(t) for t in x], 'H': np.interp(x,df.time.values,df[category])}
    else:
       d = {'dates': [datetime.fromtimestamp(t) for t in x], 'T': np.interp(x,df.time.values,df[category])}    
    new_df = pd.DataFrame(data=d)
    
    # remove night-time hour samples (before 7am and after 7pm)
    ind = np.where((new_df.dates.dt.hour<7) | (new_df.dates.dt.hour>19))[0]

    new_df[category][ind] = np.nan
    Vi[ind] = np.nan
    new_df[category+'_est'] = Vi

    # make a time-series plot showing actual and estimated
    # for just a few days 
    n1 = 0; n2=130
    fig = plt.figure(figsize=(8,6))
    ax=plt.subplot(211)	
    ax.plot_date(mdate.epoch2num(x)[n1:n2], new_df[category][n1:n2],'k', lw=2, label='Measured')
    ax.plot_date(mdate.epoch2num(x)[n1:n2], Vi[n1:n2], 'b.-', lw=2, alpha=0.5, label='Estimated from Image')
    plt.ylabel(r'$H_s$ (s)')
    date_formatter = mdate.DateFormatter('%m-%d-%y')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(image_path.split(os.sep)[0]+'_short_time_series_'+str(category)+'.png', dpi=300, bbox_inches='tight')
    plt.close()
 
    # make a time-series plot showing actual and estimated
    # for the whole time period   
    fig = plt.figure(figsize=(8,6))
    ax=plt.subplot(211)	
    ax.plot_date(mdate.epoch2num(x), new_df[category],'k', lw=2, label='Measured')
    ax.plot_date(mdate.epoch2num(x), Vi, 'b.-', lw=2, alpha=0.5, label='Estimated from Image')
    plt.ylabel(r'$H_s$ (s)')
    date_formatter = mdate.DateFormatter('%m-%d-%y')
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()
    plt.legend()    
    plt.savefig(image_path.split(os.sep)[0]+'_time_series_'+str(category)+'.png', dpi=300, bbox_inches='tight')
    plt.close()

    new_df.to_csv(image_path.split(os.sep)[0]+'_obs_est_time_series.csv')


#    T = []; V = []
#    files  = sorted(glob(image_path+os.sep+'*.'+file_ext))
#    
#    print ("[INFO] Estimate per image ...")

	
#    counter  = 1
#    for file in files:
#        if 100*(len(files)/counter) % 2 == 0:
#           print('%f percent done' % (100*(counter/len(files))))
#        pred_Y, t = get_estimate_nopara(OWG,file, samplewise_std_normalization, samplewise_center)
#        T.append(t)
#        V.append(pred_Y)
#        counter += 1
#    # call the utils.py function get_and_tidy_df        
#    _, df = get_and_tidy_df(os.path.normpath(os.getcwd()), input_csv_file, image_path, category)
#       
