## utils.py 
## Utilities for training optical wave height gauges 
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import os
import numpy as np

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