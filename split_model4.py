## split_model4.py 
## A script to split large model files to smaller files < 100 MB (so they fit on github)
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import os
import json

# load the user configs
with open(os.getcwd()+os.sep+'conf'+os.sep+'config.json') as f:    
	config = json.load(f)

print(config)
# config variables
im_size    = int(config["img_size"])
epics = int(config["num_epochs"]) ##100
data = config["category"] ##'H'

CHUNK_SIZE = int(9.9e+7)

for batch_size [16,32,64,128]:
    if data=='H':
		infile = os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(epics)+'epoch'+os.sep+data+os.sep+'model4'+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model4_'+str(batch_size)+'batch.best.hdf5'
	elif data=='T':
		infile = os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res'+os.sep+str(epics)+'epoch'+os.sep+data+os.sep+'model4'+os.sep+'batch'+str(batch_size)+os.sep+'waveperiod_weights_model4_'+str(batch_size)+'batch.best.hdf5'	
	else:
		print("Unknown category: "+str(category))
		print("Fix config file, exiting now ...")
		import sys
		sys.exit()
	
    file_number = 1
    with open(infile, 'rb') as f:
       chunk = f.read(CHUNK_SIZE)
       while chunk:
          newfile = infile.replace('.hdf5', '.c' + str(file_number)+'.hdf5')
          with open(newfile, 'wb') as chunk_file:
             chunk_file.write(chunk)
             file_number += 1
             chunk = f.read(CHUNK_SIZE)
