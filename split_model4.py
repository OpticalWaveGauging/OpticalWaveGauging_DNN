## split_model4.py 
## A script to split large model files to smaller files < 100 MB (so they fit on github)
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu
import os

CHUNK_SIZE = int(9.9e+7)
data = 'H' #'T' 
im_size=128 
epics=100 #20, 50

for batch_size [16,32,64,128]:

    infile = os.getcwd()+os.sep+'im'+str(im_size)+os.sep+'res_'+name+os.sep+str(epics)+'epoch'+os.sep+data+os.sep+'model4'+os.sep+'batch'+str(batch_size)+os.sep+'waveheight_weights_model4_'+str(batch_size)+'batch.best.hdf5'

    file_number = 1
    with open(infile, 'rb') as f:
       chunk = f.read(CHUNK_SIZE)
       while chunk:
          newfile = infile.replace('.hdf5', '.c' + str(file_number)+'.hdf5')
          with open(newfile, 'wb') as chunk_file:
             chunk_file.write(chunk)
             file_number += 1
             chunk = f.read(CHUNK_SIZE)
