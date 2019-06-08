## plt_compile_results.py 
## A script to collate all model results and make plots of model-observation mismatch
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import savemat, loadmat
from utils import *

##==========================

epics = 200
root = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN'

fig, axs = plt.subplots(4, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
E = []; EE = []		
counter2=0  

for batch_size in [16,32,64,128]:
   if batch_size==16:
      letters='abcd'
   elif batch_size==32:
      letters='efgh'
   elif batch_size==64:
      letters='ijkl'		
   else:
      letters='mnop'
	  
   for counter1 in range(4):
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'H'+os.sep+'model'+str(counter1+1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1+1)+'_200epoch'+str(batch_size)+'batch_nearshore.mat')
      E.append(dat['yhat'])
      EE.append(dat['yhat_extreme'])	  
      axs[counter1, counter2].plot(dat['test_Y'], dat['yhat'], 'k.', label = 'estimated', markersize=1)
      axs[counter1, counter2].plot(dat['extreme_Y'], dat['yhat_extreme'], 'bo', label = 'estimated, out-of-calibration', markersize=1)
      axs[counter1, counter2].plot([0.25,2.75], [0.25,2.75], 'r--', label = 'observed', lw=0.5)
      string = r'RMS (m): '+str(np.squeeze(dat['rms']))[:4] + ' ('+str(np.squeeze(dat['rms_ex']))[:4] +')  R$^2$: '+str(np.squeeze(dat['rsq']))[:4]
      axs[counter1, counter2].set_title(letters[counter2]+') '+string, fontsize=3, loc='left')	 
      axs[counter1, counter2].set_xlim(0.25, 2.75)
      axs[counter1, counter2].set_ylim(0.25,2.75)		  
      if counter1==3:
         if counter2==0:
            axs[counter1, counter2].set_xlabel('Observed $H_s$ (m)', fontsize=5)
            axs[counter1, counter2].set_ylabel(r'Estimated $H_s$ (m)', fontsize=5)		  
      plt.setp(axs[counter1, counter2].get_xticklabels(), fontsize=4)
      plt.setp(axs[counter1, counter2].get_yticklabels(), fontsize=4)
      axs[counter1, counter2].set_aspect(1.0)
   counter2 += 1
   
plt.savefig('nearshore_H_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')							
plt.close('all')
del fig		




epics = 200
root = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN'

fig, axs = plt.subplots(4, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
E = []; EE = []		
counter2=0  

for batch_size in [16,32,64,128]:
   if batch_size==16:
      letters='abcd'
   elif batch_size==32:
      letters='efgh'
   elif batch_size==64:
      letters='ijkl'		
   else:
      letters='mnop'
	  
   for counter1 in range(4):
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'H'+os.sep+'model'+str(counter1+1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1+1)+'_200epoch'+str(batch_size)+'batch_IR.mat')
      E.append(dat['yhat'])
      EE.append(dat['yhat_extreme'])	  
      axs[counter1, counter2].plot(dat['test_Y'], dat['yhat'], 'k.', label = 'estimated', markersize=1)
      axs[counter1, counter2].plot(dat['extreme_Y'], dat['yhat_extreme'], 'bo', label = 'estimated, out-of-calibration', markersize=1)
      axs[counter1, counter2].plot([0,6], [0,6], 'r--', label = 'observed', lw=0.5)
      string = r'RMS (m): '+str(np.squeeze(dat['rms']))[:4] + ' ('+str(np.squeeze(dat['rms_ex']))[:4] +')  R$^2$: '+str(np.squeeze(dat['rsq']))[:4]
      axs[counter1, counter2].set_title(letters[counter2]+') '+string, fontsize=3, loc='left')	 
      axs[counter1, counter2].set_xlim(0, 6)
      axs[counter1, counter2].set_ylim(0,6)		  
      if counter1==3:
         if counter2==0:
            axs[counter1, counter2].set_xlabel('Observed H (m)', fontsize=5)
            axs[counter1, counter2].set_ylabel(r'Estimated H (m)', fontsize=5)		  
      plt.setp(axs[counter1, counter2].get_xticklabels(), fontsize=4)
      plt.setp(axs[counter1, counter2].get_yticklabels(), fontsize=4)
      axs[counter1, counter2].set_aspect(1.0)
   counter2 += 1
   
plt.savefig('IR_H_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')							
plt.close('all')
del fig		



##==========================

epics = 200
root = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN'

fig, axs = plt.subplots(4, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
E = []; EE = []		
counter2=0  

for batch_size in [16,32,64,128]:
   if batch_size==16:
      letters='abcd'
   elif batch_size==32:
      letters='efgh'
   elif batch_size==64:
      letters='ijkl'		
   else:
      letters='mnop'
	  
   for counter1 in range(4):
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'T'+os.sep+'model'+str(counter1+1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1+1)+'_200epoch'+str(batch_size)+'batch_nearshore.mat')
      E.append(dat['yhat'])
      EE.append(dat['yhat_extreme'])	  
      axs[counter1, counter2].plot(dat['test_Y'], dat['yhat'], 'k.', label = 'estimated', markersize=1)
      axs[counter1, counter2].plot(dat['extreme_Y'], dat['yhat_extreme'], 'bo', label = 'estimated, out-of-calibration', markersize=1)
      axs[counter1, counter2].plot([7,24], [7,24], 'r--', label = 'observed', lw=0.5)
      string = r'RMS (s): '+str(np.squeeze(dat['rms']))[:4] + ' ('+str(np.squeeze(dat['rms_ex']))[:4] +')  R$^2$: '+str(np.squeeze(dat['rsq']))[:4]
      axs[counter1, counter2].set_title(letters[counter2]+') '+string, fontsize=3, loc='left')	 
      axs[counter1, counter2].set_xlim(7, 24)
      axs[counter1, counter2].set_ylim(7,24)		  
      if counter1==3:
         if counter2==0:
            axs[counter1, counter2].set_xlabel('Observed $T_p$ (s)', fontsize=5)
            axs[counter1, counter2].set_ylabel(r'Estimated $T_p$ (s)', fontsize=5)		  
      plt.setp(axs[counter1, counter2].get_xticklabels(), fontsize=4)
      plt.setp(axs[counter1, counter2].get_yticklabels(), fontsize=4)
      axs[counter1, counter2].set_aspect(1.0)
   counter2 += 1
   
plt.savefig('nearshore_T_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')							
plt.close('all')
del fig		



###============================



epics = 200
root = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN'

fig, axs = plt.subplots(4, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
E = []; EE = []		
counter2=0  

for batch_size in [16,32,64,128]:
   if batch_size==16:
      letters='abcd'
   elif batch_size==32:
      letters='efgh'
   elif batch_size==64:
      letters='ijkl'		
   else:
      letters='mnop'
	  
   for counter1 in range(4):
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'T'+os.sep+'model'+str(counter1+1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1+1)+'_200epoch'+str(batch_size)+'batch_IR.mat')
      E.append(dat['yhat'])
      EE.append(dat['yhat_extreme'])	  
      axs[counter1, counter2].plot(dat['test_Y'][::4], dat['yhat'][::4], 'k.', label = 'estimated', markersize=1)
      axs[counter1, counter2].plot(dat['extreme_Y'][::4], dat['yhat_extreme'][::4], 'bo', label = 'estimated, out-of-calibration', markersize=1)
      axs[counter1, counter2].plot([3,19], [3,19], 'r--', label = 'observed', lw=0.5)
      string = r'RMS (s): '+str(np.squeeze(dat['rms']))[:4] + ' ('+str(np.squeeze(dat['rms_ex']))[:4] +')  R$^2$: '+str(np.squeeze(dat['rsq']))[:4]
      axs[counter1, counter2].set_title(letters[counter1]+') '+string, fontsize=3, loc='left')	 
      axs[counter1, counter2].set_xlim(3, 19)
      axs[counter1, counter2].set_ylim(3,19)		  
      if counter1==3:
         if counter2==0:
            axs[counter1, counter2].set_xlabel('Observed T (s)', fontsize=5)
            axs[counter1, counter2].set_ylabel(r'Estimated T (s)', fontsize=5)		  
      plt.setp(axs[counter1, counter2].get_xticklabels(), fontsize=4)
      plt.setp(axs[counter1, counter2].get_yticklabels(), fontsize=4)
      axs[counter1, counter2].set_aspect(1.0)
   counter2 += 1
   
plt.savefig('IR_T_scatter_all_'+str(epics)+'.png', dpi=300, bbox_inches='tight')							
plt.close('all')
del fig		

