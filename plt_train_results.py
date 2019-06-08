import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ##use CPU
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import savemat, loadmat
from utils import *

##==========================

epics = 200
root = r'C:\Users\ddb265\github_clones\OpticalWaveGauging_DNN'

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for counter1 in range(1,5):
   if counter1==1:
      ax1=0; ax2=0
   elif counter1==2:
      ax1=0; ax2=1
   elif counter1==3:
      ax1=1; ax2=0
   elif counter1==4:
      ax1=1; ax2=1
	  
   for batch_size in [16,32,64,128]:
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'H'+os.sep+'model'+str(counter1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1)+'_200epoch'+str(batch_size)+'batch_nearshore.mat')
      y = np.squeeze(dat['history_val_mae'])
      #plt.plot(np.arange(len(y)), y, label = 'Model: '+str(counter1)+', batch size: '+str(batch_size))
      if batch_size==16:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'k',lw=0.5, label='Batch size: 16')
      elif batch_size==32:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'r', lw=0.5,label='Batch size: 32')
      elif batch_size==64:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'g', lw=0.5,label='Batch size: 64')
      elif batch_size==128:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'b', lw=0.5,label='Batch size: 128')		 

axs[0,0].set_title('A) MobileNetV1', fontsize=6, loc='left')
axs[0,0].set_xlim(0,200)
axs[0,0].set_ylim(-3,3)
axs[0,0].set_yticks([-2,0,2])
axs[0,0].set_yticklabels([0.25,1,4])
plt.setp(axs[0,0].get_xticklabels(), fontsize=5)
plt.setp(axs[0,0].get_yticklabels(), fontsize=5)

axs[0,1].set_title('B) MobileNetV2', fontsize=6, loc='left')
axs[0,1].set_xlim(0,200)
axs[0,1].set_ylim(-3,3)
axs[0,1].set_yticks([-2,0,2])
axs[0,1].set_yticklabels([0.25,1,4])
plt.setp(axs[0,1].get_xticklabels(), fontsize=5)
plt.setp(axs[0,1].get_yticklabels(), fontsize=5)

axs[1,0].set_title('C) InceptionV3', fontsize=6, loc='left')
axs[1,0].set_xlim(0,200)
axs[1,0].set_ylim(-3,3)
axs[1,0].set_yticks([-2,0,2])
axs[1,0].set_yticklabels([0.25,1,4])
plt.setp(axs[1,0].get_xticklabels(), fontsize=5)
plt.setp(axs[1,0].get_yticklabels(), fontsize=5)
axs[1,0].set_ylabel('Mean absolute error (m)', fontsize=6)
axs[1,0].set_xlabel(r'Training epoch', fontsize=6)	
			
axs[1,1].set_title('D) Inception-ResNetV2', fontsize=6, loc='left')
axs[1,1].set_xlim(0,200)
axs[1,1].set_ylim(-3,3)
axs[1,1].set_yticks([-2,0,2])
axs[1,1].set_yticklabels([0.25,1,4])
plt.setp(axs[1,1].get_xticklabels(), fontsize=5)
plt.setp(axs[1,1].get_yticklabels(), fontsize=5)
plt.legend(fontsize=6)

plt.savefig('nearshore_H_mae.png', dpi=300, bbox_inches='tight')
del fig
plt.close('all')



fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for counter1 in range(1,5):
   if counter1==1:
      ax1=0; ax2=0
   elif counter1==2:
      ax1=0; ax2=1
   elif counter1==3:
      ax1=1; ax2=0
   elif counter1==4:
      ax1=1; ax2=1
	  
   for batch_size in [16,32,64,128]:
      dat = loadmat(root+os.sep+'im128'+os.sep+'res'+os.sep+'200epoch'+os.sep+'H'+os.sep+'model'+str(counter1)+os.sep+'batch'+str(batch_size)+os.sep+'im128_model'+str(counter1)+'_200epoch'+str(batch_size)+'batch_IR.mat')
      y = np.squeeze(dat['history_val_mae'])
      #plt.plot(np.arange(len(y)), y, label = 'Model: '+str(counter1)+', batch size: '+str(batch_size))
      if batch_size==16:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'k', lw=0.5, label='Batch size: 16')
      elif batch_size==32:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'r', lw=0.5, label='Batch size: 32')
      elif batch_size==64:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'g', lw=0.5, label='Batch size: 64')
      elif batch_size==128:
         axs[ax1,ax2].plot(np.arange(len(y)), np.log(y), 'b', lw=0.5, label='Batch size: 128')		 

axs[0,0].set_title('A) MobileNetV1', fontsize=6, loc='left')
axs[0,0].set_xlim(0,200)
axs[0,0].set_ylim(-4,1)
axs[0,0].set_yticks([-4,-3,-2,0,1])
axs[0,0].set_yticklabels([0.063, 0.125,0.25,1,2])
plt.setp(axs[0,0].get_xticklabels(), fontsize=5)
plt.setp(axs[0,0].get_yticklabels(), fontsize=5)

axs[0,1].set_title('B) MobileNetV2', fontsize=6, loc='left')
axs[0,1].set_xlim(0,200)
axs[0,1].set_ylim(-4,1)
axs[0,1].set_yticks([-4,-3,-2,0,1])
axs[0,1].set_yticklabels([0.063, 0.125,0.25,1,2])
plt.setp(axs[0,1].get_xticklabels(), fontsize=5)
plt.setp(axs[0,1].get_yticklabels(), fontsize=5)

axs[1,0].set_title('C) InceptionV3', fontsize=6, loc='left')
axs[1,0].set_xlim(0,200)
axs[1,0].set_ylim(-4,1)
axs[1,0].set_yticks([-4,-3,-2,0,1])
axs[1,0].set_yticklabels([0.063, 0.125,0.25,1,2])
plt.setp(axs[1,0].get_xticklabels(), fontsize=5)
plt.setp(axs[1,0].get_yticklabels(), fontsize=5)
axs[1,0].set_ylabel('Mean absolute error (m)', fontsize=6)
axs[1,0].set_xlabel(r'Training epoch', fontsize=6)	
			
axs[1,1].set_title('D) Inception-ResNetV2', fontsize=6, loc='left')
axs[1,1].set_xlim(0,200)
axs[1,1].set_ylim(-4,1)
axs[1,1].set_yticks([-4,-3,-2,0,1])
axs[1,1].set_yticklabels([0.063, 0.125,0.25,1,2])
plt.setp(axs[1,1].get_xticklabels(), fontsize=5)
plt.setp(axs[1,1].get_yticklabels(), fontsize=5)
plt.legend(fontsize=6)

plt.savefig('IR_H_mae.png', dpi=300, bbox_inches='tight')
del fig
plt.close('all')
