import numpy as np
import ast
import scipy   
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing import image    
from tensorflow.python.keras.models import Model   
import sys, glob, os
import pandas as pd

def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)

def get_ResNet():
    # define ResNet50 model
    model = ResNet50(weights='imagenet')
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    ResNet_model = Model(inputs=model.input, 
        outputs=(model.layers[-4].output, model.layers[-1].output)) 
    return ResNet_model, all_amp_layer_weights
    
def ResNet_CAM(img_path, model, all_amp_layer_weights):
    # get filtered images from convolutional output + model prediction vector
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
    # change dimensions of last convolutional outpu tto 7 x 7 x 2048
    last_conv_output = np.squeeze(last_conv_output) 
    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec)
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224,224) # dim: 224 x 224
    # return class activation map
    return final_output, pred

if __name__ == '__main__':
	ResNet_model, all_amp_layer_weights = get_ResNet()

	base_dir = r'C:\Users\ddb265\github_clones\IRwaves\train'

	cat = 'H' #'H'

	df = pd.read_csv(os.path.join(base_dir, 'training-dataset.csv'))
																																							 
	df['path'] = df['id'].map(lambda x: os.path.join(base_dir,
															 'categorical',  
															 '{}.png'.format(x)))

	df['category'] = pd.cut(df['H'], 10)

	X = []
	for k in range(10):
		tmp = df.groupby('category').apply(lambda s: s.sample(1))
		for k in tmp['path']:
			X.append(k)

	levels = [0,.001,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

	origin='upper'
		
	if cat=='H':
			
		fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))

		counter=1
		for (idx, c_ax) in zip(np.arange(len(X)), m_axs.flatten()):
			print(counter)
			#img = image.load_img(X[idx], target_size=(224, 224))
			#img = image.img_to_array(img)
			c_ax.set_aspect('equal')
			#c_ax.imshow(img, cmap = 'gray', origin=origin)
			CAM, pred = ResNet_CAM(X[idx], ResNet_model, all_amp_layer_weights)
			c_ax.contourf(CAM/np.max(CAM), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
			#CS=plt.contour(CAM/np.max(CAM),levels, colors='k', linewidths=0.5, origin=origin)
			#plt.clabel(CS, inline=1, fontsize=4)
			c_ax.axis('off')		
			if counter==1:
			   c_ax.set_title('0.0 - 0.6 m', fontsize=10)
			if counter==2:
			   c_ax.set_title('0.6 - 1.2 m', fontsize=10)	
			if counter==3:
			   c_ax.set_title('1.2 - 1.8 m', fontsize=10)	   
			if counter==4:
			   c_ax.set_title('1.8 - 2.4 m', fontsize=10)	   
			if counter==5:
			   c_ax.set_title('2.4 - 3.0 m', fontsize=10)	   
			if counter==6:
			   c_ax.set_title('3.0 - 3.5 m', fontsize=10)
			if counter==7:
			   c_ax.set_title('3.5 - 4.2 m', fontsize=10)	
			if counter==8:
			   c_ax.set_title('4.2 - 4.8 m', fontsize=10)		   
			if counter==9:
			   c_ax.set_title('4.8 - 5.4 m', fontsize=10)	   
			if counter==10:
			   c_ax.set_title('5.4 - 6.0 m', fontsize=10)
			   
			counter += 1
			
		fig.savefig(cat+'_ex_per_cat_CAM.png', dpi = 600)	
		del fig
		plt.close('all')

	else:

		fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))

		counter=1
		for (idx, c_ax) in zip(np.arange(len(X)), m_axs.flatten()):
			print(counter)		
			#img = image.load_img(X[idx], target_size=(224, 224))
			#img = image.img_to_array(img)
			c_ax.set_aspect('equal')
			#c_ax.imshow(img, cmap = 'gray', origin=origin)
			CAM, pred = ResNet_CAM(X[idx], ResNet_model, all_amp_layer_weights)
			c_ax.contourf(CAM/np.max(CAM), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
			#CS=plt.contour(CAM/np.max(CAM),levels, colors='k', linewidths=0.5, origin=origin)
			#plt.clabel(CS, inline=1, fontsize=4)
			c_ax.axis('off')
			if counter==1:
			   c_ax.set_title('2.3 - 4.0 s', fontsize=10)
			if counter==2:
			   c_ax.set_title('4.0 - 5.7 s', fontsize=10)	
			if counter==3:
			   c_ax.set_title('5.7 - 7.4 s', fontsize=10)	   
			if counter==4:
			   c_ax.set_title('7.4 - 9.1 s', fontsize=10)	   
			if counter==5:
			   c_ax.set_title('9.1 - 10.8 s', fontsize=10)	   
			if counter==6:
			   c_ax.set_title('10.8 - 12.5 s', fontsize=10)
			if counter==7:
			   c_ax.set_title('12.5 - 14.2 s', fontsize=10)	
			if counter==8:
			   c_ax.set_title('14.2 - 15.9 s', fontsize=10)		   
			if counter==9:
			   c_ax.set_title('15.9 - 17.7 s', fontsize=10)	   
			if counter==10:
			   c_ax.set_title('17.7 - 19.3 s', fontsize=10)
			   
			counter += 1
			
		fig.savefig(cat+'_ex_per_cat_CAM.png', dpi = 600)	
		del fig
		plt.close('all')

	



		
    # use = 5	
    # images = glob.glob(r'C:\Users\ddb265\github_clones\IR_wavegauge\test\spill\*.jpg')[:use]
    # C1 = []; I1 = []
    # for img_path in images:
       # CAM, pred = ResNet_CAM(img_path, ResNet_model, all_amp_layer_weights)
       # C1.append(CAM)
       # I1.append(imread(img_path))	
	
    # images = glob.glob(r'C:\Users\ddb265\github_clones\IR_wavegauge\test\plunge\*.jpg')[:use]
    # C2 = []; I2 = []
    # for img_path in images:
       # CAM, pred = ResNet_CAM(img_path, ResNet_model, all_amp_layer_weights)
       # C2.append(CAM)
       # I2.append(imread(img_path))

    # images = glob.glob(r'C:\Users\ddb265\github_clones\IR_wavegauge\test\nonbreaking\*.jpg')[:use]
    # C3 = []; I3 = []
    # for img_path in images:
       # CAM, pred = ResNet_CAM(img_path, ResNet_model, all_amp_layer_weights)
       # C3.append(CAM)
       # I3.append(imread(img_path))

    # levels = [0,.001,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    # origin='upper'

    # ax=plt.subplot(251)
    # ax.set_aspect('equal')
    # plt.imshow(I1[0], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('A) Spilling 1', fontsize=6, loc='left')

    # ax=plt.subplot(252)
    # ax.set_aspect('equal')
    # plt.imshow(I1[1], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('B) Spilling 2', fontsize=6, loc='left')

    # ax=plt.subplot(253)
    # ax.set_aspect('equal')
    # plt.imshow(I1[2], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('C) Spilling 3', fontsize=6, loc='left')

    # ax=plt.subplot(254)
    # ax.set_aspect('equal')
    # plt.imshow(I1[3], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('D) Spilling 4', fontsize=6, loc='left')

    # ax=plt.subplot(255)
    # ax.set_aspect('equal')
    # plt.imshow(I1[4], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('E) Spilling 5', fontsize=6, loc='left')

    # ax=plt.subplot(256)
    # ax.set_aspect('equal')
    # plt.contourf(C1[0]/np.max(C1[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C1[0]/np.max(C1[0]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(257)
    # ax.set_aspect('equal')
    # plt.contourf(C1[1]/np.max(C1[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C1[1]/np.max(C1[1]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(258)
    # ax.set_aspect('equal')
    # plt.contourf(C1[2]/np.max(C1[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C1[2]/np.max(C1[2]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(259)
    # ax.set_aspect('equal')
    # plt.contourf(C1[3]/np.max(C1[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C1[3]/np.max(C1[3]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(2,5,10)
    # ax.set_aspect('equal')
    # plt.contourf(C1[4]/np.max(C1[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C1[4]/np.max(C1[4]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # #plt.tight_layout()
    # plt.savefig('av_actmap_spill.png', bbox_inches='tight', dpi=300)
    # plt.close()   	


    # plt.close('all')

    # ax=plt.subplot(251)
    # ax.set_aspect('equal')
    # plt.imshow(I2[0], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('F) Plunging 1', fontsize=6, loc='left')

    # ax=plt.subplot(252)
    # ax.set_aspect('equal')
    # plt.imshow(I2[1], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('G) Plunging 2', fontsize=6, loc='left')

    # ax=plt.subplot(253)
    # ax.set_aspect('equal')
    # plt.imshow(I2[2], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('H) Plunging 3', fontsize=6, loc='left')

    # ax=plt.subplot(254)
    # ax.set_aspect('equal')
    # plt.imshow(I2[3], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('I) Plunging 4', fontsize=6, loc='left')

    # ax=plt.subplot(255)
    # ax.set_aspect('equal')
    # plt.imshow(I2[4], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('J) Plunging 5', fontsize=6, loc='left')

    # ax=plt.subplot(256)
    # ax.set_aspect('equal')
    # plt.contourf(C2[0]/np.max(C2[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C2[0]/np.max(C2[0]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(257)
    # ax.set_aspect('equal')
    # plt.contourf(C2[1]/np.max(C2[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C2[1]/np.max(C2[1]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(258)
    # ax.set_aspect('equal')
    # plt.contourf(C2[2]/np.max(C2[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C2[2]/np.max(C2[2]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(259)
    # ax.set_aspect('equal')
    # plt.contourf(C2[3]/np.max(C2[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C2[3]/np.max(C2[3]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(2,5,10)
    # ax.set_aspect('equal')
    # plt.contourf(C2[4]/np.max(C2[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C2[4]/np.max(C2[4]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # #plt.tight_layout()
    # plt.savefig('av_actmap_plunge.png', bbox_inches='tight', dpi=300)
    # plt.close()   


    # plt.close('all')

    # ax=plt.subplot(251)
    # ax.set_aspect('equal')
    # plt.imshow(I3[0], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('K) Unbroken 1', fontsize=6, loc='left')

    # ax=plt.subplot(252)
    # ax.set_aspect('equal')
    # plt.imshow(I3[1], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('L) Unbroken 2', fontsize=6, loc='left')

    # ax=plt.subplot(253)
    # ax.set_aspect('equal')
    # plt.imshow(I3[2], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('M) Unbroken 3', fontsize=6, loc='left')

    # ax=plt.subplot(254)
    # ax.set_aspect('equal')
    # plt.imshow(I3[3], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('N) Unbroken 4', fontsize=6, loc='left')

    # ax=plt.subplot(255)
    # ax.set_aspect('equal')
    # plt.imshow(I3[4], cmap='gray', origin=origin)
    # plt.axis('off')
    # plt.title('O) Unbroken 5', fontsize=6, loc='left')

    # ax=plt.subplot(256)
    # ax.set_aspect('equal')
    # plt.contourf(C3[0]/np.max(C3[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C3[0]/np.max(C3[0]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(257)
    # ax.set_aspect('equal')
    # plt.contourf(C3[1]/np.max(C3[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C3[1]/np.max(C3[1]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(258)
    # ax.set_aspect('equal')
    # plt.contourf(C3[2]/np.max(C3[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C3[2]/np.max(C3[2]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(259)
    # ax.set_aspect('equal')
    # plt.contourf(C3[3]/np.max(C3[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C3[3]/np.max(C3[3]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # ax=plt.subplot(2,5,10)
    # ax.set_aspect('equal')
    # plt.contourf(C3[4]/np.max(C3[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    # CS=plt.contour(C3[4]/np.max(C3[4]),levels, colors='k', linewidths=0.5, origin=origin)
    # plt.clabel(CS, inline=1, fontsize=4)
    # plt.axis('off')

    # #plt.tight_layout()
    # plt.savefig('av_actmap_unbroken.png', bbox_inches='tight', dpi=300)
    # plt.close()   






