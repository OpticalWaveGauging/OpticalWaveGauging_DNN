# organize imports
from __future__ import print_function

#https://gogul09.github.io/software/flower-recognition-deep-learning
#https://keras.io/applications/

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB  ##DB
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os, sys, getopt
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#==============================================================
if __name__ == '__main__':
	argv = sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv,"h:c:")
	except getopt.GetoptError:
		print('python train_categorical.py -c conf_file')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print('Example usage: python extract_features_imaug.py -c conf_mobilenet')
			sys.exit()
		elif opt in ("-c"):
			configfile = arg

	# load the user configs
	with open(os.getcwd()+os.sep+'conf'+os.sep+configfile+'.json') as f:    
	  config = json.load(f)

	# config variables
	test_size     = config["test_size"]
	seed      = config["seed"]
	features_path   = config["features_path"]
	labels_path   = config["labels_path"]
	results     = config["results"]
	model_path = config["model_path"]
	train_path    = config["train_path"]
	num_classes   = config["num_classes"]
	classifier_path = config["classifier_path"]
	cm_path = config["cm_path"]

	# import features and labels
	h5f_data  = h5py.File(features_path, 'r')
	h5f_label = h5py.File(labels_path, 'r')

	features_string = h5f_data['dataset_1']
	labels_string   = h5f_label['dataset_1']

	features = np.array(features_string)
	labels   = np.array(labels_string)

	h5f_data.close()
	h5f_label.close()

	# verify the shape of features and labels
	print ("[INFO] features shape: {}".format(features.shape))
	print ("[INFO] labels shape: {}".format(labels.shape))

	print ("[INFO] training started...")
	# split the training and testing data
	(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
																	  np.array(labels),
																	  test_size=test_size,
																	  random_state=seed)

	print ("[INFO] splitted train and test data...")
	print ("[INFO] train data  : {}".format(trainData.shape))
	print ("[INFO] test data   : {}".format(testData.shape))
	print ("[INFO] train labels: {}".format(trainLabels.shape))
	print ("[INFO] test labels : {}".format(testLabels.shape))

	# use logistic regression as the model
	print ("[INFO] creating model...")
	#model = LogisticRegression(random_state=seed)
	model = LogisticRegression(C=0.5, dual=True, random_state=seed)
	model.fit(trainData, trainLabels)
	
	#model = GaussianNB()
	#model.fit(trainData, trainLabels)
	
	

# from sklearn import (manifold, datasets, decomposition, ensemble,
                     # discriminant_analysis, random_projection)


# n_samples, n_features = trainData.shape
# n_neighbors = 10

# y = trainLabels.copy()

# #----------------------------------------------------------------------
# # Scale and visualize the embedding vectors
# def plot_embedding(X, title=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    # #plt.figure()
    # #ax = plt.subplot(111)
    # col='rgb'
    # for i in range(X.shape[0]):
        # plt.text(X[i, 0], X[i, 1], str(y[i]),
                 # color=col[int(y[i])],
                 # fontdict={'weight': 'bold', 'size': 9})

    # plt.xticks([]), plt.yticks([])
    # if title is not None:
        # plt.title(title, fontsize=6)

# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(trainData)
# plot_embedding(X_pca,"Principal Components projection of the curves")
	
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(trainData)
# plot_embedding(X_iso,"Isomap projection of the curves")	
	
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
# X_ltsa = clf.fit_transform(trainData)
# plot_embedding(X_ltsa,"Local Tangent Space Alignment of the curves")	
	
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(trainData)
# plot_embedding(X_tsne, "t-SNE embedding of the curves")
			   
			   
	# use rank-1 and rank-5 predictions
	print ("[INFO] evaluating model...")
	f = open(results, "w")
	rank_1 = 0
	rank_5 = 0

	# loop over test data
	for (label, features) in zip(testLabels, testData):
		# predict the probability of each class label and
		# take the top-5 class labels
		predictions = model.predict_proba(np.atleast_2d(features))[0]
		predictions = np.argsort(predictions)[::-1][:5]

		# rank-1 prediction increment
		if label == predictions[0]:
			rank_1 += 1

		# rank-5 prediction increment
		if label in predictions:
			rank_5 += 1

	# convert accuracies to percentages
	rank_1 = (rank_1 / float(len(testLabels))) * 100
	rank_5 = (rank_5 / float(len(testLabels))) * 100

	# write the accuracies to file
	f.write("Rank-1: {:.2f}%\n".format(rank_1))
	f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

	# evaluate the model of test data
	preds = model.predict(testData)

	# write the classification report to file
	f.write("{}\n".format(classification_report(testLabels, preds)))
	f.close()

	# dump classifier to file
	print ("[INFO] saving model...")
	pickle.dump(model, open(classifier_path, 'wb'))

	# display the confusion matrix
	print ("[INFO] confusion matrix")

	# get the list of training lables
	labels = sorted(list(os.listdir(train_path)))
	labels =[t for t in labels if not t.endswith('csv')][1:]

	# plot the confusion matrix
	cm = confusion_matrix(testLabels, preds)

	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	sns.heatmap(cm,
				annot=True,
				cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)) #cmap="cubehelix") #"Set2")
				
	tick_marks = np.arange(len(labels))+.5
	plt.xticks(tick_marks, labels, rotation=45,fontsize=5)
	plt.yticks(tick_marks, labels,rotation=45, fontsize=5)		
	
# F = []	
# for k in np.where(trainLabels==0)[0]:	
   # f = trainData[k].reshape(49,1280)	
   # f = np.mean(f, axis=0) 	
   # F.append(f[:1225].reshape(35,35)) 	
	
# plt.subplot(131)	
# plt.imshow(np.mean(np.dstack(F), axis=2), cmap='gray')	
	
# F = []	
# for k in np.where(trainLabels==1)[0]:	
   # f = trainData[k].reshape(49,1280)	
   # f = np.mean(f, axis=0) 	
   # F.append(f[:1225].reshape(35,35)) 	
	
# plt.subplot(132)	
# plt.imshow(np.mean(np.dstack(F), axis=2), cmap='gray')	
		
	
# F = []	
# for k in np.where(trainLabels==2)[0]:	
   # f = trainData[k].reshape(49,1280)	
   # f = np.mean(f, axis=0) 	
   # F.append(f[:1225].reshape(35,35)) 	
	
# plt.subplot(133)	
# plt.imshow(np.mean(np.dstack(F), axis=2), cmap='gray')	
			
# plt.show()