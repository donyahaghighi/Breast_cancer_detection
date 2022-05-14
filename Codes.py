

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab 
from random import shuffle
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
import os
import csv
from sklearn.metrics import precision_recall_curve
import warnings

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# from mlxtend.preprocessing import shuffle_arrays_unison

def loadfile(file):
	df = pd.read_csv(file)
	cols = df.columns.values
	data = df.values
	return cols, data

	# plot histogram - to save code, each graph contains 8 subplot
	# and repeated 4 times to generate all feature graphs
def plotFeatures1(cols, data):
	gridsize = (6, 2)
	fig = plt.figure(figsize=(8, 20))
	fig.subplots_adjust(hspace=0.6)
	ax1 = plt.subplot2grid(gridsize, (1, 0))
	ax2 = plt.subplot2grid(gridsize, (1, 1))
	ax3 = plt.subplot2grid(gridsize, (2, 0))
	ax4 = plt.subplot2grid(gridsize, (2, 1))
	ax5 = plt.subplot2grid(gridsize, (3, 0))
	ax6 = plt.subplot2grid(gridsize, (3, 1))
	ax7 = plt.subplot2grid(gridsize, (4, 0))
	ax8 = plt.subplot2grid(gridsize, (4, 1))		

	ax1.set_title(cols[1])
	ax1.hist(sorted(data[:, 1]), bins=20)
	ax2.set_title(cols[2])
	ax2.hist(sorted(data[:, 2]), bins=20)
	ax3.set_title(cols[3])
	ax3.hist(sorted(data[:, 3]), bins=20)
	ax4.set_title(cols[4])
	ax4.hist(sorted(data[:, 4]), bins=20)
	ax5.set_title(cols[5])
	ax5.hist(sorted(data[:, 5]), bins=20)
	ax6.set_title(cols[6])
	ax6.hist(sorted(data[:, 6]), bins=20)
	ax7.set_title(cols[7])
	ax7.hist(sorted(data[:, 7]), bins=20)
	ax8.set_title(cols[8])
	ax8.hist(sorted(data[:, 8]), bins=20)
	pylab.savefig('Chart1.png', bbox_inches='tight')

	plt.clf()

	# plot some boxcharts to see the variance, but they were not used for the report
def plotFeatures2(cols, data):
	df = pd.DataFrame(data, columns=cols)
	df_num = df.convert_objects(convert_numeric=True)
	plt.figure(figsize=(8, 15))
	f, axes = plt.subplots(2, 1)
	ax1 = sns.boxplot( data=df_num[['radius_mean', 'texture_mean', 'perimeter_mean', 'texture_worst', 'perimeter_worst']],  orient='v', ax=axes[0])
	ax2 = sns.boxplot( data=df_num[['smoothness_mean', 'compactness_mean', 'fractal_dimension_mean', 'smoothness_se', 'smoothness_worst']],  orient='v', ax=axes[1])
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")

	plt.tight_layout()
	pylab.savefig('Boxchart.png', bbox_inches='tight')

	# calculate pca
def pcaFeatures(data):
	pca = PCA(n_components=3)
	pcaData = pca.fit_transform(data)
	print('Shape of PCA data: ', pcaData.shape)
	return pcaData		

	# plot PCA 3D chart
def plotPCA(labels, data):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = data[:, 0]
	ys = data[:, 1]
	zs = data[:, 2]

	ax.scatter(xs, ys, zs, c=labels)
	ax.set_title('PCA Class Distribution')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	ax.set_zlabel('PCA 3')
	pylab.savefig('PCA_Class_Distribution.png', bbox_inches='tight')

	# plt.show()

	# plot correlation heatmap
def plotCorr(cols, data):
	df = pd.DataFrame(data, columns=cols)
	dff = df.drop(['id', 'diagnosis'], axis=1)
	dff =dff.convert_objects(convert_numeric=True)
	corr = dff.corr(method='pearson')
	# hide the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
	            square=True, linewidths=.5, cbar_kws={"shrink": .5})
	plt.savefig('Correlation_Heatmap.png', bbox_inche='tight')
	# plt.show()

	# process data
def prepData(data):
	m, n = data.shape

	# normalize data
	id_class = data[:, 0:2]
	features = data[:, 2:]
	dataNorm = (features - np.mean(features, axis=0))/(np.max(features,axis=0)-np.min(features, axis=0))
	print('Shape of normalized feature data: ', dataNorm.shape)
	# print(dataNorm[0])
	# find pca
	pcaData = pcaFeatures(dataNorm)
	# assign 1 for Malignant and 0 for Beneighn
	labTemp = []
	for i in id_class[:, 1]:
		if i=='M':
			labTemp.append(1)
		else:
			labTemp.append(-1)
	# plot pca fetures in 3D
	plotPCA(labTemp, pcaData)

	# shuffle data
	tempID = id_class[:, 0].reshape(m, 1)
	targets = np.array(labTemp).reshape(m, 1)
	idTarget = np.append(tempID, targets, axis=1)

	ind_list = [i for i in range(m)]
	shuffle(ind_list)
	pcaShuffled = pcaData[ind_list]
	idTargetShuffled = idTarget[ind_list]

	return idTargetShuffled, pcaShuffled

	# train, test and evaluate
def train(id_class, pcaData, fold, algorithm, features):
	X = pcaData.astype(float)
	y = id_class[:, 1].astype(float)
	X_train, X_test, y_train, y_test = train_test_split(pcaData, y, test_size=0.20, random_state=42)

	# conduct cross validation becaues of the small sample size
	kf = KFold(n_splits=fold, random_state=None, shuffle=False)
	# initialize models
	RFC_clf = RandomForestClassifier(n_estimators=10)
	LG_clf = LogisticRegression()
	SVC_clf = svm.SVC(probability=True)
	MLPC_clf =  MLPClassifier()

	# RFC
	if algorithm == 'RFC':
		warnings.filterwarnings("ignore")		
		model = RFC_clf.fit(X_train, y_train)
		# without cross-valdidation
		predict_RFC = model.predict(X_test)
		acc = accuracy_score(predict_RFC, y_test)
		y_prob = model.predict_proba(X_test)[:, 1]
		evalRFC, roc_values_RFC = evaluate(y_test, predict_RFC, y_prob, algorithm)
		# importances = feat_importance(model, features[2:])
		# featuresSelected = ['diagnosis', 'radius_mean', 'texture__mean']
		# createTree(model, featuresSelected)
		eval_rates = evalRate(y_test, predict_RFC)
		print('Confusion Matrix - '+algorithm, evalRFC)
		print('RFC - Recall, Precision, F1: ', eval_rates)

		# with cross-validation
		acc_cv = cross_val_score(RFC_clf, X, y, scoring='accuracy', cv=kf).mean()
		accuracies = [acc, acc_cv]
		return accuracies

	# LG
	if algorithm == "LG":
		model = LG_clf.fit(X_train, y_train)
		predict_LG = model.predict(X_test)
		acc_LG = accuracy_score(predict_LG, y_test)
		y_prob_LG = model.predict_proba(X_test)[:, 1]

		# with cross-validation and grid search of hyperparameters
		tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
		LG= GridSearchCV(LG_clf, tuned_parameters, cv=fold)
		model_LG = LG.fit(X_train, y_train)
		print('Logistic Regression Best Parameters: ',  LG.best_params_)
		y_prob_cv_LG = model_LG.predict_proba(X_test)[:, 1]
		y_pred_cv_LG= np.where(y_prob_cv_LG > 0.4, 1, -1)
		# evaluations
		acc_cv_LG = model_LG.best_score_
		# generate confusion matrix and ROC curve
		evalLG, roc_values_LG = evaluate(y_test, y_pred_cv_LG, y_prob_cv_LG, algorithm)
		print('FPR, TPR & Thresholds - '+algorithm, roc_values_LG)
		eval_rates_LG = evalRate(y_test, y_pred_cv_LG)
		analyze(y_test, y_prob_cv_LG, algorithm)
		print('Confusion Matrix - '+algorithm, evalLG)
		print('LG - Recall, Precision, F1: ', eval_rates_LG)
		accuracies = [acc_LG, acc_cv_LG]
		return accuracies

	# SVM 
	if algorithm =='SVM':
		model = SVC_clf.fit(X_train, y_train)
		predict_svm = model.predict(X_test)
		acc_SVM = accuracy_score(predict_svm, y_test)

		# with cross-validation and grid search of hyperparameters
		tuned_parameters = {'gamma':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}
		SVC = GridSearchCV(SVC_clf, tuned_parameters, cv=fold)
		model_SVM = SVC.fit(X_train, y_train)
		print('SVM Best Parameters: ',  SVC.best_params_)
		y_prob_cv_SVM = model_SVM.predict_proba(X_test)[:, 1]
		y_pred_cv_SVM= np.where(y_prob_cv_SVM > 0.4, 1, -1)
		# evaluation
		acc_cv_SVM = model_SVM.best_score_
		# generate confusion matrix and ROC curve
		evalSVM, roc_values_SVM = evaluate(y_test, y_pred_cv_SVM, y_prob_cv_SVM, algorithm)
		print('FPR, TPR & Thresholds - '+algorithm, roc_values_SVM)
		eval_rates_SVM = evalRate(y_test, y_pred_cv_SVM)
		analyze(y_test, y_prob_cv_SVM, algorithm)
		print('Best parameters for SVM: ', model_SVM.best_params_)
		print('SVM - Recall, Precision, F1: ', eval_rates_SVM)
		print('Confusion Matrix - '+algorithm, evalSVM)
		accuracies = [acc_SVM, acc_cv_SVM]
		return accuracies

	# Multi-layer perceptron classifier
	if algorithm == "MLPC":
		model = MLPC_clf.fit(X_train, y_train)
		warnings.filterwarnings("ignore")
		predict_MLPC = model.predict(X_test)
		acc_MLPC = accuracy_score(predict_MLPC, y_test)
		y_prob_MLPC = model.predict_proba(X_test)[:, 1]

		# with cross-validation and grid search of hyperparameters
		parameters = {'solver': ['lbfgs'], 'max_iter': [100, 500, 1000], 'alpha': 10.0 ** -np.arange(1, 3), 'hidden_layer_sizes': np.arange(5, 12), 'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
		MLPC_clf_grid = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv=fold)
		warnings.filterwarnings("ignore", category=DeprecationWarning) 
		model_MLPC = MLPC_clf_grid.fit(X_train, y_train)
		print("MPLC Best Parameters: ", MLPC_clf_grid.best_params_)
		acc_cv_MLPC = model_MLPC.best_score_
		y_prob_cv_MLPC = model_MLPC.predict_proba(X_test)[:, 1]
		y_pred_cv_MLPC= np.where(y_prob_cv_MLPC > 0.4, 1, -1)
		evalMLPC, roc_values_MLPC =evaluate(y_test, y_pred_cv_MLPC, y_prob_cv_MLPC, algorithm)		
		print('FPR, TPR & Thresholds - '+algorithm, roc_values_MLPC)
		eval_rates_MLPC= evalRate(y_test, y_pred_cv_MLPC)
		analyze(y_test, y_prob_cv_MLPC, algorithm)
		print('MLPC - Recall, Precision, F1: ', eval_rates_MLPC)
		# print('Confusion Matrix - '+algorithm, evalMLPC)
		accuracies = [acc_MLPC, acc_cv_MLPC]
		return accuracies	

	# generate confusion matrix and plot ROC curve
def evaluate(y_test, y_pred, y_prob, algorithm):
	# calculate confusion matrix for each algorthm
	confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
	# print('Confusion Matrix - '+algorithm, confusion_matrix)

	# generate ROC curve for each algorithm
	auc_roc = metrics.roc_auc_score(y_test, y_pred)
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
	roc_values = [false_positive_rate, true_positive_rate, thresholds]
	# generate ROC AUC
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.figure(figsize=(10, 10))
	plt.title('Receiver Operating Characteristic'+' - '+algorithm)
	plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.axis('tight')
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	pylab.savefig(algorithm+' - ROC_Curve.png', bbox_inches='tight')
	# plt.show()
	return confusion_matrix, roc_values

	# calculate recall and precision rate
def evalRate(y_true, y_predict):	
	precision = metrics.precision_score(y_true,  y_predict)
	recall = metrics.recall_score(y_true, y_predict)
	f1_score = 2*precision*recall/(precision+recall)
	return [recall, precision, f1_score]

	# export roc value to csv file to check the best threshold against sklearn threshold
def analyze(y_test, y_prob, algorithm):
	y_test = pd.DataFrame(y_test)
	y_prob = pd.DataFrame(y_prob)
	roc_values = pd.concat([y_test, y_prob], axis=1, sort=False)
	roc_values.columns = ['y_true', 'y_prob']
	roc_values.to_csv('roc_values - '+algorithm+'.csv')


if __name__ == '__main__':
	# load data
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
	filename = 'Breast_Cancer_data.csv'
	cols, data = loadfile(filename)
	print('Shape of data: ', data.shape)
	print(cols)

	# plot each columns
	plotFeatures2(cols, data)
	plotFeatures1(cols, data)
	plotCorr(cols, data)
	

	# normalize and PCA transform data
	id_class, pcaData = prepData(data)

	# train, test and evaluate data
	models =['RFC', 'LG', 'SVM', 'MLPC']
	# results are stored in the 'results' dictionary
	results = {'RFC': {'Accuracy': None, 'Accuracy_cv': None}, 'LG': {'Accuracy': None, 'Accuracy_cv': None}, 'SVM': {'Accuracy': None, 'Accuracy_cv': None}, 'MLPC': {'Accuracy': None, 'Accuracy_cv': None}}
	for model in models:
		acc = train(id_class, pcaData, 10, model, cols)  # cv is set to 10
		results[model]['Accuracy'] = acc[0]
		results[model]['Accuracy_cv'] = acc[1]
	print(results)

	

