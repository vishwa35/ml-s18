#!/usr/bin/env python

import numpy as np
import scipy
from sklearn import svm, metrics, neighbors, tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
from time import time
from sklearn import preprocessing
from collections import Counter
import seaborn as sns
from sklearn.metrics import mean_squared_error


gunfile="Guns1.png"
loanfile="Loans2.png"


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=4):
	plt.figure()
	plt.title(title)
	if ylim is not None:
	    plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
	    estimator, X, y, cv=cv, n_jobs=n_jobs)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score")

	plt.legend(loc="best")
	return plt

class GunDeaths:
	def __init__(self):
		self.KNNclassifier = None
		self.DTclassifier = None
		self.DT2classifier = None
		self.SVMclassifier = None
		self.SVM2classifier = None
		self.NNclassifier = None
		self.BoostedDTclassifier = None

	def clfs(self):
		return (self.KNNclassifier, self.DTclassifier, self.SVMclassifier, self.SVM2classifier,
			self.NNclassifier, self.BoostedDTclassifier)

	def load_data(self, dir):
		df = pd.read_csv(dir, index_col=0)
		df, labels = self.preproccess(df)
		counter = Counter(np.ravel(labels))
		print counter
		sns.countplot(x='intent', data=labels)
		plt.title("Gun Deaths")
		plt.savefig("gundeaths.png")
		train, test, labels, answers = train_test_split(df, labels, stratify=labels.intent, test_size=0.3)
		return train, test, labels, answers

	def preproccess(self, df):
		df = df.loc[df['year'] == 2014]
		df = df.loc[df['month'] == 1]
		df = df.drop(['month', 'year'], axis=1)
		df['sex'] = (df['sex'] == 'M').astype(int)
		df = pd.get_dummies(df, columns=["race"])
		df = pd.get_dummies(df, columns=["place"])
		df = pd.get_dummies(df, columns=["education"])
		df.loc[df['hispanic'] <= 199, 'hispanic'] = "not-hispanic"
		df.loc[df['hispanic'] <= 209, 'hispanic'] = "spaniard"
		df.loc[df['hispanic'] <= 219, 'hispanic'] = "mexican"
		df.loc[df['hispanic'] <= 229, 'hispanic'] = "central-american"
		df.loc[df['hispanic'] <= 249, 'hispanic'] = "south-american"
		df.loc[df['hispanic'] <= 259, 'hispanic'] = "latin-american"
		df.loc[df['hispanic'] <= 269, 'hispanic'] = "puerto_rican"
		df.loc[df['hispanic'] <= 274, 'hispanic'] = "cuban"
		df.loc[df['hispanic'] <= 279, 'hispanic'] = "dominican"
		df.loc[df['hispanic'] <= 299, 'hispanic'] = "other-spanish-hispanic"
		df.loc[df['hispanic'] <= 999, 'hispanic'] = "unknown"
		df = pd.get_dummies(df, columns=["hispanic"])
		df.loc[df['intent'] == 'Homicide', 'intent'] = (1,0,0,0)
		df.loc[df['intent'] == 'Suicide', 'intent'] = (0,1,0,0)
		df.loc[df['intent'] == 'Homicide', 'intent'] = (0,0,1,0)
		df.loc[df['intent'] == 'Undetermined', 'intent'] = (0,0,0,1)
		df = df.dropna(thresh=len(df.columns))
		labels = df.filter(['intent'], axis=1)
		df = df.drop("intent", axis=1)

		return df, labels

	def train_classifiers(self, data, labels, cv=True, verbose=4):

		knn_params = {"n_neighbors": [3, 8],
							"weights": ["uniform", "distance"]}
		dt_params = {"criterion": ["gini", "entropy"],
									"max_depth": [2, 8]}
		svm_params = {"max_iter": sp_randint(250, 750)}
		svm2_params = {"decision_function_shape": ["ovo", "ovr"]}
		nn_params = {"solver": ['lbfgs', 'sgd', 'adam'],
									"learning_rate": ['constant', 'invscaling', 'adaptive'],
									"hidden_layer_sizes": [(5,2), (10,2), (10,5), (20,5)]}
		bdt_params = {"algorithm": ["SAMME", "SAMME.R"],
									"learning_rate": [0.5, 2],
									"n_estimators": [25, 60]}

		scaled = preprocessing.scale(data)
		y = labels = np.ravel(labels)

		# print ("training KNN...")
		# self.KNNclassifier = neighbors.KNeighborsClassifier()
		# if cv:
		# 	self.KNNclassifier = GridSearchCV(self.KNNclassifier, param_grid=knn_params, refit=True, verbose=verbose, n_jobs=5)
		# 	print "PARAMS: ", self.KNNclassifier.get_params
		# self.KNNclassifier.fit(data, labels)
		# if cv: print("KNN CV scores: ", cross_val_score(self.KNNclassifier, data, labels, cv=5))
		# plot_learning_curve(self.KNNclassifier, "Learning Curve: KNN Guns", data, y).savefig("KNN"+gunfile)

		# print ("training DT...")
		# self.DTclassifier = tree.DecisionTreeClassifier() # depth post-pruning
		# if cv:
		# 	self.DTclassifier = GridSearchCV(self.DTclassifier, param_grid=dt_params, refit=True, verbose=verbose, n_jobs=5)
		# 	print "PARAMS: ", self.DTclassifier.get_params
		# self.DTclassifier.fit(data, labels)
		# if cv: print("DT CV scores: ", cross_val_score(self.DTclassifier, data, y, cv=5))
		# plot_learning_curve(self.DTclassifier, "Learning Curve: DT Guns", data, y).savefig("DT"+gunfile)

		# self.DT2classifier = tree.DecisionTreeClassifier(max_depth=2) # depth post-pruning
		# self.DT2classifier.fit(data, labels)
		# plot_learning_curve(self.DT2classifier, "Learning Curve: DT Guns - pruned", data, y).savefig("DT2"+gunfile)


		# print ("training Linear SVM...")
		# self.SVMclassifier = svm.LinearSVC()
		# if cv:
		# 	# self.SVMclassifier = GridSearchCV(self.SVMclassifier, param_grid=svm_params, refit=True, verbose=verbose, n_jobs=3)
		# 	self.SVMclassifier = RandomizedSearchCV(self.SVMclassifier, param_distributions=svm_params, n_iter=4, refit=True, verbose=verbose, n_jobs=1)
		# 	print "PARAMS: ", self.SVMclassifier.get_params
		# self.SVMclassifier.fit(data, labels)
		# if cv: print("SVM CV scores: ", cross_val_score(self.SVMclassifier, data, y, cv=5))
		# plot_learning_curve(self.SVMclassifier, "Learning Curve: Linear SVM Guns", scaled, y, n_jobs=1).savefig("SVM"+gunfile)

		# print ("training SVM...")
		# self.SVMclassifier2 = svm.SVC()
		# if cv:
		# 	self.SVMclassifier2 = GridSearchCV(self.SVMclassifier2, param_grid=svm2_params, refit=True, verbose=verbose, n_jobs=3)
		# 	print "PARAMS: ", self.SVMclassifier2.get_params
		# self.SVMclassifier2.fit(data, y)
		# if cv: print("SVM2 CV scores: ", cross_val_score(self.SVMclassifier2, data, y, cv=5))
		# plot_learning_curve(self.SVMclassifier2, "Learning Curve: SVM Guns", scaled, y).savefig("SVM2"+gunfile)

		print ("training NN...")
		self.NNclassifier = MLPClassifier()
		if cv:
			self.NNclassifier = GridSearchCV(self.NNclassifier, param_grid=nn_params, refit=True, verbose=verbose, n_jobs=1)
			print "PARAMS: ", self.NNclassifier.get_params
		self.NNclassifier.fit(data, labels)
		# if cv: print("NN CV scores: ", cross_val_score(self.NNclassifier, data, labels, cv=5))
		# plot_learning_curve(self.NNclassifier, "Learning Curve: NN Guns", data, y, n_jobs=1).savefig("NN"+gunfile)

		# print ("training Boosted DT...")
		# self.BoostedDTclassifier = AdaBoostClassifier(tree.DecisionTreeClassifier())
		# if cv:
		# 	self.BoostedDTclassifier = GridSearchCV(self.BoostedDTclassifier, param_grid=bdt_params, refit=True, verbose=verbose, n_jobs=3)
		# 	print "PARAMS: ", self.BoostedDTclassifier.get_params
		# self.BoostedDTclassifier.fit(data, labels)
		# if cv: print("BoostedDT CV scores: ", cross_val_score(self.BoostedDTclassifier, data, labels, cv=5))
		# plot_learning_curve(self.BoostedDTclassifier, "Learning Curve: Boosted DT Guns", data, y).savefig("BDT"+gunfile)

	def predict(self, data):
		KNNlabels = []
		DTlabels = []
		DT2labels = []
		SVMlabels = []
		SVM2labels = []
		NNlabels = []
		BoostedDTlabels = []

		for d in data:
			# KNNlabels.append(self.KNNclassifier.predict(d.reshape(1, -1)))
			# DTlabels.append(self.DTclassifier.predict(d.reshape(1, -1)))
			# DT2labels.append(self.DT2classifier.predict(d.reshape(1, -1)))
			# SVMlabels.append(self.SVMclassifier.predict(d.reshape(1, -1)))
			# SVM2labels.append(self.SVMclassifier2.predict(d.reshape(1, -1)))
			NNlabels.append(self.NNclassifier.predict(d.reshape(1, -1)))
			# BoostedDTlabels.append(self.BoostedDTclassifier.predict(d.reshape(1, -1)))

		return (KNNlabels, DTlabels, SVMlabels, SVM2labels, NNlabels, BoostedDTlabels, DT2labels)

class Loans:
	def __init__(self):
		self.KNNclassifier = None
		self.DTclassifier = None
		self.DT2classifier = None
		self.SVMclassifier = None
		self.SVM2classifier = None
		self.NNclassifier = None
		self.BoostedDTclassifier = None

	def clfs(self):
		return (self.KNNclassifier, self.DTclassifier, self.SVMclassifier, self.SVM2classifier,
		self.NNclassifier, self.BoostedDTclassifier)

	def load_data(self, dir):
		df = pd.read_csv(dir)
		df, labels = self.preproccess(df)
		counter = Counter(np.ravel(labels))
		print counter
		sns.countplot(x='int_rate', data=labels)
		plt.title("Loan Interest Rates")
		plt.savefig("loans.png")
		train, test, labels, answers = train_test_split(df, labels, stratify=labels.int_rate, test_size=0.3)
		return train, test, labels, answers

	def preproccess(self, df):
		df = df.loc[df['issue_d'] == 'Jul-17']
		df = df.loc[df['loan_status'] == 'Current']
		df = df.loc[df['application_type'] == 'Individual']
		df = df[['funded_amnt','term', 'installment', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'int_rate']]
		df = df.dropna(thresh=len(df.columns))
		df = pd.get_dummies(df, columns=["verification_status"])
		df = pd.get_dummies(df, columns=["home_ownership"])
		df = pd.get_dummies(df, columns=["purpose"])
		df['emp_length'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
		df['term'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
		df['int_rate'].replace(regex=True,inplace=True,to_replace=r'[^\.0-9]',value=r'')

		df['int_rate'] = df['int_rate'].astype("double")
		df['term'] = df['int_rate'].astype("int")
		df['emp_length'] = df['int_rate'].astype("int")

		df.loc[df['int_rate'] <= 10.0, 'int_rate'] = "lte 10%"
		df.loc[df['int_rate'] <= 15.0, 'int_rate'] = "10-15%"
		df.loc[df['int_rate'] <= 20.0, 'int_rate'] = "15-20%"
		df.loc[df['int_rate'] <= 25.0, 'int_rate'] = "20-25%"
		df.loc[df['int_rate'] <= 31.0, 'int_rate'] = "gt 25%"

		df = df.dropna(thresh=len(df.columns))

		labels = df.filter(['int_rate'], axis=1)
		df = df.drop("int_rate", axis=1)

		return df, labels

	def train_classifiers(self, data, labels, cv=True, verbose=4):

		knn_params = {"n_neighbors": [3, 8],
							"weights": ["uniform", "distance"]}
		dt_params = {"criterion": ["gini", "entropy"],
									"max_depth": [2, 8]}
		svm_params = {"max_iter": sp_randint(250, 750)}
		svm2_params = {"decision_function_shape": ["ovo", "ovr"]}
		nn_params = {"solver": ['lbfgs', 'sgd', 'adam'],
									"learning_rate": ['constant', 'invscaling', 'adaptive'],
									"hidden_layer_sizes": [(5,2), (10,2), (10,5), (20,5)]}
		bdt_params = {"algorithm": ["SAMME", "SAMME.R"],
									"learning_rate": [0.5, 2],
									"n_estimators": [25, 60]}

		scaled = preprocessing.scale(data)
		y = labels = np.ravel(labels)

		print ("training KNN...")
		self.KNNclassifier = neighbors.KNeighborsClassifier()
		if cv:
			self.KNNclassifier = GridSearchCV(self.KNNclassifier, param_grid=knn_params, refit=True, verbose=verbose, n_jobs=5)
			print "PARAMS: ", self.KNNclassifier.get_params
		self.KNNclassifier.fit(data, labels)
		if cv: print("KNN CV scores: ", cross_val_score(self.KNNclassifier, data, labels, cv=5))
		plot_learning_curve(self.KNNclassifier, "Learning Curve: KNN Loans", data, y).savefig("KNN"+loanfile)

		print ("training DT...")
		self.DTclassifier = tree.DecisionTreeClassifier() # implement pruning
		if cv:
			self.DTclassifier = GridSearchCV(self.DTclassifier, param_grid=dt_params, refit=True, verbose=verbose, n_jobs=5)
			print "PARAMS: ", self.DTclassifier.get_params
		self.DTclassifier.fit(data, labels)
		if cv: print("DT CV scores: ", cross_val_score(self.DTclassifier, data, labels, cv=5))
		plot_learning_curve(self.DTclassifier, "Learning Curve: DT Loans", data, y).savefig("DT"+loanfile)

		self.DT2classifier = tree.DecisionTreeClassifier(max_depth=2) # implement pruning
		self.DT2classifier.fit(data, labels)
		plot_learning_curve(self.DT2classifier, "Learning Curve: DT Loans - pruned", data, y).savefig("DT2"+loanfile)


		print ("training Linear SVM...")
		self.SVMclassifier = svm.LinearSVC() # try another one also
		if cv:
			# self.SVMclassifier = GridSearchCV(self.SVMclassifier, param_grid=svm_params, refit=True, verbose=verbose, n_jobs=1)
			self.SVMclassifier = RandomizedSearchCV(self.SVMclassifier, param_distributions=svm_params, n_iter=5, refit=True, verbose=verbose, n_jobs=1)
			print "PARAMS: ", self.SVMclassifier.get_params
		self.SVMclassifier.fit(data, labels)
		if cv: print("SVM CV scores: ", cross_val_score(self.SVMclassifier, data, labels, cv=5))
		plot_learning_curve(self.SVMclassifier, "Learning Curve: Linear SVM Loans", scaled, y, n_jobs=1).savefig("SVM"+loanfile)

		print ("training SVM...")
		self.SVMclassifier2 = svm.SVC() # try another one also
		if cv:
			self.SVMclassifier2 = GridSearchCV(self.SVMclassifier2, param_grid=svm2_params, refit=True, verbose=verbose, n_jobs=3)
			print "PARAMS: ", self.SVMclassifier2.get_params
		self.SVMclassifier2.fit(data, labels)
		if cv: print("SVM2 CV scores: ", cross_val_score(self.SVMclassifier2, data, labels, cv=5))
		plot_learning_curve(self.SVMclassifier2, "Learning Curve: SVM Loans", scaled, y).savefig("SVM2"+loanfile)

		print ("training NN...")
		self.NNclassifier = MLPClassifier()
		if cv:
			self.NNclassifier = GridSearchCV(self.NNclassifier, param_grid=nn_params, refit=True, verbose=verbose, n_jobs=1)
			print "PARAMS: ", self.NNclassifier.get_params
		self.NNclassifier.fit(data, labels)
		if cv: print("NN CV scores: ", cross_val_score(self.NNclassifier, data, labels, cv=5))
		plot_learning_curve(self.NNclassifier, "Learning Curve: NN Loans", data, y, n_jobs=1).savefig("NN"+loanfile)

		print ("training Boosted DT...")
		self.BoostedDTclassifier = AdaBoostClassifier(tree.DecisionTreeClassifier())
		if cv:
			self.BoostedDTclassifier = GridSearchCV(self.BoostedDTclassifier, param_grid=bdt_params, refit=True, verbose=verbose, n_jobs=3)
			print "PARAMS: ", self.BoostedDTclassifier.get_params
		self.BoostedDTclassifier.fit(data, labels)
		if cv: print("BoostedDT CV scores: ", cross_val_score(self.BoostedDTclassifier, data, labels, cv=5))
		plot_learning_curve(self.BoostedDTclassifier, "Learning Curve: Boosted DT Loans", data, y).savefig("BDT"+loanfile)

	def predict(self, data):
		KNNlabels = []
		DTlabels = []
		DT2labels = []
		SVMlabels = []
		SVM2labels = []
		NNlabels = []
		BoostedDTlabels = []

		for d in data:
			KNNlabels.append(self.KNNclassifier.predict(d.reshape(1, -1)))
			DTlabels.append(self.DTclassifier.predict(d.reshape(1, -1)))
			DT2labels.append(self.DT2classifier.predict(d.reshape(1, -1)))
			SVMlabels.append(self.SVMclassifier.predict(d.reshape(1, -1)))
			SVM2labels.append(self.SVMclassifier2.predict(d.reshape(1, -1)))
			NNlabels.append(self.NNclassifier.predict(d.reshape(1, -1)))
			BoostedDTlabels.append(self.BoostedDTclassifier.predict(d.reshape(1, -1)))

		return (KNNlabels, DTlabels, SVMlabels, SVM2labels, NNlabels, BoostedDTlabels, DT2labels)


def stats(predicted_labels, labels, train = True, knn=True, dt=True, svm=True, nn=True, bdt=True):
	if train:
		print("\nTraining results")
	else:
		print("\nTesting results")
	print("=============================")
	if knn:
		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[0]))
		print("KNN Accuracy: ", metrics.accuracy_score(labels, predicted_labels[0]))
		print("KNN F1 score: ", metrics.f1_score(labels, predicted_labels[0], average='micro'))
	if dt:
		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[1]))
		print("DT Accuracy: ", metrics.accuracy_score(labels, predicted_labels[1]))
		print("DT F1 score: ", metrics.f1_score(labels, predicted_labels[1], average='micro'))

		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[6]))
		print("DT Accuracy: ", metrics.accuracy_score(labels, predicted_labels[6]))
		print("DT F1 score: ", metrics.f1_score(labels, predicted_labels[6], average='micro'))

	if svm:
		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[2]))
		print("SVM Accuracy: ", metrics.accuracy_score(labels, predicted_labels[2]))
		print("SVM F1 score: ", metrics.f1_score(labels, predicted_labels[2], average='micro'))

		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[3]))
		print("SVM2 Accuracy: ", metrics.accuracy_score(labels, predicted_labels[3]))
		print("SVM2 F1 score: ", metrics.f1_score(labels, predicted_labels[3], average='micro'))
	if nn:
		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[4]))
		print("NN Accuracy: ", metrics.accuracy_score(labels, predicted_labels[4]))
		print("NN F1 score: ", metrics.f1_score(labels, predicted_labels[4], average='micro'))
		print labels.unique()
		print("NN MSE: ", metrics.mean_squared_error(labels, predicted_labels[4]))
	if bdt:
		print("Confusion Matrix: \n",metrics.confusion_matrix(labels, predicted_labels[5]))
		print("Boosted DT Accuracy: ", metrics.accuracy_score(labels, predicted_labels[5]))
		print("Boosted DT F1 score: ", metrics.f1_score(labels, predicted_labels[5], average='micro'))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def main():
	# print 'x'
	print("GUN DEATHS")
	guns = GunDeaths()
	gunTrain, gunTest, gunLabels, gunTestLabels = guns.load_data("guns.csv")
	guns.train_classifiers(gunTrain, gunLabels, cv=False, verbose=0)

	predicted_labels = guns.predict(gunTrain.as_matrix())
	stats(predicted_labels, gunLabels, svm=False, knn=False, bdt=False, dt=False)

	predicted_labels = guns.predict(gunTest.as_matrix())
	stats(predicted_labels, gunTestLabels, train=False, svm=False, knn=False, bdt=False, dt=False)


	# print("LOAN INTEREST")
	# loans = Loans()
	# loanTrain, loanTest, loanLabels, loanTestLabels = loans.load_data("LoanStats_2017Q3.csv")
	# loans.train_classifiers(loanTrain, loanLabels, cv=False, verbose=0)

	# predicted_labels = loans.predict(loanTrain.as_matrix())
	# stats(predicted_labels, loanLabels)#, svm=False, nn=False, bdt=False)

	# predicted_labels = loans.predict(loanTest.as_matrix())
	# stats(predicted_labels, loanTestLabels, train=False)#, svm=False, nn=False, bdt=False)



if __name__ == "__main__":
    main()
