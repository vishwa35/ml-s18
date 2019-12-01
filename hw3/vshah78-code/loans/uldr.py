# source: https://github.com/joshuamorton/Machine-Learning

import numpy as np
from StringIO import StringIO
from pprint import pprint
import argparse
from matplotlib import pyplot as plt
from collections import Counter


from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GMM as EM
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2
from sklearn.preprocessing import scale


from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, learning_curve

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

CHECKER_CLUSTERS = 5

def create_dataset(name):
    if name == 'guns':
        dir = "guns.csv"
        CHECKER_CLUSTERS = 4
    elif name == 'loans':
        dir = "LoanStats_2017Q3.csv"
        CHECKER_CLUSTERS = 3 # 5 og. talk about this!
    else:
        print "invalid name"
        exit()
    df = pd.read_csv(dir, index_col=0)
    df, labels = preproccess(df)
    counter = Counter(np.ravel(labels))
    print counter
    # # sns.countplot(x='intent', data=labels)
    # plt.title("Gun Deaths")
    # plt.savefig("gundeaths.png")
    # print df.as_matrix().shape
    train, test, labels, answers = train_test_split(df, labels, stratify=labels.int_rate, test_size=0.3)
    return scale(train.as_matrix()), labels.as_matrix(), scale(test.as_matrix()), answers.as_matrix()

def preproccess(df):
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
    df.loc[df['int_rate'] == 'lte 10%', 'int_rate'] = 0
    df.loc[df['int_rate'] == '10-15%', 'int_rate'] = 1
    df.loc[df['int_rate'] == '15-20%', 'int_rate'] = 2
    df.loc[df['int_rate'] == '20-25%', 'int_rate'] = 3
    df.loc[df['int_rate'] == 'gt 25%', 'int_rate'] = 4

    df = df.dropna(thresh=len(df.columns))

    labels = df.filter(['int_rate'], axis=1)
    df = df.drop("int_rate", axis=1)

    return df, labels


def plot(axes, values, x_label, y_label, title, name):
    plt.clf()
    plt.plot(*values, markersize=2)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    # plt.show()
    plt.clf()


def pca(tx, ty, rx, ry):
    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    km(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wPCAtr")


def ica(tx, ty, rx, ry):
    compressor = ICA(whiten=False)  # for some people, whiten needs to be off
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wICAtr", times=10)
    km(newtx, ty, newrx, ry, add="wICAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wICAtr")


def randproj(tx, ty, rx, ry):
    compressor = RandomProjection(tx[1].size)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    # compressor = RandomProjection(tx[1].size)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wRPtr", times=10)
    km(newtx, ty, newrx, ry, add="wRPtr", times=10)
    nn(newtx, ty, newrx, ry, add="wRPtr")


def kbest(tx, ty, rx, ry):
    compressor = best(chi2)
    tx = [[abs(y) for y in x] for x in tx]
    rx = [[abs(y) for y in x] for x in rx]
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wKBtr", times=10)
    km(newtx, ty, newrx, ry, add="wKBtr", times=10)
    nn(newtx, ty, newrx, ry, add="wKBtr")


def em(tx, ty, rx, ry, add="", times=5):
    errs = []

    # this is what we will compare to
    checker = EM(n_components=CHECKER_CLUSTERS)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}

        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set

        # here we make the arguably awful assumption that for a given cluster,
        # all values in tha cluster "should" in a perfect world, belong in one
        # class or the other, meaning that say, cluster "3" should really be
        # all 0s in our truth, or all 1s there
        #
        # So clusters is a dict of lists, where each list contains all items
        # in a single cluster
        for index, val in enumerate(result):
            clusters[val].append(index)

        # then we take each cluster, find the sum of that clusters counterparts
        # in our "truth" and round that to find out if that cluster should be
        # a 1 or a 0
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}

        # the processed list holds the results of this, so if cluster 3 was
        # found to be of value 1,
        # for each value in clusters[3], processed[value] == 1 would hold
        processed = [mapper[val] for val in result]
        errs.append(sum((processed != truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "Expectation Maximization Error", "EM"+add)

    # dank magic, wrap an array cuz reasons
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onEM"+add)



def km(tx, ty, rx, ry, add="", times=5):
    #this does the exact same thing as the above
    errs = []

    checker = KM(n_clusters=CHECKER_CLUSTERS)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}
        clf = KM(n_clusters=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set
        for index, val in enumerate(result):
            clusters[val].append(index)
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
        processed = [mapper[val] for val in result]
        errs.append(sum((processed != truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "KMeans clustering error", "KM"+add)

    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onKM"+add)


def nn(tx, ty, rx, ry, add="", iterations=250):
    """
    trains and plots a neural network on the data we have
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(tx[1].size, 5, 1, bias=True)
    ds = ClassificationDataSet(tx[1].size, 1)
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.1)
    train = zip(tx, ty)
    test = zip(rx, ry)
    for i in positions:
        trainer.train()
        resultst.append(sum(np.array([(round(network.activate(t_x)) != t_y)**2 for t_x, t_y in train])/float(len(train))))
        resultsr.append(sum(np.array([(round(network.activate(t_x)) != t_y)**2 for t_x, t_y in test])/float(len(test))))
        # resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i, resultst[-1], resultsr[-1]
    plot([0, iterations, 0, 1], (positions, resultst, "ro", positions, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithms on stuff')
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name
    train_x, train_y, test_x, test_y = create_dataset(name)
    nn(train_x, train_y, test_x, test_y)
    print "nn done"
    em(train_x, train_y, test_x, test_y, times = 10)
    print "em done"
    km(train_x, train_y, test_x, test_y, times = 10)
    print "km done"
    pca(train_x, train_y, test_x, test_y)
    print "pca done"
    ica(train_x, train_y, test_x, test_y)
    print "ica done"
    randproj(train_x, train_y, test_x, test_y)
    print "rp done"
    kbest(train_x, train_y, test_x, test_y)
    print "kbest done"