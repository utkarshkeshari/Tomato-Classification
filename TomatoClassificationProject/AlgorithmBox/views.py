from django.shortcuts import render

# Create your views here.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

def readFile():
    data = pd.read_csv('Data/TomatoData.csv')
    y = data['label']
    X = data.drop('label', axis=1)
    return X,y


def randomForest(feature):
    X,y = readFile()

    clf = RandomForestClassifier(n_estimators=90,max_features=0.4, random_state=42)
    clf.fit(X, y)
    return clf.predict(feature)

def logisticRegression(feature):
    X,y = readFile()
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    return clf.predict(feature)

def kNeighborsClassifier(feature):
    X,y= readFile()

    clf = KNeighborsClassifier()
    clf.fit(X, y)
    return clf.predict(feature)

def decisionTree(feature):
    X,y = readFile()

    clf = DecisionTreeClassifier(random_state=421)
    clf.fit(X,y)
    return clf.predict(feature)

def svm(feature):
    X, y = readFile()

    clf = LinearSVC()
    clf.fit(X, y)
    return clf.predict(feature)

def gradientBoosting(feature):
    X, y = readFile()

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X, y)
    return clf.predict(feature)


def Xgboost(feature):
    X,y = readFile()
    xg_train = xgb.DMatrix(X, label=y)

    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 1
    param['num_class'] = 6

    num_round = 5
    bst = xgb.train(param, xg_train, num_round)
    return bst.predict(xgb.DMatrix(feature))

def getEstimator(clf):

    estimator = None
    if clf == 'Logistic Regression':
        estimator = LogisticRegression(random_state=42)
    elif clf == 'Random Forest':
        estimator = RandomForestClassifier(n_estimators=90,max_features=0.4, random_state=42)

    elif clf == 'XgBoost':
        estimator = XGBClassifier()

    elif clf == 'KNN':
        estimator = KNeighborsClassifier()

    elif clf == 'GradientBoosting':
        estimator = GradientBoostingClassifier(random_state=42)

    elif clf == 'SVM':
        estimator = LinearSVC(random_state=42)

    elif clf == 'Decision Tree':
        estimator = DecisionTreeClassifier(random_state=421)

    return estimator













