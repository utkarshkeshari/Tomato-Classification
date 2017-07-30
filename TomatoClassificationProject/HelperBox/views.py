from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import skimage.io as io
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from AlgorithmBox.views import logisticRegression,randomForest,kNeighborsClassifier,readFile,\
    Xgboost,getEstimator,gradientBoosting,decisionTree,svm


#___________________________________________________________________________________

def extractFeature(path):

    color = ['red', 'green', 'blue']
    csvData = {'redAvg': [], 'greenAvg': [], 'blueAvg': [],
               'red100H': [], 'green100H': [], 'blue100H': [],
               'red100V': [], 'green100V': [], 'blue100V': [],
               'red500H': [], 'green500H': [], 'blue500H': [],
               'red500V': [], 'green500V': [], 'blue500V': []
               }


    im = io.imread(path)
    for i, col in enumerate(color):
        data = im[:, :, i][10:-10, :][:, 10:-10]
        avg = np.mean(data)
        h100 = np.mean(data[100, 50:100])
        v100 = np.mean(data[50:100, 100])
        h500 = np.mean(data[500, 50:100])
        v500 = np.mean(data[50:100, 500])
        csvData[col + 'Avg'].append(avg)
        csvData[col + '100H'].append(h100)
        csvData[col + '100V'].append(v100)
        csvData[col + '500H'].append(h500)
        csvData[col + '500V'].append(v500)

    df = pd.DataFrame(csvData)
    return df

#___________________________________________________________________________________
def plot_learning_curve(request):

    clf = request.GET.get('classifier')
    X,y = readFile()

    estimator = None
    cv = None
    n_jobs = 1
    train_sizes = np.linspace(.1, 1.0, 5)

    fig = Figure()
    ax = fig.add_subplot(111)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    fig.suptitle(clf, fontsize=20)
    estimator = getEstimator(clf)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ax.legend(loc="best")

    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

#___________________________________________________________________________________


def root(request):

    return render(request, 'TomatoClassification.html')
#___________________________________________________________________________________

def predict(request):

    path = request.POST['path']
    clf = request.POST['classifier']

    predictedValue = None
    ret_value = None
    feature = extractFeature(path)

    if clf == 'Logistic Regression':
        predictedValue = logisticRegression(feature)
    elif clf == 'Random Forest':
        predictedValue = randomForest(feature)

    elif clf == 'XgBoost':
        predictedValue = Xgboost(feature)

    elif clf == 'KNN':
        predictedValue = kNeighborsClassifier(feature)

    elif clf == 'GradientBoosting':
        predictedValue = gradientBoosting(feature)

    elif clf == 'SVM':
        predictedValue = svm(feature)

    elif clf == 'Decision Tree':
        predictedValue = decisionTree(feature)


    if list(predictedValue)[0] == 1:
        ret_value = 'Predicted Tomato Class : Green using ' + str(clf)
    elif list(predictedValue)[0] == 2:
        ret_value = 'Predicted Tomato Class : Breaker using ' + str(clf)
    elif list(predictedValue)[0] == 3:
        ret_value = 'Predicted Tomato Class : Pink using ' + str(clf)
    elif list(predictedValue)[0] == 4:
        ret_value = 'Predicted Tomato Class : Red using ' + str(clf)
    elif list(predictedValue)[0] == 5:
        ret_value = 'Predicted Tomato Class : Red-Matured using ' + str(clf)


    response_dict = {}
    response_dict.update({'predictedValue': ret_value})
    print '---'*20
    print response_dict
    print '---' * 20
    return HttpResponse(json.dumps(response_dict), content_type='application/javascript')

#___________________________________________________________________________________

def getTableData(request):
    clf = request.POST['classifier']

    X,y = readFile()

    estimator = getEstimator(clf)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    confusion_arr = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test,y_pred) * 100
    accuracy = 'Accuracy : ' + str(round(accuracy,2)) + '%'
    response_dict = {}
    response_dict.update({'confusionMatrix': confusion_arr,'accuracyScore':accuracy})
    print '=='*30,'confusion matrix'
    print response_dict
    print '=='*20
    return HttpResponse(json.dumps(response_dict), content_type='application/javascript')

























