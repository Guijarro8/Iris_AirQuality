import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from statistics import mean

from ..common import split_train_test



dic_iris={0:'setosa', 
          1:'versicolor',
          2:'virginica'}


####################################################################################################
# return1: Iris Dataframe.
####################################################################################################
def load_iris_df():
    """Import Iris Dataframe."""

    #Load Data
    iris = datasets.load_iris()
    df_iris = pd.DataFrame(data = iris.data,columns = iris.feature_names)
    
    #Introduce target numeric and string
    df_iris['target'] = iris.target
    df_iris['target_name'] =df_iris['target'].map(dic_iris)
    
    return df_iris

####################################################################################################
# - df_iris: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################
def classification_OvsR(df_features,target):
    """Build the OneVsRest Classification model for the iris dataset."""
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target)

    # Classifier and evaluation
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    eval_classification(y_test, y_score,'Classification_OneVsRest')
    return clf, y_test, y_score

####################################################################################################
# - df_iris: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################
def classification_OvsO(df_features,target):
    """Build the OneVsOne Classification model for the iris dataset."""
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target)

    # Classifier and evaluation
    clf = OneVsOneClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    eval_classification(y_test, y_score, 'Classification_OneVsOne')
    return clf

####################################################################################################
# - df_iris: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################
def classification_outputcode(df_features,target):
    """Build the OutputCode Classification model for the iris dataset."""
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target)

    # Classifier and evaluation
    clf = OutputCodeClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    eval_classification(y_test, y_score, 'Classification_OutputCode')
    return clf

####################################################################################################
# - df_iris: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################
def classification_randomforest(df_features,target):
    """Build the RandomForest Classification model for the iris dataset."""
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target)

    # Classifier and evaluation
    clf = RandomForestClassifier()
    y_score = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    eval_classification(y_test, y_score, 'Classification_RandomForest')
    return clf

    GradientBoostingClassifier
####################################################################################################
# - df_iris: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################
def classification_model(df_features,target,Classifier,Classifier_name):
    """Build the RandomForest Classification model for the iris dataset."""
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target)

    # Classifier and evaluation
    clf = OneVsOneClassifier(Classifier)
    y_score = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    eval_classification(y_test, y_score, Classifier_name)
    return clf

####################################################################################################
# - y_test: Output values of test set.
# - y_score: Output values predicted by the model.
# - n_classes: Number os classes of the multiclass model.
####################################################################################################
def eval_classification(y_test, y_score, model_name,n_classes = 3):
    """Print the metrics of classification"""
    y_test = pd.DataFrame(label_binarize(y_test, classes=[0,1,2]))
    y_score = pd.DataFrame(label_binarize(y_score, classes=[0,1,2]))
    
    SEN = []
    SPE = []
    AC = []
    for i in range(n_classes):
        metrics = y_test[[i]].rename(columns={i:'true'})
        metrics['prediction'] = y_score[i]
        TN = eval_count(metrics, 0,0)
        FN = eval_count(metrics, 1,0)
        TP = eval_count(metrics, 1,1)
        FP = eval_count(metrics, 0,1)

        SEN.append(TP/(TP+FN))
        SPE.append(TN/(TN+FP))
        AC.append((TN+TP)/(TP+FN+TN+FP))

    print(f"[{model_name}] - Acurracy: {pourcentage(AC)}% - Sensitivity: {pourcentage(SEN)}% - Specificity: {pourcentage(SPE)}%")

def eval_count(df,label_true,label_predict):
    return len (df[(df['true'] == label_true) & (df['prediction'] == label_predict)])

def pourcentage(x):
    return int(mean(x)*1000)/10

