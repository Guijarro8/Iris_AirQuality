import os

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from ..common import split_train_test, percentage, print_report, std_pca, plot_pca_multiclass, save_model, load_model



dic_iris={0:'setosa', 
          1:'versicolor',
          2:'virginica'}


def load_iris_df():
    '''
    Import Iris Dataframe.
    # return1: Iris Dataframe.
    '''


    #Load Data
    iris = datasets.load_iris()
    df_iris = pd.DataFrame(data = iris.data,columns = iris.feature_names)
    
    #Introduce target numeric and string
    df_iris['target'] = iris.target
    df_iris['target_name'] =df_iris['target'].map(dic_iris)
    
    return df_iris

def classify(df_features,target,classifier):
    '''
    # - df_iris: Iris Dataframe.
    # - fields_input:  Features to input in the model of the dataframe
    # - fields_target: Output of the dataframe
    '''
        
    # Shuffle and split training and test sets
    x_train, x_test, y_train, y_test = split_train_test(df_features, target)

    if isinstance(classifier, tuple):
        clf = GridSearchCV(classifier[0], classifier[1], cv=10, n_jobs=-1)
        clf.fit(x_train, y_train.values.ravel())
        y_pred = clf.predict(x_test)
    else:
        clf = classifier.fit(x_train, y_train.values.ravel())
        y_pred = clf.predict(x_test)

    scores = {
        'accuracy_train': clf.score(x_train, y_train),
        'accuracy_test': clf.score(x_test, y_test),
        'f1_weighted': f1_score(y_test.values, y_pred, average='weighted'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'best_params': str(clf.best_params_) if isinstance(classifier, tuple) else None
    }

    return scores, clf

def load_explore_iris(): 
    '''
    Load, explore  Iris Data generating a profiling report and implement a PCA.
    # return: df_pca_iris: Standarized- PCA of the Iris Dataframe.
    # return: df_iris: Iris Dataframe.
    # return: fields_input:  Features to input in the model of the dataframe
    # return: fields_target: Output of the dataframe
    '''

    
    #Load Data

    df_iris = load_iris_df()
    print ("\n[Dataset]\n")
    display(df_iris.describe())
    fields_input =['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'] 
    fields_target= ['target']

    #Print report
    print ("\n[Pandas_profiling]\n")
    print_report(df_iris,'iris_profiling')
    
    #Implement a PCA and a overview of the data
    print (f"\n[PCA]\n ")
    df_pca_iris = std_pca(df_iris[fields_input])

    print ("\n[PCA_distribution]\n")
    plot_pca_multiclass(
                    df_iris['target_name'],
                    df_pca_iris,
                    'Iris Distribution',
                    'iris_distribution')

    return df_pca_iris, df_iris, fields_input, fields_target 

def classification_iris(df_pca_iris, df_iris, fields_input_iris, fields_target_iris):
    '''
    # - df_iris: Iris Dataframe.
    # - fields_input:  Features to input in the model of the dataframe
    # - fields_target: Output of the dataframe
    # return: results of the evaluation
    '''
    
    results={}
    #Define classifiers and params
    
    param_regularization = [10**x for x in range(-4, 4)]
    param_ranges_alpha = [1/(2*i) for i in param_regularization]
    param_solver = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    param_algorithm = ['ball_tree', 'kd_tree', 'brute']
    classifiers = {
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),    
    'svm_linear': (LinearSVC(max_iter=10000), {'C': param_regularization}),
    'ridge': (RidgeClassifier(),{'alpha':param_regularization ,'solver': param_solver}),
    'logistic_regression': (LogisticRegression (multi_class="multinomial"),{'C': param_regularization}),
    'k_neighbors':(KNeighborsClassifier(n_neighbors=3),{'algorithm': param_algorithm})
    }
    
    scores_per_classifier = ['accuracy_train', 'accuracy_test', 'f1_weighted', 'cohen_kappa', 'best_params']
    
    for key, classifier in classifiers.items(): 
        scores, clf = classify(
            df_pca_iris,
            df_iris[fields_target_iris],
            classifier
        )
        save_model(clf,f"iris_{key}")
        
        # Save results
        filepath = f"{os.path.abspath('')}/results/"
        for score in scores_per_classifier:
            results[f'{key}__{score}'] = scores[score]

        np.save(f"{filepath}[iris_clasification]{key}", scores) 

    print("[Evaluation Metrics]\n")
    for  item in results.items():
        print(item)
    




