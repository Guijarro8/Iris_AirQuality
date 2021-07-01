import pandas as pd

from sklearn import svm, datasets


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
    dic_iris={0:'setosa', 
          1:'versicolor',
          2:'virginica'}
    df_iris['target_name'] =df_iris['target'].map(dic_iris)
    
    return df_iris

def clasification_iris(df_iris):
      """Build the Clasification model for the iris dataset."""
      
      from sklearn.metrics import roc_curve, auc
      from sklearn import datasets
      from sklearn.multiclass import OneVsRestClassifier
      from sklearn.svm import LinearSVC
      from sklearn.preprocessing import label_binarize
      from sklearn.cross_validation import train_test_split
      import matplotlib.pyplot as plt

      iris = datasets.load_iris()
      X, y = iris.data, iris.target

      y = label_binarize(y, classes=[0,1,2])
      n_classes = 3

      # shuffle and split training and test sets
      X_train, X_test, y_train, y_test =\
      train_test_split(X, y, test_size=0.33, random_state=0)

      # classifier
      clf = OneVsRestClassifier(LinearSVC(random_state=0))
      y_score = clf.fit(X_train, y_train).decision_function(X_test)



####################################################################################################
# - labels: clases.
# - df_pca: DataFrame to visualize.
# - title: final detail of plot title.
# - figure_filename: name of the file exported.
####################################################################################################
def plot_pca_multiclass(labels, df_pca, title,  figure_filename=None):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()