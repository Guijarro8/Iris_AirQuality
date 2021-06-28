import pandas as pd

from sklearn import svm, datasets


####################################################################################################
# return1: Iris Dataframe.
####################################################################################################
def load_iris_df():
    """Import Iris Dataframe"""

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