import os

import pandas as pd

import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn import svm, datasets,preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

####################################################################################################
# - df: DataFrame to generate a Report
# - df_title: title of the DataFrame
####################################################################################################
def print_report(df,df_title):
    """Exports an html file with the profiling, distributions and correlations of the DataFrame variables."""
    filepath = f"{os.path.abspath('')}/reports/"
    profile = ProfileReport(df, title=f"{df_title}")
    profile.to_file(f"{filepath}{df_title}_report.html")


####################################################################################################
# - df: DataFrame to standarize.
# return: standarized and selected more-explainable PCA Dataframe.
####################################################################################################
def std_pca(df,pca_threshold=0.999):
    """Standarize a DataFrame and perform a PCA,"""
    
    # Process flow as a pipeline.
    std_pca_pipe = Pipeline([('std', StandardScaler()), ('pca', PCA())])

    # Standarize and make the PCA.
    df_pca = pd.DataFrame(std_pca_pipe.fit(X = df).transform(X = df))
                             
    # Extract the trained model.
    pca_model = std_pca_pipe.named_steps['pca']
    
     # Select non-explained variables.
    df_pca_exp_var = pca_model.explained_variance_ratio_.cumsum()
    df_pca_filter = df_pca.copy()
    for i_var in range(len(df_pca_exp_var)):
            if (df_pca_exp_var[i_var] > pca_threshold):
                del df_pca_filter[i_var]
    return df_pca_filter


####################################################################################################
# - labels: clases.
# - df_pca: DataFrame to visualize.
# - title: final detail of plot title.
# - figure_filename: name of the file exported.
####################################################################################################
def plot_pca_multiclass(labels, df_pca, title,  figure_filename=None):
    """ Plot and save a 2D distribution graph of the PCA of a multiclass dataframe"""
    figure_filepath = f"{os.path.abspath('')}/reports"
    
    # Plot the dataset and color clusters.        
    fig, ax = plt.subplots()
    groups = pd.DataFrame(df_pca, columns = [0, 1]).assign(category = labels).groupby('category')
    
    for name, points in groups:
        ax.scatter(points[0], points[1], label = name)
    ax.legend()
    plt.title(str(title));        
    
    # Export figure as image.
    if (figure_filename is not None):
        plt.savefig(f"{figure_filepath}/{figure_filename}.png")