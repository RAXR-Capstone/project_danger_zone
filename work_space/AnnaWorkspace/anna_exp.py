import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
from sklearn.cluster import KMeans

import prepare as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy.stats import f_oneway
from scipy import stats
from kmodes.kmodes import KModes

def elbow_kmode(train, n=7, var=[]):
    '''
    elbow_kmode will print out a visual graph between cost and number of clusters
    '''
    cost = []
    K = range(1,n)
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
        kmode.fit_predict(train[var])
        cost.append(kmode.cost_)

    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X)
    kmeans.predict(X)
    df['cluster'] = kmeans.predict(X)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    return df, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(30, 15))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    
    
def create_kmode_clusters(train, test, n, var=[], cluster_name):
    '''
    takes in train, test sets, the number of clusters to make, variable names, and a desired cluster column name
    and will use kmode to create the cluster groups. It will also append the predicted clusters to the train and test 
    dataframes and return them back
    '''
    kmode = KModes(n_clusters=n, init = "random", n_init = 5, verbose=0, random_state= 19)
    
    clusters = kmode.fit_predict(train[var])
    train[cluster_name] = clusters
    
    clusters_test = kmode.fit_predict(test[var])
    test[cluster_name] = clusters_test
    return train, test


def chi2(feature, target, alpha=0.05):
    '''
    performs a chi squared test on a column and the specified target. 
    Returns whether to reject the null hypothesis or not, and returns the p value.
    Default alpha value of 0.05
    '''
    alpha = a
    observed = pd.crosstab(feature, target, margins = True)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p > a:
        return print(f'Fail to reject the null hypothesis, p value is {p}')
    else:
        print(f'Reject the null hypothesis, p value is {p}')
        
