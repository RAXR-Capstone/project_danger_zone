import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools


def get_distribution(df):
    for i in df:
        plt.figure(figsize=(16,9))
        plt.title('{} Distribution'.format(i))
        plt.xlabel(i)
        plt.ylabel('count')
        df[i].astype(str).hist(grid = False, bins = 100)
        plt.show()
        
def compare_to_target(df, target):
    

    for i in df:
        if  df[target].dtype == np.float64:
            plt.figure(figsize=(16,9))
            plt.title('{} vs {}'.format(i,target))
            sns.scatterplot(data=df , y = target, x = i)
            plt.show()
        else:
            plt.figure(figsize=(16,9))
            plt.title('{} Distribution'.format(i))
            plt.xlabel(i)
            plt.ylabel('count')
            sns.histplot(data=df , x=i, hue=target)
            plt.show()
        
            
        
def plot_variable_pairs(df, cont_vars = 2):
    combos = itertools.combinations(df,cont_vars)
    for i in combos:
        plt.figure(figsize=(8,3))
        sns.regplot(data=df, x=i[0], y =i[1],line_kws={"color":"red"})
        plt.show()

        
def get_heatmap(df, target):
    '''
    This method will return a heatmap of all variables and there relation to churn
    '''
    plt.figure(figsize=(15,12))
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending=False), annot=True)
    heatmap.set_title('Feautures  Correlating with {}'.format(target))
    plt.show()
    return heatmap

