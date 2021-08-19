import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
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

def plot_map():
    m = folium.Map(location=[29.377711363953658, -98.4970935625])
    mask = train.injury_class == 1
    mask2 = train.injury_class == 0
    for i in range(0,len(train[mask])):
        folium.CircleMarker(location=[train[mask].iloc[i]['crash_latitude'], train[mask].iloc[i]['crash_longitude']], radius = 0.5, color='red').add_to(m)
    for i in range(0,len(train[mask2])):
    folium.CircleMarker(location=[train[mask2].iloc[i]['crash_latitude'], train[mask2].iloc[i]['crash_longitude']], radius = 0.5, color='blue').add_to(m)

