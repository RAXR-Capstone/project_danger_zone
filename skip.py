import pandas as pd
import os

if os.path.isfile('accident_links_continued.csv'):
    links = pd.read_csv('accident_links_continued.csv')
else:
    links = pd.read_csv('accident_links.csv')
links = links.drop_duplicates()

links = links.iloc[1: , :]
if os.path.isfile('accident_links_continued.csv'):
    links.to_csv('accident_links_continued.csv',index=False)
else:
    links.to_csv('accident_links.csv',index=False)
    