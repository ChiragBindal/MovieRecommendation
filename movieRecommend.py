#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:33:17 2023

@author: chiragbindal
"""

import pandas as pd
import numpy as np
import warnings 
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
columns_names = ["user_id" , "item_id" , "rating" , "timestamp"]
df = pd.read_csv("ml-100k/u.data" , sep='\t' , names = columns_names)
movies_titles = pd.read_csv("ml-100k/u.item" , sep="\|" , header=None , encoding='latin-1')
movies_titles = movies_titles[[0,1]]
movies_titles.columns = ["item_id" , "title"]
df = pd.merge(df,movies_titles,on="item_id")


# Exploratory Data Analysis
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['no of ratings'] = pd.DataFrame(df.groupby('title').count()['rating']);

plt.figure(figsize=(10,6))
plt.hist(ratings['no of ratings'],bins=70)
plt.show()
plt.hist(ratings['rating'],bins=70)
plt.show()

sns.jointplot(x='rating' , y='no of ratings' , data=ratings, alpha=0.5)

# Creating Movie Recommendation
movies_mat = df.pivot_table(index="user_id",columns="title" , values="rating")
print(movies_mat.head())
ratings.sort_values('no of ratings' , ascending=False)
starwars_user_ratings = movies_mat['Star Wars (1977)']
similar_to_starwars = movies_mat.corrwith(starwars_user_ratings)
print(similar_to_starwars)
corr_starwars = pd.DataFrame(similar_to_starwars , columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['no of ratings'])
print(corr_starwars.head())
corr_starwars[corr_starwars['no of ratings']>100].sort_values('Correlation' , ascending=False)

def predict_movies(movie_name) :
    movie_user_ratings = movies_mat[movie_name]
    similar_to_movie = movies_mat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie , columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['no of ratings'])
    prediction =  corr_movie[corr_movie['no of ratings']>100].sort_values('Correlation' , ascending=False)
    
    return prediction

predictions = predict_movies('Titanic (1997)')
print(predictions.head())