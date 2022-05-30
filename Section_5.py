#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:18:37 2020

@author: haythamomar
"""
import numpy as np

##simulation of data using numpy

pinapple_juice=np.random.uniform(2,500,1000).round()
pinapple_juice

#mean of pinapple juice
pinapple_juice.mean()

#median
import statistics

statistics.median(pinapple_juice)

# mode
statistics.mode(pinapple_juice)

##measure of spread
statistics.stdev(pinapple_juice)

def range_data(data):
    a=max(data)-min(data)
    return a

range_data(pinapple_juice)

### variance

statistics.variance(pinapple_juice)


##Percentile
#np.percentile(array, percentile)
#50th percentile
np.percentile(pinapple_juice, 50)

#25th pecentile
np.percentile(pinapple_juice,25)

#75th pecentile
np.percentile(pinapple_juice,75)


import pandas as pd

cars=pd.read_csv("cars (1).csv")
cars.info()

cars.describe()

# subsetting and correlation

cars.iloc[:,[12,13]].corr()

#or 

cars.columns
cars.loc[:,['horsepower','city_miles_per_galloon']].corr()

#or

cars[['horsepower','city_miles_per_galloon']].corr()

cars.columns

cars.cylenders.value_counts()

cars[["Price",'horsepower','city_miles_per_galloon','weight','length']].corr()

#correlation by dropping NA's

cars.shape

car_clean= cars.dropna(axis=0)   # dropping rows having NA value
car_clean

## continuos variables of interest

car_clean[["Price",'horsepower','city_miles_per_galloon','weight','length']].corr()

correlation_data= car_clean [["Price",'horsepower','city_miles_per_galloon','weight','length']].corr()

# to get corr data against horsepower
correlation_data.loc['horsepower','weight']


#plotting correlation
import seaborn as sns
sns.heatmap(correlation_data)

# detecting outliers
import numpy as np
 
sales= np.array([5,8,10,20,100,2,65,18,32,25,200,9,15])
 
first= np.percentile(sales, 25)
third= np.percentile(sales, 75)
 
IQR= third-first

upper_thresold= third +  IQR*1.5
lowe_thresold = first - IQR*1.5

def outlier_function(x):
    first= np.percentile(sales, 25)
    third= np.percentile(sales, 75)
    
    IQR= third-first
    
    upper_thresold= third+  IQR*1.5
    lower_thresold = first - IQR*1.5
    
    outliers={'upper_outliers': x[x>upper_thresold],
              'lower_outliers': x[x<lower_thresold]}
    

#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

pricing= pd.read_excel('pricing.xlsx')

X=pricing.Price.values.reshape(1,-1)
y=pricing.Demand

model=LinearRegression()

model.fit(X,y)

array=np.array([1,2,3,4,5,6,7,8])
array.reshape(2,4)
              
array.reshape(4,2)
array.reshape(2,2,2)    

model.coef_
model.intercept_

pricing['prediction']=model.predict(X)

pricing

#distributions

from scipy.stats import norm,normaltest,kstest

#reference=https://stackoverflow.com/users/4258483/pasindu-tennage

skus= pd.read_csv('sku_distributions.csv')

dist_names= ["norm","exponweib","weibull_max","weibull_min","pareto","genext"]

apple_juice = np.array(skus['apple_juice'])

mean= apple_juice.mean()

sd= apple_juice.std()

kstest(apple_juice,'norm',arg=(mean,sd))

result=[]
parameters= {}
