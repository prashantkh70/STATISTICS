#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 19:46:03 2020

@author: haythamomar
"""

import pandas as pd
pinapple= pd.read_csv('pinapple_juice.csv')


#we will work on pineapple juice data that has the demand and the price every day for this juice.
#- Fit the demand of this Distribution to normal demand .
import numpy as np
from scipy.stats import norm,normaltest,kstest

import scipy.stats as st

juice= np.array(pinapple['Pinapple juice'])

mean= juice.mean()
sd= juice.std()






##what is the p-value after the fit ?

kstest(juice, 'norm',args=(mean,sd))






#Make a linear regression using lm function y~x and outline the coefficients and the intercept.

import pandas as pd
from sklearn.linear_model import LinearRegression

X=pinapple[['Price']]
y= pinapple[['Pinapple juice']]
model=LinearRegression().fit(X,y)

model.coef_
model.intercept_







