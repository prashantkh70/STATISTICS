# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:10:32 2022

@author: prash
"""

import pandas as pd
import numpy as np

pinapple= pd.read_csv("Pinapple_juice.csv")

# we will work on the pinapple juice data has the demand and the price every day for this juice.
#fit the demand of this distribution to normal demand

from scipy.stats import norm,normaltest,kstest
import scipy.stats as st

juice= np.array(pinapple['Pinapple juice'])

mean= juice.mean()
mean
sd=juice.std()


##what is the p-value after the fit?

kstest(juice, 'norm', args=(mean,sd))


# Make a linear regression using lm fucntion y~x and outline the coefficeints and the intercept.

import pandas as pd
from sklearn.linear_model import LinearRegression

X=pinapple[["Price"]]
y= pinapple[['Pinapple juice']]
model= LinearRegression().fit(X,y)

model.coef_
model.intercept_



