# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:32:51 2022

@author: prash
"""


import pandas as pd
import numpy as np
#reference=https://stackoverflow.com/users/4258483/pasindu-tennage

from scipy.stats import norm,normaltest,kstest
skus= pd.read_csv('sku_distributions.csv')

dist_names= ["norm","exponweib","weibull_max","weibull_min","pareto","genextreme"]

apple_juice = np.array(skus['apple_juice'])

mean= apple_juice.mean()

sd= apple_juice.std()

kstest(apple_juice,'norm',args=(mean,sd))

result=[]
parameters= {}

import scipy.stats as st
norm_param= getattr(st,'norm')

norm_param.fit(apple_juice)

for dist in dist_names:
    param= getattr(st, dist)
    fitting=param.fit(apple_juice)
    test= kstest(apple_juice,dist,args=fitting)
    result.append([dist,test])
    print("The result for dist" + dist + 'is' + str(test))
    
result    
    