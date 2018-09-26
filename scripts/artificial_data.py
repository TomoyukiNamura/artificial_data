#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人工データ発生実験
"""


### 初期処理 =======================================================

## ディレクトリ変更
import os
os.chdir('/Users/tomoyuki/python_workspace/artificial_data')
os.getcwd()

## パッケージ読み込み
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
import statsmodels.formula.api as smf

## 設定ファイル，関数ファイル読み込み
from cfg import script_config as conf
from functions import plot_functions as pf
from functions import data_generator as dg
from functions import machine_learning as ml



### グラフ描画 =======================================================
from cfg import script_config as conf
graph = pd.DataFrame(conf.linear_coef,index = conf.linear_coef.keys())
pf.plotGraph(graph=graph, name_dependent=conf.name_dependent, name_response=conf.name_response)
#pf.plotGraph(graph=graph,output='output/graph.png')



### 線形ガウス乱数発生 =======================================================
from cfg import script_config as conf

## 人工データ発生
df = dg.generateLinearGaussData(n_train=conf.n_data, intercept=conf.intercept, linear_coef=conf.linear_coef, gauss_std=conf.gauss_std)

## 散布図マトリックス・相関行列を表示
#pf.scatterAndCorrMat(df,conf.name_dependent,conf.name_response,corr_method='pearson')
pf.scatterAndCorrMat(df,corr_method='pearson')


### 訓練データと評価データを作成 =======================================================
from cfg import script_config as conf
train,test = dg.devideTrainTest(df=df, name_dependent=conf.name_dependent, name_response=conf.name_response,rate_train=conf.rate_train, normalize_X=conf.normalize_X, normalize_Y=conf.normalize_Y)



## 線形回帰(statsmodels)=========================
print(smf.OLS(endog=train['Y'],exog=train['X']).fit().summary())
#print(smf.OLS(endog=train['Y'],exog=train['X']['X2']).fit().summary())
#print(smf.OLS(endog=train['Y'],exog=train['X']['X3']).fit().summary())



## 機械学習手法適用=========================
from cfg import script_config as conf
ml_result = ml.machineLearning(train,test,param_lm=conf.param_lm, param_ridge=conf.param_ridge, 
                               param_lasso=conf.param_lasso, param_svr=conf.param_svr)






from cfg import script_config as conf
# モデル初期化
model_X6 = LinearRegression()
model_X6.fit(train['X'].loc[:,['X2','X3']], train['X'].loc[:,['X6']])
X6_pred = model_X6.predict(test['X'].loc[:,['X2','X3']])

model_X7 = LinearRegression()
model_X7.fit(train['X'].loc[:,['X4','X5']], train['X'].loc[:,['X7']])
X7_pred = model_X7.predict(test['X'].loc[:,['X4','X5']])


test_pred = pd.DataFrame({
        'X6':np.reshape(X6_pred,(X6_pred.shape[0],)),
        'X7':np.reshape(X7_pred,(X7_pred.shape[0],))
        })


depen_list = ['X6','X7']
model_X8 = LinearRegression()
model_X8.fit(train['X'].loc[:,depen_list], train['Y'].loc[:,['X8']])
X8_pred = model_X8.predict(test_pred)

X8_pred = model_X8.predict(test['X'].loc[:,depen_list])


np.sqrt(mean_squared_error(X8_pred,test['Y'].loc[:,['X8']]))


depen_list = ['X2','X3','X4','X5']
model_X8 = LinearRegression()
model_X8.fit(train['X'].loc[:,depen_list], train['Y'].loc[:,['X8']])
X8_pred = model_X8.predict(test['X'].loc[:,depen_list])





from mpl_toolkits.mplot3d import axes3d, Axes3D

fig = plt.figure()
ax = Axes3D(fig) 

angle = 360-0
ax.view_init(00, angle)
ax.scatter(train['X'][conf.name_dependent[0]], train['X'][conf.name_dependent[1]], train['Y'][conf.name_response[0]],  c=train['Y'].values.T[0]/len(set(train['Y'].values.T[0])))

plt.xlabel(conf.name_dependent[0])
plt.ylabel(conf.name_dependent[1])
    
plt.show()



def matplotlib_rotate(df, name_dependent, name_response, dataname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    X1 = df[name_dependent[0]].values
    X2 = df[name_dependent[1]].values
    Y  = df[name_response[0]].values

    ax.scatter(X1, X2, Y, c=Y/len(set(Y)))
    
    plt.xlabel(name_dependent[0])
    plt.ylabel(name_dependent[1])
#    plt.zlabel(name_response[0])
    
    ax.view_init(45, 45)

    for angle in range(0, 360):
#        ax.view_init(30, angle)
#        plt.savefig(f"output/test_{angle}.png")
#        plt.show()


#
#
#tmp_x = np.linspace(-4,4,100)
#tmp_y = lasso.intercept_ + tmp_x*lasso.coef_[1] 
#tmp_y2 = 0+tmp_x
#
#plt.scatter(df_norm['X3'],df_norm['X4'])
#plt.plot(tmp_x,tmp_y,color='orange')
#plt.plot(tmp_x,tmp_y2,color='red')
#plt.grid()
#
