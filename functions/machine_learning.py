#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR


def machineLearning(train,test,param_lm=None,param_ridge=None,param_lasso=None,param_svr=None):
    
    result = {}
    
    ## 線形回帰 =============================
    if param_lm !=None:
        
        result['lm'] = {}
        
        print('線形回帰=============================')
        
        # モデル初期化
        linear = LinearRegression()
        
        # 学習
        linear.fit(train['X'], train['Y'])
        result['lm']['model'] = deepcopy(linear)
        
        # 学習結果
        print(f'・学習結果')
        print(f'  coef')
        print(f'    intercept: {linear.intercept_[0]}') # 切片
        for i in range(len(linear.coef_[0])):  
            tmp_X_name = train['X'].columns[i]
            print(f'    {tmp_X_name}: {linear.coef_[0][i]}') # 回帰係数
        
        print('')
        print('  R2:',linear.score(train['X'], train['Y']))# 決定係数
        print('')
        
        # 予測
        if test['X'].shape[0]>0:
            linear_pred = linear.predict(test['X'])
            
            # RMSE
            print(f'・予測結果')
            result['lm']['RMSE'] =  np.sqrt(mean_squared_error(linear_pred,test['Y']))
            print('  RMSE:' , result['lm']['RMSE'])
            print('\n')
        
        
        
        
    ## リッジ回帰 =============================
    if param_ridge !=None:
    
        result['ridge'] = {}
        
        print('リッジ回帰=============================')
        
        # モデル初期化
        ridge = Ridge(alpha=0.1)
        
        # 学習
        ridge.fit(train['X'], train['Y'])
        result['ridge']['model'] = deepcopy(ridge)
        
        # 学習結果
        print(f'・学習結果')
        print(f'  coef')
        print(f'    intercept: {ridge.intercept_[0]}') # 切片
        for i in range(len(ridge.coef_[0])):  
            tmp_X_name = train['X'].columns[i]
            print(f'    {tmp_X_name}: {ridge.coef_[0][i]}') # 回帰係数
        
        print('')
        print('  R2:',ridge.score(train['X'], train['Y']))# 決定係数
        print('')
        
        # 予測
        if test['X'].shape[0]>0:
            ridge_pred = ridge.predict(test['X'])
            
            # RMSE
            print(f'・予測結果')
            result['ridge']['RMSE'] =  np.sqrt(mean_squared_error(ridge_pred,test['Y']))
            print('  RMSE:' , result['ridge']['RMSE'])
            print('\n')
        
        
    
    ## lasso回帰 =============================
    if param_lasso !=None:
        
        result['lasso'] = {}
        
        print('lasso===============================')
        
        # モデル初期化
        lasso = Lasso(alpha=0.01)
        
        # 学習
        lasso.fit(train['X'], train['Y'])
        result['lasso']['model'] = deepcopy(lasso)
        
        # 学習結果
        print(f'・学習結果')
        print(f'  coef')
        print(f'    intercept: {lasso.intercept_[0]}') # 切片
        for i in range(len(lasso.coef_)):  
            tmp_X_name = train['X'].columns[i]
            print(f'    {tmp_X_name}: {lasso.coef_[i]}') # 回帰係数
        
        print('')
        print('  R2:',lasso.score(train['X'], train['Y']))# 決定係数
        print('')
        
        # 予測
        if test['X'].shape[0]>0:
            lasso_pred = lasso.predict(test['X'])
            
            # RMSE
            print(f'・予測結果')
            result['lasso']['RMSE'] =  np.sqrt(mean_squared_error(lasso_pred,test['Y']))
            print('  RMSE:' , result['lasso']['RMSE'])
            print('\n')
            
            
            
            
    ## SVR =============================
    if param_svr !=None:
        
        result['svr'] = {}
        
        print('SVR =============================')
        
        # モデル初期化
        svr = SVR(C=param_svr['C'], epsilon=param_svr['epsilon'], kernel=param_svr['kernel'], 
                  gamma=param_svr['gamma'], degree=param_svr['degree'], coef0=param_svr['coef0'])
        
        # 学習
        svr.fit(train['X'], train['Y'])
        result['svr']['model'] = deepcopy(svr)
        
        # 学習結果
        print(f'・学習結果')
        print(f'  coef')
        print(f'    intercept: {svr.intercept_[0]}') # 切片
        for i in range(len(svr.coef_[0])):  
            tmp_X_name = train['X'].columns[i]
            print(f'    {tmp_X_name}: {svr.coef_[0][i]}') # 回帰係数
        
        print('')
        print('  R2:',svr.score(train['X'], train['Y']))# 決定係数
        print('')
        
        # 予測
        if test['X'].shape[0]>0:
            svr_pred = svr.predict(test['X'])
            
            # RMSE
            print(f'・予測結果')
            result['svr']['RMSE'] =  np.sqrt(mean_squared_error(svr_pred,test['Y']))
            print('  RMSE:' , result['svr']['RMSE'])
            print('\n')
        
            
    return result
    