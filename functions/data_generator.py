#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from copy import deepcopy
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

def generateLinearGaussData(n_train, intercept, linear_coef, gauss_std):
    
    # 結果を初期化
    df = {}
    
    for i in range(len(linear_coef)):
        # 人工データ発生対象の変数名取得
        i_var_name = list(linear_coef.keys())[i]
        
        # 線型結合部を初期化(n_train次元の零ベクトル生成)
        linear_combination = np.zeros(n_train)
        
        # 線型結合部を計算
        for j in range(len(linear_coef)):
            # 線型結合に含む変数名を取得
            j_var_name = list(linear_coef.keys())[j]
            
            # 取得した変数にかかる係数が0でない場合，係数と変数の積を線型結合部に和算する
            if linear_coef[i_var_name][j]!=0:
                linear_combination += linear_coef[i_var_name][j] * df[j_var_name]
        
        # 誤差項~N(0,gauss_std[i_var_name]**2)を計算
        epsilon = np.random.randn(n_train) * gauss_std[i_var_name]
        
        # 線型結合部+誤差項を，現在の変数の人工データとして保存
        df[i_var_name] = intercept[i_var_name] + linear_combination + epsilon
        
    # 結果をpandas形式に変換
    df = pd.DataFrame(df)
    
    # 出力
    return df



def devideTrainTest(df,name_dependent,name_response,rate_train,normalize_X=False,normalize_Y=False):
    
    # 訓練・評価データ数を定義
    n_train = round(df.shape[0] * rate_train)
    n_test  = df.shape[0] - n_train
    
    # 説明変数と目的変数を分割
    tmp_X = deepcopy(df.loc[:,name_dependent])
    tmp_Y = deepcopy(df.loc[:,name_response])
    
    # 訓練データと評価データを作成
    train = {}
    train['X'] = deepcopy(tmp_X.iloc[0:n_train,:])
    train['Y'] = deepcopy(tmp_Y.iloc[0:n_train,:])
    
    test = {}
    test['X'] = deepcopy(tmp_X.iloc[n_train:(n_train+n_test),:])
    test['Y'] = deepcopy(tmp_Y.iloc[n_train:(n_train+n_test),:])
    
    # 説明変数を標準化
    if normalize_X and (train['X'].shape[0]>1):
        # 列名を保存
        X_colnames = train['X'].columns
        
        # 標準化のための平均，標準偏差を保存
        sc_X = StandardScaler()
        sc_X.fit(train['X'])
        
        # 標準化
        train['X'] = pd.DataFrame(sc_X.transform(train['X']))
        
        if test['X'].shape[0]>0:
            test['X']  = pd.DataFrame(sc_X.transform(test['X']))
        
        # 元の列名を適用
        train['X'].columns = X_colnames
        test['X'].columns = X_colnames
                
    # 目的変数を標準化
    if normalize_Y and (train['Y'].shape[0]>1):
        # 列名を保存
        Y_colnames = train['Y'].columns
        
        # 標準化のための平均，標準偏差を保存
        sc_Y = StandardScaler()
        sc_Y.fit(train['Y'])
        
        # 標準化
        train['Y'] = pd.DataFrame(sc_Y.transform(train['Y']))
        
        if test['Y'].shape[0]>0:
            test['Y']  = pd.DataFrame(sc_Y.transform(test['Y']))
        
        # 元の列名を適用
        train['Y'].columns = Y_colnames
        test['Y'].columns = Y_colnames

    # 出力
    return train,test