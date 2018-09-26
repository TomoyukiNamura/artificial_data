#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from copy import deepcopy
import time


## 初期設定=======================================

sleep_second = 0.05
# 閾値設定
tol = 0.001

# テストデータの被験者数と面談者数を設定
N = 29  # <- 被験者数を設定
K = 10  # <- 面談者数を設定



## テストデータ作成========================================
y_true = np.round(np.random.rand(N)*2+2)
lambda_true =  np.array(abs(np.random.rand(K)))
#lambda_true =  np.array([3,1,0.1,0.01,5])
#lambda_true =  np.array([3,1,0.1,0.01,5,3,1,0.5,6,0.8])

result = {}
for j in range(K):
    result[f'a{j}']=[]
    for i in range(N):
        result[f'a{j}'].append(np.round(list(np.random.randn(1)*lambda_true[j] + y_true[i])[0]))


result = pd.DataFrame(result)


yy = result.values



### 斎藤さんプログラムでのテストデータ========================================
#yy= np.array(
#        [[2,1,3],
#         [2,4,2],
#         [3,3,1],
#         [4,5,4],
#         [2,5,3]
#         ])
#
## テストデータの被験者数と面談者数
#N = yy.shape[0]
#K = yy.shape[1]




## EMアルゴリズム=======================================
# y(i)とλ(j)初期化
y_pred = np.zeros(N).reshape(-1,1)
lambda_pred = np.ones(K).reshape(1,-1)/2

# 差分を初期化
delta_y = 1.0
delta_lambda = 1.0

# 繰り返し回数を初期化
i = 0


while(delta_y>tol or delta_lambda>tol):
    # 繰り返し回数を更新
    i+=1
    print(f'i={i}')
    
    # 更新前のy(i)とλ(j)を保存
    prior_y_pred = deepcopy(y_pred)
    prior_lambda_pred = deepcopy(lambda_pred)
    
    # y(i)の更新
    y_pred = np.dot(yy,lambda_pred.T)/np.sum(lambda_pred)
    
    # λ(j)の更新
    lambda_pred = np.sum((yy-y_pred)**2,axis=0).reshape(1,-1)/N
    
    # 更新前のy(i)とλ(j)との差分を計算
    delta_y = np.max(abs(y_pred - prior_y_pred))
    delta_lambda = np.max(abs(lambda_pred - prior_lambda_pred))
    
    # 差分を表示
    print(f'delta_y: {delta_y}')
    print(f'delta_lambda: {delta_lambda}')
    print('\n')
    
    # 0.5秒停止
    time.sleep(sleep_second) 
    




yy_pred2 = y_pred[:,0]
np.dot(y_true-yy_pred2,(y_true-yy_pred2).T)

yy_mean = np.mean(yy,axis=1)
np.dot(y_true-yy_mean,(y_true-yy_mean).T)


np.round(lambda_pred/np.sum(lambda_pred),2)
np.round(lambda_true/np.sum(lambda_true),2)
