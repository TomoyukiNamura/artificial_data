#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pandas.plotting import scatter_matrix


def plotGraph(graph, name_dependent, name_response, k=0.7,node_size=1000, node_color='cyan', edge_color='k', width=1.0, output=None):
    # グラフ表示
    G = nx.DiGraph()
    
    # ノードの追加    
    G.add_nodes_from(graph.index) 
    
    # ノードの色の定義
    node_color = np.array(['darkgray'] * graph.shape[0])
    
    is_dependent = [graph.index[i] in name_dependent for i in range(len(graph.index))]
    node_color[is_dependent] = 'cyan'
    
    is_response = [graph.index[i] in name_response for i in range(len(graph.index))]
    node_color[is_response] = 'orange'
    
    node_color = list(node_color)
    
    
    # エッジの追加
    edge_list = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if i!=j and graph.iloc[i][j]!=0:
                edge_list.append((graph.index[i],graph.index[j]))
    
    G.add_edges_from(edge_list)  
    
    
    pos = nx.spring_layout(G, k=k)
    nx.draw_networkx(G,pos,node_size=node_size,node_color=node_color, edge_color=edge_color, width=width)
    
    plt.tick_params(color='white')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    if output != None:
        plt.savefig(output)  
        
    plt.show()


def scatterAndCorrMat(df,name_dependent=None,name_response=None,corr_method='pearson'):
    ## plot対象dfを作成
    if name_dependent!=None and name_response!=None:
        df_target = df.loc[:,name_dependent+name_response]
        
    elif name_dependent!=None and name_response==None:
        df_target = df.loc[:,name_dependent]
        
    elif name_dependent==None and name_response!=None:
        df_target = df.loc[:,name_response]
        
    else :
        df_target = df
    
    ## 散布図マトリックスを表示
    #scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    sns.pairplot(df_target)
    plt.show()
    
    ## 相関行列(ピアソンの積率相関係数)をヒートマップで表示
    corr_mat = df_target.corr(method=corr_method)
    sns.heatmap(corr_mat,cmap='seismic',vmin=-1.0,vmax=1.0,center=0,annot=True,fmt='.3f',
                xticklabels=corr_mat.columns.values,yticklabels=corr_mat.columns.values)
    plt.show()
    
    