# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:20:56 2024

@author: shang
"""

import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
plt.rcParams['font.size']=8
plt.rcParams['font.family']='Arial'
print('load all packages successfully')

input_dat=sys.argv[1]
x_trait=sys.argv[2]
y_trait=sys.argv[3]
outputf=sys.argv[4]
'''
input_dat="C:/Users/shang/liu_project/trade_off/arabi/01phenotype/pheno_326_accession.csv"
x_trait='Fresh weight'
y_trait='Protein content'
outputf='C:/Users/shang/liu_project/trade_off/arabi/01phenotype/FPT'
'''
color2=(71/255,75/255,87/255)#low pf
color1=(182/255,98/255,134/255)#high pf


#preprocess data
scale=MinMaxScaler()
dat_pheno=pd.read_csv(input_dat,sep=',',index_col=0)
scale.fit(dat_pheno)
dat_pheno_mn=scale.transform(dat_pheno)
dat_pheno_mn=pd.DataFrame(dat_pheno_mn)
dat_pheno_mn.index=dat_pheno.index
dat_pheno_mn.columns=dat_pheno.columns
print('Maximum-minimum scale has been completed')

#get the convex hull points
dat_pheno_mn=dat_pheno_mn[[x_trait,y_trait]]
dat_pheno_mn_np=np.array(dat_pheno_mn)
dat_pheno_mn_np=dat_pheno_mn_np.astype(np.float32)
hull_points=cv2.convexHull(dat_pheno_mn_np)
hull_points=[(i,j) for i,j in zip(hull_points[:,0,0],hull_points[:,0,1])]
#filter out points from top-right convex hull
slope_1d,intercept_1d=np.polyfit(dat_pheno_mn_np[:,0], dat_pheno_mn_np[:,1], 1)
expect_points=[(i,j) for i,j in hull_points if (slope_1d*i+intercept_1d)<j]
expect_points_x=np.array([i[0] for i in expect_points])
expect_points_y=np.array([i[1] for i in expect_points])
print('Identification of convex-hull')

#fit pareto frontier functions
def get_func(expect_points_x,expect_points_y,num):  
    coefs = np.polyfit(expect_points_x, expect_points_y, num)
    p=np.poly1d(coefs)
    yfit = p(expect_points_x) 
    yresid = expect_points_y - yfit 
    SSresid = sum(pow(yresid, 2)) 
    SStotal = expect_points_y.shape[0] * np.var(expect_points_y) 
    r2 = 1 - SSresid/SStotal 
    return (r2,p,coefs)
dic_fit_funcs={}
for i in range(1,4):
    dic_fit_funcs[i]=get_func(expect_points_x, expect_points_y, i)
print('Get the functions for pareto frontier')

def fit_func(fit_list,poly_item):
    y_sery=[poly_item(i) for i in fit_list]
    return (fit_list,y_sery)
x_fit_list=dat_pheno_mn[x_trait].tolist()
x_fit_list.sort()
poly_x_list,poly_y_list=[],[]
for i in [1,2,3]:
    poly_x,poly_y=fit_func(x_fit_list,dic_fit_funcs[i][1])
    poly_x_list.append(poly_x)
    poly_y_list.append(poly_y)

#PF score calculating
def min_distance(poly1d_item,point_x,point_y):    
    min_f = lambda x: pow(x-point_x, 2) + pow(poly1d_item(x)-point_y, 2)
    min_res = minimize(min_f, 0)
    result=distance.euclidean((point_x,point_y),(min_res.x[0],poly1d_item(min_res.x[0])))
    return result

dat_pheno_pf_list=[]
for i_ in [1,2,3]:    
    dat_pheno_pf=pd.DataFrame()
    dat_pheno_pf[x_trait]=[i for i in dat_pheno_mn[x_trait]]
    dat_pheno_pf[y_trait]=[i for i in dat_pheno_mn[y_trait]]
    dat_pheno_pf['%s_%s_pf'%(x_trait,y_trait)]=[min_distance(dic_fit_funcs[i_][1],i,j) for i,j in zip(dat_pheno_mn[x_trait],dat_pheno_mn[y_trait])]  
    dat_pheno_pf.index=dat_pheno_mn.index
    dat_pheno_pf=dat_pheno_pf.sort_index()
    dat_pheno_pf.to_csv(outputf+'%s_%s_pareto_frontier_pheno_%d.csv'%(x_trait,y_trait,i_))
    dat_pheno_pf_list.append(dat_pheno_pf)

#plot the scatter for different functions
longyuan_kubei_cmap=LinearSegmentedColormap.from_list("my_cmap", [color2,color1], N=dat_pheno_mn.shape[0])

fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,figsize=(4.5,1.5),dpi=600)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.scatter([i for i in dat_pheno_pf_list[0][x_trait]],[i for i in dat_pheno_pf_list[0][y_trait]],c=[i for i in dat_pheno_pf_list[0]['%s_%s_pf'%(x_trait,y_trait)]],cmap=longyuan_kubei_cmap,s=0.4)
ax1.plot(poly_x_list[0],poly_y_list[0],color=color2,lw=1)
#ax1.tick_params(labelbottom=False)
ax1.set_xlabel('%s'%x_trait)
ax1.set_ylabel('%s'%y_trait)
ax1.set_ylim(0,1)
#ax1.set_title('${y=ax+b}$')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.scatter([i for i in dat_pheno_pf_list[1][x_trait]],[i for i in dat_pheno_pf_list[1][y_trait]],c=[i for i in dat_pheno_pf_list[1]['%s_%s_pf'%(x_trait,y_trait)]],s=0.4,cmap=longyuan_kubei_cmap)
ax2.plot(poly_x_list[1],poly_y_list[1],color=color2,lw=1)
ax2.tick_params(labelleft=False)
ax2.set_xlabel('%s'%x_trait)
#ax2.set_title('${y=ax^2+bx+c}$')
#ax2.set_ylabel('trait 2')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.scatter([i for i in dat_pheno_pf_list[2][x_trait]],[i for i in dat_pheno_pf_list[2][y_trait]],c=[i for i in dat_pheno_pf_list[2]['%s_%s_pf'%(x_trait,y_trait)]],s=0.4,cmap=longyuan_kubei_cmap)
ax3.plot(poly_x_list[2],poly_y_list[2],color=color2,lw=1)
ax3.tick_params(labelleft=False)
ax3.set_xlabel('%s'%x_trait)
#ax3.set_ylabel('trait 2')
#ax3.set_title('${y=ax^3+bx^2+cx+d}$')
fig.tight_layout()
fig.savefig(outputf+'_fit_functions.svg')
fig.clf()

#plot the pf score distribution
fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,figsize=(4.5,1.5),dpi=600)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.hist(dat_pheno_pf_list[0]['%s_%s_pf'%(x_trait,y_trait)],color=color2)
ax1.set_xlabel('IP score')
ax1.set_ylabel('Number')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.hist(dat_pheno_pf_list[1]['%s_%s_pf'%(x_trait,y_trait)],color=color2)
ax2.tick_params(labelleft=False)
ax2.set_xlabel('IP score')
#ax2.set_ylabel('trait 2')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.hist(dat_pheno_pf_list[2]['%s_%s_pf'%(x_trait,y_trait)],color=color2)
ax3.tick_params(labelleft=False)
ax3.set_xlabel('IP score')
#ax3.set_ylabel('trait 2')
fig.tight_layout()
fig.savefig(outputf+'_pf_hist.svg')
fig.clf()

#plot the process for trade-off quantification

for i in [1,2,3]: 
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,figsize=(3,1.5),dpi=600)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.scatter(dat_pheno_mn[x_trait],dat_pheno_mn[y_trait],color=color1,s=0.4)
    ax1.scatter(expect_points_x,expect_points_y,color=color2,s=0.4)
    ax1.set_xlabel('%s'%x_trait)
    ax1.set_ylabel('%s'%y_trait)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.scatter(dat_pheno_mn[x_trait],dat_pheno_mn[y_trait],color=color1,s=0.4)
    ax2.scatter(expect_points_x,expect_points_y,color=color2,s=0.4)
    ax2.plot(poly_x_list[i-1],poly_y_list[i-1],color=color2,lw=1)
    ax2.tick_params(labelleft=False)
    ax2.set_xlabel('%s'%x_trait)
#ax2.set_ylabel('trait 2')

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.scatter([i for i in dat_pheno_pf_list[i-1][x_trait]],[i for i in dat_pheno_pf_list[i-1][y_trait]],c=[i for i in dat_pheno_pf_list[2]['%s_%s_pf'%(x_trait,y_trait)]],s=0.4,cmap=longyuan_kubei_cmap)
    ax3.plot(poly_x_list[i-1],poly_y_list[i-1],color=color2,lw=1)
    ax3.tick_params(labelleft=False)
    ax3.set_xlabel('%s'%x_trait)
#ax3.set_ylabel('trait 2')

    fig.tight_layout()
    fig.savefig(outputf+'_pf_process%d.svg'%i)
    fig.clf()


print('linear function:',pearsonr(dat_pheno_pf_list[0][x_trait],dat_pheno_pf_list[0]['%s_%s_pf'%(x_trait,y_trait)]))
print('quadratic function:',pearsonr(dat_pheno_pf_list[1][x_trait],dat_pheno_pf_list[1]['%s_%s_pf'%(x_trait,y_trait)]))
print('cubic function:',pearsonr(dat_pheno_pf_list[2][x_trait],dat_pheno_pf_list[2]['%s_%s_pf'%(x_trait,y_trait)]))
fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,figsize=(6,1.5),dpi=600)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.scatter(dat_pheno_pf_list[0][x_trait],dat_pheno_pf_list[0]['%s_%s_pf'%(x_trait,y_trait)],color=(182/255,98/255,134/255),s=1)
ax1.set_xlabel('%s'%x_trait)
ax1.set_ylabel('PF score')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.scatter(dat_pheno_pf_list[1][x_trait],dat_pheno_pf_list[1]['%s_%s_pf'%(x_trait,y_trait)],color=(182/255,98/255,134/255),s=1)
ax2.tick_params(labelleft=False)
ax2.set_xlabel('%s'%x_trait)
#ax2.set_ylabel('trait 2')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.scatter(dat_pheno_pf_list[2][x_trait],dat_pheno_pf_list[2]['%s_%s_pf'%(x_trait,y_trait)],color=(182/255,98/255,134/255),s=1)
ax3.tick_params(labelleft=False)
ax3.set_xlabel('%s'%x_trait)
#ax3.set_ylabel('trait 2')
fig.tight_layout()
fig.savefig(outputf+'_pf_xtrait.svg')
fig.clf()



