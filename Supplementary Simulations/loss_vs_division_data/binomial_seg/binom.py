# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:43:58 2025

@author: Jayson Cortez
"""

import numpy as np
import warnings
import pandas as pd
import time
import pickle


####### replication only, binomial segregation ############
##################################################
class cell:
    def __init__(self,index,init_clusters,parent,birth,tau):
        '''Initializes cell attributes
        param:: index: cell index
        param:: clusters: count of n-clusters
        param:: x: plasmid trajectory in time 
        param:: t: time stamps for x
        param:: parent: parent cell
        param:: birth: birth time
        return:: None
        '''
        self.index=index
        self.init_clusters=init_clusters
        self.fin_clusters=[]
        self.x = []
        self.t = []
        self.parent = parent
        self.birth = birth
        self.div = birth+tau

def pop_check(cells,T):
    b_filter = [x for x in cells if x.t[0] <= T]
    d_filter = [x for x in b_filter if x.t[-1] >T]
    return d_filter #N_pop


def time_approx(s,xi,y_tot,eps):
    D=k_tr*xi
    V=V0*np.exp(mu*s)
    C=K*y_tot/(s0*6.022e2*n)
    a_prime = (mu*n*D*C/V)*(1+C/V)**(-n-1)
    a = D/(1+C/V)**n 
    return (-a+np.sqrt(a**2-2*a_prime*np.log(eps)))/a_prime

def ssa_vol(y0,x0,tau,beta,gamma,sigma,max_size):
    t_x= [0]
    x = [x0]
    i=0
    #Tj= 0
    #Sj = np.log(1.0/np.random.uniform(0,1,1))
    while t_x[i]<tau:
        epsilon=np.random.uniform(0,1)
        t_j = time_approx(t_x[-1],x[-1],x[-1],epsilon)
        t_x= np.append(t_x,t_x[-1]+t_j)
        x=np.append(x,[x[i]+1])
        i+=1
    y=np.zeros(max_size)
    y[0]=x[-1]
    return y,x,t_x

def sim_cells(current,beta,gamma,sigma,max_size):
    y0 = current.init_clusters
    y0[1]=y0[0]
    x0 = np.dot(np.arange(1,max_size),y0[1:max_size])
    y0[1]=0 
    if x0>0:
        yy,xx,tt = ssa_vol(y0,x0,current.div-current.birth,beta,gamma,sigma,max_size)
    else:
        yy = y0
        xx = np.array([0,0])
        tt = np.array([0,current.div-current.birth])
        clust = np.zeros(max_size)
        ty = np.array([0,current.div-current.birth])
    return yy,xx,tt

def lineage_sim(c,tau,beta,gamma,sigma,max_size):
    Line=[]
    current_cell = c 
    N=0
    length = 1
    status =1
    copy_n=[]
    while status == 1:
        #print('############## new cell ##############')
        zz0 = np.copy(current_cell.init_clusters) 
        yy,xx,tt= sim_cells(current_cell,beta,gamma,sigma,max_size)
        current_cell.fin_clusters = yy
        current_cell.x = xx
        current_cell.t = tt+current_cell.birth  
        current_cell.init_clusters = zz0

        #Line.extend([current_cell])
        copy_n.append([xx[-1]])
        rand = np.random.binomial(xx[-1],0.5,1)[0]
        if np.random.uniform(0,1,1)<0.5:
            x0 = rand
        else:
            x0= xx[-1]-rand
        
        yy=np.zeros(max_size)
        yy[0]=x0
    
        daughter = cell(N+1,yy,current_cell.index,current_cell.t[-1],tau)
        
        if current_cell.x[-1]==x0 or x0==0 or length==5000:
            status = 0
        else:
            length+=1
            status = 1
            current_cell = daughter
            N+=1
    return copy_n


##################################################
n =  2.56  #4.84 #2.56 #8.85
s0 = 0.034 #0.06 #0.034 #0.08  
tau = 40             #30-min doubling time
k_tr = 2.84    
K = 14.5/1.26  
mu = np.log(2)/tau
V0 = 0.8
rho = 0.13

x0=20          #number of plasmids in the initial cell
max_size=200   #total number of possible clusters, can be increased to accomodate larger sims

true_beta = 0.015
true_sigma = 0.08
true_gamm = 0.05

##################################################

warnings.filterwarnings('ignore')
repeats = 1
loss = np.zeros(repeats)
start_time = time.perf_counter()
for k in range(repeats):
    y0 = np.zeros(2*max_size)
    y0[0]=x0
    c= cell(0,y0,[],0,tau)   #index,clusters,parent,birth
    copy_n = lineage_sim(c,tau,true_beta,true_gamm,true_sigma,max_size)
print('Loss prob',np.mean(loss),'err',np.std(loss))
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
np.savetxt("tau"+str(tau)+".csv", copy_n, delimiter=",",header=" ")
