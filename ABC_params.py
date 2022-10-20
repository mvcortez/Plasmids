#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pickle
#from numba import jit
from scipy.special import rel_entr
import random

n =2.5540
##################################################
tau = 60             #60-min doubling time
k_tr = 2.73       
K = 11.35/1.26  
s0 =48.4144/(1.4*n)        #4.62500352/n #WT   #13.58594784/n  #rom-1
##################################################

x0=20
max_size=1000
Nmax =2**10
y0 = np.zeros(2*max_size)
y0[0]=x0
rho = 0.1
#exp_data = [0.04,0.18,0.48,0.24,0.06]   #from paper
exp_data = array([0.04101562, 0.25820312, 0.47617188, 0.19570313, 0.02890625]) #trial sims
time_stamp =500
thresh = 7
epsilon = 0.20
L=2000

# In[13]:
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
        self.div = birth+tau+np.random.normal(0,1)

#@jit
def find_tot_plasmids(y1,y2,max_size):
    y1[0]+=y1[1]   #because of plasmid shedding y[1] may have a nonzero value, we transfer this to y[0].
    y2[0]+=y2[1]
    y1[1]=y1[0]
    y2[1]=y2[0]
    y_tot1 = np.dot(np.arange(1,max_size),y1[1:max_size])
    y_tot2 = np.dot(np.arange(1,max_size),y2[1:max_size])            
    y_tot = y_tot1+y_tot2
    y1[1]=0
    y2[1]=0
    return y1,y2,y_tot1,y_tot2,y_tot 

#@jit
def pop_check(cells,T):
    b_filter = [x for x in cells if x.t[0] <= T]
    d_filter = [x for x in b_filter if x.t[-1] >T]
    return d_filter #N_pop

#@jit
def hazards(y,c_idx,num_reax,N,y_totp,y_tot,beta,gamma,sigma,max_size):
    a =np.zeros(num_reax)                                     #reaction hazards 
    v=np.zeros([num_reax,max_size])                           #reaction vectors    
    a[0]=(k_tr*y_totp)/(1+K*y_tot/(s0*n))**n
    v[0,0] = 1
    a[1] = 0.5*beta*y[0]*(y[0]-1)
    v[1,0] = -2
    v[1,2] = 1

    if N>1:
        species_idx =np.zeros([2,num_reax])
        count=1
        for i in np.arange(1,N):   #plasmid recruitment
            count+=1
            a[count] = beta*y[0]*y[c_idx[i]]
            species_idx[:,count] = np.array([c_idx[i],0])
        rec_count=count
                
        for i in np.arange(1,N):   #cluster combination
            for j in np.arange(i,N):
                count+=1
                if i==j:
                    a[count] = 0.5*gamma*y[c_idx[i]]*(y[c_idx[j]]-1)
                else:
                    a[count] = gamma*y[c_idx[i]]*y[c_idx[j]]
                species_idx[:,count] = [c_idx[i],c_idx[j]]
        cluster_count = count
            
        for i in np.arange(1,N): #plasmid shedding
            count+=1
            a[count] = sigma*y[c_idx[i]]*c_idx[i]
            species_idx[:,count] = np.array([c_idx[i],0])
                    
        for i in range(len(a)): #reaction vectors
            if i==0:                         #plasmid replication
                v[i,0] = 1
            elif i==1:                       #plasmid duplex formation
                v[i,0] = -2
                v[i,2] = 1
            elif (2<=i and i<=rec_count):    #plasmid recruitment to a cluster
                v[i,0] = -1
                v[i,int(species_idx[0,i])] = -1
                v[i,int(species_idx[0,i])+1] = 1
            elif (i>rec_count and i<=cluster_count):   #cluster combination
                if (int(species_idx[0,i])==int(species_idx[1,i])):
                    v[i,int(species_idx[0,i])] = -2
                else:
                    v[i,int(species_idx[0,i])] = -1
                    v[i,int(species_idx[1,i])] = -1
                v[i,int(np.sum(species_idx[:,i]))] = 1
            else:                            #plasmid shedding         
                v[i,0] = 1
                v[i,int(species_idx[0,i])] = -1
                v[i,int(species_idx[0,i])-1] = 1   
    return a,v


#@jit
def ssa(y0,x0,tau,beta,gamma,sigma,max_size):
    y=y0               #plasmid count count; y[0]=x[0], y[1]=0, y[i] = number of cluster of size i     
    x=[x0]
    t_x = [0]
    t_y=0             #reaction time 
    y_tot1=x0
    y_tot2 = 0
    y1 = y[:max_size]
    y2 = y[max_size:2*max_size] 
    y_tot=y_tot1+y_tot2
    while t_y<tau:
        ############################### pole 1 ##################################
        dummy = y1[0]                                               
        y1[0] = 1
        c_idx1 = np.where(y1>0)[0]                                  #indices of species with non-zero state except possibly free plasmids
        y1[0] = dummy
        N1 = len(c_idx1)                                            #number of non-zero species
        num_reax1 = N1+1+int(((N1-1)**2-N1+1)/2)+2*(N1-1)           #possible number of reactions given the system state 
        a1,v1 = hazards(y1,c_idx1,num_reax1,N1,y_tot1,y_tot,beta,gamma,sigma,max_size)     
        
        ############################### pole 2 ##################################
        dummy = y2[0]
        y2[0]=1
        c_idx2 = np.where(y2>0)[0]                                  
        y2[0]=dummy
        N2 = len(c_idx2)
        num_reax2 = N2+1+int(((N2-1)**2-N2+1)/2)+2*(N2-1)  
        a2,v2 = hazards(y2,c_idx2,num_reax2,N2,y_tot2,y_tot,beta,gamma,sigma,max_size)

        a = np.append(a1,a2,axis=0)
        a = np.append(a,[rho*y1[0],rho*y2[0]])
        sum_a=np.sum(a)
        cumsum_a=np.cumsum(a,axis=0)
        epsilon=np.random.uniform(0,1,2)
        j=np.min(np.where(epsilon[0]*sum_a<cumsum_a)) 
        t_f=(np.log(1/epsilon[1]))/sum_a
                     
        if t_y+t_f<tau:  
            t_y+=t_f 
            if (j==num_reax1+num_reax2):
                y2[0]+=1
                y1[0]-=1
            elif (j==num_reax1+num_reax2+1):
                y2[0]-=1
                y1[0]+=1
            elif (num_reax1<=j and j<num_reax1+num_reax2):
                y2+=v2[j-num_reax1,:]  
            else:
                y1+=v1[j,:]    
                
            y1,y2,y_tot1,y_tot2,y_tot = find_tot_plasmids(y1,y2,max_size)
            if (j==0 or j==num_reax1):
                x= np.append(x,y_tot)
                t_x =np.append(t_x,[t_y])
        else:
            t_y=tau
            x=np.append(x,x[-1])
            t_x =np.append(t_x,[t_y])  
    y = np.append(y1,y2,axis=0)
    return y,x,t_x

#@jit
def sim_cells(current,beta,gamma,sigma,max_size):
    y0 = current.init_clusters
    y0[1]=y0[0]
    x0 = np.dot(np.arange(1,max_size),y0[1:max_size])
    y0[1]=0 
    if x0>0:
        yy,xx,tt = ssa(y0,x0,current.div-current.birth,beta,gamma,sigma,max_size)
    else:
        yy = y0
        xx = np.array([0,0])
        tt = np.array([0,current.div-current.birth])
    return yy,xx,tt

def cluster_data(time_stamp,thresh,cells,max_size,exp_data):
    gg = pop_check(cells,time_stamp)
    cluster_num = np.zeros(len(gg))
    for i in range(len(gg)):
        yy = gg[i].fin_clusters
        cluster_num[i] = np.sum(yy[thresh:max_size])+np.sum(yy[max_size+thresh:])    
    bin_count = np.bincount(cluster_num.astype(int))
    if len(bin_count)>=5:
        bin_count_last = np.sum(bin_count[4:])
        bins = bin_count[:5]
        bins[4] =bin_count_last
    else:
        bins = np.zeros(5)
        bins[:len(bin_count)]=bin_count
    heights = bins/len(cluster_num)
    #print(heights)
    #print(exp_data)
    KLD = sum(rel_entr(heights,exp_data))
    return KLD 

def pop_sim(Q,tau,beta,gamma,sigma,max_size,Nmax):
    cells =[]  #processed cells
    N=0  #cell index
    keep = 1500  #keep number of living cells
    time_stamp = (np.log2(keep)+1)*60 +30 # approximate time "keep" number of cells are produced  
    while len(cells)<=Nmax:
        current = Q[0]
        yy,xx,tt = sim_cells(current,beta,gamma,sigma,max_size)
        #update current cell
        current.fin_clusters = yy
        current.x = xx
        current.t = tt+current.birth   
        cells.extend([current])  
            
        if len(Q)>=keep:
            rem_cells = random.sample(Q[1:],1)  #cells to be removed
            new_Q = [x for x in Q if x not in rem_cells]
            Q = new_Q.copy()
        
        Q.remove(current)  #remove current cell from Q
        
        #generate daughter cells  [index,init_clusters,parent,birth,tau]
        daughter1 = cell(N+1,np.append(yy[:max_size],np.zeros(max_size)),current.index,current.t[-1],tau)
        daughter2 = cell(N+2,np.append(yy[max_size:],np.zeros(max_size)),current.index,current.t[-1],tau)
        N+=2
        Q.extend([daughter1,daughter2])  #add daugthers to Q
        Q = sorted(Q, key=lambda x: x.div, reverse=False) #sort Q wrt cell birth time
        
        #if len(cells)%201==0:
        #    print('cells',len(cells),'Q',len(Q))  
    return cells,Q


# In[16]:
beta = np.zeros(L)
sigma = np.zeros(L)
gamma = np.zeros(L)

beta[0]=0.05
sigma[0]=0.05
gamma[0]=0.05

beta_std = 0.01
sigma_std = 0.01
gamma_std = 0.01

for i in range(L-1):
    beta_prop = -1
    while beta_prop<0:
        beta_prop = beta[i]+np.random.normal(0,beta_std)
    c= cell(0,y0,[],0,tau)   #index,clusters,parent,birth
    Q=[c]
    cells, Q = pop_sim(Q,tau,beta_prop,gamma[i],sigma[i],max_size,Nmax)
    KLD = cluster_data(time_stamp,thresh,cells,max_size,exp_data)
    if np.abs(KLD)<epsilon:
        beta[i+1]=beta_prop
    else:
        beta[i+1]=beta[i]
    
    gamma_prop = -1
    while gamma_prop<0:
        gamma_prop = gamma[i]+np.random.normal(0,gamma_std)
    c= cell(0,y0,[],0,tau)   #index,clusters,parent,birth
    Q=[c]
    cells, Q = pop_sim(Q,tau,beta[i+1],gamma_prop,sigma[i],max_size,Nmax)
    KLD = cluster_data(time_stamp,thresh,cells,max_size,exp_data)
    if np.abs(KLD)<epsilon:
        gamma[i+1]=gamma_prop
    else:
        gamma[i+1]=gamma[i]

    sigma_prop = -1
    while sigma_prop<0:
        sigma_prop = sigma[i]+np.random.normal(0,sigma_std)
    c= cell(0,y0,[],0,tau)   #index,clusters,parent,birth
    Q=[c]
    cells, Q = pop_sim(Q,tau,beta[i+1],gamma[i+1],sigma_prop,max_size,Nmax)
    KLD = cluster_data(time_stamp,thresh,cells,max_size,exp_data)
    if np.abs(KLD)<epsilon:
        sigma[i+1]=sigma_prop
    else:
        sigma[i+1]=sigma[i]

    print('beta',beta[i+1],'gamma',gamma[i+1],'sigma',sigma[i+1])
        

columns=' '
np.savetxt("beta.csv", beta, delimiter=",",header=columns)
np.savetxt("gamma.csv", gamma, delimiter=",",header=columns)
np.savetxt("sigma.csv", sigma, delimiter=",",header=columns)

# In[ ]:




