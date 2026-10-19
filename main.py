import numpy as np
import warnings
import pickle
import time

##################################################
class cell:
    def __init__(self,index,init_clusters,parent,birth,tau):
        """Initialize a cell in the simulated lineage.

        Parameters
        ----------
        index : int
            Identifier assigned to this cell.
        init_clusters : array-like
            Initial state vector for cluster counts at the two cell poles.
            The state is stored as two consecutive blocks of length
            ``max_size`` by the simulation code.
        parent : int or list
            Identifier of the parent cell (the initial cell uses ``[]``).
        birth : float
            Absolute simulation time at which this cell is born.
        tau : float
            Nominal cell-division interval. The actual division time is
            perturbed by a normally distributed random offset.

        Returns
        -------
        None
        """
        self.index=index
        self.init_clusters=init_clusters
        self.fin_clusters=[]
        self.x = []
        self.t = []
        self.parent = parent
        self.birth = birth
        self.div = birth+tau+np.random.normal(0,1)

# Reconcile the free-plasmid bookkeeping state at each pole and calculate
# total plasmid counts for pole 1, pole 2, and the whole cell.
# The function modifies y1 and y2 in place before returning them.
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

# Return cells that have been born by time T and whose recorded trajectory
# extends beyond T. This helper is not called elsewhere in the shown script.
def pop_check(cells,T):
    b_filter = [x for x in cells if x.t[0] <= T]
    d_filter = [x for x in b_filter if x.t[-1] >T]
    return d_filter #N_pop

# Build the reaction-hazard vector and state-change matrix for one pole.
# Each row of v corresponds to a possible reaction; each column represents
# a cluster-size state. The hazards depend on the current state and rates.
def hazards(y,c_idx,num_reax,N,beta,gamma,sigma,max_size,t_f,y_totp,y_tot): 
    a = np.zeros(num_reax)                                     #reaction hazards 
    v = np.zeros([num_reax,max_size])                           #reaction vectors    
    #V = V0*np.exp(mu*t_f)
    a[0]= 1 #(k_tr*y_totp)/(1+K*y_tot/(s0*6.022e2*n*V))**n    #not needed since computation of reaction time is different 
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
                if (int(species_idx[0,i])==int(species_idx[1,i]) and y[int(species_idx[1,i])]>1 ):
                    v[i,int(species_idx[0,i])] = -2
                    v[i,int(np.sum(species_idx[:,i]))] = 1
                elif not(int(species_idx[0,i])==int(species_idx[1,i])):
                    v[i,int(species_idx[0,i])] = -1
                    v[i,int(species_idx[1,i])] = -1
                    v[i,int(np.sum(species_idx[:,i]))] = 1
            else:  #plasmid shedding         
                if int(species_idx[0,i]) == 2:   
                    v[i,0] = 2
                    v[i,int(species_idx[0,i])] = -1 
                else:
                    v[i,0] = 1
                    v[i,int(species_idx[0,i])] = -1
                    v[i,int(species_idx[0,i])-1] = 1 
    return a,v

# Approximate the waiting time to the next plasmid-replication event while
# accounting for the time-varying cell volume. Uses global model parameters.
def time_approx(s,xi,y_tot,eps):
    D=k_tr*xi
    V=V0*np.exp(mu*s)
    C=K*y_tot/(s0*6.022e2*n)
    #a_prime = (mu*n*D*C/V)*np.exp(-mu*s)*(1+C/V)**(-n-1)
    a_prime = (mu*n*D*C/V)*(1+C/V)**(-n-1)
    a = D/(1+C/V)**n 
    return (-a+np.sqrt(a**2-2*a_prime*np.log(eps)))/a_prime

# Simulate plasmid/cluster reactions in a cell over its lifetime using a
# stochastic simulation loop. The two blocks of y represent the two poles.
# Returns the final cluster state, total-plasmid trajectory and its times,
# the sequence of cluster states, and the reaction-event times.
def ssa_vol(y0,x0,tau,beta,gamma,sigma,max_size):
    y=y0               #plasmid cluster count; y[0]=x[0], y[1]=0, y[i] = number of cluster of size i     
    x=[x0]
    t_x = [0]
    t_y=0             #reaction time 
    y_tot1=x0
    y_tot2 = 0
    clust = [y]
    ty = []
    y1 = y[:max_size]
    y2 = y[max_size:2*max_size] 
    y_tot=y_tot1+y_tot2
    while t_y<tau:
        #print('########## new reaction')
        ############################### pole 1 ##################################    
        if y_tot1>0:
            dummy = y1[0]                                               
            y1[0] = 1
            c_idx1 = np.where(y1>0)[0]                                  #indices of species with non-zero state except possibly free plasmids
            y1[0] = dummy
            N1 = len(c_idx1)                                            #number of non-zero species
            num_reax1 = N1+1+int(((N1-1)**2-N1+1)/2)+2*(N1-1)           #possible number of reactions given the system state 
            a1,v1 = hazards(y1,c_idx1,num_reax1,N1,beta,gamma,sigma,max_size,t_y,y_tot1,y_tot)     
        else:
            num_reax1 = 1
            a1 = [0]
            v1 = np.zeros([1,max_size])
        
        ############################### pole 2 ##################################
        if y_tot2>0:
            dummy = y2[0]
            y2[0]=1
            c_idx2 = np.where(y2>0)[0]                                  
            y2[0]=dummy
            N2 = len(c_idx2)
            num_reax2 = N2+1+int(((N2-1)**2-N2+1)/2)+2*(N2-1)  
            a2,v2 = hazards(y2,c_idx2,num_reax2,N2,beta,gamma,sigma,max_size,t_y,y_tot2,y_tot)
            a = np.append(a1,a2,axis=0)
       
        else:
            num_reax2 = 1
            a2 = [0]
            a = np.append(a1,a2,axis=0)
            v2 = np.zeros([1,max_size])
            
        a = np.append(a,[rho*y1[0],rho*y2[0]])
        epsilon=np.random.uniform(0,1,len(a))
        t_j  = np.zeros(num_reax1+num_reax2+2)
        
        for i in range(len(a)):
            if a[i]==0:
                t_j[i] = np.inf
            else:
                if i == 0: #replication in pole 1
                    t_j[i] = time_approx(t_y,y_tot1,y_tot,epsilon[i])
                elif i == num_reax1:  #replication in pole 2
                    t_j[i] = time_approx(t_y,y_tot2,y_tot,epsilon[i])
                else:
                    t_j[i] = -np.log(epsilon[i])/a[i]
        j=np.argmin(t_j)
        
        if (j==num_reax1+num_reax2): #plasmid transfer pole 1 to 2
            y2[0]+=1
            y1[0]-=1
        elif (j==num_reax1+num_reax2+1): #plasmid transfer pole 2 to 1
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
        t_y+=t_j[j]
        ty =np.append(ty,[t_y]) 
        clust = np.vstack((clust,np.append(y1,y2,axis=0)))
    y = np.append(y1,y2,axis=0)
    return y,x,t_x,clust,ty

# Simulate the intracellular state of one cell from birth to division.
# If the initial plasmid count is zero, the reaction simulation is skipped.
def sim_cells(current,beta,gamma,sigma,max_size):
    y0 = current.init_clusters
    y0[1]=y0[0]
    x0 = np.dot(np.arange(1,max_size),y0[1:max_size])
    y0[1]=0 
    if x0>0:
        yy,xx,tt,clust,ty = ssa_vol(y0,x0,current.div-current.birth,beta,gamma,sigma,max_size)
    else:
        yy = y0
        xx = np.array([0,0])
        tt = np.array([0,current.div-current.birth])
        clust = np.zeros(max_size)
        ty = np.array([0,current.div-current.birth])
    return yy,xx,tt,clust,ty

# Follow one lineage by repeatedly simulating a cell and selecting one of
# its two daughter-pole states at random. The run stops when plasmid count
# is unchanged across division, the selected daughter has no plasmids, or
# the lineage reaches the 5000-cell limit.
def lineage_sim(c,tau,beta,gamma,sigma,max_size):
    Line=[]
    current_cell = c 
    N=0
    length = 1
    status =1
    while status == 1:
        #print('############## new cell ##############')
        zz0 = np.copy(current_cell.init_clusters) 
        yy,xx,tt,clust,ty= sim_cells(current_cell,beta,gamma,sigma,max_size)
        current_cell.fin_clusters = yy
        current_cell.x = xx
        current_cell.t = tt+current_cell.birth  
        current_cell.init_clusters = zz0
        
        Line.extend([current_cell])
        #if len(Line)>20:
        #    Line.remove(Line[0]) 
        
        if np.random.uniform(0,1)<0.5:
            daughter = cell(N+1,np.append(yy[:max_size],np.zeros(max_size)),current_cell.index,current_cell.t[-1],tau)
        else:
            daughter = cell(N+1,np.append(yy[max_size:],np.zeros(max_size)),current_cell.index,current_cell.t[-1],tau)
        x0 = np.dot(np.arange(1,max_size),daughter.init_clusters[1:max_size])+daughter.init_clusters[0]
        
        if current_cell.x[-1]==x0 or x0==0: #or len(Line)==5000:
            status = 0
        else:
            length+=1
            status = 1
            current_cell = daughter
            #print(current_cell.init_clusters)
            N+=1
        
        if len(Line)%50==0:
            print(len(Line))
    return Line#,length,clust,ty

##################################################
# Hill-like exponent used in the volume-dependent replication expression.
n =  2.56  
s0 = 0.034   
tau = 30             #30-min doubling time
k_tr = 2.17  
K = 15.8/1.26  
mu = np.log(2)/tau
V0 = 1.2
rho = 0.13

# Initial number of free plasmids in the starting cell.
x0=25          #number of plasmids in the initial cell
max_size=200   #total number of possible clusters, can be increased to accomodate larger sims

# Rate parameters for plasmid duplex formation/recruitment, cluster
# combination, and plasmid shedding, respectively.
true_beta = 0.015
true_sigma = 0.08
true_gamm = 0.05

##################################################

# Suppress warning messages during the simulation run.
warnings.filterwarnings('ignore')
# Number of independent lineage simulations to run.
repeats = 1
loss = np.zeros(repeats)
start_time = time.perf_counter()
# Initialize the starting cell with x0 free plasmids at pole 1 and simulate
# its lineage. The resulting cell objects are collected in Line.
for k in range(repeats):
    y0 = np.zeros(2*max_size)
    y0[0]=x0
    c= cell(0,y0,[],0,tau)   #index,clusters,parent,birth
    #Line,length,clust,ty = lineage_sim(c,tau,true_beta,true_gamm,true_sigma,max_size)
    Line = lineage_sim(c,tau,true_beta,true_gamm,true_sigma,max_size)
    #loss[k] = 1./length
print('Loss prob',np.mean(loss),'err',np.std(loss))
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
#np.savetxt("cluster_evolution.csv", clust, delimiter=",",header=" ")
#np.savetxt("cluster_time.csv", ty, delimiter=",",header=" ")
#np.savetxt(str(rho)+".csv", loss, delimiter=",",header=" ")

# Save the simulated lineage objects to a pickle file named for tau.
file_name = 'tau'+str(tau)+'.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(Line, file)

# Summarize the final total-plasmid count recorded for each cell in Line.
length = np.zeros(len(Line))
for i in range(len(Line)):
    length[i] = Line[i].x[-1]
    #print(length[i])
print("mean",np.mean(length))
print("CV",np.std(length)/np.mean(length))
    