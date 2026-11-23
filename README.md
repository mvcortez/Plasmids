This is a repository of computer programs used in the simulations in the paper "Plasmid clustering slows redistribution driving a trade-off between growth and plasmid stability"

1. main.py  - The main skeleton program for implementing the nonhomogeneous CTMC
2. Lineage simulation and characterization: contains the program for the visualization of lineage simulations
3. ODE analog model: contains the program corresponding to the deterministic ODE analog of the stochastic model
4. Parameter fitting:
          - Clustering parameter: contains the programs for implementing the ABC algorithm for inferring the clustering parameters
          - Replication parameter: contains the programs for implementing the ABC algorithm for inferring the replication parameters
5. Supplementary Simulations: contains programs for
          - Comparing the Binomial and No-clustering models
          - Determining the effects of clustering on population variance
          - Effects of clustering on inheritance asymmetry
          - Cross analysis of the effects of plasmid copy number and generation time to loss probability    

Notes on Required Data:
1. lineage_tau30.pkl - can be generated using the Lineage code
2. replication.pkl - can be generated using the “effects of clustering on variance” code
3. cluster.pkl - can be generated using main.py and renaming the output accordingly. Set the replication parameters to tau=30.  

      

