#!/usr/bin/python
#
# simulate an OU process.  using van den Berg's solution.  

import numpy as np

# helper methods for example code

def OUProcess(x0=0,mu_n=0,lambda_n=0.1,sigma_n=0.1,dt_n=0.01):
    ''' Simulate OU process using van den Bergs methods'''

    # constants
    x_l = [x0]
    t_max_n = 100.
    t_n = 0
    t_l = [0]
    # constants
    exp_n = np.exp(-lambda_n*dt_n)
    while t_l[-1] < (t_max_n + dt_n):
        x_t0_n = x_l[-1]
        n_n = np.random.randn()
        term1_n = x_t0_n*exp_n
        term2_n = mu_n*(1-exp_n)
        term3_n = sigma_n*np.sqrt((1.-np.power(exp_n,2))/(2*lambda_n))*n_n
        x_t1_n = term1_n + term2_n + term3_n
        x_l.append(x_t1_n)
        t_l.append(t_l[-1] + dt_n)
    #endwhile
    x_v = np.array(x_l)
    t_v = np.array(t_l)
    return x_v, t_v
#enddef


def JSD(v1,v2):
    ''' compute jensen-shannon divergence '''
    half=(v1+v2)/2
    kl_1_v = v1*np.log2(v1/half)
    kl_2_v = v2*np.log2(v2/half)
    kl_1_v[np.isnan(kl_1_v)] = 0.
    kl_2_v[np.isnan(kl_2_v)] = 0.
    return 0.5*sum(kl_1_v)+0.5*sum(kl_2_v)
#enddef


def WeightedMedian(data_M,weight_M,i):
    ''' compute weighted median on data matrix.  modeled on
    R weighted.median function. '''

    # find samples
    sample_v = np.nonzero(~np.isnan(data_M[:,0]))[0]
    sample_i = np.nonzero(sample_v==i)[0][0]
    sample_M = data_M[sample_v,:]
    weight_v = weight_M[i,sample_v]

    # sort each matrix column
    sort_M = np.argsort(sample_M,0)

    # find halfway points for weights in each column
    otu_n = sort_M.shape[1]
    exp_v = np.zeros((1,otu_n))[0]
    for col_i in range(otu_n):
        mid_i = np.nonzero(np.cumsum(weight_v[sort_M[:,col_i]]) > 0.5)[0][0]
        median_n = sample_M[sort_M[mid_i,col_i],col_i]
        exp_v[col_i] = median_n
    #endfor
    
    # return vector of values
    return exp_v

#enddef
