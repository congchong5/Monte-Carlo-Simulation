# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:16:28 2018

@author: Kuan-Wei
"""

import numpy as np
from scipy.stats import norm

# Set parameters
T=1# maturity time:1 year
r=0.01# interest rates
sigma=0.2
K=100# Strike price
S0=110
m=12 # number of period
delta_t=T/m #t(j)-t(j-1)
def AsianOption(N):
    # Generate different paths
    s=np.zeros(shape=(m+1,N))# Create a metric to store paths
    s[0,]=S0
    # Create a metric to store arithmetic and geometric payoff
    ArithPayoff=np.array([])
    GeoPayoff=np.array([])
    for i in range(N):
        for j in range(1,m+1):
            s[j,i]=s[j-1,i]*np.exp((r-0.5*sigma*sigma)*delta_t+sigma*np.sqrt(delta_t)*np.random.normal(0,1))
        ArithPayoff=np.append(ArithPayoff,max(0,np.mean(s[1:m,i])-K))
        GeoPayoff=np.append(GeoPayoff,max(0,np.exp(np.mean(np.log(s[1:m,i])))-K))
    # Calculate arithmetric and geometric Asian call option price
    ArithCall=np.exp(-r*T)*ArithPayoff
    ArithAsianPrice=np.mean(ArithCall)
    GeoCall=np.exp(-r*T)*GeoPayoff
    #GeoAsianPrice=np.mean(GeoCall)
    # Geometric price of Asian call option by BS-Model
    GBM_T=1/(m*m)*np.sum(range(m+1))
    GBM_sigma_sq=0
    for i in range(m+1):
        GBM_sigma_sq=GBM_sigma_sq+(sigma*sigma)/(m**3*GBM_T)*(2*i-1)*(m+1-i)
        GBM_sigma=np.sqrt(GBM_sigma_sq)
    GBM_delta=(sigma*sigma)/2-(GBM_sigma_sq)/2
    d1=(np.log(S0/K)+(r-GBM_delta+GBM_sigma_sq/2)*GBM_T)/(GBM_sigma*np.sqrt(GBM_T))
    d2=d1-GBM_sigma*np.sqrt(GBM_T)
    GBM_AsianPrice=np.exp(-GBM_delta*GBM_T-r*(T-GBM_T))*S0*norm.cdf(d1)-np.exp(-r*T)*K*norm.cdf(d2)
    # Control variates variance reduction technique implementationm
    b=np.cov(ArithCall,GeoCall)[0,1]/np.var(ArithCall) # Find the optimal b
    ControlArithCall=ArithCall-b*(GeoCall-[GBM_AsianPrice]*N)
    ControlArithAsianPrice=np.mean(ControlArithCall)
    # Calculate the variance 
    OrigVar=np.std(ArithCall,ddof=1)/np.sqrt(N)
    ControlVar=np.std(ControlArithCall,ddof=1)/np.sqrt(N)
    # Return data
    price=[ArithAsianPrice,ControlArithAsianPrice]
    error=[OrigVar,ControlVar]
    rho=np.cov(ArithCall,GeoCall)[0,1]/(np.std(ArithCall)*np.std(GeoCall))
    return(price,error,rho)


for N in [1000,10000,100000,1000000]:
    MonteCarloAsian=AsianOption(N)
    price=MonteCarloAsian[0]
    error=MonteCarloAsian[1]
    rho=MonteCarloAsian[2]
    print('N:',N)
    print('Asian Call Option Price by 1st Algorithm:%.4f'%price[0])
    print('Asian Call Option Price by 2nd Algorithm:%.4f'%price[1])
    print('Error Estimation by 1st Algorithm:%.4f'%error[0])
    print('Error Estimation by 2nd Algorithm:%.4f'%error[1])
    print('Rho:%.4f'%rho)