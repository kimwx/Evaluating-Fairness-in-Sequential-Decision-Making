#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import time
import multiprocessing
import itertools
import sys

def gen_explore_bk(T):
    count = 1
    explore_bk = []
    while len(explore_bk) < T:
        explore_bk = explore_bk+ [1,1]+[0]*count
        count += 2
    return explore_bk


def opt_pt(df):
    dfmd = df.head(len(df)-1)
    #res = minimize(cal_revenue_minus, 0.1, args = (dfmd))
    res = smf.logit(formula='apply ~ -1+Tier+ termclass+ partnerbin+Term+rate+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+epoch:company_price', data=dfmd).fit(disp=0)
    revs = []
    prices = np.linspace(0.02,0.2,10)
    for price in prices:
        df.at[len(df)-1,'company_price'] = price
        prob = res.predict(df.iloc[len(df)-1])
        rev = price*prob.values[0]
        revs.append(rev)
    maxindex = revs.index(max(revs))
    return (prices[maxindex])

def opt_pt_demand_fair(df,phi):
    dfmd = df.head(len(df)-1)
    #res = minimize(cal_revenue_minus, 0.1, args = (dfmd))
    res = smf.logit(formula='apply ~ -1+Tier+ termclass+ partnerbin+Term+rate+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+epoch:company_price', data=dfmd).fit(disp=0)
    revs = []
    prices = np.linspace(0.02,0.2,10)
    for price in prices:
        df.at[len(df)-1,'company_price'] = price
        prob = res.predict(df.iloc[len(df)-1])
        if abs(prob.values[0]-res.predict(df.iloc[len(df)-3]).values[0])<= phi:
            rev = price*prob.values[0]
            revs.append(rev)
        else:
            revs.append(0)
    maxindex = revs.index(max(revs))
    return (prices[maxindex])

def opt_pt_group_price_fair(df,phi):
    dfmd = df.head(len(df)-1)
    res = smf.logit(formula='apply ~ -1+Tier+ termclass+ partnerbin+Term+rate+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+epoch:company_price', data=dfmd).fit(disp=0)
    revs = []
    prices = np.linspace(0.02,0.2,10)
    for price in prices:
        df.at[len(df)-1,'company_price'] = price
        groupedprices = df.groupby('State')['company_price'].mean()
        if max(groupedprices)-min(groupedprices)<= phi:
            prob = res.predict(df.iloc[len(df)-1])
            rev = price*prob.values[0]
            revs.append(rev)
        else:
            revs.append(0)
    maxindex = revs.index(max(revs))
    return (prices[maxindex])

def opt_pt_group_demand_fair(df,phi):
    dfmd = df.head(len(df)-1)
    res = smf.logit(formula='apply ~ -1+Tier+ termclass+ partnerbin+Term+rate+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+epoch:company_price', data=dfmd).fit(disp=0)
    revs = []
    probs = []
    prices = np.linspace(0.02,0.2,10)
    for price in prices:
        df.at[len(df)-1,'company_price'] = price
        prob = res.predict(df.iloc[len(df)-1])
        df.at[len(df)-1,'apply'] = prob.values[0]
        groupeddemand = df.groupby('State')['apply'].mean()
        if max(groupeddemand)-min(groupeddemand)<= phi:
            rev = price*prob.values[0]
            revs.append(rev)
            probs.append(prob)
        else:
            revs.append(0)
            probs.append(0)
    maxindex = revs.index(max(revs))
    return (prices[maxindex], probs[maxindex])

def ex_predict_demand(df,p):
    dfmd = df.head(len(df)-1)
    res = smf.logit(formula='apply ~ -1+Tier+ termclass+ partnerbin+Term+rate+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+epoch:company_price', data=dfmd).fit(disp=0)
    df.at[len(df)-1,'company_price'] = p
    prob = res.predict(df.iloc[len(df)-1])
    return prob.values[0]
# In[14]:


def ILQX(df,exp_ps,start):
    T = len(df)
    res = smf.logit(formula='apply ~ Tier+ termclass+ partnerbin+Term+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+rate:company_price+epoch:company_price', data=df).fit(disp=0)
    res.summary()
    np.random.seed()
    dfround = df.sample(frac = 1)
    explore_bk = gen_explore_bk(T)
    revenue = 0
    for t in range(start,T):
        if explore_bk[t-start] == 1:
            pt = exp_ps[t%2]
        else:
            pt = opt_pt(dfround.iloc[:t])
        dfround.at[t,'company_price'] = pt
        real_prob = res.predict(dfround.iloc[t])
        revenue += real_prob.values[0]*pt
        dfround.at[t,'apply'] = np.random.binomial(1, real_prob)   
        pmeanstate = dfround.groupby('State')['company_price'].mean()
        applymeanstate = dfround.groupby('State')['apply'].mean()
    return (revenue,pmeanstate.values,applymeanstate.values)


def project_price(pt,p0,phi):
    if abs(pt-p0) <= phi:
        return pt
    else:
        if pt>p0:
            return p0+phi
        else:
            return p0-phi
        
def ILQX_price_fair(df,exp_ps,start,phi):
    T = len(df)
    res = smf.logit(formula='apply ~ Tier+ termclass+ partnerbin+Term+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+rate:company_price+epoch:company_price', data=df).fit(disp=0)
    res.summary()
    np.random.seed()
    dfround = df.sample(frac = 1)
    explore_bk = gen_explore_bk(T)
    revenue = 0
    for t in range(start,T):
        if explore_bk[t-start] == 1:
            pt = exp_ps[t%2]
        else:
            pt = opt_pt(dfround.iloc[:t])
            pt = project_price(pt,dfround.iloc[t-2]['company_price'],phi)
        dfround.at[t,'company_price'] = pt
        real_prob = res.predict(dfround.iloc[t])
        revenue += real_prob.values[0]*pt
        dfround.at[t,'apply'] = np.random.binomial(1, real_prob)   
    return (revenue) 

def ILQX_group_price_fair(df,exp_ps,start,phi):
    T = len(df)
    res = smf.logit(formula='apply ~ Tier+ termclass+ partnerbin+Term+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+rate:company_price+epoch:company_price', data=df).fit(disp=0)
    res.summary()
    np.random.seed()
    dfround = df.sample(frac = 1)
    explore_bk = gen_explore_bk(T)
    revenue = 0
    for t in range(start,T):
        if explore_bk[t-start] == 1:
            pt = exp_ps[t%2]
        else:
            pt = opt_pt_group_price_fair(dfround.iloc[:t],phi)
        dfround.at[t,'company_price'] = pt
        real_prob = res.predict(dfround.iloc[t])
        revenue += real_prob.values[0]*pt
        dfround.at[t,'apply'] = np.random.binomial(1, real_prob)   
#         pmeanstate = dfround.groupby('State')['company_price'].mean()
#         applymeanstate = dfround.groupby('State')['apply'].mean()
    return (revenue) #,pmeanstate.values,applymeanstate.values)

def ILQX_demand_fair(df,exp_ps,start,phi):
    T = len(df)
    res = smf.logit(formula='apply ~ Tier+ termclass+ partnerbin+Term+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+rate:company_price+epoch:company_price', data=df).fit(disp=0)
    res.summary()
    np.random.seed()
    dfround = df.sample(frac = 1)
    explore_bk = gen_explore_bk(T)
    revenue = 0
    for t in range(start,T):
        if explore_bk[t-start] == 1:
            pt = exp_ps[t%2]
        else:
            pt = opt_pt_demand_fair(dfround.iloc[:t],phi)
        dfround.at[t,'company_price'] = pt
        real_prob = res.predict(dfround.iloc[t])
        revenue += real_prob.values[0]*pt
        dfround.at[t,'apply'] = np.random.binomial(1, real_prob)   
    return (revenue)



def ILQX_group_demand_fair(df,exp_ps,start,phi):
    T = len(df)
    res = smf.logit(formula='apply ~ Tier+ termclass+ partnerbin+Term+onemonth+rate1+epoch+Tier:company_price+ termclass:company_price+ partnerbin:company_price+Term:company_price+rate:company_price+epoch:company_price', data=df).fit(disp=0)
    res.summary()
    np.random.seed()
    dfround = df.sample(frac = 1)
    explore_bk = gen_explore_bk(T)
    revenue = 0
    for t in range(start,T):
        if explore_bk[t-start] == 1:
            pt = exp_ps[t%2]
        else:
            optres = opt_pt_group_demand_fair(dfround.iloc[:t],phi)
            pt = optres[0]
        dfround.at[t,'company_price'] = pt
        real_prob = res.predict(dfround.iloc[t])
        revenue += real_prob.values[0]*pt
        dfround.at[t,'apply'] = np.random.binomial(1, real_prob)   
    return (revenue)

def compute_ILQX_fair(params):
    df = pd.read_csv('datacleaned.csv')  
    #print(params)
    phi = float(params[0])
    np.random.seed()
    exp_ps = [0.1,0.05]
    start = 5000
    rev = ILQX(df,exp_ps,start)
    rev_pf = ILQX_price_fair(df[:1200],exp_ps,start,phi)
    rev_df = ILQX_demand_fair(df[:1200],exp_ps,start,phi)
    rev_gpf = ILQX_group_price_fair(df[:1200],exp_ps,start,phi)
    rev_gdf = ILQX_group_demand_fair(df,exp_ps,start,phi)
    return(phi,rev,rev_pf,rev_df,rev_gpf,rev_gdf)


# In[ ]:
if __name__ == "__main__":
    time_start = time.time()

    print(sys.argv[1])
    print(sys.argv[2])
    start = float(sys.argv[1])
    end = float(sys.argv[2])
    numx  = sys.argv[3]
    print(numx)
    xlist = np.linspace(start,end,int(numx))
    nums = range(int(sys.argv[4]))


    paramlist = list(itertools.product(xlist,nums))
    #print(paramlist)
    #Generate processes equal to the number of cores
    pool = multiprocessing.Pool()
    
    #Distribute the parameter sets evenly across the cores
    res  = pool.map(compute_ILQX_fair,paramlist)
    pool.close()
    pool.join()
    print(res)
    print("--- %s seconds ---" % (time.time() - time_start))

