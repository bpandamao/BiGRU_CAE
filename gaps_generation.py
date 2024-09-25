import numpy as np

# Set the lambda parameter for the exponential distribution
def ge_unsche(t,delta_t=5,lambda_value = 0.75):
    w = np.ones(int(np.random.exponential(scale=1/lambda_value))*24*60*60/delta_t)
    while len(w) <= len(t):
        durations = int(np.random.uniform(4,8))*60*60
        tt = np.zeros(int(durations/delta_t)) 
        w = np.append(w, tt)
        w = np.append(w, np.ones(int(np.random.exponential(scale=1/lambda_value))*24*60*60/delta_t))
    return w[:len(t)]

def ge_sche(t,delta_t=5,fix=3.5, period=7):
    w=np.ones(np.random.randint(0,period*24*60*60/delta_t))
    while len(w) <= len(t):
        os_array = np.zeros(fix*60*60/delta_t)
        w = np.append(w, os_array)
        w = np.append(w, np.ones(period*24*60*60/delta_t))
    return w[:len(t)]

def ge_gap(t):
    w1=ge_sche(t)
    w2=ge_unsche(t)
    return w1*w2