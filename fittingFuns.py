from math import sqrt
import vars
import numpy as np
import lmfit
import dataFuns as dF
import pyDOE

def kmDiff(graph):
    diff = 0
    for i in range(vars.days):
        diff += (graph[i] - vars.idealKM[i]) ** 2
        
    return diff

#LHS, changing only the vars that are set to vary. Only for KM
#Outputs multiple sets of parameters
def lhsKM(n_samples=100):
    varying_params = [p for p in vars.params.values() if p.vary]
    bounds = [(p.min, p.max) for p in varying_params]
    samples = pyDOE.lhs(len(varying_params), samples=n_samples)
    scaled_samples = np.zeros_like(samples)
    
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = samples[:, i] * (high - low) + low

    output = []
    for sample in scaled_samples:
        sample_params = lmfit.Parameters()
        for p_name in vars.params:
            sample_params.add(vars.params[p_name])

        for i, param_name in enumerate([p.name for p in varying_params]):
            sample_params[param_name].value = sample[i]
        output.append(sample_params)
    
    return output

#Objective function
def residualKM(params):
    KM = dF.parsToKM(params)
    diff = kmDiff(KM)
    return diff

#def lhsSize(n_samples=200):
    pars = lmfit.Parameters()
    params.add('g', value=0.02, min=0.01, max=0.06, vary=True)
    params.add('d', value=0.15, min=0.03, max=0.25, vary=True)
    params.add('r', value=0.04, min=0, max=0.05, vary=True)


#def residualSize(params, ideal):
    size = 
    for i in range(vars.days):
