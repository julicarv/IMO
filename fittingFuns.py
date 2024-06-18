import vars
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib as plt
import lmfit
import models
import dataFuns as dF
import pyDOE

def graphDiff(graph):
    diff = 0
    for i in range(vars.days):
        diff += (graph[i] - vars.ideal[i]) ** 2
    return diff

#LHS, changing only the vars that are set to vary
#Outputs multiple sets of parameters
def lhs(n_samples=100):
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
def residual(params, n):
    allParams = np.empty((n, 3), dtype=np.float16)

    allParams[:, 0] = sp.stats.lognorm.rvs(params.get('gDistr'), scale=np.exp(params.get('g')), size=n)
    allParams[:, 1] = sp.stats.lognorm.rvs(params.get('dDistr'), scale=np.exp(params.get('d')), size=n)
    allParams[:, 2] = sp.stats.lognorm.rvs(params.get('rDistr'), scale=np.exp(params.get('r')), size=n)

    sizes = models.gdr(allParams, True)
    sizes = dF.transform(sizes)
    progs = dF.findProgs(sizes)
    KM = dF.makeKM(progs)

    diff = graphDiff(KM)
    
    return diff