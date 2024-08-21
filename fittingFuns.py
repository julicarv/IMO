import math
import models
import vars
import numpy as np
import lmfit
import dataFuns as dF
import pyDOE
import copy
import scipy as sp
import itertools

#Outputs multiple sets of parameters
def lhsKM(n_samples=vars.lhs_km):
    varying_params = [p for p in vars.params.values() if p.vary]
    bounds = [(p.min, p.max) for p in varying_params]
    samples = pyDOE.lhs(len(varying_params), samples=n_samples)
    scaled_samples = np.zeros_like(samples)
    
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = samples[:, i] * (high - low) + low # type: ignore

    output = []
    for sample in scaled_samples:
        sample_params = copy.deepcopy(vars.params)

        for i, param_name in enumerate(p.name for p in varying_params):
            sample_params[param_name].value = sample[i]
        output.append(sample_params)

    return output

def residualKM(params):
    KM = dF.parsToKM(params)
    return sum((KM[i] - vars.idealKM[i]) ** 2 for i in range(vars.days))

#This will soon disappear
def lhsSize(bounds, n_samples=vars.lhs_size):
    params = lmfit.Parameters()
    params.add('g', min=bounds[0][0], max=bounds[0][1])
    params.add('d', min=bounds[1][0], max=bounds[1][1])
    params.add('r', min=bounds[2][0], max=bounds[2][1])
    #bounds = [(params[p_name].min, params[p_name].max) for p_name in params]

    samples = pyDOE.lhs(3, samples=n_samples)
    scaled_samples = np.zeros_like(samples)
    
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = samples[:, i] * (high - low) + low # type: ignore
    
    output = []
    for sample in scaled_samples:
        sample_params = copy.deepcopy(params)
        
        for i, param_name in enumerate(params):
            sample_params[param_name].value = sample[i]
        output.append(sample_params)

    return output

#Deal with NaNs
def residualSize(args):
    params, scanDates, ideal = args
    paramvals = [params[p].value for p in params]
    size = models.gdr(paramvals, scanDates)
    return sum(math.log2(size[i] / ideal[i]) ** 2 for i in range(vars.numScans))

#Currently takes parameters, make it to take in numbers
def bestLHS(diffFun, *args, n=vars.topLHS):
    bestparams_w_diffs = []

    for params in args[0]:
        diffArgs = params if (len(args) == 1) else [params] + list(args[1:])
        currentDiff = diffFun(diffArgs)

        if len(bestparams_w_diffs) < n:
            bestparams_w_diffs.append((params, currentDiff))
            continue

        maxDiff = max(bestparams_w_diffs, key=lambda x: x[1])[1]

        if currentDiff < maxDiff:
            maxIndex = bestparams_w_diffs.index(max(bestparams_w_diffs, key=lambda x: x[1]))
            bestparams_w_diffs[maxIndex] = (params, currentDiff)

    bestparams_w_diffs.sort(key=lambda x: x[1])

    bestparams, diffs = zip(*bestparams_w_diffs)
    return bestparams, diffs

def gSizeBounds(sizes, scanDates, maxB):
    i = len(sizes)-1
    while(i > 1 and sizes[i] > np.log10(vars.maxSize)):
        i -= 1
    
    slope = (sizes[i] - sizes[i - 1]) / (scanDates[i] - scanDates[i - 1])
    uCoeff = 3 if sizes[i - 1] == np.log10(vars.minSize) else 999
    
    #Delete this after you have fixed the math
    uCoeff = 4
    
    uBound = min(uCoeff*slope, maxB)
    return (slope, uBound)       
   
#This will replace both lhs generating functions  
def inHull(hull, n, paramNames=['g', 'd', 'r']):
    dim = len(hull.points[0])
    simplexComplex = sp.spatial.Delaunay(hull.points)
    
    vol = sum(simplexVol(hull, simplex) for simplex in simplexComplex.simplices)
    sampleNs = [math.ceil(simplexVol(hull, simplex) * n / vol) for simplex in simplexComplex.simplices]
    
    samples = []
    for i in range(len(simplexComplex.simplices)):
        points = hull.points[simplexComplex.simplices[i]]
        lhsSamples = pyDOE.lhs(dim+1, samples=sampleNs[i])
        
        for sample in lhsSamples: # type: ignore
            samples.append(sum([sample[i]*points[i] for i in range(dim+1)]) / sum(sample))
    
    samples = numsToPars(samples, paramNames)
    return samples

#Based on the assumption that len(paramNames) != 1
def numsToPars(nums, paramNames):
    if(len(nums[0]) == 1): 
        params = lmfit.Parameters()
        for i in range(len(paramNames)):
            params.add(paramNames[i], value=nums[i])
        return params
    else:
        params = []
        for i in range(len(nums)):
            paramList = lmfit.Parameters()
            
            for j in range(len(paramNames)):
                paramList.add(paramNames[j], value=nums[i][j])
            params.append(paramList)
        return params
    
def parsToNums(params):
    if len(params) == 1:
        return [params[p].value for p in params]
    else:
        return [[params[i][p].value for p in params[i]] for i in range(len(params))]


def hullExpand(hull):
    centroid = np.mean([point for point in hull.vertices], axis=0)
    
    newPoints = hull.vertices + vars.epsilon * (hull.vertices - centroid) / np.linalg.norm(hull.vertices - centroid, axis=1)
    newHull = sp.spatial.ConvexHull(newPoints)
    
    return newHull

def simplexVol(hull, simplex):
    points = hull.points[simplex]
    V = points[1:] - points[0]
    det = np.linalg.det(V)
    volume = abs(det) / math.factorial(len(points)-1)
    return volume

def boundsToPoints(bounds):
    return list(itertools.product(*bounds))

